"""Output-only contour correction and object completion."""
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import Sam2Model, Sam2Processor

from .cues import SAM_MODEL

SAM2_MASK_GRID = 256
AOT_DECODER_DIRECTORY = Path("models/gsedit_scd/sam2_decoder_aot")
FEATURE_TRANSFORM = "symmetric_max_min_per_group"


def symmetric_side_features(features):
    """Make the paired query/reference SAM features side-invariant.

    Input columns are [query(3), reference(3)] for each of the three groups
    (mask logits, predicted IoU, their product), with each side sorted by its
    predicted IoU. Each group becomes [max(query, reference)(3), min(...)(3)],
    so an object present on either side yields the same features. Products are
    formed per side before this transform. Output keeps 18 columns.
    """
    parts = []
    for group in range(3):
        query = features[:, 6 * group:6 * group + 3]
        reference = features[:, 6 * group + 3:6 * group + 6]
        parts.append(torch.maximum(query, reference))
        parts.append(torch.minimum(query, reference))
    return torch.cat(parts, 1)


def affine_log_bf(models):
    """Fold scaler -> logistic -> Platt into log_bf = x @ weight + bias."""
    scaler, classifier, calibrator = models["scaler"], models["classifier"], models["calibrator"]
    coefficient = classifier.coef_[0] / scaler.scale_
    intercept = float(classifier.intercept_[0] - np.dot(scaler.mean_ / scaler.scale_, classifier.coef_[0]))
    platt_a = float(calibrator.coef_[0][0])
    platt_b = float(calibrator.intercept_[0])
    weight = torch.tensor(platt_a * coefficient, dtype=torch.float32, device="cuda")
    bias = platt_a * intercept + platt_b
    return weight, bias


def components_from_support(support):
    """Connected components of the decision support: boxes plus a label image.

    Box convention matches ``exact_components`` in the original evaluator.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(support.astype(np.uint8), connectivity=8)
    height, width = support.shape
    boxes = []
    for label in range(1, count):
        x, y, w, h = stats[label, :4]
        x1, y1 = float(x), float(y)
        x2, y2 = float(x + w - 1), float(y + h - 1)
        if x2 <= x1:
            if x2 < width - 1:
                x2 += 1.0
            else:
                x1 = max(0.0, x1 - 1.0)
        if y2 <= y1:
            if y2 < height - 1:
                y2 += 1.0
            else:
                y1 = max(0.0, y1 - 1.0)
        boxes.append([x1, y1, x2, y2])
    return boxes, labels.astype(np.int32)


@torch.inference_mode()
def preprocess_gpu(processor, target_uint8, reference_float):
    """Run the official Sam2ImageProcessorFast on CUDA uint8 tensors (no CPU round trip)."""
    target = target_uint8.permute(2, 0, 1).contiguous()
    reference = reference_float.float().mul(255.0).round().clamp(0.0, 255.0).to(torch.uint8).contiguous()
    inputs = processor.image_processor(images=[target, reference], return_tensors="pt")
    return inputs["pixel_values"], inputs["original_sizes"].cpu()


def drop_subcell_components(boxes, label_image, height, width):
    """Keep only components larger than one decoder cell in at least one direction; relabel them 1..K in order."""
    cell_w = width / SAM2_MASK_GRID; cell_h = height / SAM2_MASK_GRID
    keep = [k for k, (x1, y1, x2, y2) in enumerate(boxes) if (x2 - x1 + 1.0) > cell_w or (y2 - y1 + 1.0) > cell_h]
    remap = np.zeros(len(boxes) + 1, dtype=np.int32)
    for new, old in enumerate(keep, start=1):
        remap[old + 1] = new
    return [boxes[k] for k in keep], remap[label_image], len(boxes) - len(keep)


def component_peaks(probability, label_image, count):
    """One prompt point per component: its highest-probability pixel (always inside the component)."""
    flat_labels = label_image.ravel(); flat_prob = probability.ravel()
    indices = np.flatnonzero(flat_labels > 0)
    labels = flat_labels[indices]
    values = flat_prob[indices]
    maxima = np.full(count + 1, -np.inf, dtype=flat_prob.dtype)
    np.maximum.at(maxima, labels, values)
    is_peak = values == maxima[labels]
    # Preserve the first pixel in row-major order when maxima are tied.
    peak = np.full(count + 1, flat_labels.size, dtype=np.int64)
    np.minimum.at(peak, labels[is_peak], indices[is_peak])
    ys, xs = np.divmod(peak[1:], probability.shape[1])
    return [[float(x), float(y)] for x, y in zip(xs, ys)]


def load_aot_decoder(directory):
    """Load the four offline-compiled decoder variants; never compile online."""
    runners = {}
    for kind in ("box", "point"):
        for case in ("single", "multiple"):
            folder = Path(directory) / (kind + "_" + case)
            libraries = list(folder.glob("*.so"))
            if len(libraries) != 1:
                raise FileNotFoundError(f"Expected one prepared AOT decoder library in {folder}")
            runners[kind, case] = torch._C._aoti.AOTIModelContainerRunnerCuda(
                str(libraries[0]), 1, "cuda", str(folder)
            )

    def forward(image_embeddings, image_positional_embeddings, sparse_prompt_embeddings,
                dense_prompt_embeddings, multimask_output, high_resolution_features,
                attention_similarity=None, target_embedding=None, **kwargs):
        kind = "box" if image_embeddings.shape[0] == 2 else "point"
        case = "single" if sparse_prompt_embeddings.shape[1] == 1 else "multiple"
        tensors = (image_embeddings, image_positional_embeddings, sparse_prompt_embeddings,
                   dense_prompt_embeddings, *high_resolution_features)
        return tuple(runners[kind, case].run([value.contiguous() for value in tensors]))

    return forward


class ObjectRefiner:
    def __init__(self, checkpoint, chunk=16, aot_directory=None):
        if aot_directory is not None and not 1 <= chunk <= 64:
            raise ValueError("The prepared AOT decoder supports prompt batches from 1 to 64.")
        with open(checkpoint, "rb") as stream:
            models = pickle.load(stream)
        if models.get("feature_transform") != FEATURE_TRANSFORM:
            raise ValueError(f"{checkpoint} was not fitted on {FEATURE_TRANSFORM} features; "
                             "refit the contour classifier with gsedit_scd.train scene --contour.")
        self.weight, self.bias = affine_log_bf(models)
        self.processor = Sam2Processor.from_pretrained(SAM_MODEL, local_files_only=True)
        self.model = Sam2Model.from_pretrained(
            SAM_MODEL, local_files_only=True, dtype=torch.float16
        ).eval().cuda()
        self.chunk = chunk
        if aot_directory is not None:
            self.model.mask_decoder.forward = load_aot_decoder(aot_directory)

    @torch.inference_mode()
    def warmup(self):
        """Warm SAM with synthetic inputs without consuming the inference RNG state."""
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            target = torch.zeros((1024, 1024, 3), dtype=torch.uint8, device=self.weight.device)
            reference = torch.zeros((3, 1024, 1024), device=self.weight.device)
            for _ in range(2):
                pixels, sizes = preprocess_gpu(self.processor, target, reference)
                embeddings = self.model.get_image_embeddings(pixels.to(dtype=self.model.dtype))
                for count in (1, min(3, self.chunk)):
                    boxes = [[[128., 128., 896., 896.]] * count] * 2
                    prompts = self.processor(original_sizes=sizes, input_boxes=boxes, return_tensors="pt")
                    decoded = self.model(image_embeddings=embeddings, multimask_output=True,
                                         input_boxes=prompts["input_boxes"].to(self.weight.device))
                    self.processor.post_process_masks(decoded.pred_masks.float(), sizes, binarize=False)
                    prompts = self.processor(original_sizes=sizes[:1],
                        input_points=[[[[512., 512.]]] * count], input_labels=[[[1]] * count], return_tensors="pt")
                    decoded = self.model(image_embeddings=[x[:1] for x in embeddings], multimask_output=True,
                        input_points=prompts["input_points"].to(self.weight.device),
                        input_labels=prompts["input_labels"].to(self.weight.device))
                    self.processor.post_process_masks(decoded.pred_masks.float(), sizes[:1], binarize=False)
            torch.cuda.synchronize()

    @torch.inference_mode()
    def __call__(self, result, target_uint8, reference, prior_log_odds):
        probability = result["probability"].clone()
        support = (probability >= 0.5).cpu().numpy()
        boxes, labels = components_from_support(support)
        boxes, labels, _ = drop_subcell_components(boxes, labels, *support.shape)
        if not boxes:
            return probability
        device = probability.device
        pixel_values, sizes = preprocess_gpu(self.processor, target_uint8, reference.clamp(0, 1))
        embeddings = self.model.get_image_embeddings(pixel_values.to(device, dtype=self.model.dtype))
        label_image = torch.from_numpy(labels).to(device)
        logit = torch.log(probability.clamp_min(1e-12)) - torch.log1p(-probability.clamp_max(1 - 1e-12))

        # Correct boundaries using paired query/reference box prompts.
        for start in range(0, len(boxes), self.chunk):
            chunk = boxes[start:start + self.chunk]
            prompts = self.processor(original_sizes=sizes, input_boxes=[chunk, chunk], return_tensors="pt")
            decoded = self.model(image_embeddings=embeddings, multimask_output=True,
                                 input_boxes=prompts["input_boxes"].to(device))
            masks = self.processor.post_process_masks(decoded.pred_masks.float(), sizes, binarize=False)
            scores = decoded.iou_scores.float()
            order = torch.argsort(scores, dim=-1, descending=True, stable=True)
            sorted_scores = torch.gather(scores, 2, order)
            ys, xs = torch.nonzero((label_image > start) & (label_image <= start + len(chunk)), as_tuple=True)
            if not ys.numel():
                continue
            component = label_image[ys, xs] - 1 - start
            mask_logits, quality = [], []
            for side in (0, 1):
                raw = masks[side][component, :, ys, xs]
                mask_logits.append(torch.gather(raw, 1, order[side][component]))
                quality.append(sorted_scores[side][component])
            mask_logits, quality = torch.cat(mask_logits, 1), torch.cat(quality, 1)
            features = symmetric_side_features(torch.cat((mask_logits, quality, mask_logits * quality), 1))
            correction = (features @ self.weight + self.bias).clamp_max(0)
            values = probability[ys, xs]
            probability[ys, xs] = torch.where(
                values < 1, torch.sigmoid(logit[ys, xs] + correction), values
            )

        # Complete surviving objects using target-only point prompts. Their
        # evidence is read from the pixel inference result, never written back.
        support = (probability >= 0.5).cpu().numpy()
        boxes, labels = components_from_support(support)
        boxes, labels, _ = drop_subcell_components(boxes, labels, *support.shape)
        if not boxes:
            return probability
        points = component_peaks(probability.cpu().numpy(), labels, len(boxes))
        target_embeddings = [embedding[:1] for embedding in embeddings]
        evidence = 2 * result["score"]
        covered = result["covered"]
        threshold = -prior_log_odds
        fill = torch.zeros_like(probability)
        for start in range(0, len(points), self.chunk):
            chunk = points[start:start + self.chunk]
            prompts = self.processor(original_sizes=sizes[:1], input_points=[[[point] for point in chunk]],
                                     input_labels=[[[1] for _ in chunk]], return_tensors="pt")
            decoded = self.model(image_embeddings=target_embeddings, multimask_output=True,
                                 input_points=prompts["input_points"].to(device),
                                 input_labels=prompts["input_labels"].to(device))
            masks = self.processor.post_process_masks(decoded.pred_masks.float(), sizes[:1], binarize=False)[0]
            quality = decoded.iou_scores.float()[0]
            binary = masks > 0
            binary = binary[torch.arange(len(chunk), device=device), quality.argmax(1)][:, None]
            inside = binary & covered[None, None]
            count = inside.flatten(2).sum(-1).float()
            pooled = (evidence[None, None] * inside).flatten(2).sum(-1) / count.clamp_min(1)
            passing = (count > 0) & (pooled >= threshold)
            if bool(passing.any()):
                value = torch.where(passing, torch.sigmoid(pooled - threshold), 0)
                fill = torch.maximum(fill, (value[:, :, None, None] * inside.float()).flatten(0, 1).amax(0))
        return torch.maximum(probability, fill)
