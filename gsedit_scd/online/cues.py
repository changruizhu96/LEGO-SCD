"""The retained O-SCD photometric and SAM2 cues."""
import torch
import torch.nn.functional as F

from gsedit_scd.vendor.oscd.utils.loss_utils import ssim

SAM_MODEL = "facebook/sam2.1-hiera-tiny"


@torch.no_grad()
def frame_cues(original_image, rendered_image, sam, height, width,
               *, normalize_before_resize=False, image_embeddings=None):
    ssim_map = ssim(rendered_image, original_image, window_size=11, map=True).mean(dim=0)
    l1_image = torch.abs(rendered_image - original_image).mean(dim=0)
    photo = 0.8 * l1_image + (1.0 - 0.8) * (1.0 - ssim_map)
    vmin, vmax = photo.min(), photo.max()
    raw_photo = photo.float()
    photo = (photo - vmin) / (vmax - vmin + 1e-8)
    if image_embeddings is None:
        rendered = F.interpolate(rendered_image.unsqueeze(0), size=(1024, 1024), mode="nearest").squeeze()
        original = F.interpolate(original_image.unsqueeze(0), size=(1024, 1024), mode="nearest").squeeze()
        with torch.inference_mode():
            image_embeddings = sam.get_image_embeddings(torch.stack([rendered, original]).half())[-1]
    raw_feature = (image_embeddings[0] - image_embeddings[1]).abs().mean(dim=0).float()
    # Suppression training normalizes on the SAM grid; likelihood fitting
    # and online inference retain normalization on the image grid.
    if normalize_before_resize:
        feature = (raw_feature - raw_feature.min()) / (raw_feature.max() - raw_feature.min() + 1e-8)
        feature = F.interpolate(feature[None, None], size=(height, width), mode="bilinear", align_corners=False).squeeze()
    raw_feature = F.interpolate(raw_feature[None, None], size=(height, width), mode="bilinear", align_corners=False).squeeze()
    if not normalize_before_resize:
        feature = (raw_feature - raw_feature.min()) / (raw_feature.max() - raw_feature.min() + 1e-8)
    # This second rescaling is part of the frozen cue implementation.
    feature = ((feature - vmin) / (vmax - vmin + 1e-8)).clamp(0, 1)
    candidate = (2.0 * (0.5 * photo + 0.5 * feature)).unsqueeze(0)
    return candidate, photo.unsqueeze(0), raw_photo, raw_feature
