"""Reference-image observations shared by likelihood and prior fitting."""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from gsedit_scd.data.mask_geometry import find_image
from gsedit_scd.online.cues import frame_cues
from gsedit_scd.online.pipeline import cached_view


class ReferenceObservations:
    def __init__(self, source, camera_cache, gaussians, sam):
        self.source, self.camera_cache = Path(source), Path(camera_cache)
        self.gaussians, self.sam = gaussians, sam
        self.manifest = json.loads((self.camera_cache / "manifest.json").read_text())
        self.height, self.width = self.manifest["height"], self.manifest["width"]
        self.records = {row["image_name"]: row for row in self.manifest["frames"]}
        self.pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
        self.background = torch.zeros(3, device=gaussians.get_xyz.device)
        self.views = {}
        self.reference_images = {}
        # Enabled only for a scene fit's likelihood-to-prior pass. CPU tensors
        # retain the exact dtype; no image encoding or quantization is applied.
        self.measurements = None

    def rgb(self, path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        if image.shape[:2] != (self.height, self.width):
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().to(self.background.device) / 255

    def mask(self, path, soft=False, *, size=None):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED if soft else cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.resize(image, size if size is not None else (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        if soft:
            image = image.astype(np.float32)
            if float(image.max(initial=0.0)) > 1.0:
                image /= 65535.0
            image = np.clip(image, 0, 1)
        else:
            image = image > 127
        return torch.from_numpy(image).to(self.background.device)

    def view(self, name):
        if name not in self.views:
            payload = torch.load(self.camera_cache / self.records[name]["file"], map_location="cpu", weights_only=True)
            view = cached_view(payload, self.height, self.width, self.gaussians, self.pipe,
                               self.background, with_measurement=False)
            real = self.rgb(find_image(self.source, name))
            self.views[name] = view, real
        return self.views[name]

    @torch.no_grad()
    def prepare_measurements(self, rows, control_rows):
        # Preserve the original batch-two encoder kernels. Batch-one encoding
        # changes FP16 features; pairing queries saves repeated reference encodes.
        reference_embeddings = {}
        tasks = [(row, "event_image") for row in rows]
        tasks.extend((row, "h0_image") for row in control_rows)
        for start in range(0, len(tasks), 2):
            block = tasks[start:start + 2]
            prepared, missing = [], {}
            for row, image_key in block:
                view, real = self.view(row["camera_name"])
                path = row["reference_image"]
                if path not in self.reference_images:
                    self.reference_images[path] = self.rgb(path)
                reference = self.reference_images[path]
                path = row[image_key]
                image = self.reference_images[path] if path in self.reference_images else self.rgb(path)
                observed = (real + image - reference).clamp(0, 1)
                query = F.interpolate(observed[None], size=(1024, 1024), mode="nearest")[0].half()
                prepared.append((view, observed, query))
                if row["camera_name"] not in reference_embeddings:
                    missing[row["camera_name"]] = F.interpolate(
                        view.reference_render[None], size=(1024, 1024), mode="nearest"
                    )[0].half()
            with torch.inference_mode():
                if missing:
                    images = list(missing.values())
                    embeddings = self.sam.get_image_embeddings(
                        torch.stack(images if len(images) == 2 else images * 2)
                    )[-1]
                    for index, name in enumerate(missing):
                        reference_embeddings[name] = embeddings[index].clone()
                images = [entry[2] for entry in prepared]
                embeddings = self.sam.get_image_embeddings(
                    torch.stack(images if len(images) == 2 else images * 2)
                )[-1]
            for index, ((row, image_key), (view, observed, _)) in enumerate(zip(block, prepared)):
                pair = torch.stack((reference_embeddings[row["camera_name"]], embeddings[index]))
                response = frame_cues(
                    observed, view.reference_render, self.sam, self.height, self.width,
                    image_embeddings=pair,
                )[0].detach()
                key = (row["camera_name"], row[image_key], row["reference_image"],
                       torch.backends.cuda.matmul.allow_tf32)
                self.measurements[key] = response.cpu()

    def observe(self, row, image_key="event_image"):
        view, real = self.view(row["camera_name"])
        key = (row["camera_name"], row[image_key], row["reference_image"],
               torch.backends.cuda.matmul.allow_tf32)
        if self.measurements is not None and key in self.measurements:
            view.probability_measurement = self.measurements[key].to(self.background.device)
            return view
        reference_path = row["reference_image"]
        if reference_path not in self.reference_images:
            self.reference_images[reference_path] = self.rgb(reference_path)
        reference = self.reference_images[reference_path]
        image_path = row[image_key]
        image = self.reference_images[image_path] if image_path in self.reference_images else self.rgb(image_path)
        observed = (real + image - reference).clamp(0, 1)
        view.probability_measurement = frame_cues(observed, view.reference_render, self.sam,
                                                 self.height, self.width)[0].detach()
        if self.measurements is not None and image_key == "event_image":
            self.measurements[key] = view.probability_measurement.cpu()
        return view
