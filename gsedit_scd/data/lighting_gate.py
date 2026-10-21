from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from gsedit_scd.vendor.oscd.scene.gaussian_model import GaussianModel
from gsedit_scd.data.mask_geometry import find_image, make_camera, resized_rgb
from gsedit_scd.data.lighting_model import canonical_normals, directional_relighting_admissible, reliable_object_gaussians

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build the reference-only directional-relighting gate required by "
            "semantic event nuisance generation."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--object-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", type=int, default=6)
    parser.add_argument("--max-side", type=int, default=1008)
    parser.add_argument("--fit-max-side", type=int, default=320)
    return parser.parse_args()


def select_evenly(values, count):
    count = min(int(count), len(values))
    if count < 1:
        raise ValueError("at least one reference gate view is required")
    positions = np.linspace(0, len(values) - 1, count, dtype=np.int64)
    return [int(values[int(position)]) for position in positions]


@torch.no_grad()
def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    source = args.source.resolve()
    object_map_path = args.object_map.resolve()
    latent_map = torch.load(
        object_map_path, map_location="cpu", weights_only=False
    )
    camera_records = json.loads(
        (source / "reference_reconstruction" / "cameras.json").read_text()
    )
    camera_indices = select_evenly(
        latent_map.get("camera_indices", ()), args.views
    )
    model_path = (
        source
        / "reference_reconstruction"
        / "point_cloud"
        / "iteration_30000"
        / "point_cloud.ply"
    )
    gaussians = GaussianModel(3, 3)
    gaussians.load_ply(str(model_path))
    object_gaussians, component_count = reliable_object_gaussians(
        latent_map,
        gaussians.get_xyz.shape[0],
        gaussians.get_xyz.device,
    )
    normals = canonical_normals(
        gaussians, [record["position"] for record in camera_records]
    )
    views = []
    for camera_index in camera_indices:
        record = camera_records[camera_index]
        image = resized_rgb(
            find_image(source, record["img_name"]), args.max_side
        )
        views.append(make_camera(record, image))
    pipe = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
    )
    background = torch.zeros(3, device=gaussians.get_xyz.device)
    directional_admissible, statistics = directional_relighting_admissible(
        views,
        gaussians,
        pipe,
        background,
        normals,
        object_gaussians,
        args.fit_max_side,
    )
    manifest = {
        "source": str(source),
        "object_map": str(object_map_path),
        "method": "reference_directional_relighting_gate_v1",
        "reference_gate": {
            "directional_admissible": bool(directional_admissible),
            "camera_indices": camera_indices,
            **statistics,
        },
        "reliable_component_count": component_count,
        "target_images_used": False,
        "target_gt_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
