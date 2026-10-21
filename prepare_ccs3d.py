"""Adapt an extracted CCS3D scene without modifying the original data or reconstruction."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from gsedit_scd.vendor.oscd.dataloaders.read_write_model import qvec2rotmat, read_model


def prepare(dataset, reconstruction, output, exclude_reference=()):
    dataset, reconstruction = dataset.resolve(), reconstruction.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    sparse = dataset / "sparse/0"
    cameras, images, _ = read_model(str(sparse), include_points=False)
    reference = {path.name: path for path in (dataset / "train-pre").glob("*.png")}
    if not reference:
        raise FileNotFoundError(f"No reference PNGs in {dataset / 'train-pre'}")
    excluded = set(exclude_reference)
    if excluded - reference.keys():
        raise ValueError(f"Excluded reference names not found: {sorted(excluded - reference.keys())}")
    posed = {image.name for image in images.values()}
    names = (reference.keys() & posed) - excluded
    if not names:
        raise ValueError("No posed reference images remain.")

    # Use train-pre poses, not the cameras.json bundled with the pretrained model.
    records = []
    for image in sorted(images.values(), key=lambda value: value.name):
        if image.name not in names:
            continue
        camera = cameras[image.camera_id]
        if camera.model == "SIMPLE_PINHOLE":
            fx = fy = float(camera.params[0])
        elif camera.model == "PINHOLE":
            fx, fy = map(float, camera.params[:2])
        else:
            raise ValueError(f"Unsupported camera model: {camera.model}")
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = qvec2rotmat(image.qvec)
        world_to_camera[:3, 3] = image.tvec
        camera_to_world = np.linalg.inv(world_to_camera)
        records.append({"id": len(records), "img_name": Path(image.name).stem,
                        "width": int(camera.width), "height": int(camera.height),
                        "position": camera_to_world[:3, 3].tolist(),
                        "rotation": camera_to_world[:3, :3].tolist(), "fx": fx, "fy": fy})

    queries = sorted((dataset / "test-post").glob("whole_*.png"))
    if not queries:
        raise FileNotFoundError(f"No test-post PNGs in {dataset}")
    masks = []
    for query in queries:
        index = int(query.stem.split("_")[-1])
        pre = dataset / "gt-pre-mask" / f"gt_{index:03d}.png"
        post = dataset / "gt-post-mask" / f"gt_{index:03d}.png"
        for path in (pre, post):
            if not path.is_file():
                raise FileNotFoundError(path)
        masks.append((query.name, pre, post))
    ply = reconstruction / "point_cloud/iteration_30000/point_cloud.ply"
    if not ply.is_file():
        raise FileNotFoundError(ply)

    for directory in ("reference_scene/images", "reference_scene/sparse",
                      "reference_reconstruction/point_cloud/iteration_30000",
                      "inference_scene/images", "gt_mask", "gt_pre_mask"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    for name in sorted(names):
        with Image.open(reference[name]) as image:
            image.convert("RGB").save(output / "reference_scene/images" / name)
    for query in queries:
        with Image.open(query) as image:
            image.convert("RGB").save(output / "inference_scene/images" / query.name)
    for name, pre, post in masks:
        shutil.copy2(pre, output / "gt_pre_mask" / name)
        shutil.copy2(post, output / "gt_mask" / name)
    # These large, read-only inputs stay in the extracted dataset/model directories.
    (output / "reference_scene/sparse/0").symlink_to(sparse, target_is_directory=True)
    (output / "reference_reconstruction/point_cloud/iteration_30000/point_cloud.ply").symlink_to(ply)
    (output / "reference_reconstruction/cameras.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"Prepared {len(records)} reference images and {len(queries)} test-post queries in {output}")
    print(f"Without COLMAP poses: {sorted(reference.keys() - posed)}")
    print(f"Explicit reference exclusions: {sorted(excluded)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Extracted CCS3D/<scene> directory.")
    parser.add_argument("--reconstruction", type=Path, required=True, help="Extracted models/CCS3D/<scene> directory.")
    parser.add_argument("--output", type=Path, required=True, help="New prepared scene directory.")
    parser.add_argument("--exclude-reference", nargs="*", default=[], metavar="FILENAME",
                        help="Optional explicit reference exclusions when reproducing a recorded subset.")
    args = parser.parse_args()
    prepare(args.dataset, args.reconstruction, args.output, args.exclude_reference)


if __name__ == "__main__":
    main()
