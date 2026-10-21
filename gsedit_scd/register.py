"""Register images; save query cues or reference poses for offline learning."""
import argparse
import json
from pathlib import Path

from .online.cache import prepare_frontend_cache_write, save_frontend_cache_frame, write_frontend_cache_manifest
from .online.registration import ReferenceRegistrationError, registered_frames, write_frontend_cache
from .vendor.oscd.utils.utils_fly import get_image_names


def prepare_references(source, output, prepared_source, resolution):
    """Check a separate reference subset before edit generation; leave source data intact."""
    source, output, prepared_source = source.resolve(), output.resolve(), prepared_source.resolve()
    if prepared_source == source or source in prepared_source.parents:
        raise ValueError("The checked source must be outside the original scene directory.")
    if prepared_source.exists() and any(prepared_source.iterdir()):
        raise FileExistsError(f"Checked source must be empty: {prepared_source}")
    # Scene and parent names are used to form the existing object IDs.
    if prepared_source.parts[-2:] != source.parts[-2:]:
        raise ValueError("Keep the source's final two directory names, e.g. checked/CCS3D/bedroom.")
    names = get_image_names(str(source / "reference_scene/images"))
    if not names:
        raise ValueError("No reference images to check.")
    cameras_path = source / "reference_reconstruction/cameras.json"
    cameras_text = cameras_path.read_text()
    cameras = json.loads(cameras_text)
    prepare_frontend_cache_write(output)
    for folder in ("reference_scene/images", "reference_reconstruction"):
        (prepared_source / folder).mkdir(parents=True, exist_ok=True)
    for entry in source.iterdir():
        if entry.name not in ("reference_scene", "reference_reconstruction"):
            (prepared_source / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    for folder, excluded in (("reference_scene", "images"), ("reference_reconstruction", "cameras.json")):
        for entry in (source / folder).iterdir():
            if entry.name != excluded:
                (prepared_source / folder / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    for name in names:
        (prepared_source / "reference_scene/images" / name).symlink_to(source / "reference_scene/images" / name)
    checked_cameras = prepared_source / "reference_reconstruction/cameras.json"
    checked_cameras.write_text(cameras_text)
    excluded = []
    while True:
        payloads = []
        try:
            for payload, view in registered_frames(str(prepared_source), resolution=resolution,
                                                  reference_images=True, extract_cues=False):
                payloads.append(payload)
                height, width = view.image_height, view.image_width
            break
        except ReferenceRegistrationError as error:
            (prepared_source / "reference_scene/images" / error.image_name).unlink()
            excluded.append({"image_name": error.image_name, "reason": "too_few_pose_inliers"})
            print(f"Excluded reference {error.image_name}; checking the remaining reference set again.", flush=True)
            retained = {Path(name).stem for name in get_image_names(str(prepared_source / "reference_scene/images"))}
            if not retained:
                raise RuntimeError("No reference images passed registration.") from error
            selected = [dict(camera) for camera in cameras if camera["img_name"] in retained]
            for index, camera in enumerate(selected):
                camera["id"] = index
            checked_cameras.write_text(json.dumps(selected, indent=2) + "\n")
        # Discard all poses from a failed pass: the next pass uses a different map.

    records = []
    for payload in payloads:
        index = payload["frame_index"]
        relative = save_frontend_cache_frame(output, index, payload)
        records.append({"index": index, "image_name": payload["image_name"], "file": relative})
    write_frontend_cache_manifest(output, {"height": height, "width": width, "frames": records,
                                           "frame_count": len(records), "reference_images": True})
    (output / "reference_check.json").write_text(json.dumps({
        "input_source": str(source), "prepared_source": str(prepared_source), "resolution": resolution,
        "input_count": len(names), "retained_count": len(records), "excluded": excluded,
        "retained_images": [payload["image_name"] for payload in payloads],
    }, indent=2) + "\n")
    print(f"Reference check: {len(names)} input, {len(excluded)} excluded, {len(records)} retained.", flush=True)
    print(f"Use --source {prepared_source} for edit generation, fitting and subsequent scene processing.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-images", action="store_true",
                        help="Check reference registration and save the accepted reference subset and poses before generating edits.")
    parser.add_argument("--prepared-source", type=Path,
                        help="Checked scene output for --reference-images; defaults to <output>/source/<parent>/<scene>.")
    parser.add_argument("--resolution", type=int, default=4)
    args = parser.parse_args()
    if args.prepared_source and not args.reference_images:
        parser.error("--prepared-source requires --reference-images")
    if args.reference_images:
        source = args.source.resolve()
        prepared = args.prepared_source or args.output / "source" / source.parent.name / source.name
        prepare_references(source, args.output, prepared, args.resolution)
    else:
        write_frontend_cache(str(args.source), args.output, resolution=args.resolution)


if __name__ == "__main__":
    main()
