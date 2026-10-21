"""Download and prepare PASLCD or CCS3D, including the shared SAM2 weights."""
import argparse
from pathlib import Path, PurePosixPath
import zipfile

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError


def ensure_sam2():
    model = "facebook/sam2.1-hiera-tiny"
    required = ("config.json", "model.safetensors", "preprocessor_config.json", "processor_config.json")
    try:
        cached = Path(snapshot_download(model, local_files_only=True))
    except LocalEntryNotFoundError:
        cached = None
    if cached is not None and all((cached / name).is_file() for name in required):
        return
    print("Downloading missing SAM2 model files before training or inference.", flush=True)
    snapshot_download(model, allow_patterns=list(required))


def require_empty(directory):
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError(
            f"Destination is not empty: {directory}. Use the prepared data directly, "
            "or choose a new --data-dir; existing files will not be overwritten.")


def extract(archive, destination, root):
    require_empty(destination / root)
    with zipfile.ZipFile(archive) as source:
        members = [item for item in source.infolist()
                   if PurePosixPath(item.filename).is_relative_to(root)]
        if not members or any(".." in PurePosixPath(item.filename).parts for item in members):
            raise ValueError(f"Expected archive contents under {root}: {archive}")
        print(f"Extracting {archive} to {destination}", flush=True)
        source.extractall(destination, members=members)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("PASLCD", "CCS3D"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Parent directory for prepared datasets and downloads (default: data).")
    args = parser.parse_args()
    output = args.data_dir / args.dataset
    downloads = args.data_dir / "downloads"
    require_empty(output)
    if args.dataset == "PASLCD":
        archive = hf_hub_download("ChamudithaJay/PASLCD", "PASLCD_online.zip",
                                  repo_type="dataset", local_dir=downloads)
        extract(archive, args.data_dir, "PASLCD")
    else:
        from evaluate_ccs3d import CCS3D_FRAME_COUNTS
        from prepare_ccs3d import prepare

        require_empty(downloads / "CCS3D")
        require_empty(downloads / "models/CCS3D")
        archive = hf_hub_download("zr-zhou/CCS3D", "CCS3D.zip", repo_type="dataset",
                                  local_dir=downloads / "ccs3d-data")
        extract(archive, downloads, "CCS3D")
        models = hf_hub_download("zr-zhou/CCS3D-pretrained", "CCS3D.zip",
                                 local_dir=downloads / "ccs3d-models")
        extract(models, downloads, "models/CCS3D")
        for scene in CCS3D_FRAME_COUNTS:
            prepare(downloads / "CCS3D" / scene, downloads / "models/CCS3D" / scene,
                    output / scene)
    ensure_sam2()
    print(f"Prepared {args.dataset}: {output}", flush=True)


if __name__ == "__main__":
    main()
