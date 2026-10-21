"""O-SCD query registration and cue extraction, independent of the evidence model."""
import os
import random
import time
from argparse import Namespace

import numpy as np
import torch
from tqdm import tqdm
from transformers import Sam2Model

from gsedit_scd.vendor.oscd.dataloaders.image_dataset import ImageDataset
from .cues import SAM_MODEL, frame_cues
from .cache import prepare_frontend_cache_write, save_frontend_cache_frame, write_frontend_cache_manifest
from gsedit_scd.vendor.oscd.gaussian_renderer import render
from gsedit_scd.vendor.oscd.poses.feature_detector import Detector
from gsedit_scd.vendor.oscd.poses.matcher import Matcher
from gsedit_scd.vendor.oscd.poses.pose_initializer import PoseInitializer, get_reference_keyframes
from gsedit_scd.vendor.oscd.poses.triangulator import Triangulator
from gsedit_scd.vendor.oscd.scene.gaussian_model import GaussianModel
from gsedit_scd.vendor.oscd.scene.cameras import Camera
from gsedit_scd.vendor.oscd.scene.dense_extractor import DenseExtractor
from gsedit_scd.vendor.oscd.scene.keyframe import Keyframe

COMPILE_SAM_ENCODER = True
# O-SCD pose-estimation and matching constants (values of the O-SCD configuration), read by the shared pose classes
POSE_ARGS = {'fundmat_samples': 2000,
 'iters_miniba_incr': 20,
 'match_max_error': 0.002,
 'min_num_inliers': 100,
 'num_keyframes_for_triangulation': 8,
 'num_kpts': 6144,
 'num_pts_miniba_incr': 2000,
 'num_reference_keyframes': 4,
 'pnpransac_samples': 2000,
 'pyr_levels': 1}


class ReferenceRegistrationError(RuntimeError):
    def __init__(self, image_name):
        self.image_name = image_name
        super().__init__(f"Reference image {image_name}: too few inliers for pose initialization")


def _dataset_args(source_path, resolution, test_hold, num_loader_threads):
    return Namespace(source_path=source_path, resolution=resolution, test_hold=test_hold, masks_dir="",
                     use_colmap_poses=False, eval_poses=False, num_loader_threads=num_loader_threads, start_at=0, **POSE_ARGS)


def load_reference(args, width, height):
    """Reference keyframes with triangulated 3D points, exactly as O-SCD builds them."""
    max_error = max(args.match_max_error * width, 1.5)
    matcher = Matcher(args.fundmat_samples, max_error)
    triangulator = Triangulator(args.num_kpts, args.num_keyframes_for_triangulation, max_error)
    pose_initializer = PoseInitializer(width, height, triangulator, matcher, max_error, args)
    dense_extractor = DenseExtractor(width, height)
    detector = Detector(args.num_kpts, width, height)
    reference_dataset = ImageDataset(args, instance="ref")
    keyframes = []
    focal = fovx = fovy = None
    for frame_id in range(len(reference_dataset)):
        image, info = reference_dataset.getnext()
        focal, fovx, fovy = info["focal"], info["FovX"], info["FovY"]
        keyframes.append(Keyframe(image, info, detector(image), info["Rt"], frame_id, focal, dense_extractor, triangulator, args))
    for i in range(len(keyframes)):
        for j in range(i + 1, len(keyframes)):
            matcher(keyframes[i].desc_kpts, keyframes[j].desc_kpts, remove_outliers=True, update_kpts_flag="inliers", kID=i, kID_other=j)
    pose_initializer.init_focal(focal)
    for keyframe in keyframes:
        keyframe.update_3dpts(keyframes)
    return keyframes, detector, matcher, pose_initializer, fovx, fovy


def registered_frames(source_path, resolution=4, test_hold=5, max_inference_frames=-1,
                         num_loader_threads=8, pipe=None, quiet=False, reference_images=False, gaussians=None,
                         extract_cues=True, refinement_rgb=False, cache_cues=True, warmup=None, on_ready=None):
    torch.manual_seed(0); torch.cuda.manual_seed(0); torch.cuda.manual_seed_all(0); np.random.seed(0); random.seed(0)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = not reference_images
    try:
        args = _dataset_args(source_path, resolution, test_hold, num_loader_threads)
        dataset = ImageDataset(args, instance="ref" if reference_images else "inf",
                               refinement_rgb=refinement_rgb)
        height, width = dataset.get_image_size()
        # Finish initial loader CUDA transfers before reference setup captures graphs.
        # The queue is not replenished until the first getnext() below.
        for future in tuple(dataset.preload_queue.queue):
            future.result()
        torch.cuda.synchronize()
        if extract_cues and gaussians is None:
            gaussians = GaussianModel(3, 3)
            gaussians.load_ply(os.path.join(source_path, "reference_reconstruction", "point_cloud", "iteration_30000", "point_cloud.ply"))
        background = torch.zeros(3, dtype=torch.float32, device="cuda")
        if pipe is None:
            pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
        keyframes, detector, matcher, pose_initializer, fovx, fovy = load_reference(args, width, height)
        if extract_cues:
            sam = Sam2Model.from_pretrained(SAM_MODEL, local_files_only=True).half().to("cuda")
        if extract_cues and COMPILE_SAM_ENCODER:  # compiled encoder retains online numerics
            sam.get_image_embeddings = torch.compile(sam.get_image_embeddings, mode="max-autotune")
            dummy = torch.zeros((3, height, width), device="cuda").half()
            for _ in range(min(10, len(dataset))):
                frame_cues(dummy, dummy, sam, height, width)
            torch.cuda.synchronize()

        def register(image, info, frame_id):
            desc_kpts = detector(image)
            previous = get_reference_keyframes(n=args.num_reference_keyframes, keyframes=keyframes, matcher=matcher, desc_kpts=desc_kpts)
            estimate = pose_initializer.init_inference_pose(previous, desc_kpts, frame_id, False, image)
            if reference_images and estimate is None:
                raise ReferenceRegistrationError(info["name"])
            Rt, pose_stats = estimate
            R = np.transpose(Rt[:3, :3].cpu().numpy()); T = Rt[:3, 3].cpu().numpy()
            uid = f"{frame_id:04d}"; image_name = info["name"].split(".")[0]
            view = Camera(colmap_id=uid, R=R, T=T, FoVx=fovx, FoVy=fovy, image=image[:3, ...], gt_alpha_mask=None, image_name=image_name, uid=uid)
            if refinement_rgb:
                view.refinement_rgb = info["refinement_rgb"]
            payload = {
                "frame_index": frame_id, "image_name": image_name, "is_test": bool(info["is_test"]),
                "R": torch.as_tensor(R).clone(), "T": torch.as_tensor(T).clone(), "FoVx": float(fovx), "FoVy": float(fovy),
            }
            if extract_cues:
                with torch.no_grad():
                    rendered = render(view, gaussians, pipe, background)["render"]
                    view.reference_render = rendered.detach()
                candidate, photometric, raw_photometric, raw_feature = frame_cues(view.original_image[:3, ...], rendered, sam, height, width)
                view.probability_measurement = candidate.detach()
                if cache_cues:
                    payload.update({
                        "candidate_map": candidate.detach().cpu(),
                        "normalized_photometric_map": photometric.detach().cpu(),
                        "normalized_feature_map": (candidate - photometric).detach().cpu(),
                        "raw_photometric_residual": raw_photometric.detach().cpu(),
                        "raw_feature_residual": raw_feature.detach().cpu(),
                    })
            if isinstance(pose_stats, dict):
                for key in ("covariance_se3", "covariance_valid", "residual_rms_pixels", "information_condition_number",
                            "match_count", "pnp_inlier_count", "miniba_inlier_count"):
                    if key in pose_stats:
                        value = pose_stats[key]
                        payload["pose_" + key] = value.detach().cpu() if torch.is_tensor(value) else value
            return payload, view

        if warmup is not None:
            torch.cuda.synchronize()
            started = time.perf_counter()
            # Pose estimation overwrites reference match entries and consumes RNG.
            matches = [dict(keyframe.desc_kpts.matches) for keyframe in keyframes]
            numpy_state, python_state = np.random.get_state(), random.getstate()
            warmup_tf32 = torch.backends.cuda.matmul.allow_tf32
            try:
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    stats = warmup(register, height, width)
            finally:
                for keyframe, saved in zip(keyframes, matches):
                    keyframe.desc_kpts.matches.clear()
                    keyframe.desc_kpts.matches.update(saved)
                np.random.set_state(numpy_state)
                random.setstate(python_state)
                torch.backends.cuda.matmul.allow_tf32 = warmup_tf32
                torch.cuda.synchronize()
            stats["seconds"] = time.perf_counter() - started

        n_frames = len(dataset) if max_inference_frames <= 0 else min(len(dataset), max_inference_frames)
        if on_ready is not None:
            on_ready(n_frames)
        for frame_id in tqdm(range(n_frames), desc="Query registration and cues", disable=quiet):
            image, info = dataset.getnext()
            yield register(image, info, frame_id)
    finally:
        if reference_images:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def write_frontend_cache(source_path, cache_dir, resolution=4, reference_images=False):
    prepare_frontend_cache_write(cache_dir)
    records = []
    height = width = None
    # Offline learning recomputes cues from each edited/control image pair.
    for payload, view in registered_frames(source_path, resolution=resolution,
                                          reference_images=reference_images,
                                          extract_cues=not reference_images):
        height, width = view.image_height, view.image_width
        index = payload["frame_index"]
        relative = save_frontend_cache_frame(cache_dir, index, payload)
        records.append({"index": index, "image_name": payload["image_name"], "file": relative})
    write_frontend_cache_manifest(cache_dir, {
        "height": height, "width": width, "frames": records,
        "frame_count": len(records), "reference_images": reference_images,
    })
