# Third-party code in `gsedit_scd/vendor/oscd/`

This directory contains the registration, camera, Gaussian, and rendering primitives
used by **O-SCD** (Galappaththige, Lai, Windrim, Dansereau, Suenderhauf, Miller:
"Changes in Real Time: Online Scene Change Detection with Multi-View Fusion",
CVPR 2026; upstream commit 3abfeaaaef112ba9666b03993edf7d5a91fcaf74), whose code base is in turn built on:

- **on-the-fly-nvs** (Inria, GRAPHDECO) — pose estimation (`poses/`), dense features, keyframes, datasets, renderer
  interface; files carry "Copyright (C) 2025, Inria".  License: `LICENSE.md` (research use only).
- **3D Gaussian Splatting** (Inria / MPII) — `scene/gaussian_model.py`, `scene/cameras.py`, `gaussian_renderer/`,
  `utils/*_utils.py`, `arguments/`.  License: Gaussian-Splatting license, included in `LICENSE.md`.
- **XFeat** (verlab, "XFeat: Accelerated Features for Lightweight Image Matching", CVPR 2024) —
  `scene/extractor_model.py`, Apache License 2.0 (modified by O-SCD to increase the output resolution).
- **COLMAP** (ETH Zurich and UNC Chapel Hill) — `dataloaders/read_write_model.py`, BSD 3-Clause (header in file).
- **FastGS** — the CUDA rasterizer used for rendering lives in `submodules/diff-gaussian-rasterization_fastgs`
  (Gaussian-Splatting license; this repository patches a tile-pass bound and an opacity guard).

This is a subset copied from the project's retained O-SCD implementation. Import
paths use `gsedit_scd.vendor.oscd`; the RANSAC CUDA include directory is resolved
relative to its source file. Existing project modifications in the retained pose
and renderer primitives are preserved. The method's evidence model and shared raw
cue extraction live outside this vendor directory.
