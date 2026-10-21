# Third-party code

The included dependencies keep their original notices and license terms.

| Code | Source | License location |
| --- | --- | --- |
| Registration and scene primitives | O-SCD; on-the-fly-nvs; 3D Gaussian Splatting | `gsedit_scd/vendor/oscd/LICENSE.md` and file headers |
| O-SCD-derived cue extraction and registration (`gsedit_scd/online/cues.py`, `gsedit_scd/online/registration.py`) | O-SCD; on-the-fly-nvs; 3D Gaussian Splatting | `gsedit_scd/vendor/oscd/LICENSE.md` |
| FastGS rasterizer with method-specific aggregation operations and local fixes | FastGS / 3D Gaussian Splatting | `submodules/diff-gaussian-rasterization_fastgs/LICENSE.md` |
| Nearest-neighbour CUDA extension | simple-knn | `submodules/simple-knn/LICENSE.md` |
| GLM headers | OpenGL Mathematics | `submodules/diff-gaussian-rasterization_fastgs/third_party/glm/copying.txt` |
| XFeat model and feature utilities | XFeat | `gsedit_scd/vendor/oscd/LICENSE-XFeat` and notices in the relevant vendor files |
| COLMAP model reader | COLMAP | BSD notice in `gsedit_scd/vendor/oscd/dataloaders/read_write_model.py` |

Additional details are in `gsedit_scd/vendor/oscd/NOTICE.md`. Some bundled dependencies
restrict use to research and evaluation; this repository does not replace their
terms with a blanket permissive license.

The two O-SCD-derived files above reorganize query registration and cue extraction;
local changes include reference checks, cache handling, initialization, and timing
hooks. They are excluded from the project's MIT grant.

SAM2, XFeat weights, PASLCD, and CCS3D are obtained separately from their respective
providers. Their licenses and dataset terms continue to apply. Fitted parameters
and rendered edits have separate notices in `pretrained/LICENSE.md` and
`examples/initialization/LICENSE.md`.
