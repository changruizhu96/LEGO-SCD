# Learning from Automatic 3D Gaussian Splatting Edits for Online Bayesian Scene Change Detection

**Changrui Zhu · Simon Julier**  
University College London

[Paper](https://arxiv.org/abs/2610.28490) ·
[Project page](https://changruizhu96.github.io/work/lego-scd/)

The method learns scene-specific evidence models from
automatic reference edits, then detects changes through causal Bayesian evidence
fusion and accumulation. Contour filtering and object completion refine the output
without modifying the accumulated history.

[Install](#install) · [Quick start](#quick-start) ·
[Train your own parameters](#train-your-own-parameters) ·
[Timing](#online-timing) · [Evaluation details](#evaluation-details)

## Install

Run the command blocks below in a Bash terminal. Directory listings are labelled
**Expected directory structure** and are for comparison, not terminal input.
Follow the commands for your chosen dataset and route. Alternative modes and
optional examples are labelled; they are not additional required steps.

Clone the repository, which includes the fitted scene parameters and fixed
initialization examples; these need no separate download:

```bash
git clone https://github.com/changruizhu96/LEGO-SCD.git
cd LEGO-SCD
```

`gsedit_scd` is the Python package name. Run the following commands from the
repository root. Use a checkout path without spaces for compiler caches.

On Linux x86_64 with Conda available and a working NVIDIA GPU driver, run:

```bash
bash install.sh
```

This creates a **new, separate environment** at `~/.local/share/gsedit-scd`, installs
Python 3.12, PyTorch 2.11.0, torchvision 0.26.0, the CUDA 12.8 toolkit components,
GCC/G++ 13 and the Python dependencies, then compiles the included FastGS and
simple-knn extensions. The GPU architecture is detected automatically. Existing
environments and the system CUDA installation are not modified.

The installer builds our modified FastGS sources, including the required local
fixes. **Do not replace them with the upstream version.**

After installation, activate the environment in your terminal:

```bash
conda activate ~/.local/share/gsedit-scd
```

Activate it again in each new terminal. CUDA/compiler settings and project-local
temporary/cache settings are applied automatically on activation; no manual `export`
commands are needed. The software stack has been tested on an RTX 5090.

<details>
<summary>Choose another environment location</summary>

The destination must not already exist. To use another location:

```bash
bash install.sh "$HOME/.local/share/gsedit-scd-custom"
conda activate "$HOME/.local/share/gsedit-scd-custom"
```

If installation stops partway through, the script leaves that environment in
place and reports the failure. It does not remove or overwrite an existing
environment when run again. Conda itself and the NVIDIA driver must already be
installed; the script does not install system packages or change shell startup files.

</details>

SAM2 is checked automatically during data preparation and before training or
inference. Complete cached model files are reused without network access; missing
files are downloaded from `facebook/sam2.1-hiera-tiny` into the Hugging Face cache.
Downloads finish before compilation and inference, outside the timing window.

Registration uses XFeat from `verlab/accelerated_features` through PyTorch Hub.
Its first use requires internet access unless the model and weights are already
cached. SAM2 and XFeat weights are not included in this source repository.

## Quick start

Use this route to run the supplied checkpoints. Complete installation, then follow
one dataset section below. Editing and offline learning are not needed.
If you want to train your own models, use the same data preparation steps, then
continue to [Train your own parameters](#train-your-own-parameters).

### Get PASLCD Full-20

**Run in the terminal to download, extract, and prepare PASLCD:**

```bash
python prepare_data.py PASLCD
```

This downloads the authors' [PASLCD online release](https://huggingface.co/datasets/ChamudithaJay/PASLCD/blob/main/PASLCD_online.zip),
which includes both instances of all ten scenes, reference reconstructions, and
query labels. It also checks the shared SAM2 model files. No manual unzip step is
needed. If this dataset is already prepared, skip this command and run evaluation.

The extracted scenes should be under `data/PASLCD/Instance_1/` and
`data/PASLCD/Instance_2/`. Matching fitted parameters for all 20 instances are
included under `pretrained/PASLCD/`.

**Expected directory structure after extraction (not commands):**
Cantina is shown as an example.

```text
data/PASLCD/Instance_1/Cantina/
  reference_scene/images/
  reference_scene/sparse/0/
  reference_reconstruction/cameras.json
  reference_reconstruction/point_cloud/iteration_30000/point_cloud.ply
  inference_scene/images/
  gt_mask/                         # evaluation only
```

The method starts from a reconstructed reference scene. Query images and query
change labels are not used for offline fitting. Ground-truth masks are used only
for evaluation.

#### Run PASLCD Full-20

The full pipeline automatically checks for the shared SAM2 acceleration files.
If they are missing, it compiles them from a reference image before inference.
**This is compilation, not retraining**, and only needs to happen once in the same
GPU and PyTorch environment. All scenes in PASLCD and CCS3D reuse these files.
`--pixel-only` skips decoder compilation.

Run all 20 instances and evaluate both pixel and refined outputs:

```bash
python run_evaluation.py PASLCD
```

Add `--timing` to also measure speed, or `--timing-only` to measure speed without
accuracy evaluation. See [Online timing](#online-timing) for the measurement procedure.

Results are saved to `outputs/evaluation/PASLCD/full/metrics.json`. Each instance
has its own `predictions/` directory and starts with a fresh history.
For pixel-only inference without decoder compilation:

```bash
python run_evaluation.py PASLCD --pixel-only
```

Pixel-only results go to `outputs/evaluation/PASLCD/pixel/`.

<details>
<summary>Try Cantina first (25 frames, pixel-only)</summary>

This uses a separate output directory, so a later Full-20 run needs no path changes.

```bash
python run_evaluation.py PASLCD --scene Cantina --instance Instance_1 --pixel-only \
  --output outputs/try_cantina
```

</details>

### CCS3D

#### Prepare CCS3D

**Run in the terminal to download, extract, and prepare all four CCS3D scenes:**

```bash
python prepare_data.py CCS3D
```

This downloads the [CCS3D dataset](https://huggingface.co/datasets/zr-zhou/CCS3D)
and the authors' [reference reconstructions](https://huggingface.co/zr-zhou/CCS3D-pretrained),
extracts them, runs the existing adapter for all four scenes, and checks SAM2.
There is no separate extraction command or scene loop to run. If CCS3D is already
prepared, skip this command and run evaluation.

**Expected directory structure after preparation (not commands):**
`<scene>` stands for `livingroom`, `desk`, `bookcase`, or `bedroom`.

```text
data/CCS3D/<scene>/
  reference_scene/images/
  reference_scene/sparse/0/
  reference_reconstruction/cameras.json
  reference_reconstruction/point_cloud/iteration_30000/point_cloud.ply
  inference_scene/images/
  gt_mask/                         # evaluation only
  gt_pre_mask/                     # evaluation only
```

Only posed `train-pre` images are used as references; queries are `test-post`, in
filename order. `train-post` is not used. The adapter makes RGB image copies,
renames both sets of GT masks to match query filenames, and rebuilds
`cameras.json` from the reference COLMAP poses. It does not use the camera list
bundled with the pretrained reconstruction, which corresponds to different views.
Extracted originals remain under `data/downloads/CCS3D/` and
`data/downloads/models/CCS3D/`. Keep these folders: prepared scenes link to their
COLMAP files and reconstructions. Original files are not modified.

After preparation, the adapted scenes contain:

| Scene | Reference images | Test-post queries |
| --- | ---: | ---: |
| Livingroom | 89 | 5 |
| Desk | 96 | 4 |
| Bookcase | 94 | 5 |
| Bedroom | 160 | 4 |

Each prepared scene has the same reference/inference layout as PASLCD, plus
`gt_pre_mask/` alongside the post-state `gt_mask/`. Both are evaluation-only.

#### Run all four scenes

SAM2 acceleration files are checked automatically, just as for PASLCD. Existing
files are reused; if missing, they are compiled from a CCS3D reference image.
`--pixel-only` skips this step.

Run all four scenes and evaluate both pixel and refined outputs:

```bash
python run_evaluation.py CCS3D
```

Results are saved to `outputs/evaluation/CCS3D/full/summary.json`, with metrics
for each scene alongside its predictions. For pixel-only inference without decoder
compilation:

```bash
python run_evaluation.py CCS3D --pixel-only
```

<details>
<summary>Use a different data location</summary>

`--data-dir` changes the parent directory for downloads and prepared datasets.
For example, for PASLCD:

```bash
python prepare_data.py PASLCD --data-dir datasets
python run_evaluation.py PASLCD --data-root datasets/PASLCD
```

The same option works for CCS3D. For training, pass the prepared dataset directory
with `run_training.py --data-root`. Preparation requires empty destination folders
and never overwrites an existing dataset. Downloaded archives are kept under
`<data-dir>/downloads/`.

</details>

### Checkpoints and outputs

| Dataset | Bundled parameters | Coverage |
| --- | --- | --- |
| PASLCD | `pretrained/PASLCD/Instance_<1 or 2>/<scene>/` | 20 instances, 500 query frames |
| CCS3D | `pretrained/CCS3D/<scene>/` | Four scenes, 18 query frames |

The parameters occupy about 88 MB and include `suppression.pt`, `likelihood.json`,
`unchanged.pt`, `prior.json`, `contour.pkl`, and `split.json`. Use the corresponding
reference reconstructions from the data instructions. Unchanged-response statistics
are indexed by Gaussian; replacing, pruning, or reordering the reconstruction
requires fitting new parameters.

Detection registers raw queries online. Each query uses the previous history before
updating it. Predictions are saved in `probability/` and `mask/`; full refinement
also writes `refined_probability/` and `refined_mask/`. PNG probabilities use 16-bit
export values; internal evidence and probabilities remain floating point.
The first run initializes the models and compiles the SAM2 encoder.

Existing prediction directories are not overwritten. For a new run, set a new
`--output`; to score saved predictions, use [Evaluate existing predictions](#evaluate-existing-predictions).

<details>
<summary>Optional: compile SAM2 acceleration files in advance</summary>

The evaluation and timing commands do this automatically for the full pipeline.
To prepare the files separately, use any reference image, for example:

```bash
python -m gsedit_scd.compile_decoder \
  --reference-image data/PASLCD/Instance_1/Cantina/reference_scene/images/Inst_1_IMG_3065.jpg \
  --output models/gsedit_scd/sam2_decoder_aot
```

The output directory must not already exist. If an interrupted compilation left
incomplete files, move that directory aside and rerun the evaluation or timing
command. Existing files are never overwritten automatically. If you change the GPU
or PyTorch environment, move the old compiled directory aside so it can be rebuilt.

</details>

## Train your own parameters

This route uses the same installed environment and prepared datasets as Quick start.
Reference checking, editing, and fitting use reference data only; query labels are
used only by evaluation. The following commands run all three stages in order for
each instance, stopping if a stage fails:

```bash
# PASLCD: all ten scenes and both instances
python run_training.py PASLCD
```

```bash
# CCS3D: all four scenes
python run_training.py CCS3D
```

Outputs are connected automatically:

| Output | PASLCD | CCS3D |
| --- | --- | --- |
| Checked scenes | `data/checked/PASLCD/Instance_<1 or 2>/<scene>/` | `data/checked/CCS3D/<scene>/` |
| Training outputs | `outputs/PASLCD/Instance_<1 or 2>/<scene>/` | `outputs/CCS3D/<scene>/` |

Each training directory contains `reference_cues/`, `edits/training_events.tsv`,
and `parameters/`. Outputs for the stages being run must be new or empty. No
completed stage is silently skipped or overwritten. Custom locations can be set
with `--data-root`, `--checked-root`, and `--output`.

<details>
<summary>Run the three stages individually</summary>

Use these instead of the all-stage command when you want to inspect intermediate
outputs. Run them in order; reference checking is required for this training route.

```bash
python run_training.py PASLCD --stage reference
python run_training.py PASLCD --stage edits
python run_training.py PASLCD --stage fit
```

```bash
python run_training.py CCS3D --stage reference
python run_training.py CCS3D --stage edits
python run_training.py CCS3D --stage fit
```

**Reference checking** saves re-estimated poses and a separate scene with accepted
reference images. Images failing the pose-inlier check are excluded, and registration
restarts on the remaining set. Original data and query membership are preserved;
`reference_check.json` records the retained references and exclusions.

**Automatic editing** generates recolouring, removal, translation, and relighting
pairs with labels. PASLCD uses six proposal views and a maximum image side of 1008;
CCS3D uses all available reference views and a maximum side of 800. `--views` can
override the proposal-view count; it is not the number of fitting images. Generated
TSV paths are repository-relative, so keep training outputs inside the checkout.

**Offline fitting** learns suppression, evidence mapping, unchanged-response
statistics, the pixel prior, and contour filtering. Registration and fitting use
resolution 4 for PASLCD and 2 for CCS3D. The existing reference split and fitting
procedure are unchanged.

</details>

<details>
<summary>Train just Cantina first</summary>

The trial has separate checked-data and training directories, so it does not block
a later Full-20 training run:

```bash
python run_training.py PASLCD --scene Cantina --instance Instance_1 \
  --checked-root data/checked_trial/PASLCD --output outputs/training_trial/PASLCD
```

Detect and evaluate this trial (decoder compilation is automatic when needed):

```bash
python run_evaluation.py PASLCD --scene Cantina --instance Instance_1 \
  --data-root data/checked_trial/PASLCD --training-root outputs/training_trial/PASLCD \
  --output outputs/evaluation_trial/PASLCD
```

</details>

### Detect with your fitted parameters

Use the same evaluation entry, pointing it to your checked data and training outputs.
The full pipeline automatically prepares the shared decoder if needed:

```bash
python run_evaluation.py PASLCD --data-root data/checked/PASLCD \
  --training-root outputs/PASLCD --output outputs/evaluation/PASLCD/trained
```

```bash
python run_evaluation.py CCS3D --data-root data/checked/CCS3D \
  --training-root outputs/CCS3D --output outputs/evaluation/CCS3D/trained
```

`--training-root` selects existing fitted parameters; it does not start training.
Add `--pixel-only` for detection without refinement or decoder compilation.
Both datasets use the same entry and options; PASLCD additionally uses an instance level.

## Online timing

Use the same evaluation command for all three modes. Timing currently supports
PASLCD only:

```bash
python run_evaluation.py PASLCD                # accuracy only
python run_evaluation.py PASLCD --timing       # accuracy, then one timing pass
python run_evaluation.py PASLCD --timing-only  # one timing pass only
```

Choose one command. All modes accept the same `--pixel-only`, `--scene`,
`--instance`, `--data-root`, `--training-root`, and `--output` options.
Timing uses the same selected inputs and fitted parameters as evaluation.
Each timing pass covers all 20 instances once (500 frames), unless a single
instance is selected, with a fresh process and history for each sequence.

**Why a separate pass?** Accuracy evaluation saves predicted masks and probability
maps. Encoding and writing these images adds overhead between frames and changes
continuous FPS. With `--timing`, the first pass saves predictions for accuracy;
the second measures detection without those writes. This gives both results in
one command while keeping the existing timing protocol. Timing itself runs only
once; `--timing-only` skips the accuracy pass entirely. Any required decoder
compilation runs before the inference process starts. Compilation and initialization
are excluded from the reported sequence time and FPS.

Timing results go under `<output>/timing/`. The default full-pipeline summary is
`outputs/evaluation/PASLCD/full/timing/full_summary.json`; pixel-only timing uses
`outputs/evaluation/PASLCD/pixel/timing/pixel_summary.json`.
Accuracy results retain their usual paths. Existing run directories are not
overwritten; use a new `--output` for a new run. To score existing predictions
and add timing, use `--evaluate-only --timing`.

For your own fitted parameters and generated initialization edits:

```bash
python run_evaluation.py PASLCD --timing \
  --data-root data/checked/PASLCD --training-root outputs/PASLCD \
  --output outputs/evaluation/PASLCD/trained
```

For a single Cantina timing run:

```bash
python run_evaluation.py PASLCD --timing-only --scene Cantina --instance Instance_1 \
  --output outputs/timing_cantina
```

<details>
<summary>Summarize saved timings or request repeated measurements</summary>

The standalone timing entry remains available and also defaults to one pass.
To summarize the default combined command's timing files without inference:

```bash
python run_timing.py PASLCD --summarize-only \
  --output outputs/evaluation/PASLCD/full/timing
```

Use the same scene selection and `--pixel-only` setting as the original run.
For optional repeated measurements, use `run_timing.py --repeats N` with the dataset
and a new `--output`; use the same `--repeats N` when summarizing those records.
Missing or incomplete timing records cause an error rather than a partial summary.

</details>

### Timing protocol

The bundled initialization images in `examples/initialization/PASLCD/` are fixed
reference edits, not test images. Each is used twice to exercise empty and
populated history, as in the reported Full-20 timing. For fitted parameters,
the command uses `edits/training_events.tsv` under each training directory;
the inference entry selects the first identity-condition edit whose reference
camera belongs to the saved evidence-learning subset and uses it twice.

After initialization, accumulated evidence, observation counts, reference matches,
and random-number states are restored. The selected paths and edit initialization
time are saved in `warmup.json`.

Timing starts after scene/model loading, reference setup, SAM compilation, and
initialization have finished. Query prefetch starts during initialization, as in
normal inference. The timed interval includes remaining input waits, registration,
rendering, evidence computation, history updates, refinement when enabled, and
inter-frame overhead. It ends when the last mask is ready on the GPU. Every frame,
including the first, is counted; no first-three-frame exclusion is applied.
CUDA is synchronized at each mask output, without extra synchronization between stages.
PNG encoding/writing and metric evaluation are outside this interval.

`timing.json` reports initialization separately (including edit initialization),
first-frame latency, individual frame latencies, total sequence time, and
`sequence_fps = frames / sequence_seconds`. Initialization excludes Python startup
and imports. The mean individual latency excludes bookkeeping between frames, so use
the measured sequence time for throughput. Benchmark mode uses eight PyTorch CPU
threads and records the GPU, PyTorch/CUDA versions, inputs, and parameters.

The summary computes FPS as **total frames / total sequence seconds**, both
across repeats for each instance and across the complete dataset. It does not
average individual FPS values. Initialization and edit-initialization times are
listed separately; edit initialization is part of initialization, not an extra
term to add. Per-frame timings remain in each run's `timing.json`.

The retained full-pipeline PASLCD measurement is 17.2 FPS on an RTX 5090 under this
timing protocol. The public CLI has also been checked on Cantina (Instance 1):
25 frames at 15.57 FPS, compared with 15.65 FPS in the retained two-run Cantina
measurement. This single-scene check is separate from the Full-20 result.

## Evaluation details

### Evaluate existing predictions

These commands score saved Quick start predictions without running inference:

```bash
python run_evaluation.py PASLCD --evaluate-only
python run_evaluation.py CCS3D --evaluate-only
```

For other results, supply the same `--data-root`, `--output`, `--pixel-only`, and
scene selection used for detection. The expected prediction layout is
`<output>/Instance_<1 or 2>/<scene>/predictions/` for PASLCD and
`<output>/<scene>/predictions/` for CCS3D. Terminal summaries show percentages;
JSON metrics use [0, 1].

### Full-20 evaluation

The PASLCD evaluator matches filename stems, resizes saved probability maps to
the ground-truth resolution, and applies threshold 0.5. Missing predictions,
unexpected frames, duplicate stems, or unreadable images stop evaluation.
JSON metrics use the range [0, 1]; terminal summaries show percentages.

The Full-20 evaluator first checks all 20 instances: each must have exactly 25 query
images, corresponding GT masks, and predictions for every requested output. It then
averages frames within each instance and weights all 20 instances equally. The JSON
contains both pixel and full results, per-instance scores, and per-frame TP/FP/FN.
The dataset score is not pooled-pixel IoU.

The arXiv results are:

| Output | Mean IoU (%) | F1 (%) |
| --- | ---: | ---: |
| Pixel | 60.36 | 73.80 |
| Full | 67.64 | 78.87 |

### CCS3D evaluation

**Evaluation protocol.** The target is the union of the pre-state and post-state
change masks. Evaluate the exported `mask/` and `refined_mask/` images, which are
thresholded at 0.5 during inference. Binary predictions and pre-state masks are
resized to the post-state GT canvas with nearest-neighbour interpolation, then
read with a value threshold of 127. The evaluator reports foreground precision,
recall, F1, IoU, and per-frame TP/FP/FN. Missing labels or predictions cause an error.
It averages frames within each scene and then weights the four scenes equally;
it does not pool pixels or all 18 frames into one score.
Full-dataset evaluation requires all four distinct scenes and their complete
per-frame results: Livingroom 5, Desk 4, Bookcase 5, and Bedroom 4. Missing scenes,
incorrect frame counts, duplicate frame records, or mismatched frames between
prediction stages stop aggregation without writing a summary.

The retained paper predictions under this protocol give:

| Output | Mean IoU (%) | F1 (%) |
| --- | ---: | ---: |
| Pixel | 66.83 | 79.47 |
| Full | 75.82 | 85.90 |

These numbers use the **pre ∪ post** protocol, not the original benchmark's
post-state-only protocol. Interpolating probability maps before thresholding is
also a different evaluation; do not use the PASLCD evaluator to reproduce this
table. Re-evaluating retained predictions verifies the metric implementation, not
a fresh training/inference reproduction.

## Code layout

| Location | Purpose |
| --- | --- |
| `install.sh` | Create an isolated Conda environment, configure CUDA, and build the included extensions |
| `gsedit_scd/generate.py`, `gsedit_scd/data/` | Automatic object edits, relighting, and change labels |
| `gsedit_scd/train.py`, `gsedit_scd/learning/` | Fit response suppression, likelihood-ratio mapping, unchanged-response statistics, pixel prior, and contour filter |
| `gsedit_scd/infer.py`, `gsedit_scd/online/` | Query registration, current-view support, historical support, closed-form Bayes factors, and refinement |
| `gsedit_scd/register.py` | Check the offline reference subset and save its poses, or cache registered query observations |
| `gsedit_scd/compile_decoder.py` | Compile the shared SAM2 mask decoder before inference |
| `prepare_data.py` | Download, extract, and prepare either dataset; check shared SAM2 model files |
| `prepare_ccs3d.py` | Adapt extracted CCS3D images, COLMAP cameras, reconstruction, and labels |
| `run_training.py` | Check references, generate edits, and fit all scene parameters |
| `run_evaluation.py` | Run and evaluate either dataset, with pretrained or fitted parameters |
| `run_timing.py` | Run PASLCD timing and summarize sequence throughput across instances and repeats |
| `evaluate.py` | PASLCD single-instance and Full-20 probability-map evaluation |
| `evaluate_ccs3d.py` | CCS3D binary-mask evaluation and equal-weight scene aggregation |
| `pretrained/` | Fitted parameters for all 20 PASLCD instances and four CCS3D scenes |
| `examples/initialization/PASLCD/` | One fixed reference edit per instance for timing initialization |
| `gsedit_scd/vendor/oscd/`, `submodules/` | Required third-party registration and rendering code |

## License

Original code is licensed under [MIT](LICENSE.md). Our fitted
[checkpoints](pretrained/LICENSE.md) and [edit samples](examples/initialization/LICENSE.md)
are licensed under CC BY 4.0; third-party code retains its original licenses.
The original datasets and reference reconstructions are obtained separately from
their respective authors.

## Third-party code

Registration and scene primitives are adapted from O-SCD and its dependencies.
See [THIRD_PARTY.md](THIRD_PARTY.md) for attribution and license locations.
Datasets, SAM2/XFeat weights, reference reconstructions, and machine-specific
compiled binaries are obtained or built separately. The repository includes fitted
scene parameters in `pretrained/` and 20 rendered PASLCD edits in
`examples/initialization/PASLCD/` for timing initialization.

## Validation status

The new installation script has passed syntax and existing-environment protection
checks; a full clean installation using this script has not yet been run.

Verified on Linux with Python 3.12, PyTorch 2.11.0, CUDA 12.8, and an RTX 5090.
A separate environment built both included CUDA extensions and all four SAM2
decoder variants from this package. Cantina completed the full offline workflow,
as well as 25-frame inference and evaluation using the bundled parameters.
Checkpoint-based inference required no edit generation or fitting; its pixel and
full mIoU differed from the preceding inference check by less than 0.01 percentage
points. All 24 parameter sets load and match the Gaussian counts of their reference
reconstructions.

The PASLCD evaluator was checked against retained Full-20 predictions, and the
CCS3D adapter and evaluator against retained four-scene data. Completeness checks
reject missing frames or scenes. The public timing entry was verified on Cantina.
Fresh inference from this release package has been checked on Cantina; the full
20-instance and four-scene results above come from the retained experiments.

## Citation

```bibtex
@article{zhu2026learning,
  title={Learning from Automatic 3D Gaussian Splatting Edits for Online Bayesian Scene Change Detection},
  author={Zhu, Changrui and Julier, Simon},
  journal={arXiv preprint arXiv:2610.28490},
  year={2026},
  url={https://arxiv.org/abs/2610.28490}
}
```
