# XD-RobustPVOS / DAG-RS

**Cross-Domain and Compound-Degradation Robust Promptable Video Object
Segmentation via Downstream-Agreement-Guided Restoration** — training-free.

This repository is the experimental package behind the paper: a reproducible
degradation benchmark, a training-free restoration-selection method (**DAG-RS**,
Downstream-Agreement-Guided Restoration Selection), and an evaluation protocol
with error bars and significance tests. Everything runs on a single 16 GB GPU.

Two conventions are worth stating up front, because they are what makes the
numbers in `results/` checkable:

* **One inference path.** The method, every baseline and every ablation arm are
  the same loop in `xdrp/pipeline.py`, selected by configuration flags only. No
  comparison arm has an implementation of its own.
* **Every number has an owner.** Any value that reaches the paper is produced by
  a named script and reconciled against the raw result files by a guard script
  (`scripts/118`–`128`); `scripts/00_smoke_test.py` runs the whole chain.

**Shipped:** the code (`xdrp/`, `scripts/`), the configuration (`configs/`), and
the results (`results/`).
**Not shipped:** the datasets, the model weights, the manuscript source, and the
internal working notes. Section 3.8 and `scripts/_release_gate.py` state which
checks therefore report `SKIP` instead of running.

### Contents

1. [Environment setup](#1-environment-setup)
2. [Data preparation](#2-data-preparation)
3. [Experimental steps](#3-experimental-steps)
4. [Directory structure](#4-directory-structure)
5. [FAQ](#5-faq)

---

## 1. Environment setup

### 1.1 Hardware and driver requirements

The reference machine is an **RTX 5080 Laptop GPU** (Blackwell, compute
capability **sm_120**, 17.1 GB = 15.9 GiB usable VRAM). Blackwell is the binding
constraint: a PyTorch/CUDA build that predates sm_120 support installs without
complaint and then fails at the first forward pass with

```
CUDA error: no kernel image is available for execution on the device
```

| Component | Requirement | Reason |
|---|---|---|
| NVIDIA driver | ≥ R570 (576.02+ recommended) | CUDA 12.8 runtime |
| PyTorch | ≥ 2.7.0 (2.9+ recommended) | first release with native sm_120 kernels |
| Wheels | **cu128** | CUDA 12.8 build; on Windows, PyPI defaults to a CPU wheel |

On an older GPU none of this applies — the constraint is the hardware, not the
code.

### 1.2 Create the environment

Two routes. Every result in this repository was produced in a conda environment
named **`py12-torch`** (Python 3.12.13):

```bash
conda activate py12-torch
```

This is what was verified on the reference machine:

| | |
|---|---|
| Python | 3.12.13 |
| torch | 2.12.0.dev20260408+cu128 |
| torchvision | 0.27.0.dev20260407+cu128 |
| CUDA runtime | 12.8 |
| `torch.cuda.is_available()` | `True` |
| Device | NVIDIA GeForce RTX 5080 Laptop GPU, capability `(12, 0)`, 17.1 GB |
| Other | numpy 2.5.2 · opencv 5.0.0 · scipy 1.18.0 · pandas 3.0.5 · matplotlib 3.11.1 · PyYAML 6.0.3 · tqdm 4.69.0 · Pillow 12.3.0 · scikit-image 0.26.0 · `sam2` importable |

Or build one from scratch:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Linux / macOS
```

### 1.3 Install PyTorch first, from the cu128 index

The order is not cosmetic — the GPU wheel has to be in place before anything that
might pull a CPU build in behind it.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verify immediately:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
# expected: 2.x.x+cu128  True  (12, 0)
```

`(12, 0)` is the value to check. Anything else means an older wheel was resolved.
Use `--index-url` rather than `--extra-index-url`: the latter lets pip fall back
to the CPU wheel on PyPI without printing a warning.

### 1.4 Install the remaining dependencies

```bash
pip install -r requirements.txt
```

Covers numpy, OpenCV, SciPy, Matplotlib, PyYAML, tqdm, pandas, Pillow (the
DAVIS/MOSE palette-PNG label reader) and scikit-image (the boundary-metric
cross-check and the F4 contour overlay). It deliberately does **not** pin torch
— see 1.3.

### 1.5 Install SAM 2.1 and fetch the checkpoints

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e .
cd checkpoints && bash download_ckpts.sh     # sam2.1_hiera_{tiny,small,base_plus,large}.pt
```

**A failed CUDA-extension build during this install is not a problem.** SAM 2's
`_C` extension only accelerates mask post-processing (filling small holes,
removing fragments); without it SAM 2 falls back to a pure-Python path with
identical segmentation output. The message
`Failed to build the SAM 2 CUDA extension during installation` can be ignored, and
installing the CUDA toolkit or Visual Studio build tools to silence it is wasted
effort.

Point the code at the checkpoint (defaults shown):

```yaml
# configs/default.yaml
backend:
  checkpoint: checkpoints/sam2.1_hiera_large.pt
  model_cfg: configs/sam2.1/sam2.1_hiera_l.yaml
```

### 1.6 Choosing a checkpoint on a 16 GB card

| Checkpoint | Weights | Comfortable on 24 GB | Recommendation at 16 GB |
|---|---|---|---|
| `sam2.1_hiera_large` | ~900 MB | yes | usable; set `cache_size: 2–4` |
| `sam2.1_hiera_base_plus` | ~320 MB | yes | best choice while iterating |
| `sam2.1_hiera_small` | ~180 MB | yes | use for the full sweeps |

Start from the default `cache_size: 4`. On OOM drop it to 1, or set
`cache_embeddings: false`.

---

## 2. Data preparation

Two sources, and only two, feed every number in the paper: **DAVIS-2017 val**
(2.1) and a subset of **MOSEv2** (2.2). Neither is committed — `data/` is
excluded — so everything in this section is something you fetch or build locally.

### 2.1 DAVIS-2017 — the primary benchmark

Download **only** this archive:

> https://davischallenge.org/davis2017/code.html  
> → **Semi-supervised** → **TrainVal** → **480p** →
> `DAVIS-2017-trainval-480p.zip` (~560 MB)

**Do not use the *Unsupervised* TrainVal archive.** The DAVIS release notes state
that its sequences match Semi-supervised but its annotations differ: it carries no
first-frame annotation. This protocol is the standard PVOS setting, in which the
first frame's ground-truth mask *is* the prompt — `DavisReader.object_ids()` reads
`label(seq, 0)` and `first_mask()` reads frame 0. The unsupervised archive raises
no error; it simply cannot supply the prompt.

**Do not use Test-Dev / Test-Challenge.** They publish images and first-frame
annotations only; later frames are scored server-side and never released, so J&F
cannot be computed locally. The paper's tables use the 30-sequence val split.

Expected layout (flattened):

```
data/DAVIS/JPEGImages/480p/<seq>/<00000>.jpg
data/DAVIS/Annotations/480p/<seq>/<00000>.png
data/DAVIS/ImageSets/2017/val.txt
```

Two traps, and both are silent:

1. **One directory level too many.** The zip unpacks into a top-level folder.
   `DavisReader` accepts `<root>`, `<root>/DAVIS` and `<root>/davis`, so
   `JPEGImages` has to end up directly under `data/DAVIS/`; otherwise the reader
   raises `FileNotFoundError` at start-up.
2. **A missing `val.txt` sweeps the whole dataset.** If
   `ImageSets/2017/val.txt` is absent, `DavisReader.sequences()` falls back to
   *every* sequence under `JPEGImages` (train 60 + val 30 = 90) without raising,
   which triples the runtime of the decision experiment.

Verify in ten seconds, with no GPU:

```bash
python scripts/01_build_benchmark.py
```

The `Dataset` block must print **`sequences=30 frames=...`**. `sequences=90` means
the split file was not found; `[WARN] dataset unavailable` means the directory was
not flattened. On the reference download the val split holds **30 sequences and 61
annotated instances** (17 of the sequences have more than one object), which is
what `protocol.max_objects: 0` scores.

**Use 480p, not Full-Resolution.** The published DAVIS metric is defined at 480p,
and SAM 2 resizes its input to 1024 regardless, so full resolution only multiplies
the cost of the degradation operators while breaking comparability with published
VOS numbers. Download Full-Resolution separately if you later want an
input-resolution study.

**Optional, small, worth having:** the same page, Object categories → TrainVal →
480p (`DAVIS-2017_semantics-480p.zip`). It carries per-frame semantic classes and
the object-to-superclass mapping, which support a per-category breakdown table. Do
not download Scribbles.

**Before submitting**, re-score one sequence with the official
`davis2017-evaluation` package and compare it against `xdrp/metrics.py`; the two
must agree to numerical precision.

### 2.2 The cross-source subset (MOSEv2 mini)

The cross-source check needs a second, independent VOS source that is *not* an
artefact of DAVIS. This subset is materialised into the **MOSE** layout — not the
DAVIS one — and is read by `MoseReader`:

```bash
python scripts/112_prepare_mose_mini.py     # parquet shards -> data/MOSE_mini
python scripts/113_check_mose_mini.py       # validate; writes VALIDATION.txt
```

```
data/MOSE_mini/JPEGImages/<video_id>/<5-digit>.jpg
data/MOSE_mini/Annotations/<video_id>/<5-digit>.png   <- palette PNG; the palette
                                                         index IS the object id
data/MOSE_mini/{PROVENANCE.json,VALIDATION.txt,manifest.csv}
```

There is no `480p/` level and no `ImageSets/` here, so `DavisReader` would reject
this tree (`missing .../JPEGImages/480p`). The run uses
`configs/_tau013_mose.yaml`, which is `configs/_tau013.yaml` with **exactly three
non-comment lines changed** — `dataset.kind: mose`,
`dataset.root: data/MOSE_mini`, `dataset.split: valid`. Every other key,
including the protocol and the DAG-RS hyper-parameters, is byte-identical. That is
what makes "one protocol, two sources" a fact rather than an argument; the
configuration says so itself, and `diff` confirms it.

Because the label PNGs are palette-encoded with the palette index as the object
id, they must be read with the project's own `xdrp/datasets.py::read_label_png`,
never with OpenCV.

**Why the subset comes from the `train` split:** MOSE publishes first-frame
annotations only for *val* and keeps the test set server-side, so a whole-clip
J&F cannot be computed from MOSE val. It is unusable here, not merely unused —
which is exactly why the second source is a subset of MOSEv2 **train**.

Its provenance is recorded beside the data in `data/MOSE_mini/PROVENANCE.json` and
has to be carried into the paper verbatim:

* source: `FudanCVL/MOSEv2` **train** split, mirrored at
  `modelscope.cn/merve/mosev2-mini`;
* licence: **CC BY-NC-SA 4.0 — non-commercial academic use only**;
* it is a **community 120-video subset, not the official 2149-video MOSEv2
  release**, and the paper must say so, citing MOSE (ICCV 2023) and MOSEv2.

Two consequences: results on this subset are reported as a *within-source* gap
comparison, never as an absolute level comparable to DAVIS; and the
non-commercial licence governs any redistribution of what is derived from it.

### 2.3 Additional sources for the cross-domain row (optional)

Only needed if you extend the cross-domain claim beyond 2.2, which the shipped
protocol does not do. Take a source from outside the driving domain, from a
properly licensed public dataset; third-party web video of unknown provenance
must not be used. Either add a reader to the `READERS` registry in
`xdrp/datasets.py`, or convert the source into the **DAVIS** layout of 2.1 —
`JPEGImages/480p/<seq>/` plus `Annotations/480p/<seq>/`, under a directory you
then set as `dataset.root` — and reuse `DavisReader`.

### 2.4 Aligning with the original RobustPVOS release (optional)

The RobustPVOS project page publishes two real-world test sets (ACDC-Video, 149
sequences; MVSeg, 202 sequences). To compare against them directly, download and
lay them out as in 2.3, inherit their original licences, and state the source in
the paper.

---

## 3. Experimental steps

### 3.0 Protocol and cost unit

All protocol settings live in `configs/default.yaml` and are **frozen before any
test-set evaluation**; changing them after seeing results invalidates the
protocol. The defaults that matter:

| Setting | Value | Meaning |
|---|---|---|
| `protocol.frame_stride` | 4 | temporal subsampling, ~10–15 frames per DAVIS sequence |
| `protocol.max_objects` | 0 | score **every** annotated instance (the official DAVIS protocol) |
| `protocol.dilation` | 2 | boundary-F tolerance, in pixels |
| `protocol.seed` | 0 | degradation RNG seed |
| `sweep.degradations` | 8 single (+ `clean` as reference) | Tier-1 degradations |
| `sweep_compound.degradations` | 4 | compound degradations — the headline setting |
| `sweep.levels` | 1.0 – 5.0 | severity levels |

`--degradations single` resolves to the **8 single degradations and does not
include `clean`**; ask for `--degradations clean,fog,...` when a degraded-vs-clean
retention column is needed. `compound` resolves to the four compound names, whose
definitions and order are part of the benchmark
(`xdrp/degradations.py::COMPOUND_DEGRADATIONS`).

**The cost unit of this project is encoder calls per frame.** One 1024×1024 image
encoding takes ~0.25–0.4 s on the reference machine, so the encoding count
predicts wall-clock almost exactly:

| Arm | Encoder calls / frame | Note |
|---|---|---|
| `greedy`, `cascade`, `fixed_op`, `random_op`, `oracle_clean` | 1.0 | one encoding per frame |
| `sam2video` | 1.0 | one backbone forward per frame **plus a memory bank**; the only arm that is not a per-frame backend — it tracks the whole clip at once (`xdrp/video_backend.py`) |
| `dagrs`, default (K=5, J=3) | 1 + 3/5 = 1.6 | J extra evaluations on decision frames |
| `dagrs` (K=5, J=13) | 3.6 | J at its sweep maximum |
| `dagrs` (K=1, J=3) | 4.0 | a decision on every frame |
| `equal_tta` | 13.0 | all 13 operators at every frame; an ablation control, and by far the most expensive arm |

Read that table before picking a first command: `equal_tta` alone accounts for
roughly 72 % of a four-arm baseline sweep. Section 3.10 gives the measured totals
and the cheapest ways to cut them.

### 3.1 Step 0 — self-check, no data and no weights

```bash
python scripts/00_smoke_test.py
python scripts/05_analysis.py --self-test
```

`00_smoke_test.py` drives the whole pipeline on synthetic video with a dummy
backend: reproducible degradations, the operator bank, the Dirichlet invariants,
the J/F metrics, the statistics, the full DAG-RS loop, configuration plumbing, and
the guard chain of 3.8. Do not continue if it exits non-zero.

A large part of that suite exists because a class of defect is invisible in the
result tables: a mechanism that never fires, a config flag the YAML round-trip
drops, a propagation arm that silently receives the restoration bank, a protocol
that scores one instance instead of all of them. Each known instance of that class
is now an assertion.

`05_analysis.py --self-test` verifies the correlation machinery against data with
a known answer: independent data must give ρ ≈ 0, monotone data |ρ| > 0.9. It also
regression-tests the rules that keep the central claim honest — the clean
reference must not enter the correlation, different segmentation arms must not be
pooled, multiple instances on one frame must be averaged before fitting, and the
verdict must be taken on effect size rather than p.

**Synthetic data cannot decide the paper's central question.**
`99_synth_dataset.py` together with `--backend dummy` verifies that the pipeline is
connected; its J&F values are degenerate, because the dummy backend pins its
colour model to the first frame and any global photometric change makes it lose
the target, so past frame 1 almost everything scores 0. Four different arms
produce an identical correlation on it because their target vectors are the same
string of zeros. Use it to check plumbing, never to read a direction.

### 3.2 Step 1 — build the benchmark

Report only — writes no files, and nothing downstream reads it:

```bash
python scripts/01_build_benchmark.py
```

Materialising writes the degraded frames to PNG:

```bash
python scripts/01_build_benchmark.py --degradations single   --levels 1,2,3,4,5 --materialize data/XD-RobustPVOS
python scripts/01_build_benchmark.py --degradations compound --levels 1,2,3,4,5 --materialize data/XD-RobustPVOS
```

Materialising is **not a prerequisite** for the experiments: every cell is
generated on the fly from the original dataset, and only `--materialize` writes
degraded frames to disk. What it buys is bit-identical degradations across OpenCV
versions and machines, plus saved CPU on repeated sweeps. Do it for the
camera-ready archive, not on day one.

### 3.3 Step 2 — the decision run

Run this first. The paper's central correlation either survives it or does not, and
that has to be known before anything else is scheduled.

```bash
# (a) pipeline check, no model and no data
python scripts/02_run_baselines.py --quick --backend dummy --out results/_smoke_baselines.csv

# (b) real SAM path check: one arm, one sequence, cheapest setting
python scripts/02_run_baselines.py --modes greedy --max-sequences 1 \
    --frame-stride 16 --levels 3 --degradations fog --out results/_smoke_sam.csv

# (c) the decision sweep: the single arm the correlation is computed on
python scripts/02_run_baselines.py --modes greedy --degradations single \
    --levels 1,2,3,4,5 --frame-stride 4 --resume --out results/baselines_raw.csv

# (d) the decision itself
python scripts/05_analysis.py --study iqa --iqa-mode greedy
```

Three points that are easy to get wrong:

* `--quick` does **not** switch to the dummy backend — `backend.kind` stays
  `sam2`, so `--quick` alone still requires SAM 2 and a checkpoint. Pass
  `--backend dummy` explicitly to test the plumbing without a model.
* Always pass `--out` for a trial run. The default output paths are fixed names,
  so a trial without `--out` overwrites the real result file.
* The correlation is computed on **one** segmentation arm (`--iqa-mode`), by
  design. If the direction looks right, re-run (d) with the cheap arms
  `cascade,fixed_op,random_op` to obtain the per-arm sensitivity table that
  answers "is this an artefact of the arm you picked?".

### 3.4 Step 3 — the baseline sweep

```bash
# the four cheap arms: ~20 h
python scripts/02_run_baselines.py --modes greedy,cascade,fixed_op,random_op \
    --degradations single --levels 1,2,3,4,5 --resume --out results/baselines_raw.csv

# equal_tta on its own: 13x the cost of greedy
python scripts/02_run_baselines.py --modes equal_tta \
    --degradations single --levels 1,2,3,4,5 --resume --out results/baselines_equal_tta.csv
```

Available arms: `greedy` (main baseline), `greedy_gate`, `greedy_reseed`,
`greedy_robust`, `cascade`, `fixed_op`, `equal_tta`, `random_op`, `oracle_clean`
(upper-bound reference, fed the clean frames) and `sam2video` (the released
SAM 2.1 video predictor, which anchors the absolute numbers). `--modes` takes any
comma-separated subset.

`equal_tta` is the only arm that can be restricted to a subset of sequences
without weakening the comparison, since it is a control. If you restrict it, say
so in the protocol section.

### 3.5 Step 4 — the method

```bash
python scripts/03_run_dagrs.py --degradations single   --levels 1,2,3,4,5 --resume
python scripts/03_run_dagrs.py --degradations compound --levels 1,2,3,4,5 --resume
```

**Re-prompt convention (`hard` vs `soft`).** The propagated anchor is re-prompted
either as a binarised mask (`hard`, the default, faithful to the released
baselines) or by carrying the segmenter's own mask probability across frames
(`soft`), which lets the boundary be re-decided from the current image. `soft` is
a **global protocol switch**: it applies to the method and to the baselines alike,
so it must be reported as a protocol setting, with the A/B as an appendix
ablation.

```bash
# the cheap half first: six sequences, one process, byte-identical degraded frames
python scripts/92_ab_mask_prompt.py \
    --seq lab-coat,drift-straight,bmx-trees,shooting,cows,blackswan \
    --degradations clean,fog --levels 3 --frame-stride 1 --mode greedy

# the full-stride A/B (stride 1 multiplies the frames per cell by ~5)
python scripts/02_run_baselines.py --modes greedy --degradations single --levels 1,2,3,4,5 \
    --mask-prompt soft --frame-stride 1 --resume --out results/soft_baselines.csv
```

`92_ab_mask_prompt.py` reports the per-frame and per-cell `area_ratio` — the
direct runaway indicator, `last`/`max`, where > 3 counts as runaway — alongside
J&F. `cows` and `blackswan` are the control group: a convention that fixes the
runaways but damages sequences that were already fine is not a net gain.

### 3.6 Step 5 — ablations

```bash
python scripts/04_ablation.py --list           # arm legend
python scripts/04_ablation.py                  # the six trunk arms A0,A1,A3,A5,A6,A7 (~40 h)
python scripts/04_ablation.py --variants all   # 11 arms + three sweeps (>150 h)
```

A0–A10 are described in the script's own legend. Every arm reuses the identical
inference path and differs only in configuration, so no arm can quietly benefit
from a different implementation; an arm whose mechanism is not implemented raises,
rather than producing a duplicate row.

### 3.7 Step 6 — cost and memory

```bash
python scripts/06_profile.py --degradation fog --level 4 --seqs 3
```

Reports encoder calls, wall-clock FPS and peak GPU memory per arm — the source of
the complexity table in the paper, and of the per-frame cost unit used in 3.0.

### 3.8 Step 7 — aggregation, and the guard chain

```bash
python scripts/05_analysis.py --study all      # tables and figures from the CSV results
python scripts/124_fig_build.py                # the data figures + the figure manifest
python scripts/128_fig4_qualitative.py         # FIG-4, the qualitative panel
```

`--study all` covers the five analysis studies: correlation (`iqa`), significance
tests (`stats`), vacuity and gating diagnostics (`vacuity`), the result tables
(`tables`) and the figures (`figs`). Point `--out-dir` at a scratch directory to
preview a study on an incomplete sweep, instead of writing half-finished tables
into `results/`.

The numbers that go into the paper are then reconciled by the guard chain, which
`00_smoke_test.py` runs end to end:

| Guard | Owns |
|---|---|
| `118_p1_doc_sync.py` | the primary-protocol (P1) numbers |
| `121_sec5_table_audit.py` | the Section-5 table rows |
| `122_ms_doc_guard.py` | document-level numbers, wording, keys, sizes |
| `123_ref_guard.py` | bibliography integrity |
| `124_fig_build.py` | the data figures — build, manifest and refusal controls |
| `125_cl3_table_rows.py` | the correlation table and its restatements |
| `126_table7_rows.py` | the complexity table |
| `127_make_docx.py` | the Word manuscript (local only — see 3.9) |
| `128_fig4_qualitative.py` | FIG-4 selection and provenance |

Several of these read internal working documents that are **not distributed**.
`scripts/_release_gate.py` is the single declaration of what exists only locally,
and it is what turns "my input is missing" into a reported `SKIP` rather than a
silent pass or a hard failure:

```bash
python scripts/_release_gate.py --check
python scripts/_release_gate.py --self-test
```

In a clone, the affected suites print `SKIP-UNPUBLISHED-INPUT` and the run summary
lists them under `Skipped suites`. In the authors' checkout, where every declared
input exists, a `SKIP` would mean a document went missing — which is why the gate
is audited in both directions and is not allowed to skip quietly. A `SKIP` is not
a pass: it means a check could not run here.

`results/figs/FIGURES.json` is the figure manifest. Each entry names the artifact
a figure is built from and a SHA-1 digest of the exact numbers drawn, so a figure
whose source table has drifted is detectable rather than merely suspected.

### 3.9 Step 8 — build the Word manuscript (authors' checkout only)

```bash
python scripts/127_make_docx.py             # build
python scripts/127_make_docx.py --check     # rebuild to a temp file and compare
python scripts/127_make_docx.py --self-test
```

Converts the manuscript source into the submitted `.docx`: front matter and the
drafting notes are dropped, figures are embedded after the paragraph that
introduces them, and every caption fact is asserted against the figure manifest.
It is fail-closed — a partial conversion cannot be reported as a build, and
`--check` fails if the manuscript changes after the fact.

This step requires the manuscript source, which is not distributed, so it cannot
run in a clone; `--self-test` exercises the transform logic without it.

### 3.10 Budget, and how to reduce it

Measured on the reference machine (RTX 5080 16 GB, SAM 2.1 Hiera-L, bf16). One
cell is 30 sequences × ~13 frames at stride 4; `--degradations single` is 8
degradations × 5 levels = 40 cells per configuration, i.e. 1200 cells for one arm.

Per frame, at 480p:

| Stage | Time | Charged per |
|---|---|---|
| SAM 2 encode + decode | ~0.25 s | instance |
| Farneback flow (one pair) | ~0.09 s | instance |
| Degradation rendering | ~0.03 s | sequence |
| IQA panel (8 metrics) | ~0.24 s | sequence (cached per frame) |

⇒ one cell (≈ 2.03 instances × 13 frames) ≈ **15 s**. Extrapolated totals:

| Command | Cells | Measured / extrapolated |
|---|---|---|
| `--modes greedy --degradations single` | 1200 | ~5 h |
| `greedy,cascade,fixed_op,random_op` | 4800 | ~20 h |
| plus `equal_tta` | +1200 | +43 h |
| `03_run_dagrs.py --degradations single` | 1200 | ~8 h |
| `03_run_dagrs.py --degradations compound` | 600 | ~4 h |
| `04_ablation.py` (six trunk arms) | 7200 | ~40 h |
| `04_ablation.py --variants all` | — | >150 h |

Levers, in order of value: `--frame-stride 8` (halves the cost),
`--levels 1,3,5` (5 levels → 3), `--max-sequences 10` (a third of the cost, and
the correlation still has thousands of points), restrict degradations to those the
paper actually reports, and switch to `sam2.1_hiera_small`.

Any reduction has to be stated in the paper's protocol section, applied
**identically to every arm**, and the reported numbers must all come from one
stride — mixing strides within a table is not allowed.

### 3.11 Long sweeps and resuming

Every sweep writes incrementally and skips completed cells when `--resume` is set,
so a multi-hour sweep is run by calling the same command repeatedly:

```bash
python scripts/02_run_baselines.py --modes greedy --degradations single \
    --levels 1,2,3,4,5 --frame-stride 4 --resume --out results/baselines_raw.csv
```

Without `--resume`, results are written once at the end of a clean run, and a run
killed part-way leaves nothing behind. `scripts/_sweep_progress.py` reports how
much of a sweep is done and the remaining time from the CSV alone, without
touching the GPU. For very long sweeps the protocol can also be split across
processes with `--seq-offset`, then reassembled with `scripts/94_merge_sweeps.py`.

### 3.12 Runs that are deliberately not part of the protocol

* **`03_run_dagrs.py --ood`** — the unseen-severity transfer study was withdrawn
  (authors' decision, 2026-09-23). The severity axis is already answered directly
  by levels 1–5, so `--ood` would only interpolate; the cross-domain evidence is
  carried by the independent MOSEv2 subset instead. The flag is kept for audit and
  is not run in the submission pipeline.
* **Synthetic correlations** — see 3.1. `99_synth_dataset.py` checks plumbing
  only.
* **Retired ablation arms** — `04_ablation.py` raises for an arm that is declared
  but not implemented, rather than measuring nothing.

### 3.13 Command reference

| Script | Purpose |
|---|---|
| `00_smoke_test.py` | full-chain self-check; no data, no weights |
| `01_build_benchmark.py` | benchmark report / materialisation |
| `02_run_baselines.py` | the baselines, `greedy` … `sam2video` |
| `03_run_dagrs.py` | the method (single and compound); `--resume`, `--seq-offset` |
| `04_ablation.py` | A0–A10; `--list` for the legend |
| `05_analysis.py` | iqa / stats / vacuity / tables / figs |
| `06_profile.py` | encoder calls, FPS, peak GPU memory |
| `_sweep_progress.py` | sweep completion and ETA, GPU-free |
| `90_diagnose_greedy.py` | per-frame geometry and prompt ablation (diagnostic) |
| `91_flow_audit.py` | flow-propagation direction/strength audit — **re-run after any change to `xdrp/flow.py`** |
| `92_ab_mask_prompt.py` | `hard` vs `soft` re-prompt, same-frame A/B |
| `93_trace_runaway.py` | runaway onset and trajectory shape, from `92_*` output |
| `94_merge_sweeps.py` | merge chunked sweeps; `--trim` drops truncated cells |
| `95_attr_runaway.py` | same-frame attribution of runaways |
| `96_ab_accept.py` | propagation acceptance strategies, A/B |
| `97_flow_chain_area.py` | flow-chain area and IoU, GPU-free |
| `98_vac_gate_audit.py` | gating reachability / replay / information, GPU-free |
| `99_synth_dataset.py` | synthetic dataset for plumbing checks |
| `100_pair_probe_report.py` | paired per-arm report, GPU-free |
| `101_pair_frame_diff.py` | per-frame A/B trace for one sequence |
| `102_descriptor_contract.py` | descriptor-width contract (loads the model) |
| `103_video_postproc_ab.py` | effect of SAM 2's low-resolution hole filling |
| `104_video_offload_ab.py` | whether CPU offload buys the VRAM it is enabled for |
| `105`–`107` | evaluator-protocol A/B, self-check, and rescoring under the released convention |
| `108_compound_vs_member.py` | compound degradations vs their single members |
| `109_iqa_config_level.py` | configuration-level correlation, GPU-free |
| `110_ablation_table.py`, `111_severity_table.py` | the ablation and severity tables |
| `112`–`117` | cross-source subset preparation, table, denominator rule and verification |
| `118`–`126` | the guard chain of 3.8 |
| `127_make_docx.py`, `128_fig4_qualitative.py` | the Word manuscript and FIG-4 |
| `_common.py`, `_release_gate.py`, `eval_protocol_common.py` | shared: CLI/config helpers, release gate, released evaluator |

All sweep scripts share `--quick`, `--backend dummy`, `--max-sequences`,
`--frame-stride`, `--max-frames`, `--max-objects`, `--out`, `--resume`,
`--seq-offset`, `--config`, `--seed`, `--mask-prompt` and `--no-iqa`.

---

## 4. Directory structure

```
xd-robustpvos/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   ├── default.yaml                every hyper-parameter (frozen before evaluation)
│   └── sam2.1/                     SAM 2.1 model configs
├── xdrp/
│   ├── degradations.py             reproducible degradation engine: 8 single + 4 compound, continuous severity
│   ├── operators.py                training-free restoration bank (M = 13)
│   ├── evidence.py                 downstream-agreement scoring, Dirichlet aggregation, degradation profile
│   ├── flow.py                     Farneback flow and chained mask warping
│   ├── sam_backend.py              SAM 2.1 wrapper (model-agnostic) plus the dummy backend
│   ├── video_backend.py            SAM 2.1 video predictor — the memory-bank arm
│   ├── pipeline.py                 the single inference loop: method, baselines, ablations
│   ├── benchmark.py                protocol, degraded-sequence loading, sweep driver
│   ├── metrics.py                  J / F / J&F
│   ├── iqa.py                      PSNR/SSIM and no-reference statistics
│   ├── stats.py                    bootstrap CI, Wilcoxon, Holm, Spearman, Cliff's delta
│   ├── datasets.py                 DAVIS / MOSE / YouTube-VOS readers
│   └── viz.py                      plotting
├── scripts/                        00 environment self-check · 0x experiment entry points ·
│                                   9x, 1xx diagnostics, table and figure guards
├── results/                        outputs: CSV, LaTeX, figures
│   ├── p1/                         the primary-protocol table values
│   ├── figs/                       FIG-1 … FIG-5 and the figure manifest
│   └── ablation/                   per-arm ablation CSVs
├── docs/                           self-test reports (the manuscript source is not distributed)
└── data/                           datasets — download separately, never committed
```

---

## 5. FAQ

**`CUDA error: no kernel image is available for execution on the device`**
A PyTorch wheel without sm_120 kernels. Reinstall from the cu128 index:
`pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128`

**`torch.cuda.is_available()` is `False`**
Either the driver is older than R570, or pip resolved to a CPU wheel. On Windows
a plain `pip install torch` from PyPI gives the CPU build; the `--index-url` in
1.3 is what selects the GPU one.

**`Failed to build the SAM 2 CUDA extension`**
Ignore it. Mask post-processing falls back to Python with identical output, so
installing the CUDA toolkit to silence the message is not worth the time.

**Out of memory**
Set `cache_size: 1`, or `cache_embeddings: false`, or switch to
`sam2.1_hiera_base_plus`, in `configs/default.yaml`.

**`FileNotFoundError` on the dataset**
The archive was not flattened — see 2.1. `JPEGImages` must sit directly under
`data/DAVIS/`.

**Do I need YouTube-VOS, or MOSE val, to run this?**
No, and neither would work. The paper uses exactly two sources — DAVIS-2017 val
(2.1) and the MOSEv2 subset (2.2) — and this repository contains no
`data/YouTubeVOS/` and no `data/MOSE/`. YouTube-VOS 2019 val was considered and
left out of scope; it is also *not* cross-domain evidence, because the only
cross-source evidence is 2.2. MOSE val fails for a harder reason: it ships
first-frame annotations only, so a whole-clip J&F cannot be computed from it.

Note that `xdrp/datasets.py` does register a `youtubevos` reader, and every
`configs/*.yaml` carries the comment `# davis | mose | youtubevos`. That is the
reader's *capability* surface, not a record of what was run — no configuration
here selects it and nothing under `results/` was produced from it.

**The sweep is taking about three times the estimate**
Check the `Dataset` line printed by `scripts/01_build_benchmark.py`.
`sequences=90` means `val.txt` was not found and the train split is being swept
too.

**A guard reports `SKIP-UNPUBLISHED-INPUT`**
Expected in a clone. Those suites read internal working documents that are
deliberately not distributed; the authoritative list is declared in
`scripts/_release_gate.py`. A `SKIP` is not a pass — it means the check could not
run here.

**The propagated mask drifts, or flow looks wrong**
Run `python scripts/91_flow_audit.py`. Flow *direction* is the one thing in the
propagation step that results cannot reveal, because a reversed warp degrades the
metrics smoothly instead of failing. Any change to `xdrp/flow.py` must be followed
by this audit.

**Results are not reproducible**
Check, in order: the `seed` in `configs/default.yaml`; whether the benchmark was
materialised (on-the-fly degradation can differ across OpenCV versions); whether
any `dagrs` hyper-parameter was changed. Once the benchmark is materialised, an
identical configuration must reproduce bit-identically — which
`00_smoke_test.py` asserts.

**A non-ASCII path breaks OpenCV**
Keep the checkout on a pure-ASCII path on Windows.

**How do I know the results in `results/` are the current ones?**
The P1 products in `results/p1/` are what the paper's tables are built from, and
the guard chain of 3.8 fails if a document restates a number that no longer
matches them.

---

## License and citation

The code is released under the MIT licence — see `LICENSE`.

What this repository actually uses, and what you must cite alongside it:
**SAM 2** (Apache-2.0; the segmenter), **DAVIS-2017** (the primary benchmark), and
**MOSEv2** (the second source, through the community mini subset — CC BY-NC-SA
4.0, non-commercial academic use only). A benchmark that appears in the
literature but is not used here is deliberately *not* listed; §5 answers the
common cases, YouTube-VOS and MOSE val. If the RobustPVOS test sets (ACDC-Video,
MVSeg) are used, their original licences are inherited.

The complete bibliography is submitted with the paper rather than duplicated here,
where it would silently go stale; the manuscript source is not distributed with
this repository.
