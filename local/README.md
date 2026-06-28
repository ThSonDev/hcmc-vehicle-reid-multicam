# Local pipeline

Standalone multi-camera vehicle ReID pipeline: plain Python scripts plus a local Kafka.
Everything lives under this folder (`local/`) — its own `.venv`, dependencies, Kafka
compose, source, ground truth, and runtime data. The one external dependency is the
vendored `torchreid` fork at `../osnet/deep-person-reid` (shared with the Airflow stack),
which `setup.sh` installs editable into the venv.

Work from inside `local/`:

```bash
cd local
```

All commands below assume that, with the venv active (`source .venv/bin/activate`).

## Architecture (short)

Three camera videos are streamed as JPEG frames over Kafka; one consumer per camera runs
YOLO detection + ByteTrack + OSNet feature extraction, and the downstream cameras match
their vehicles back to the cam1 "source" gallery.

```
producer ──▶ video_reid_stream ──▶ cam1  ──▶ reid_gallery_stream ─┐
  (3 videos)                        (source)                       │
                                    cam2, cam3 ◀───────────────────┘
                                       │  match against cam1 gallery
                                       ▼
                                 reid_matches ──▶ visualizer
```

- **cam1** — source only; detects/tracks and publishes per-track embeddings to the gallery.
- **cam2** — matches frame-by-frame against the cam1 gallery (travel-time gated; one cam1 id used once). A periodic timestamp-based sweep (`evict_stale_state`, ~every heartbeat) evicts stale gallery entries **and** ID locks so memory stays bounded on long/looping runs and ids free up for reuse (same policy as cam3, extended to cam2's lock tables).
- **cam3** — buffers a "best shot" per track and matches on track exit; emits *new vehicle* when no match (it sits on a new road).
- **visualizer** — joins match events into one row per source vehicle.
- **monitor** — samples CPU/GPU/RAM/disk to `logs/`.

Kafka topics: `video_reid_stream` (frames), `reid_gallery_stream` (cam1 embeddings),
`reid_matches` (match events). Infra config (broker, topics, model/video paths) lives in
[`src/config.py`](src/config.py).

For the camera setup (resolutions, GT files, road layout/views) and a plain-language
walkthrough of the producer, see [`architecture.md`](architecture.md).

## Layout

```
local/
  run.py               orchestrator: launches every component (entry point)
  setup.sh             one-command environment bootstrap (creates .venv here)
  docker-compose.yaml  local Kafka (single-node KRaft, port 9092)
  requirements.txt     pinned dependencies
  .python-version      3.9.19
  README.md            this file
  architecture.md      camera setup + producer logic (plain language)
  src/
    producer.py          streams 3 videos -> topic video_reid_stream
    consumer_cam1.py     source: detect + track + publish gallery embeddings
    consumer_cam2.py     frame-by-frame matcher vs the cam1 gallery
    consumer_cam3.py     best-shot matcher (new road); emits new vehicles
    match_visualizer.py  2-pane match UI
    monitor.py           CPU/GPU/RAM/disk sampler
    report.py            run digest (Markdown + plots); --eval grades vs GT
    eval_metrics.py      trackeval HOTA/MOTA/IDF1 (called by report --eval)
    standalone_tracker.py  offline detect+track (no Kafka), auto-evaluates
    config.py log_utils.py gui_utils.py reid_utils.py   shared modules
  gt/                  gt_cam{1,2,3}.txt ground truth (global cross-camera IDs)
  data/                input videos               (gitignored)
  weights/             YOLO .pt + OSNet .pth      (gitignored)
  results/             res_cam*.txt MOT outputs   (gitignored)
  logs/                per-run JSONL + report.md  (gitignored)
  temp/                scratch                    (gitignored)
```

Paths are anchored to the `local/` folder via `config.ROOT`, so scripts run the same from
inside `local/` (`python run.py`) or from the repo root (`python local/run.py`).

## Requirements

- **Python 3.9.x** (tested 3.9.19, pinned in `.python-version`)
- [**uv**](https://docs.astral.sh/uv/) for dependency management
- Docker (local Kafka)
- A C compiler (`gcc`) — builds the `torchreid` Cython extension
- NVIDIA GPU + driver (optional but recommended; CPU works, slower)

## Setup

One command (idempotent), from inside `local/`:

```bash
./setup.sh
```

It creates `.venv` (Python 3.9.19) inside `local/`, installs `requirements.txt`, then
installs `supervision` and the editable `torchreid` (from `../osnet/deep-person-reid`) with
`--no-deps` (they must stay out of the resolvable block or their dep trees fight the pinned
legacy stack), and creates the `data/ weights/ results/ logs/ temp/` directories.

Then supply the gitignored assets:

- **Videos** → `data/cam1_640.mp4`, `data/cam2_640.mp4`, `data/cam3_640.mp4`
- **Weights** → `weights/cam1.pt`, `weights/cam2.pt` (YOLO), `weights/osnet_cam123.pth` (OSNet)

> Always run Python via the venv: `source .venv/bin/activate`, `uv run …`, or
> `.venv/bin/python …`. Never use the system `python`/`pip`.

## Run

With the venv active:

1. **Start Kafka** (must be up first, or consumers receive nothing):
   ```bash
   docker compose up -d
   ```
2. **Run everything** with one command:
   ```bash
   python run.py        # monitor + cam1/2/3 + visualizer + producer
   ```
   Five OpenCV windows open; matches show up in the visualizer. Stop with `quit` or Ctrl-C.
3. **See what happened** — concise the logs into a report + plots:
   ```bash
   python report.py     # -> logs/<stamp>_report.md + logs/images/*.png
   ```

`run.py` runs each component in its own process (its own window) and gives you a small
control console:

```
> status              # show what's running
> stop cam2           # stop just cam2
> restart cam3        # restart one component
> quit                # stop everything (or Ctrl-C)
```

Useful flags: `--only cam1,cam2`, `--exclude monitor`, `--no-monitor`, and `--once`
(stream each video a single pass, then auto-stop — use this before evaluating). Or press
`q` inside a camera window to close just that one.

**Remote / headless (SSH, no display):** add `--headless` to skip all OpenCV windows —
components run logic-only and write to console + JSONL, so you drive everything from the
report and logs. With a detached/piped stdin (no TTY) the control console is skipped and the
pipeline just runs until Ctrl-C, so `nohup python run.py --headless &` is safe.

```bash
python run.py --once --headless    # single pass, no windows -> then report --eval
```

### Run each component standalone

Each component also runs on its own (from inside `local/`, venv active):

```bash
python src/producer.py          # streams 3 videos -> topic video_reid_stream
python src/consumer_cam1.py     # source: detect + track + publish gallery embeddings
python src/consumer_cam2.py     # matches against cam1 gallery
python src/consumer_cam3.py     # same, with best-shot buffering on track exit
python src/match_visualizer.py  # 2-pane UI consuming reid_matches
python src/monitor.py           # system resource sampler
```

## Logs

Every component logs to the console **and** to a structured `logs/<stamp>_<component>.jsonl`
(size-rotated, one run per `<stamp>`) — re-read with e.g.
`jq 'select(.event=="match")' logs/<stamp>_cam2.jsonl`, or just run `report.py`. `run.py`
exports one shared `<stamp>` per launch so every component of a run logs together.

`run.py` additionally tees each component's raw stdout/stderr to
`logs/<stamp>_<component>.out` and writes orchestrator events to
`logs/<stamp>_run.jsonl` — including any component that dies unexpectedly, with its exit
signal (e.g. `SIGSEGV`/`SIGABRT`) and the tail of its output. This captures crashes that
never reach the JSONL logger (an uncaught exception, or a native CUDA/Qt abort). Heartbeats
fire on wall-clock time, so a frame-starved consumer still beats with `fps=0` rather than
looking dead — handy when diagnosing whether a component crashed or just got no data.

The consumers also **degrade instead of dying** on the common transient faults:
per-frame YOLO/OSNet inference is wrapped so a corrupt frame or a CUDA OOM (the 4 GB GPU)
just skips that frame (`event=inference_error` / `extract_error`) and keeps going; every
`producer.produce` tolerates a full local queue (`event=buffer_full`); and each consumer
`producer.flush()`es on shutdown so the last match/gallery events are delivered, not lost.
Grep these events in the JSONL to see whether a run hit GPU pressure or Kafka backpressure.

## Evaluate against ground truth

```bash
python run.py --once     # one clean pass (writes results/res_cam*.txt)
python report.py --eval  # grades ReID + detection/tracking vs gt/gt_cam*.txt
```

`--eval` needs the run to be a **single pass** (`run.py --once`): the producer otherwise
loops the video forever, which inflates counts and desyncs the live IDs in the logs from the
res files. It also needs `trackeval` installed:

```bash
pip install git+https://github.com/JonathonLuiten/TrackEval.git
```

What `--eval` reports:

- **ReID** — GT uses global cross-camera IDs, so a match is correct iff both live ByteTrack
  tracks map to the same GT global id. Precision/recall is reported per camera pair (cam2 is
  downstream of cam1; cam3's `is_new` calls are graded against cam1's global-id set).
- **Detection + tracking** — `src/eval_metrics.py` runs `trackeval` (HOTA/CLEAR/Identity).
  It scales the 640p res boxes up to the GT resolution and scores only on the frames the
  pipeline actually produced.

`src/eval_metrics.py` also runs standalone; `src/standalone_tracker.py` does offline
detect+track (no Kafka, no ReID) and auto-invokes it.

## Clean runs

Topic data is retained ~2 minutes (see `docker-compose.yaml`) and consumers use
`auto.offset.reset=latest`, so back-to-back runs are largely independent. To force a clean
slate (e.g. reruns within a couple of minutes, or to reclaim disk):

```bash
docker compose down -v   # wipe broker data + committed offsets
docker compose up -d
```

`run.py` pre-creates every topic at startup, so even a freshly wiped (cold) broker is safe
— consumers subscribe to topics that already exist and so don't miss cam1's gallery on the
first frames (which previously caused a ~5-minute stall and 0 matches).

Result files (`results/res_cam*.txt`) are truncated each run, but only for the cameras that
actually ran — run a full 3-camera `--once` before trusting `report.py --eval`.
