# HCMC Multi-camera Vehicle ReID

Multi-camera vehicle Re-Identification pipeline for HCMC traffic footage (cam1 → cam2/cam3
matching over Kafka, YOLO + ByteTrack + OSNet).

## Architecture

Three camera videos are streamed as JPEG frames over Kafka; one consumer per camera runs
YOLO detection + ByteTrack tracking + OSNet feature extraction, and the downstream cameras
match their vehicles back to the cam1 "source" gallery.

```
producer ──▶ video_reid_stream ──▶ cam1  ──▶ reid_gallery_stream ─┐
  (3 videos)                        (source)                       │
                                    cam2, cam3 ◀───────────────────┘
                                       │  match against cam1 gallery
                                       ▼
                                 reid_matches ──▶ visualizer
```

- **cam1** — source only; detects/tracks and publishes per-track embeddings to the gallery.
- **cam2** — matches frame-by-frame against the cam1 gallery (travel-time gated; one cam1 id used once).
- **cam3** — buffers a "best shot" per track and matches on track exit; emits *new vehicle* when no match (it sits on a new road).
- **visualizer** — joins match events into one row per source vehicle.
- **monitor** — samples CPU/GPU/RAM/disk to `logs/`. Every component logs structured JSONL; `local/report.py` concises a run into a Markdown digest + plots.

Kafka topics: `video_reid_stream` (frames), `reid_gallery_stream` (cam1 embeddings),
`reid_matches` (match events). Infra config (broker, topics, model/video paths) lives in `local/config.py`.

> There is also an Airflow + Spark Structured Streaming variant under `airflow/` (Spark consumers
> write matches to Postgres, Streamlit UI). It's a separate stack — don't mix it with the local scripts.

## Project structure

The local pipeline lives in `local/`. Artifact paths (`data/ weights/ results/ logs/`,
`gt_cam*.txt`) are anchored to the project root, so you can run from **inside `local/`**
(`cd local && python run.py`) or from the **repo root** (`python local/run.py`) — either works.

```
.
├── local/                    # ← the local pipeline (run these)
│   ├── run.py                # orchestrator: launches every component
│   ├── producer.py           # streams 3 videos -> Kafka topic video_reid_stream
│   ├── consumer_cam1.py      # source: detect + track + publish gallery embeddings
│   ├── consumer_cam2.py      # frame-by-frame matcher vs the cam1 gallery
│   ├── consumer_cam3.py      # best-shot matcher (new road); emits new vehicles
│   ├── match_visualizer.py   # 2-pane match UI
│   ├── monitor.py            # CPU/GPU/RAM/disk sampler
│   ├── report.py             # run digest (Markdown + plots); --eval grades vs GT
│   ├── eval_metrics.py       # trackeval HOTA/MOTA/IDF1 (called by report --eval)
│   ├── standalone_tracker.py # offline detect+track (no Kafka), auto-evaluates
│   ├── config.py             # Kafka / topics / model + video paths (env-overridable)
│   ├── log_utils.py          # structured console + JSONL logging
│   ├── gui_utils.py          # headless-aware OpenCV wrappers (REID_HEADLESS)
│   └── reid_utils.py         # FeatureExtractor, box merging, MOT result writer
├── airflow/                  # separate Airflow + Spark variant (don't mix)
├── osnet/                    # vendored torchreid fork + OSNet training helpers
├── fine_tune_yolo/           # YOLO fine-tuning (MOT -> YOLO dataset)
├── demos/                    # throwaway box-drawing demos
├── data/                     # input videos               (gitignored)
├── weights/                  # YOLO .pt + OSNet .pth       (gitignored)
├── results/                  # res_cam*.txt MOT outputs    (gitignored)
├── logs/                     # per-run JSONL + report.md + images/ (gitignored)
├── gt_cam{1,2,3}.txt         # ground truth (global cross-camera IDs)
├── requirements.txt
├── setup.sh                  # one-command bootstrap
└── docker-compose.yaml       # local Kafka (KRaft, port 9092)
```

## Requirements

- **Python 3.9.x** (tested 3.9.19, pinned in `.python-version`)
- [**uv**](https://docs.astral.sh/uv/) for dependency management
- Docker (local Kafka)
- A C compiler (`gcc`) — builds the `torchreid` Cython extension
- NVIDIA GPU + driver (optional but recommended; CPU works, slower)

## Setup

One command (idempotent):

```bash
./setup.sh
```

It creates `.venv` (Python 3.9.19), installs `requirements.txt`, then installs
`supervision` and the editable `torchreid` with `--no-deps` (they must stay out of the
resolvable block or their dep trees fight the pinned legacy stack), and creates the
`data/ weights/ results/ logs/ temp/` directories.

Then supply the gitignored assets:
- **Videos** → `data/cam1_640.mp4`, `data/cam2_640.mp4`, `data/cam3_640.mp4`
- **Weights** → `weights/cam1.pt`, `weights/cam2.pt` (YOLO), `weights/osnet_cam123.pth` (OSNet)

> Always run Python via the venv: `source .venv/bin/activate`, `uv run …`, or `python …`.

## Run the local pipeline

**Tutorial (the happy path):** commands below run from inside `local/` (`cd local` once);
they also work from the repo root as `python local/<script>.py`.

1. **Start Kafka** (must be up before anything else, or consumers receive nothing):
   ```bash
   docker compose up -d              # local Kafka (KRaft, port 9092) — from the repo root
   ```
2. **Run everything** with one command:
   ```bash
   cd local
   python run.py           # monitor + cam1/2/3 + visualizer + producer
   ```
   Five OpenCV windows open; matches show up in the visualizer. Stop with `quit` or Ctrl-C.
3. **See what happened** — concise the logs into a report + plots:
   ```bash
   python report.py        # -> logs/<stamp>_report.md + logs/images/*.png
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
(stream each video a single pass, then auto-stop — use this before evaluating).
Or press `q` inside a camera window to close just that one.

**Remote / headless (SSH, no display):** add `--headless` to skip all OpenCV windows —
components run logic-only and write to console + JSONL, so you drive everything from the
report and logs. With a detached/piped stdin (no TTY) the control console is skipped and
the pipeline just runs until Ctrl-C, so `nohup python run.py --headless &` is safe.

```bash
python run.py --once --headless    # single pass, no windows -> then report --eval
```

### Evaluate against ground truth

```bash
python run.py --once     # one clean pass (writes results/res_cam*.txt)
python report.py --eval  # grades ReID + detection/tracking vs gt_cam*.txt
```

`--eval` needs the run to be a single pass (`run.py --once`) and `trackeval` installed
(`pip install git+https://github.com/JonathonLuiten/TrackEval.git`). It scales the 640p res
boxes up to the ground-truth resolution and scores only on the frames the pipeline produced.

### Logs & standalone runs

Every component logs to the console **and** to a structured `logs/<stamp>_<component>.jsonl`
(size-rotated, one run per `<stamp>`) — re-read with e.g.
`jq 'select(.event=="match")' logs/<stamp>_cam2.jsonl`, or just run `report.py`.

Prefer separate terminals? Each component also runs standalone (from inside `local/`):

```bash
python producer.py
python consumer_cam1.py    # cam2 / cam3 likewise
python match_visualizer.py
python monitor.py          # system resource monitor (CPU/GPU/RAM/disk)
```
