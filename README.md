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
- **monitor** — samples CPU/GPU/RAM/disk to `logs/`. Every component logs structured JSONL; `report.py` concises a run into a Markdown digest + plots.

Kafka topics: `video_reid_stream` (frames), `reid_gallery_stream` (cam1 embeddings),
`reid_matches` (match events). Infra config (broker, topics, model/video paths) lives in `config.py`.

> There is also an Airflow + Spark Structured Streaming variant under `airflow/` (Spark consumers
> write matches to Postgres, Streamlit UI). It's a separate stack — don't mix it with the local scripts.

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

**Tutorial (the happy path):**

1. **Start Kafka** (must be up before anything else, or consumers receive nothing):
   ```bash
   docker compose up -d              # local Kafka (KRaft, port 9092)
   ```
2. **Run everything** with one command:
   ```bash
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

### Evaluate against ground truth

```bash
python run.py --once       # one clean pass (writes results/res_cam*.txt)
python report.py --eval    # grades ReID + detection/tracking vs gt_cam*.txt
```

`--eval` needs the run to be a single pass (`--once`) and `trackeval` installed
(`pip install git+https://github.com/JonathonLuiten/TrackEval.git`). It scales the 640p res
boxes up to the ground-truth resolution and scores only on the frames the pipeline produced.

### Logs & standalone runs

Every component logs to the console **and** to a structured `logs/<stamp>_<component>.jsonl`
(size-rotated, one run per `<stamp>`) — re-read with e.g.
`jq 'select(.event=="match")' logs/<stamp>_cam2.jsonl`, or just run `report.py`.

Prefer separate terminals? Each component also runs standalone:

```bash
python producer.py
python consumer_cam1.py    # cam2 / cam3 likewise
python match_visualizer.py
python monitor.py          # system resource monitor (CPU/GPU/RAM/disk)
```
