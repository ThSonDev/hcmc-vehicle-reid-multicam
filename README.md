# HCMC Multi-camera Vehicle ReID

Multi-camera vehicle Re-Identification pipeline for HCMC traffic footage. Three cameras
(cam1/cam2/cam3) stream JPEG frames over Kafka; per-camera consumers run YOLO detection +
ByteTrack tracking + OSNet feature extraction, and the downstream cameras (cam2, cam3)
match their vehicles back to the cam1 "source" gallery.

## Two implementations

The same pipeline exists in two parallel stacks — pick one, don't mix them:

```
.
├── local/      Self-contained standalone pipeline: its own .venv, requirements,
│               Kafka (docker compose), source (src/), ground truth (gt/), and data.
│               Run directly on the host. This is the current focus.
│               -> see local/README.md for everything (setup, run, eval).
│
└── airflow/    Containerized Airflow + Spark Structured Streaming variant:
                Spark consumers write matches to Postgres, a Streamlit UI reads them.
                Self-contained (its own Kafka, Postgres, etc.). Documented later.
```

The project is built mainly around the Airflow stack; the **local** stack is the
standalone reference implementation and the focus of the current work.

## Where to start

- **Local pipeline (here, now):** [`local/README.md`](local/README.md) — requirements,
  setup, how to run, architecture, and evaluation. Plain-language internals live in
  [`local/architecture.md`](local/architecture.md).
- **Airflow pipeline:** lives under `airflow/` (own `docker-compose.yaml`). Detailed docs
  are a later sprint.

## Repository layout

```
local/        self-contained local pipeline (run.py, src/, gt/, setup.sh,
              docker-compose.yaml, requirements.txt, .venv, data/weights/results/logs)
airflow/      Airflow + Spark variant (separate stack)
osnet/        vendored torchreid fork + OSNet training helpers (shared by both stacks)
fine_tune_yolo/  YOLO fine-tuning (MOT -> YOLO dataset)
demos/        throwaway box-drawing demos
```
