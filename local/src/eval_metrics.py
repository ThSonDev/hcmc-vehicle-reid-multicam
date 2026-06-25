"""Single-camera tracking evaluation (HOTA / MOTA / IDF1) against the ground truth.

Adapts to the pipeline automatically:
  - Stream resolution is **probed from the source video** (`config.VIDEO_SOURCES`), so any
    video resolution works with no edits; res boxes are scaled up to the GT resolution.
  - Consumers process only some frames (e.g. every 3rd). We evaluate **only on the frames
    the pipeline produced** (res ∩ gt), so whatever the skip setting is, it's handled — and
    trackeval doesn't score un-processed frames as all-misses (which would deflate HOTA/MOTA).
  - GT has 3 vehicle classes (1=car, 2=truck, 3=bus); all are mapped to one class so every
    vehicle is scored (TrackEval evaluates a single "pedestrian" class).

The only thing configured below is GT_RESOLUTION (the coordinate space the annotations
were made in) — a stable property of the dataset, not of the pipeline.

Run:  python eval_metrics.py        # expects gt_cam*.txt in local/gt/, res in local/results/
"""
import os
import shutil
from collections import Counter

import cv2
import numpy as np

import config

# TrackEval still uses numpy aliases removed in numpy >= 1.20.
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

GT_DIR = config.GT_DIR              # local/gt/ (gt_cam*.txt live there)
RES_DIR = config.RESULTS_DIR
TEMP_ROOT = config.TEMP_EVAL_DIR
TRACKER_NAME = "MyLocalAlgo"
EVAL_CLASS = 1            # map every vehicle class -> this single evaluated class

# Ground-truth annotation resolution per camera (the coordinate space of gt_cam*.txt).
# Stream resolution is detected from the source video, so only this stays configured.
GT_RESOLUTION = {
    "cam1": (1920, 1080),
    "cam2": (1920, 1080),
    "cam3": (1440, 1080),
}


def read_mot(path):
    """Read a MOT file -> list of (frame, id, x, y, w, h, conf, cls, vis)."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            fr, tid = int(float(p[0])), int(float(p[1]))
            x, y, w, h = (float(v) for v in p[2:6])
            conf = float(p[6]) if len(p) > 6 else 1.0
            cls = int(float(p[7])) if len(p) > 7 else EVAL_CLASS
            vis = float(p[8]) if len(p) > 8 else 1.0
            rows.append((fr, tid, x, y, w, h, conf, cls, vis))
    return rows


def write_mot(path, rows):
    with open(path, 'w') as f:
        for fr, tid, x, y, w, h, conf, cls, vis in rows:
            f.write(f"{fr},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.2f},{cls},{vis:.4f}\n")


def create_seqinfo(path, name, w, h, length):
    with open(path, 'w') as f:
        f.write(f"[Sequence]\nname={name}\nimDir=img1\nframeRate=30\n"
                f"seqLength={length}\nimWidth={w}\nimHeight={h}\nimExt=.jpg\n")


def stream_resolution(seq, res_rows):
    """Resolution of the res coordinate space. Probe the source video first (works for
    any video resolution); fall back to the res box extent if the video isn't available."""
    path = config.VIDEO_SOURCES.get(seq)
    if path and os.path.exists(path):
        cap = cv2.VideoCapture(path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            return float(w), float(h)
    mx = my = 1.0
    for r in res_rows:
        mx, my = max(mx, r[2] + r[4]), max(my, r[3] + r[5])
    return mx, my


def detect_step(frames):
    """Most common frame gap in res -> the consumers' effective skip interval."""
    fs = sorted(frames)
    if len(fs) < 2:
        return 1
    gaps = Counter(fs[i + 1] - fs[i] for i in range(len(fs) - 1))
    return gaps.most_common(1)[0][0]


def stage_sequence(seq, gt_out_dir, res_out_file):
    """Stage one sequence for trackeval; return (scored_frames, total_gt_frames).

    Scales res up to GT resolution, keeps only frames present in both res and gt, and
    remaps every vehicle class to one class.
    """
    res = read_mot(os.path.join(RES_DIR, f"res_{seq}.txt"))
    gt = read_mot(os.path.join(GT_DIR, f"gt_{seq}.txt"))

    ow, oh = GT_RESOLUTION[seq]
    sw, sh = stream_resolution(seq, res)
    sx, sy = ow / sw, oh / sh

    res_frames = {r[0] for r in res}
    gt_frames = {g[0] for g in gt}
    common = res_frames & gt_frames

    res_out = [(fr, tid, x * sx, y * sy, w * sx, h * sy, conf, EVAL_CLASS, 1.0)
               for (fr, tid, x, y, w, h, conf, cls, vis) in res if fr in common]
    gt_out = [(fr, tid, x, y, w, h, 1.0, EVAL_CLASS, vis)
              for (fr, tid, x, y, w, h, conf, cls, vis) in gt if fr in common]

    os.makedirs(os.path.join(gt_out_dir, "gt"), exist_ok=True)
    write_mot(os.path.join(gt_out_dir, "gt", "gt.txt"), gt_out)
    create_seqinfo(os.path.join(gt_out_dir, "seqinfo.ini"), seq, ow, oh,
                   max(common) if common else 1)
    write_mot(res_out_file, res_out)

    print(f"   {seq}: stream {int(sw)}x{int(sh)} -> GT {ow}x{oh} (scale x{sx:.2f} y{sy:.2f}) | "
          f"scored on {len(common)} frames (every ~{detect_step(res_frames)}, gt has {len(gt_frames)})")
    return len(common), len(gt_frames)


def setup_environment():
    print(f">>> Building evaluation layout under {TEMP_ROOT}")
    if os.path.exists(TEMP_ROOT):
        shutil.rmtree(TEMP_ROOT)
    gt_root = os.path.join(TEMP_ROOT, "gt")
    tracker_data = os.path.join(TEMP_ROOT, "trackers", TRACKER_NAME, "data")
    os.makedirs(tracker_data, exist_ok=True)

    valid, coverage = [], {}
    for seq in GT_RESOLUTION:
        if not os.path.exists(os.path.join(RES_DIR, f"res_{seq}.txt")):
            print(f"⚠️  Missing results/res_{seq}.txt -> skip {seq}")
            continue
        if not os.path.exists(os.path.join(GT_DIR, f"gt_{seq}.txt")):
            print(f"⚠️  Missing gt_{seq}.txt -> skip {seq}")
            continue
        proc, _ = stage_sequence(seq, os.path.join(gt_root, seq),
                                 os.path.join(tracker_data, f"{seq}.txt"))
        coverage[seq] = proc
        valid.append(seq)
    return valid, coverage


def run_evaluation():
    try:
        import trackeval
    except ImportError:
        print("trackeval not installed.")
        print("Install: pip install git+https://github.com/JonathonLuiten/TrackEval.git")
        return

    valid_seqs, coverage = setup_environment()
    if not valid_seqs:
        print("Nothing to evaluate.")
        return
    print(f"\n>>> Computing metrics for: {valid_seqs}")

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({'DISPLAY_LESS_PROGRESS': True, 'PRINT_CONFIG': False,
                        'TIME_PROGRESS': False, 'USE_PARALLEL': False})

    ds = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds.update({
        'GT_FOLDER': os.path.join(TEMP_ROOT, 'gt'),
        'TRACKERS_FOLDER': os.path.join(TEMP_ROOT, 'trackers'),
        'BENCHMARK': 'MotChallenge2DBox', 'SPLIT_TO_EVAL': 'train',
        'TRACKERS_TO_EVAL': [TRACKER_NAME], 'CLASSES_TO_EVAL': ['pedestrian'],
        'SEQ_INFO': {s: None for s in valid_seqs}, 'SKIP_SPLIT_FOL': True,
        'PRINT_CONFIG': False,
    })

    evaluator = trackeval.Evaluator(eval_config)
    datasets = [trackeval.datasets.MotChallenge2DBox(ds)]
    metric_cfg = {'PRINT_CONFIG': False}
    metrics = [trackeval.metrics.HOTA(metric_cfg), trackeval.metrics.CLEAR(metric_cfg),
               trackeval.metrics.Identity(metric_cfg)]

    output_res, _ = evaluator.evaluate(datasets, metrics)
    print_summary(output_res, coverage)


def print_summary(output_res, coverage):
    """Compact table (report.py reads this section back into the run report)."""
    try:
        res = output_res['MotChallenge2DBox'][TRACKER_NAME]
    except (KeyError, TypeError):
        print("Could not read evaluation results.")
        return

    print("\n================ SUMMARY (single-camera tracking) ================")
    print("Scored only on frames the pipeline produced; res boxes scaled to GT resolution.")
    print(f"{'SEQ':<12}{'HOTA':>8}{'DetA':>8}{'AssA':>8}{'MOTA':>8}{'IDF1':>8}{'frames':>9}")
    seqs = [s for s in res if s != 'COMBINED_SEQ']
    if 'COMBINED_SEQ' in res:
        seqs.append('COMBINED_SEQ')
    for seq in seqs:
        cls = res[seq].get('pedestrian')
        if not cls:
            continue
        hota = float(np.mean(cls['HOTA']['HOTA'])) * 100
        deta = float(np.mean(cls['HOTA']['DetA'])) * 100
        assa = float(np.mean(cls['HOTA']['AssA'])) * 100
        mota = float(cls['CLEAR']['MOTA']) * 100
        idf1 = float(cls['Identity']['IDF1']) * 100
        name = 'COMBINED' if seq == 'COMBINED_SEQ' else seq
        frames = '' if seq == 'COMBINED_SEQ' else str(coverage.get(seq, ''))
        print(f"{name:<12}{hota:>8.2f}{deta:>8.2f}{assa:>8.2f}{mota:>8.2f}{idf1:>8.2f}{frames:>9}")
    print("==================================================================")
    print("HOTA=overall  DetA=detection  AssA=association(ID)  MOTA=MOT accuracy  IDF1=ID F1")


if __name__ == "__main__":
    run_evaluation()
