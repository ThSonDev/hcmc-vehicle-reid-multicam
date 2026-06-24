"""Run report: a concise Markdown digest for AI + 3 PNG plots, with optional GT eval.

The pipeline's JSONL logs capture everything (heartbeats every ~5s, system samples
every ~2s) -> thousands of near-identical lines that bury the crucial signals. This tool
reads ONE run's logs (grouped by the `<stamp>_` filename prefix), concises them into:

  1. logs/<stamp>_report.md   - token-light Markdown for an AI (or you) to read
  2. logs/images/<stamp>_{resources,pipeline,combined}.png - plots for a human

With --eval it also grades the run against the ground truth (gt_cam*.txt + results/res_cam*.txt):
detection + tracking (HOTA/MOTA/IDF1 via eval_metrics.py / trackeval) and cross-camera ReID
correctness (the GT uses GLOBAL ids, so a match is correct iff both live tracks map to the
same global id). ReID eval needs a single clean pass -> run the pipeline with `run.py --once`.

Usage:
    python report.py                       # digest the latest run (+ plots)
    python report.py --run 17-22-28_22-06-2026
    python report.py --list                # list available runs
    python report.py --eval                # also evaluate vs ground truth
    python report.py --no-plots            # markdown only
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime

import config

LOG_DIR = config.LOG_DIR
IMAGES_DIR = os.path.join(LOG_DIR, "images")
RESULTS_DIR = config.RESULTS_DIR
GT_DIR = config.GT_DIR

# <stamp>_<component>.jsonl  (+ optional rotation suffix .1 .2 ...)
_FILE_RE = re.compile(r"^(\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{4})_(.+?)\.jsonl(?:\.\d+)?$")
_STAMP_FMT = "%H-%M-%S_%d-%m-%Y"

PIPELINE_COMPONENTS = ["producer", "cam1", "cam2", "cam3", "visualizer"]
SEQS = {"cam1": (1920, 1080), "cam2": (1920, 1080), "cam3": (1440, 1080)}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def list_runs(log_dir=LOG_DIR):
    """Return run stamps found in log_dir, newest first."""
    stamps = set()
    for path in glob.glob(os.path.join(log_dir, "*.jsonl*")):
        m = _FILE_RE.match(os.path.basename(path))
        if m:
            stamps.add(m.group(1))
    return sorted(stamps, key=lambda s: datetime.strptime(s, _STAMP_FMT), reverse=True)


def load_run(stamp, log_dir=LOG_DIR):
    """Load every JSONL line for one run, grouped by component and sorted by ts.

    Merges rotation backups (`.jsonl`, `.jsonl.1`, ...) so early lines aren't lost.
    """
    by_comp = defaultdict(list)
    for path in glob.glob(os.path.join(log_dir, f"{stamp}_*.jsonl*")):
        m = _FILE_RE.match(os.path.basename(path))
        if not m:
            continue
        comp = m.group(2)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    by_comp[comp].append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    for comp in by_comp:
        by_comp[comp].sort(key=lambda e: e.get("ts", ""))
    return dict(by_comp)


def _t0(by_comp):
    """Earliest timestamp across all components (run start), as datetime."""
    times = [e["ts"] for evs in by_comp.values() for e in evs if "ts" in e]
    return datetime.fromisoformat(min(times)) if times else None


def _secs(ts, t0):
    return (datetime.fromisoformat(ts) - t0).total_seconds() if t0 else 0.0


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def _series(events, event_name, field):
    """Collect (ts, value) for events of a given name where `field` is present."""
    out = []
    for e in events:
        if e.get("event") == event_name and field in e:
            out.append((e["ts"], e[field]))
    return out


def _stats(values):
    if not values:
        return None
    return {"avg": round(sum(values) / len(values), 1), "min": round(min(values), 1),
            "max": round(max(values), 1), "n": len(values)}


def summarize_component(comp, events):
    """One compact dict per component (heartbeats/samples collapsed to stats)."""
    hb = [e for e in events if e.get("event") == "heartbeat"]
    errors = [e for e in events if e.get("level") in ("ERROR", "WARNING")]
    exit_ev = next((e for e in events if e.get("event") in ("exit", "worker_stop")), None)
    out = {"events": len(events), "errors": len(errors)}

    if comp == "producer":
        fps = [v for _, v in _series(hb, "heartbeat", "sent_fps")]
        out["sent_fps"] = _stats(fps)
        sent = [e.get("sent_total") for e in events if "sent_total" in e]
        out["sent_total"] = max(sent) if sent else 0
        out["replays"] = sum(1 for e in events if e.get("event") == "replay")
        out["done"] = any(e.get("event") == "stream_done" for e in events)
    elif comp in ("cam1", "cam2", "cam3"):
        out["fps"] = _stats([v for _, v in _series(hb, "heartbeat", "fps")])
        frames = [e.get("frames") for e in events if "frames" in e]
        out["frames"] = max(frames) if frames else 0
        if comp == "cam1":
            sent = [e.get("gallery_sent") for e in events if "gallery_sent" in e]
            out["gallery_sent"] = max(sent) if sent else 0
        else:
            m = [e.get("matches") for e in events if "matches" in e]
            out["matches"] = max(m) if m else 0
            if comp == "cam3":
                nv = [e.get("new") for e in events if "new" in e]
                out["new_vehicles"] = max(nv) if nv else 0
    if exit_ev:
        out["exit"] = {k: v for k, v in exit_ev.items()
                       if k not in ("ts", "level", "component", "msg")}
    return out


def collect_matches(by_comp):
    """Per downstream cam: scores + a few exemplars, from `event=match` log lines."""
    out = {}
    for comp, id_field in (("cam2", "cam2_id"), ("cam3", "cam3_id")):
        matches = [e for e in by_comp.get(comp, []) if e.get("event") == "match"]
        scores = [e["score"] for e in matches if "score" in e]
        info = {"count": len(matches), "score": _stats(scores)}
        if scores:
            ranked = sorted(matches, key=lambda e: e.get("score", 0))
            info["worst"] = [(e.get("cam1_id"), e.get(id_field), e.get("score")) for e in ranked[:3]]
            info["best"] = [(e.get("cam1_id"), e.get(id_field), e.get("score")) for e in ranked[-3:]]
        if comp == "cam3":
            info["new_vehicles"] = sum(1 for e in by_comp.get("cam3", [])
                                       if e.get("event") == "new_vehicle")
        out[comp] = info
    return out


def summarize_system(events, disk_warn=140.0):
    samples = [e for e in events if e.get("event") == "sample"]
    if not samples:
        return None
    def col(name):
        return [e[name] for e in samples if name in e]
    breaches = sum(1 for e in samples if e.get("disk_w_mbs", 0) >= disk_warn)
    return {
        "samples": len(samples),
        "cpu_pct": _stats(col("cpu_pct")), "ram_pct": _stats(col("ram_pct")),
        "swap_used_gb": _stats(col("swap_used_gb")),
        "gpu_pct": _stats(col("gpu_pct")), "gpu_mem_mb": _stats(col("gpu_mem_mb")),
        "disk_w_mbs": _stats(col("disk_w_mbs")), "disk_r_mbs": _stats(col("disk_r_mbs")),
        "disk_breaches": breaches, "disk_warn": disk_warn,
    }


def collect_errors(by_comp):
    """Deduped WARNING/ERROR lines with counts + first/last ts."""
    agg = {}
    for comp, events in by_comp.items():
        for e in events:
            if e.get("level") not in ("ERROR", "WARNING"):
                continue
            key = (comp, e.get("level"), e.get("msg", ""))
            rec = agg.setdefault(key, {"count": 0, "first": e.get("ts"), "last": e.get("ts")})
            rec["count"] += 1
            rec["last"] = e.get("ts")
    return agg


def build_anomalies(comp_summaries, system, matches, errors):
    """The short 'look here first' list."""
    out = []
    for key, rec in errors.items():
        comp, level, msg = key
        out.append(f"{comp} {level} x{rec['count']}: {msg}")
    for comp in ("cam1", "cam2", "cam3"):
        s = comp_summaries.get(comp)
        if s and s.get("frames", 0) == 0:
            out.append(f"{comp} received 0 frames (Kafka down? producer not running?)")
    for comp in ("cam2", "cam3"):
        if matches.get(comp, {}).get("count", 0) == 0 and comp_summaries.get(comp, {}).get("frames", 0) > 0:
            out.append(f"{comp} produced 0 matches despite receiving frames")
    if system and system["disk_breaches"] > 0:
        out.append(f"disk write crossed {system['disk_warn']} MB/s x{system['disk_breaches']} "
                   f"(peak {system['disk_w_mbs']['max']} MB/s)")
    return out


# --------------------------------------------------------------------------- #
# Evaluation (--eval): detection + tracking (trackeval) and ReID (global ids)
# --------------------------------------------------------------------------- #
def _load_mot(path):
    """frame -> list of (track_id, (x, y, w, h)). MOT: frame,id,x,y,w,h,conf,cls,vis."""
    frames = defaultdict(list)
    if not os.path.exists(path):
        return frames
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            fr, tid = int(float(p[0])), int(float(p[1]))
            box = (float(p[2]), float(p[3]), float(p[4]), float(p[5]))
            frames[fr].append((tid, box))
    return frames


def _iou(a, b):
    ax1, ay1, aw, ah = a; bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _global_ids(gt_frames):
    return {tid for dets in gt_frames.values() for tid, _ in dets}


def _frame_extent(frames):
    """Approximate frame width/height from the furthest box edge (for scale detection)."""
    mx = my = 1.0
    for dets in frames.values():
        for _, (x, y, w, h) in dets:
            mx, my = max(mx, x + w), max(my, y + h)
    return mx, my


def _rescale(frames, sx, sy):
    return {fr: [(tid, (x * sx, y * sy, w * sx, h * sy)) for tid, (x, y, w, h) in dets]
            for fr, dets in frames.items()}


def _match_scale(res_frames, gt_frames):
    """Bring res boxes into GT coordinate space (the producer may stream a downsized
    video, e.g. *_640.mp4, so res is at stream resolution while GT is at original)."""
    rx, ry = _frame_extent(res_frames)
    gx, gy = _frame_extent(gt_frames)
    sx, sy = gx / rx, gy / ry
    if abs(sx - 1) > 0.05 or abs(sy - 1) > 0.05:
        return _rescale(res_frames, sx, sy)
    return res_frames


def build_id_map(res_frames, gt_frames, iou_thr=0.5):
    """Map live ByteTrack id -> GT global id by per-frame IoU majority vote."""
    votes = defaultdict(Counter)
    for fr, dets in res_frames.items():
        gts = gt_frames.get(fr, [])
        for lid, lbox in dets:
            best_iou, best_gid = iou_thr, None
            for gid, gbox in gts:
                v = _iou(lbox, gbox)
                if v >= best_iou:
                    best_iou, best_gid = v, gid
            if best_gid is not None:
                votes[lid][best_gid] += 1
    return {lid: c.most_common(1)[0][0] for lid, c in votes.items()}


def eval_reid(by_comp):
    """Grade cross-camera matches against the GLOBAL-id ground truth."""
    res = {c: _load_mot(os.path.join(RESULTS_DIR, f"res_{c}.txt")) for c in SEQS}
    gt = {c: _load_mot(os.path.join(GT_DIR, f"gt_{c}.txt")) for c in SEQS}
    if not any(res[c] for c in SEQS):
        return {"available": False,
                "note": "results/res_cam*.txt not found - run the pipeline (ideally `run.py --once`) first."}

    maps = {c: build_id_map(_match_scale(res[c], gt[c]), gt[c]) for c in SEQS}
    g1_ids = _global_ids(gt["cam1"])
    out = {"available": True, "pairs": {}}

    for cam, id_field in (("cam2", "cam2_id"), ("cam3", "cam3_id")):
        matches = [e for e in by_comp.get(cam, []) if e.get("event") == "match"]
        correct = wrong = unknown = 0
        correct_globals = set()
        for m in matches:
            g_src = maps["cam1"].get(m.get("cam1_id"))
            g_dst = maps[cam].get(m.get(id_field))
            if g_src is None or g_dst is None:
                unknown += 1
            elif g_src == g_dst:
                correct += 1
                correct_globals.add(g_src)
            else:
                wrong += 1
        graded = correct + wrong
        matchable = g1_ids & _global_ids(gt[cam])
        out["pairs"][f"cam1->{cam}"] = {
            "matches": len(matches), "correct": correct, "wrong": wrong,
            "unmappable": unknown,
            "precision": round(correct / graded, 3) if graded else None,
            "recall": round(len(correct_globals) / len(matchable), 3) if matchable else None,
            "matchable_gt": len(matchable),
        }

    # cam3 "new vehicle" verdicts: correct iff the global id is absent from cam1.
    new_events = [e for e in by_comp.get("cam3", []) if e.get("event") == "new_vehicle"]
    nc = nw = nu = 0
    for e in new_events:
        g = maps["cam3"].get(e.get("cam3_id"))
        if g is None:
            nu += 1
        elif g not in g1_ids:
            nc += 1
        else:
            nw += 1
    out["new_vehicle"] = {"count": len(new_events), "correct": nc, "wrong": nw, "unmappable": nu}
    return out


def eval_tracking():
    """Run eval_metrics.py (trackeval) and extract the HOTA/MOTA/IDF1 summary lines."""
    if not any(os.path.exists(os.path.join(RESULTS_DIR, f"res_{c}.txt")) for c in SEQS):
        return {"available": False, "note": "results/res_cam*.txt not found."}
    try:
        eval_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_metrics.py")
        proc = subprocess.run([sys.executable, eval_script],
                              capture_output=True, text=True, timeout=600)
    except (subprocess.SubprocessError, OSError) as e:
        return {"available": False, "note": f"eval_metrics.py failed to run: {e}"}
    text = proc.stdout + "\n" + proc.stderr
    if "trackeval not installed" in text:
        return {"available": False, "note": "trackeval not installed "
                "(pip install git+https://github.com/JonathonLuiten/TrackEval.git)."}
    marker = "SUMMARY (single-camera tracking)"
    if marker in text:
        # Just the clean table eval_metrics.py prints (keeps the AI digest tight).
        block = text.split(marker, 1)[1]
        keep = [ln.rstrip() for ln in block.splitlines() if ln.strip()][:10]
    else:
        keep = [ln for ln in text.splitlines()
                if any(tok in ln for tok in ("HOTA", "MOTA", "IDF1", "COMBINED"))][:30]
    return {"available": True, "summary": keep, "raw": text}


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _samp_series(system_events, field, t0):
    xs, ys = [], []
    for e in system_events:
        if e.get("event") == "sample" and field in e:
            xs.append(_secs(e["ts"], t0)); ys.append(e[field])
    return xs, ys


def _hb_series(events, field, t0):
    xs, ys = [], []
    for e in events:
        if e.get("event") == "heartbeat" and field in e:
            xs.append(_secs(e["ts"], t0)); ys.append(e[field])
    return xs, ys


def plot_resources(system_events, t0, path, disk_warn=140.0):
    plt = _setup_plt()
    fig, ax = plt.subplots(figsize=(11, 5))
    for field, label in (("cpu_pct", "CPU %"), ("ram_pct", "RAM %"), ("gpu_pct", "GPU %")):
        xs, ys = _samp_series(system_events, field, t0)
        if xs:
            ax.plot(xs, ys, label=label, linewidth=1.2)
    ax.set_xlabel("seconds since run start"); ax.set_ylabel("utilization %"); ax.set_ylim(0, 100)
    ax2 = ax.twinx()
    xs, ys = _samp_series(system_events, "disk_w_mbs", t0)
    if xs:
        ax2.plot(xs, ys, label="disk write MB/s", color="tab:red", alpha=0.6, linewidth=1.0)
        ax2.axhline(disk_warn, color="tab:red", linestyle="--", alpha=0.4)
    ax2.set_ylabel("disk write MB/s")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    ax.set_title("System resources over time")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_pipeline(by_comp, t0, path):
    plt = _setup_plt()
    fig, ax = plt.subplots(figsize=(11, 5))
    xs, ys = _hb_series(by_comp.get("producer", []), "sent_fps", t0)
    if xs:
        ax.plot(xs, ys, label="producer send FPS", linewidth=1.2)
    for cam in ("cam1", "cam2", "cam3"):
        xs, ys = _hb_series(by_comp.get(cam, []), "fps", t0)
        if xs:
            ax.plot(xs, ys, label=f"{cam} FPS", linewidth=1.2)
    ax.set_xlabel("seconds since run start"); ax.set_ylabel("FPS")
    ax2 = ax.twinx()
    for cam in ("cam2", "cam3"):
        xs, ys = _hb_series(by_comp.get(cam, []), "matches", t0)
        if xs:
            ax2.plot(xs, ys, label=f"{cam} matches (cumulative)", linestyle=":", linewidth=1.4)
    ax2.set_ylabel("cumulative matches")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    ax.set_title("Pipeline throughput & matches over time")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_combined(system_events, by_comp, t0, path, disk_warn=140.0):
    plt = _setup_plt()
    fig, (top, bot) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for field, label in (("cpu_pct", "CPU %"), ("gpu_pct", "GPU %")):
        xs, ys = _samp_series(system_events, field, t0)
        if xs:
            top.plot(xs, ys, label=label, linewidth=1.2)
    top.set_ylabel("utilization %"); top.set_ylim(0, 100)
    tw = top.twinx()
    xs, ys = _samp_series(system_events, "disk_w_mbs", t0)
    if xs:
        tw.plot(xs, ys, label="disk write MB/s", color="tab:red", alpha=0.6)
        tw.axhline(disk_warn, color="tab:red", linestyle="--", alpha=0.4)
    tw.set_ylabel("disk write MB/s")
    top.legend(loc="upper left"); tw.legend(loc="upper right")
    top.set_title("Resources (top) vs pipeline (bottom) - shared time axis")
    for cam in ("cam1", "cam2", "cam3"):
        xs, ys = _hb_series(by_comp.get(cam, []), "fps", t0)
        if xs:
            bot.plot(xs, ys, label=f"{cam} FPS", linewidth=1.2)
    bot.set_ylabel("FPS"); bot.set_xlabel("seconds since run start")
    bw = bot.twinx()
    for cam in ("cam2", "cam3"):
        xs, ys = _hb_series(by_comp.get(cam, []), "matches", t0)
        if xs:
            bw.plot(xs, ys, label=f"{cam} matches", linestyle=":", linewidth=1.4)
    bw.set_ylabel("cumulative matches")
    bot.legend(loc="upper left"); bw.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _fmt_stats(s):
    return f"{s['avg']} ({s['min']}-{s['max']})" if s else "n/a"


def write_markdown(stamp, by_comp, comp_summaries, system, matches, errors,
                   anomalies, duration, reid, tracking, plots, out_path):
    L = []
    comps = " ".join(sorted(by_comp))
    L.append(f"# Run {stamp}")
    L.append(f"\nDuration {duration} - components: {comps}\n")

    L.append("## Highlights / anomalies")
    L += [f"- {a}" for a in anomalies] or ["- none"]
    L.append("")

    L.append("## Per-component")
    for comp in ["producer", "cam1", "cam2", "cam3", "visualizer"]:
        s = comp_summaries.get(comp)
        if not s:
            continue
        if comp == "producer":
            extra = f"send FPS {_fmt_stats(s['sent_fps'])} - sent_total {s['sent_total']} - replays {s['replays']} - done {s['done']}"
        elif comp == "cam1":
            extra = f"FPS {_fmt_stats(s['fps'])} - frames {s['frames']} - gallery_sent {s['gallery_sent']}"
        elif comp in ("cam2", "cam3"):
            extra = f"FPS {_fmt_stats(s['fps'])} - frames {s['frames']} - matches {s['matches']}"
            if comp == "cam3":
                extra += f" - new_vehicles {s['new_vehicles']}"
        else:
            extra = f"{s['events']} events"
        L.append(f"- **{comp}**: {extra} - errors {s['errors']}")
    L.append("")

    L.append("## Match quality (from logs)")
    for cam in ("cam2", "cam3"):
        m = matches.get(cam, {})
        line = f"- **cam1->{cam}**: {m.get('count', 0)} matches, score {_fmt_stats(m.get('score'))}"
        if cam == "cam3":
            line += f", new_vehicles {m.get('new_vehicles', 0)}"
        L.append(line)
        if m.get("worst"):
            L.append(f"  - lowest scores (cam1_id,{cam}_id,score): {m['worst']}")
    L.append("")

    if system:
        L.append("## System")
        L.append(f"- CPU {_fmt_stats(system['cpu_pct'])}% | RAM {_fmt_stats(system['ram_pct'])}% "
                 f"| swap {_fmt_stats(system['swap_used_gb'])} GB")
        L.append(f"- GPU {_fmt_stats(system['gpu_pct'])}% | GPU mem {_fmt_stats(system['gpu_mem_mb'])} MB")
        L.append(f"- disk write {_fmt_stats(system['disk_w_mbs'])} MB/s "
                 f"- breaches > {system['disk_warn']}: {system['disk_breaches']}")
        L.append("")

    if reid is not None:
        L.append("## ReID evaluation vs GT (global ids)")
        if not reid.get("available"):
            L.append(f"- skipped: {reid.get('note')}")
        else:
            for pair, r in reid["pairs"].items():
                L.append(f"- **{pair}**: {r['matches']} matches | precision {r['precision']} "
                         f"recall {r['recall']} | correct {r['correct']} wrong {r['wrong']} "
                         f"unmappable {r['unmappable']} | matchable in GT {r['matchable_gt']}")
            nv = reid.get("new_vehicle", {})
            L.append(f"- **cam3 new-vehicle calls**: {nv.get('count', 0)} | correct {nv.get('correct')} "
                     f"wrong {nv.get('wrong')} unmappable {nv.get('unmappable')}")
        L.append("")

    if tracking is not None:
        L.append("## Detection & tracking vs GT (trackeval)")
        if not tracking.get("available"):
            L.append(f"- skipped: {tracking.get('note')}")
        else:
            L.append("```")
            L += tracking["summary"]
            L.append("```")
        L.append("")

    if errors:
        L.append("## Errors & warnings (deduped)")
        for (comp, level, msg), rec in sorted(errors.items(), key=lambda kv: -kv[1]["count"]):
            L.append(f"- [{level}] {comp} x{rec['count']} ({rec['first']} .. {rec['last']}): {msg}")
        L.append("")

    if plots:
        L.append("## Plots")
        L += [f"- {os.path.relpath(p)}" for p in plots]
        L.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Concise run report (Markdown digest + plots + optional GT eval).")
    ap.add_argument("--run", help="Run stamp to report (default: latest)")
    ap.add_argument("--list", action="store_true", help="List available runs and exit")
    ap.add_argument("--eval", action="store_true", help="Evaluate vs GT (detection+tracking+ReID)")
    ap.add_argument("--no-plots", action="store_true", help="Skip PNG plots (markdown only)")
    ap.add_argument("--top", type=int, default=5, help="(reserved) number of exemplars to show")
    ap.add_argument("--log-dir", default=LOG_DIR, help="Where to read JSONL logs")
    ap.add_argument("--out", default=LOG_DIR, help="Output directory for the markdown report")
    ap.add_argument("--disk-warn", type=float, default=140.0, help="Disk-write threshold (MB/s)")
    args = ap.parse_args()

    runs = list_runs(args.log_dir)
    if args.list:
        if not runs:
            print(f"No runs found in {args.log_dir}/")
        for s in runs:
            print(s)
        return

    if not runs:
        sys.exit(f"No runs found in {args.log_dir}/ (expected <stamp>_<component>.jsonl files).")
    stamp = args.run or runs[0]
    if stamp not in runs:
        sys.exit(f"Run '{stamp}' not found. Available: {', '.join(runs) or '(none)'}")

    print(f"[report] Run {stamp}")
    by_comp = load_run(stamp, args.log_dir)
    t0 = _t0(by_comp)
    all_ts = [e["ts"] for evs in by_comp.values() for e in evs if "ts" in e]
    duration = "n/a"
    if all_ts:
        secs = (datetime.fromisoformat(max(all_ts)) - datetime.fromisoformat(min(all_ts))).total_seconds()
        duration = f"{int(secs // 60)}m{int(secs % 60)}s"

    comp_summaries = {c: summarize_component(c, evs) for c, evs in by_comp.items()
                      if c in PIPELINE_COMPONENTS}
    system = summarize_system(by_comp.get("system", []), args.disk_warn)
    matches = collect_matches(by_comp)
    errors = collect_errors(by_comp)
    anomalies = build_anomalies(comp_summaries, system, matches, errors)

    reid = tracking = None
    if args.eval:
        print("[report] Evaluating ReID vs GT (global ids)...")
        reid = eval_reid(by_comp)
        print("[report] Running detection/tracking eval (trackeval)...")
        tracking = eval_tracking()

    plots = []
    if not args.no_plots:
        try:
            os.makedirs(IMAGES_DIR, exist_ok=True)
            sysmod = by_comp.get("system", [])
            p_res = os.path.join(IMAGES_DIR, f"{stamp}_resources.png")
            p_pipe = os.path.join(IMAGES_DIR, f"{stamp}_pipeline.png")
            p_comb = os.path.join(IMAGES_DIR, f"{stamp}_combined.png")
            plot_resources(sysmod, t0, p_res, args.disk_warn)
            plot_pipeline(by_comp, t0, p_pipe)
            plot_combined(sysmod, by_comp, t0, p_comb, args.disk_warn)
            plots = [p_res, p_pipe, p_comb]
            print(f"[report] Wrote 3 plots to {IMAGES_DIR}/")
        except ImportError:
            print("[report] matplotlib not installed - skipping plots "
                  "(uv pip install matplotlib==3.7.5).")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{stamp}_report.md")
    write_markdown(stamp, by_comp, comp_summaries, system, matches, errors,
                   anomalies, duration, reid, tracking, plots, out_path)
    print(f"[report] Wrote {out_path}")


if __name__ == "__main__":
    main()
