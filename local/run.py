"""Local pipeline orchestrator: run every component with one command.

Each component runs as its own process (its own OpenCV window) in its own process group,
so you can stop / restart one without affecting the others.

Examples:
    python run.py                     # run everything (monitor + 3 cams + viz + producer)
    python run.py --only cam1,cam2    # run only cam1 and cam2
    python run.py --exclude monitor   # run everything except monitor
    python run.py --no-monitor        # skip the resource monitor
    python run.py --headless          # no GUI windows (SSH / no display)
    python run.py --once --headless   # single pass, no windows (remote eval run)

While running, type commands in this terminal:
    stop <name>     stop one component       (e.g. stop cam2)
    start <name>    start one component again
    restart <name>  restart it
    status          show status
    quit / Ctrl-C   stop everything and exit

(Or press 'q' in a camera's OpenCV window to close just that one.)

Run from inside local/ with the venv active: `cd local && python run.py`
(after `source .venv/bin/activate`, or `.venv/bin/python run.py`).
"""
import argparse
import datetime
import os
import signal
import subprocess
import sys
import time

# Component scripts live in src/ (next to this file). Resolving them absolutely lets
# run.py work regardless of the current working directory.
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")

# name -> argv passed to the interpreter
COMPONENTS = {
    "monitor": ["monitor.py"],
    "cam1": ["consumer_cam1.py"],
    "cam2": ["consumer_cam2.py"],
    "cam3": ["consumer_cam3.py"],
    "viz": ["match_visualizer.py"],
    "producer": ["producer.py"],
}

# Consumers/viz/monitor must be ready before the producer streams frames
# (consumers use auto.offset.reset=latest), so the producer starts last.
START_ORDER = ["monitor", "cam1", "cam2", "cam3", "viz", "producer"]

STOP_TIMEOUT = 8.0  # seconds to wait for a clean exit before forcing

procs = {}  # name -> Popen


def is_running(name):
    p = procs.get(name)
    return p is not None and p.poll() is None


def start(name):
    if is_running(name):
        print(f"[run] '{name}' is already running.")
        return
    script = COMPONENTS[name]
    argv = [sys.executable, os.path.join(SRC, script[0]), *script[1:]]
    # Own process group + no stdin (stdin is reserved for the controller)
    procs[name] = subprocess.Popen(argv, start_new_session=True, stdin=subprocess.DEVNULL)
    print(f"[run] > start '{name}' (pid {procs[name].pid})")


def stop(name):
    if not is_running(name):
        print(f"[run] '{name}' is not running.")
        return
    p = procs[name]
    pgid = os.getpgid(p.pid)
    # SIGINT -> raises KeyboardInterrupt so the component cleans up (close consumer, log exit)
    print(f"[run] x stop '{name}' (pid {p.pid})...")
    _signal_and_wait(p, pgid)


def _signal_and_wait(p, pgid):
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        deadline = time.time() + (STOP_TIMEOUT if sig == signal.SIGINT else 3.0)
        while time.time() < deadline:
            if p.poll() is not None:
                return
            time.sleep(0.2)


def status():
    print("[run] Status:")
    for name in START_ORDER:
        if name in procs:
            p = procs[name]
            state = f"RUNNING (pid {p.pid})" if p.poll() is None else f"STOPPED (exit {p.returncode})"
        else:
            state = "-"
        print(f"   {name:10s} {state}")


def shutdown_all():
    print("\n[run] Stopping everything...")
    # producer first (stop feeding data), then the rest
    for name in reversed(START_ORDER):
        if is_running(name):
            stop(name)
    print("[run] Stopped. Bye.")


def wait_for_producer(drain):
    """--once mode: wait for the producer to finish its single pass, let consumers
    drain the Kafka backlog, then stop everything (a clean run boundary for report.py)."""
    print("[run] --once: streaming a single pass. Waiting for the producer to finish... (Ctrl-C to abort)")
    while is_running("producer"):
        time.sleep(0.5)
    print(f"[run] Producer done. Draining consumers for {drain}s...")
    time.sleep(drain)


def wait_forever():
    """No interactive console available (e.g. detached / piped stdin): just keep the
    components running and block until Ctrl-C, instead of exiting and tearing them down."""
    print("[run] No TTY for the control console; running until Ctrl-C.")
    while any(is_running(n) for n in procs):
        time.sleep(1.0)


def control_loop():
    print("[run] Type 'status', 'stop <name>', 'start <name>', 'restart <name>', 'quit'. (Ctrl-C to exit)")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        parts = line.split()
        cmd, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else None)

        if cmd in ("quit", "exit"):
            break
        elif cmd == "status":
            status()
        elif cmd in ("stop", "start", "restart"):
            if arg not in COMPONENTS:
                print(f"[run] Invalid name: {arg}. Choose: {', '.join(COMPONENTS)}")
                continue
            if cmd == "stop":
                stop(arg)
            elif cmd == "start":
                start(arg)
            else:
                stop(arg)
                start(arg)
        else:
            print(f"[run] Unknown command: {cmd}")


def main():
    ap = argparse.ArgumentParser(description="Orchestrate the local ReID pipeline.")
    ap.add_argument("--only", help="Run only these components (comma-separated)")
    ap.add_argument("--exclude", help="Skip these components (comma-separated)")
    ap.add_argument("--no-monitor", action="store_true", help="Don't run monitor.py")
    ap.add_argument("--producer-delay", type=float, default=3.0,
                    help="Wait (seconds) after starting consumers before starting the producer")
    ap.add_argument("--once", action="store_true",
                    help="Single pass: stream each video once, then stop everything "
                         "(no interactive console). Needed for report.py --eval.")
    ap.add_argument("--drain", type=float, default=15.0,
                    help="With --once: seconds to let consumers finish after the producer ends")
    ap.add_argument("--headless", action="store_true",
                    help="No GUI windows (for SSH / no display). Components only log to "
                         "console + JSONL. Sets REID_HEADLESS=1 for every component.")
    args = ap.parse_args()

    selected = list(START_ORDER)
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        bad = want - set(COMPONENTS)
        if bad:
            ap.error(f"Invalid name(s) in --only: {', '.join(bad)}")
        selected = [n for n in START_ORDER if n in want]
    if args.exclude:
        drop = {s.strip() for s in args.exclude.split(",")}
        selected = [n for n in selected if n not in drop]
    if args.no_monitor and "monitor" in selected:
        selected.remove("monitor")

    if not selected:
        ap.error("No components selected to run.")

    # Shared per-run stamp so all components log to logs/<stamp>_<component>.jsonl
    os.environ["REID_LOG_STAMP"] = datetime.datetime.now().strftime("%H-%M-%S_%d-%m-%Y")
    if args.once:
        os.environ["REID_REPLAY"] = "0"  # producer streams a single pass then stops
    if args.headless:
        os.environ["REID_HEADLESS"] = "1"  # disable OpenCV windows in every component

    print(f"[run] Will run: {', '.join(selected)}")
    try:
        for name in selected:
            if name == "producer" and any(is_running(n) for n in selected if n != "producer"):
                print(f"[run] Waiting {args.producer_delay}s for consumers to be ready...")
                time.sleep(args.producer_delay)
            start(name)
            time.sleep(0.5)  # small stagger while models load
        if args.once and "producer" in selected:
            wait_for_producer(args.drain)
        elif sys.stdin.isatty():
            control_loop()
        else:
            wait_forever()
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_all()


if __name__ == "__main__":
    main()
