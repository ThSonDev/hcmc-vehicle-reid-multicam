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

Robustness: each component's stdout/stderr is teed to `logs/<stamp>_<component>.out`
(and echoed to this terminal, prefixed with the component name), so a crash that
bypasses the JSONL logger -- an uncaught exception or a native CUDA/cuDNN segfault --
is still captured. A watchdog thread notices any component that dies unexpectedly,
prints the cause (exit code / signal) and the tail of its output, and records it in
`logs/<stamp>_run.jsonl`.

Run from inside local/ with the venv active: `cd local && python run.py`
(after `source .venv/bin/activate`, or `.venv/bin/python run.py`).
"""
import argparse
import datetime
import os
import signal
import subprocess
import sys
import threading
import time

# Component scripts live in src/ (next to this file). Resolving them absolutely lets
# run.py work regardless of the current working directory.
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
LOG_DIR = os.path.join(HERE, "logs")

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
TAIL_LINES = 25     # lines of captured output to show when a component dies

procs = {}          # name -> Popen
out_files = {}      # name -> open file object capturing its stdout/stderr
out_paths = {}      # name -> path of that file
pumps = {}          # name -> tee thread

run_log = None      # structured run-level logger (set in main())

# Watchdog state (guarded by _watch_lock).
_watch_lock = threading.Lock()
_expected = set()        # components we started and expect to stay alive
_reported_dead = set()   # components whose death we've already reported
_watcher_stop = threading.Event()
_shutting_down = False


def is_running(name):
    p = procs.get(name)
    return p is not None and p.poll() is None


def _log(level, msg, **fields):
    """Mirror an event to the run-level JSONL (if up) and the terminal."""
    if run_log is not None:
        getattr(run_log, level)(msg, extra=fields)
    else:
        print(f"[run] {msg}")


def ensure_topics(broker, topics):
    """Create every pipeline topic up front (idempotent) so no consumer ever
    subscribes to a not-yet-existing topic. Without this, a topic created lazily by
    the first producer (e.g. cam1's reid_gallery_stream) isn't seen by already-
    subscribed consumers until librdkafka's 5-minute metadata refresh -- with
    auto.offset.reset=latest they miss everything in between (0 matches)."""
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": broker})
    try:
        existing = set(admin.list_topics(timeout=10).topics.keys())
    except Exception as e:
        _log("error", "could not reach broker to pre-create topics", event="topics_error",
             broker=broker, error=str(e))
        return False

    todo = [t for t in topics if t not in existing]
    if not todo:
        _log("info", "all topics already present", event="topics_ready", topics=list(topics))
        return True

    futures = admin.create_topics([NewTopic(t, num_partitions=1, replication_factor=1) for t in todo])
    ok = True
    for t, fut in futures.items():
        try:
            fut.result()
            _log("info", f"created topic '{t}'", event="topic_created", topic=t)
        except Exception as e:
            if "already exists" in str(e).lower():  # raced with broker auto-create
                _log("info", f"topic '{t}' already exists", event="topic_exists", topic=t)
            else:
                ok = False
                _log("error", f"failed to create topic '{t}'", event="topic_error", topic=t, error=str(e))
    return ok


def _pump(name, stream, fobj):
    """Tee one child's merged stdout/stderr to its .out file and this terminal."""
    prefix = f"[{name}] "
    try:
        for line in stream:
            fobj.write(line)
            fobj.flush()
            sys.stdout.write(prefix + line)
            sys.stdout.flush()
    except (ValueError, OSError):
        pass  # stream closed during shutdown


def start(name):
    if is_running(name):
        print(f"[run] '{name}' is already running.")
        return
    script = COMPONENTS[name]
    argv = [sys.executable, os.path.join(SRC, script[0]), *script[1:]]

    out_path = os.path.join(LOG_DIR, f"{os.environ['REID_LOG_STAMP']}_{name}.out")
    out_paths[name] = out_path
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        fobj = open(out_path, "a", buffering=1, encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"[run] WARN: cannot open {out_path} ({e}); capturing to terminal only.")
        fobj = None

    # Own process group + no stdin (stdin is reserved for the controller). Merge
    # stderr into stdout and pipe both so we can tee them to file + terminal; this is
    # what captures crashes that never reach the JSONL logger.
    procs[name] = subprocess.Popen(
        argv, start_new_session=True, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, errors="replace",
    )
    if fobj is not None:
        out_files[name] = fobj
        t = threading.Thread(target=_pump, args=(name, procs[name].stdout, fobj), daemon=True)
        t.start()
        pumps[name] = t

    with _watch_lock:
        _expected.add(name)
        _reported_dead.discard(name)
    _log("info", "component started", event="start", component=name, pid=procs[name].pid)
    print(f"[run] > start '{name}' (pid {procs[name].pid})")


def stop(name):
    if not is_running(name):
        print(f"[run] '{name}' is not running.")
        return
    p = procs[name]
    pgid = os.getpgid(p.pid)
    with _watch_lock:
        _expected.discard(name)  # mark intentional so the watchdog stays quiet
    # SIGINT -> raises KeyboardInterrupt so the component cleans up (close consumer, log exit)
    print(f"[run] x stop '{name}' (pid {p.pid})...")
    _log("info", "component stopping", event="stop", component=name, pid=p.pid)
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


def _signal_name(rc):
    """Map a negative Popen returncode to a signal name (e.g. -11 -> 'SIGSEGV')."""
    try:
        return signal.Signals(-rc).name
    except ValueError:
        return f"signal {-rc}"


def _tail_out(name):
    path = out_paths.get(name)
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.readlines()[-TAIL_LINES:]
    except OSError:
        return []


def _report_exit(name, rc):
    """Announce (loudly, if abnormal) that a watched component has exited."""
    if rc == 0:
        _log("info", f"'{name}' exited normally", event="component_exit",
             component=name, code=0)
        print(f"[run] '{name}' exited normally (code 0).")
        return

    detail = _signal_name(rc) if rc < 0 else f"code {rc}"
    sig = _signal_name(rc) if rc < 0 else ""
    tail = _tail_out(name)
    hint = ""
    if rc == -signal.SIGSEGV or rc == -signal.SIGABRT:
        hint = " (native crash -- likely CUDA/cuDNN/driver; see the tail + faulthandler dump)"
    elif rc == -signal.SIGKILL:
        hint = " (SIGKILL -- possibly OOM-killed; check `dmesg`/system log)"

    print(f"\n[run] !! '{name}' DIED UNEXPECTEDLY ({detail}){hint}")
    if out_paths.get(name):
        print(f"[run]    full output: {out_paths[name]}")
    if tail:
        print(f"[run]    last {len(tail)} line(s):")
        for ln in tail:
            print(f"   | {ln.rstrip()}")
    _log("error", f"'{name}' died unexpectedly", event="component_crash",
         component=name, code=rc, signal=sig, tail="".join(tail)[-4000:])


def _watch_loop():
    """Poll every started component; report the first unexpected exit of each."""
    while not _watcher_stop.wait(1.0):
        if _shutting_down:
            continue
        with _watch_lock:
            names = list(_expected)
        for name in names:
            p = procs.get(name)
            if p is None:
                continue
            rc = p.poll()
            if rc is None:
                continue
            with _watch_lock:
                if name in _reported_dead:
                    continue
                _reported_dead.add(name)
                _expected.discard(name)
            _report_exit(name, rc)


def status():
    print("[run] Status:")
    for name in START_ORDER:
        if name in procs:
            p = procs[name]
            if p.poll() is None:
                state = f"RUNNING (pid {p.pid})"
            else:
                rc = p.returncode
                state = f"STOPPED ({_signal_name(rc) if rc < 0 else f'exit {rc}'})"
        else:
            state = "-"
        print(f"   {name:10s} {state}")


def shutdown_all():
    global _shutting_down
    _shutting_down = True
    _watcher_stop.set()
    print("\n[run] Stopping everything...")
    _log("info", "shutting down", event="shutdown")
    # producer first (stop feeding data), then the rest
    for name in reversed(START_ORDER):
        if is_running(name):
            stop(name)
    for f in out_files.values():
        try:
            f.close()
        except OSError:
            pass
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
    global run_log

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

    # Shared per-run stamp so all components log to logs/<stamp>_<component>.jsonl.
    # Must be set before importing log_utils (it reads REID_LOG_STAMP at import time).
    os.environ["REID_LOG_STAMP"] = datetime.datetime.now().strftime("%H-%M-%S_%d-%m-%Y")
    if args.once:
        os.environ["REID_REPLAY"] = "0"  # producer streams a single pass then stops
    if args.headless:
        os.environ["REID_HEADLESS"] = "1"  # disable OpenCV windows in every component

    sys.path.insert(0, SRC)
    import config
    from log_utils import setup_logging
    run_log = setup_logging("run")

    # Pre-create all Kafka topics before launching consumers (see ensure_topics).
    ensure_topics(config.BROKER, config.TOPICS)

    # Start the watchdog before launching anything so an early crash is caught.
    watcher = threading.Thread(target=_watch_loop, daemon=True)
    watcher.start()

    _log("info", "orchestrator start", event="run_start", components=selected,
         once=args.once, headless=args.headless)
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
        _watcher_stop.set()


if __name__ == "__main__":
    main()
