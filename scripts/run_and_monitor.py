"""Unified experiment runner with live monitoring for Idea C + Cloud Baselines."""

import argparse
import csv
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# CSV paths for progress tracking
IDEA_C_CSV = PROJECT_ROOT / "idea-c-quantization" / "results" / "experiments.csv"
BASELINES_CSV = PROJECT_ROOT / "shared" / "results" / "cloud_baselines.csv"

# ANSI colors
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Idea C: 25 model×quant variants; Baselines: 4 models (default subset)
IDEA_C_VARIANTS = 25
BASELINE_MODELS_DEFAULT = 4

_interrupted = False
_procs: list[subprocess.Popen] = []


def resolve_python() -> str:
    """Resolve venv python path explicitly (avoids nohup PATH issues)."""
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    print(f"{YELLOW}Warning: .venv/bin/python not found, using sys.executable{RESET}")
    return sys.executable


def count_csv_rows(path: Path) -> int:
    """Count data rows in CSV (excluding header). Returns 0 if missing."""
    if not path.exists():
        return 0
    try:
        with open(path) as f:
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return 0


def sum_csv_column(path: Path, column: str) -> float:
    """Sum a numeric column in CSV. Returns 0.0 if missing."""
    if not path.exists():
        return 0.0
    try:
        total = 0.0
        with open(path) as f:
            for row in csv.DictReader(f):
                val = row.get(column, "")
                if val:
                    total += float(val)
        return total
    except Exception:
        return 0.0


def stream_output(proc: subprocess.Popen, tag: str, color: str) -> None:
    """Read lines from subprocess stdout and print with colored prefix."""
    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        if not line:
            break
        # Pass through \r lines (tqdm) and normal lines
        stripped = line.rstrip("\n")
        if "\r" in line and not line.startswith("\n"):
            print(f"{color}[{tag}]{RESET} {stripped}", end="\r", flush=True)
        else:
            print(f"{color}[{tag}]{RESET} {stripped}", flush=True)


def print_summary(limit: int, running: dict[str, subprocess.Popen | None]) -> None:
    """Print progress summary header."""
    lines = [f"\n{BOLD}{'='*60}", "  EXPERIMENT MONITOR", f"{'='*60}{RESET}"]

    if running.get("idea-c") is not None:
        rows = count_csv_rows(IDEA_C_CSV)
        expected = limit * IDEA_C_VARIANTS
        pct = (rows / expected * 100) if expected else 0
        proc = running["idea-c"]
        status = "running" if proc.poll() is None else f"exited ({proc.returncode})"
        lines.append(
            f"  {CYAN}[C] Idea C:{RESET} {rows}/{expected} rows "
            f"({pct:.1f}%) — {status}"
        )

    if running.get("baselines") is not None:
        rows = count_csv_rows(BASELINES_CSV)
        expected = limit * BASELINE_MODELS_DEFAULT
        pct = (rows / expected * 100) if expected else 0
        cost = sum_csv_column(BASELINES_CSV, "cost_usd")
        proc = running["baselines"]
        status = "running" if proc.poll() is None else f"exited ({proc.returncode})"
        lines.append(
            f"  {YELLOW}[B] Baselines:{RESET} {rows}/{expected} rows "
            f"({pct:.1f}%) — {status} — ${cost:.4f}"
        )

    lines.append(f"{BOLD}{'='*60}{RESET}\n")
    print("\n".join(lines), flush=True)


def handle_interrupt(running: dict[str, subprocess.Popen | None]) -> None:
    """Prompt user to kill or detach on Ctrl+C."""
    alive = {k: p for k, p in running.items() if p and p.poll() is None}
    if not alive:
        print(f"\n{GREEN}All processes already finished.{RESET}")
        return

    pids = ", ".join(f"{k}={p.pid}" for k, p in alive.items())
    print(f"\n{BOLD}Interrupted.{RESET} Running PIDs: {pids}")
    print("  [k] Kill experiments")
    print("  [c] Continue in background (detach)")

    try:
        choice = input("Choice: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        choice = "k"

    if choice == "c":
        print(f"\n{GREEN}Detaching. PIDs still running:{RESET}")
        for name, proc in alive.items():
            print(f"  {name}: PID {proc.pid}")
        print("Monitor with: ps aux | grep 'run_experiments\\|run_baselines'")
        # Detach — don't wait or kill
        for proc in alive.values():
            try:
                proc.stdout.close()  # type: ignore[union-attr]
            except Exception:
                pass
    else:
        print(f"\n{RED}Killing processes...{RESET}")
        for name, proc in alive.items():
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            print(f"  {name} (PID {proc.pid}): terminated")


def launch(
    python: str, script: str, extra_args: list[str]
) -> subprocess.Popen:
    """Launch a subprocess with merged stdout/stderr."""
    cmd = [python, str(PROJECT_ROOT / script)] + extra_args
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    _procs.append(proc)
    return proc


def main() -> None:
    global _interrupted

    parser = argparse.ArgumentParser(description="Run & monitor experiments")
    parser.add_argument("--limit", type=int, default=200, help="Page limit (default: 200)")
    parser.add_argument(
        "--only", choices=["idea-c", "baselines"],
        help="Run only one experiment set",
    )
    args = parser.parse_args()

    python = resolve_python()
    print(f"{BOLD}Python:{RESET} {python}")
    print(f"{BOLD}Limit:{RESET} {args.limit}")

    running: dict[str, subprocess.Popen | None] = {"idea-c": None, "baselines": None}
    threads: list[threading.Thread] = []

    # Launch Idea C
    if args.only != "baselines":
        proc_c = launch(
            python,
            "idea-c-quantization/src/run_experiments.py",
            ["--limit", str(args.limit)],
        )
        running["idea-c"] = proc_c
        t = threading.Thread(target=stream_output, args=(proc_c, "C", CYAN), daemon=True)
        t.start()
        threads.append(t)
        print(f"{CYAN}[C] Idea C started (PID {proc_c.pid}){RESET}")

    # Launch Baselines
    if args.only != "idea-c":
        proc_b = launch(
            python,
            "shared/run_baselines.py",
            [
                "--models", "gpt-4o", "qwen-72b", "llama-70b", "mistral-large",
                "--limit", str(args.limit),
            ],
        )
        running["baselines"] = proc_b
        t = threading.Thread(target=stream_output, args=(proc_b, "B", YELLOW), daemon=True)
        t.start()
        threads.append(t)
        print(f"{YELLOW}[B] Baselines started (PID {proc_b.pid}){RESET}")

    # Install SIGINT handler
    def sigint_handler(sig, frame):
        global _interrupted
        _interrupted = True

    signal.signal(signal.SIGINT, sigint_handler)

    # Main loop: print summary every 10s, check for completion
    try:
        while True:
            time.sleep(10)

            if _interrupted:
                handle_interrupt(running)
                break

            print_summary(args.limit, running)

            # Check if all processes finished
            alive = [p for p in running.values() if p and p.poll() is None]
            if not alive:
                break
    except KeyboardInterrupt:
        handle_interrupt(running)
        return

    # Wait for reader threads to drain
    for t in threads:
        t.join(timeout=5)

    # Final summary
    print_summary(args.limit, running)

    # Report exit codes
    for name, proc in running.items():
        if proc is not None:
            code = proc.returncode
            color = GREEN if code == 0 else RED
            print(f"{color}{name}: exit code {code}{RESET}")

    print(f"\n{BOLD}Done.{RESET}")


if __name__ == "__main__":
    main()
