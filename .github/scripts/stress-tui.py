#!/usr/bin/env python3
"""Build the release test targets once, then exercise every concurrency level."""

import argparse
import os
import shlex
import subprocess


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("iterations must be positive")
    return number


def run_stress(iterations):
    command = shlex.split(os.environ.get("CARGO", "cargo")) + [
        "test", "--release", "-p", "oxidelake-tui", "--locked",
    ]
    subprocess.run(command + ["--no-run", "--timings"], check=True)
    for threads, weight in ((1, 20), (4, 40), (16, 40)):
        count = max(1, weight * iterations // 100)
        for iteration in range(1, count + 1):
            print(f"::group::{threads} threads, iteration {iteration}/{count}", flush=True)
            try:
                subprocess.run(command + ["--", f"--test-threads={threads}"], check=True)
            finally:
                print("::endgroup::", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=positive_integer, default=os.environ.get("ITERS", "100"))
    run_stress(parser.parse_args().iterations)
