#!/usr/bin/env python3
"""Classify documentation changes and enforce the required CI results."""

import json
import os
from pathlib import PurePosixPath
import re
import subprocess
import sys


RUST_JOBS = {"fmt", "clippy", "test", "metal", "msrv", "docs", "deny"}
ALWAYS_JOBS = {"changes", "release-scripts", "skill-version", "zizmor"}
ROOT_DOCS = {"CHANGELOG.md", "CONTRIBUTING.md", "AGENTS.md", "CODE_OF_CONDUCT.md", "SECURITY.md"}


def documentation_only(paths):
    # Unknown files, empty diffs, and README packaging changes take the full gate.
    return bool(paths) and all(
        path in ROOT_DOCS
        or (PurePosixPath(path).parts[0] == "docs" and path.endswith(".md"))
        for path in paths
    )


def needs_rust(event, workflow, base, head):
    # Reusable release gates and manual benchmarks always verify the whole tree.
    if workflow != "CI" or event not in {"push", "pull_request"}:
        return True
    if not all(re.fullmatch(r"[0-9a-f]{40}", sha or "") for sha in (base, head)):
        return True
    try:
        changed = subprocess.run(
            ["git", "diff", "--name-only", "--no-renames", "-z", base, head, "--"],
            check=True, capture_output=True,
        ).stdout.decode("utf-8").split("\0")
    except (subprocess.CalledProcessError, UnicodeDecodeError):
        return True
    return not documentation_only([path for path in changed if path])


def gate_passes(needs):
    if not isinstance(needs, dict) or set(needs) != RUST_JOBS | ALWAYS_JOBS:
        return False
    if any(not isinstance(job, dict) for job in needs.values()):
        return False
    changes = needs["changes"]
    outputs = changes.get("outputs", {})
    rust = outputs.get("rust") if isinstance(outputs, dict) else None
    if rust not in {"true", "false"}:
        return False
    for name, job in needs.items():
        allowed = {"success", "skipped"} if name in RUST_JOBS and rust == "false" else {"success"}
        if job.get("result") not in allowed:
            return False
    return True


def main():
    if sys.argv[1:] == ["changes"]:
        rust = needs_rust(*(os.environ.get(key, "") for key in ("EVENT", "WORKFLOW", "BASE_SHA", "HEAD_SHA")))
        print(f"rust={str(rust).lower()}")
    elif sys.argv[1:] == ["gate"]:
        needs = json.loads(os.environ["NEEDS"])
        print(json.dumps(needs, indent=2))
        if not gate_passes(needs):
            sys.exit("CI failed: a required job failed, was cancelled, is missing, or was unexpectedly skipped")
    else:
        sys.exit("usage: ci-policy.py changes|gate")


if __name__ == "__main__":
    main()
