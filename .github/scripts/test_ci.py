"""Regression tests for selective gates, feature reuse, and stress coverage."""

from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent.parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = load_script("ci-policy")
stress = load_script("stress-tui")


def results(rust="true"):
    needs = {name: {"result": "success"} for name in policy.RUST_JOBS | policy.ALWAYS_JOBS}
    needs["changes"]["outputs"] = {"rust": rust}
    return needs


class GateTests(unittest.TestCase):
    def test_full_gate_requires_every_job(self):
        good = results()
        self.assertTrue(policy.gate_passes(good))
        for name in good:
            for result in ("failure", "cancelled", "skipped", "", None):
                with self.subTest(job=name, result=result):
                    bad = deepcopy(good)
                    bad[name]["result"] = result
                    self.assertFalse(policy.gate_passes(bad))

    def test_documentation_skips_never_hide_failures(self):
        good = results("false")
        for name in policy.RUST_JOBS:
            good[name]["result"] = "skipped"
        self.assertTrue(policy.gate_passes(good))
        for name in good:
            for result in ("failure", "cancelled"):
                with self.subTest(job=name, result=result):
                    bad = deepcopy(good)
                    bad[name]["result"] = result
                    self.assertFalse(policy.gate_passes(bad))
        for name in policy.ALWAYS_JOBS:
            bad = deepcopy(good)
            bad[name]["result"] = "skipped"
            self.assertFalse(policy.gate_passes(bad))

    def test_incomplete_or_malformed_results_fail(self):
        for name in results():
            bad = results()
            del bad[name]
            self.assertFalse(policy.gate_passes(bad))
        for value in (None, [], {}, {"changes": "success"}):
            self.assertFalse(policy.gate_passes(value))
        for outputs in ({}, None, {"rust": ""}, {"rust": True}):
            bad = results()
            bad["changes"]["outputs"] = outputs
            self.assertFalse(policy.gate_passes(bad))

    def test_cli_fails_when_a_required_job_fails(self):
        needs = results("false")
        needs["metal"]["result"] = "failure"
        run = subprocess.run(
            ["python3", str(SCRIPTS / "ci-policy.py"), "gate"],
            env={**os.environ, "NEEDS": json.dumps(needs)}, capture_output=True,
        )
        self.assertNotEqual(run.returncode, 0)


class ChangeTests(unittest.TestCase):
    def test_only_known_documentation_can_skip_rust(self):
        self.assertTrue(policy.documentation_only(["docs/architecture.md", "CHANGELOG.md"]))
        for changed in ([], ["README.md"], ["Cargo.lock"], ["Makefile"],
                        ["docs/example.rs"], [".github/workflows/ci.yml"],
                        ["crates/oxidelake-device/kernels/metal/filter_project.metal"],
                        ["docs/readme.md", "crates/oxidelake-core/src/lib.rs"],
                        ["docs/not-really.md\nCargo.toml"]):
            self.assertFalse(policy.documentation_only(changed), changed)

    def test_release_manual_and_invalid_refs_run_everything(self):
        with patch.object(policy.subprocess, "run") as run:
            for event, workflow, base in (("push", "release", "a" * 40),
                                           ("workflow_dispatch", "CI", "a" * 40),
                                           ("pull_request", "CI", ""),
                                           ("push", "CI", "--help")):
                self.assertTrue(policy.needs_rust(event, workflow, base, "b" * 40))
            run.assert_not_called()

    def test_missing_commit_falls_back_to_full_gate(self):
        with patch.object(policy.subprocess, "run", side_effect=subprocess.CalledProcessError(128, "git")):
            self.assertTrue(policy.needs_rust("push", "CI", "0" * 40, "b" * 40))

    def test_git_diff_tracks_deletions_renames_and_newline_names(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            def git(*args):
                return subprocess.check_output(["git", "-C", directory, *args], stderr=subprocess.DEVNULL).decode().strip()
            git("init", "-q")
            git("config", "user.name", "CI Test")
            git("config", "user.email", "ci-test@example.invalid")
            (repo / "src").mkdir()
            (repo / "docs").mkdir()
            (repo / "src/lib.rs").write_text("fn original() {}\n")
            git("add", ".")
            git("commit", "-qm", "initial")
            base = git("rev-parse", "HEAD")
            (repo / "docs/guide.md").write_text("Guide\n")
            git("add", ".")
            git("commit", "-qm", "docs")
            docs = git("rev-parse", "HEAD")
            def classify(before, after):
                run = subprocess.run(
                    ["python3", str(SCRIPTS / "ci-policy.py"), "changes"], cwd=directory,
                    env={**os.environ, "EVENT": "pull_request", "WORKFLOW": "CI", "BASE_SHA": before, "HEAD_SHA": after},
                    check=True, capture_output=True, text=True,
                )
                return run.stdout.strip()
            self.assertEqual(classify(base, docs), "rust=false")
            git("mv", "src/lib.rs", "docs/former-code.md")
            git("commit", "-qm", "rename source to documentation")
            renamed = git("rev-parse", "HEAD")
            self.assertEqual(classify(docs, renamed), "rust=true")
            (repo / "docs/example.md\nCargo.toml").write_text("not a markdown path\n")
            git("add", ".")
            git("commit", "-qm", "newline path")
            self.assertEqual(classify(renamed, git("rev-parse", "HEAD")), "rust=true")


class StressTests(unittest.TestCase):
    def test_one_build_keeps_all_thread_counts_and_iterations(self):
        with patch.dict(os.environ, {"CARGO": "cargo"}), patch.object(stress.subprocess, "run") as run, redirect_stdout(io.StringIO()):
            stress.run_stress(100)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(sum("--no-run" in command for command in commands), 1)
        self.assertIn("--no-run", commands[0])
        self.assertEqual(Counter(command[-1] for command in commands[1:]), {
            "--test-threads=1": 20, "--test-threads=4": 40, "--test-threads=16": 40,
        })
        self.assertTrue(all("--locked" in command and "--release" in command for command in commands))

    def test_small_run_still_exercises_every_thread_count(self):
        with patch.object(stress.subprocess, "run") as run, redirect_stdout(io.StringIO()):
            stress.run_stress(1)
        self.assertEqual(run.call_count, 4)

    def test_build_or_test_failure_stops_the_run(self):
        for fail_at in (0, 1, 21):
            with self.subTest(fail_at=fail_at):
                outcomes = [None] * fail_at + [subprocess.CalledProcessError(1, "cargo")]
                with patch.object(stress.subprocess, "run", side_effect=outcomes) as run, redirect_stdout(io.StringIO()):
                    with self.assertRaises(subprocess.CalledProcessError):
                        stress.run_stress(100)
                self.assertEqual(run.call_count, fail_at + 1)

    def test_reject_invalid_iterations_before_building(self):
        for value in ("0", "-1", "abc", "1; echo bad"):
            with self.subTest(value=value), self.assertRaises((ValueError, stress.argparse.ArgumentTypeError)):
                stress.positive_integer(value)


class MetalBuildTests(unittest.TestCase):
    def test_normal_and_device_tests_select_the_same_build_graph(self):
        def command(target):
            output = subprocess.check_output(["make", "-n", target], cwd=ROOT, text=True)
            return next(line.strip().rstrip("; \\") for line in output.splitlines() if line.strip().startswith(("cargo test ", "OXIDE_BACKEND=metal cargo test ")))
        normal = command("test-metal")
        device = command("test-metal-device").removeprefix("OXIDE_BACKEND=metal ")
        self.assertEqual(device.split(" -- --ignored")[0], normal)
        self.assertIn("-p oxidelake-tui", normal)
        self.assertIn("--features oxidelake-runtime/metal", normal)


class MetadataTests(unittest.TestCase):
    def run_metadata(self, overrides=None, packaged=True):
        metadata = {"name": "oxidelake-core", "publish": None, "readme": "README.md",
                    "repository": "https://github.com/vyncint/oxidelake", "homepage": "https://github.com/vyncint/oxidelake",
                    "documentation": "https://docs.rs/oxidelake-core", "keywords": ["gpu"],
                    "categories": ["science"], "description": "Core types"}
        metadata.update(overrides or {})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / ".github/scripts"
            scripts.mkdir(parents=True)
            script = scripts / "check-crate-metadata.sh"
            script.write_text((SCRIPTS / script.name).read_text())
            (root / "metadata.json").write_text(json.dumps({"packages": [metadata]}))
            cargo = root / "cargo"
            cargo.write_text('#!/bin/sh\ncase "$1" in\nmetadata) cat metadata.json ;;\npackage) printf "%s\\n" "$PACKAGED_FILE" ;;\n*) exit 2 ;;\nesac\n')
            cargo.chmod(0o755)
            return subprocess.run(
                ["/bin/bash", str(script)], text=True, capture_output=True,
                env={**os.environ, "PATH": directory + os.pathsep + os.environ["PATH"],
                     "PACKAGED_FILE": "README.md" if packaged else "Cargo.toml"},
            )

    def test_metadata_gate_works_with_the_platform_bash(self):
        run = self.run_metadata()
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_empty_fields_wrong_docs_and_missing_readme_fail(self):
        for overrides in ({"readme": None}, {"repository": ""}, {"documentation": "https://example.invalid"}, {"keywords": []}):
            with self.subTest(overrides=overrides):
                self.assertNotEqual(self.run_metadata(overrides).returncode, 0)
        self.assertNotEqual(self.run_metadata(packaged=False).returncode, 0)


class ToolchainTests(unittest.TestCase):
    def test_msrv_uses_rustup_even_when_path_cargo_is_not_a_proxy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("Makefile", "Cargo.toml"):
                (root / name).write_text((ROOT / name).read_text())
            cargo = root / "cargo"
            cargo.write_text('#!/bin/sh\necho "PATH Cargo does not support +toolchain" >&2\nexit 9\n')
            cargo.chmod(0o755)
            rustup = root / "rustup"
            rustup.write_text('#!/bin/sh\ncase "$1" in\ntoolchain) exit 0 ;;\nrun) test "$3" = cargo && test "$4" = check ;;\n*) exit 2 ;;\nesac\n')
            rustup.chmod(0o755)
            run = subprocess.run(["make", "msrv"], cwd=root, text=True, capture_output=True,
                                 env={**os.environ, "PATH": directory + os.pathsep + os.environ["PATH"]})
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main()
