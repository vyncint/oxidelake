"""Regression tests for selective gates, feature reuse, and stress coverage."""

from collections import Counter
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import http.server
import importlib.util
import io
import json
import os
from pathlib import Path
import hashlib
import socket
import subprocess
import tempfile
import textwrap
import threading
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent.parent


def workflow_job(workflow, job):
    """The lines of `job` in .github/workflows/`workflow`, its key line included.

    Indentation-based rather than a YAML parse: the standard library has no
    YAML reader, and these files keep every job at two spaces.
    """
    lines = (ROOT / ".github/workflows" / workflow).read_text().splitlines()
    start = lines.index(f"  {job}:")
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].startswith("  ") and not lines[i].startswith("   ") and lines[i].strip()
                and not lines[i].lstrip().startswith("#")), len(lines))
    return lines[start:end]


def workflow_step(workflow, name):
    """The lines of the step called `name` in .github/workflows/`workflow`."""
    lines = (ROOT / ".github/workflows" / workflow).read_text().splitlines()
    start = lines.index(next(line for line in lines if line.strip() == f"- name: {name}"))
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip()
                and len(lines[i]) - len(lines[i].lstrip()) <= indent), len(lines))
    return lines[start:end]


def step_condition(workflow, name):
    """The step's `if:` expression, or None."""
    return next((line.strip()[len("if: "):] for line in workflow_step(workflow, name)
                 if line.strip().startswith("if: ")), None)


def step_script(workflow, name):
    """The step's `run: |` block, dedented: the script the runner executes."""
    lines = workflow_step(workflow, name)
    start = next(i for i, line in enumerate(lines) if line.strip() == "run: |")
    key = len(lines[start]) - len(lines[start].lstrip())
    body = []
    for line in lines[start + 1:]:
        if line.strip() and len(line) - len(line.lstrip()) <= key:
            break
        body.append(line)
    return textwrap.dedent("\n".join(body)) + "\n"


@contextmanager
def stand_in(statuses):
    """A local HTTP server answering with `statuses` in turn, then the last one.

    Yields its base URL and the list of paths it was asked for.
    """
    asked = []
    remaining = list(statuses)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            asked.append(self.path)
            status = remaining.pop(0) if len(remaining) > 1 else remaining[0]
            self.send_response(status)
            self.end_headers()
            self.wfile.write(b'{"errors":[{"detail":"stand-in"}]}')

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", asked
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


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


class SkillVersionTests(unittest.TestCase):
    """The vendored termlens skill must name the dependency's major.minor."""

    def run_check(self, skill_version, dep_line):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / ".github/scripts"
            scripts.mkdir(parents=True)
            script = scripts / "check-skill-version.sh"
            script.write_text((SCRIPTS / script.name).read_text())
            skill = root / ".claude/skills/termlens/SKILL.md"
            skill.parent.mkdir(parents=True)
            header = f"Written against **termlens {skill_version}**.\n" if skill_version else ""
            skill.write_text("# Testing terminal programs with termlens\n\n" + header)
            (root / "Cargo.toml").write_text(
                f'[workspace.package]\nrust-version = "1.94.1"\n[workspace.dependencies]\n{dep_line}\n'
            )
            (root / "docs").mkdir()
            for document in ("SPEC.md", "dependencies.md"):
                (root / "docs" / document).write_text("termlens 0.10\nrust-version 1.94.1\n")
            return subprocess.run(["/bin/bash", str(script)], text=True, capture_output=True, cwd=directory)

    def test_a_matching_skill_passes_and_a_patch_release_is_not_drift(self):
        for skill, dep in (("0.10.1", 'termlens = { version = "0.10", features = ["serde"] }'),
                           ("0.10.0", 'termlens = "0.10"'),
                           ("0.10.7", 'termlens = { version = "0.10.1" }')):
            with self.subTest(skill=skill, dep=dep):
                run = self.run_check(skill, dep)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_drift_fails_in_both_directions_and_says_so(self):
        for skill, dep in (("0.9.0", 'termlens = { version = "0.10", features = ["serde"] }'),
                           ("0.11.0", 'termlens = "0.10"'),
                           ("1.0.0", 'termlens = "0.10"')):
            with self.subTest(skill=skill, dep=dep):
                run = self.run_check(skill, dep)
                self.assertNotEqual(run.returncode, 0)
                self.assertIn("::error::", run.stdout)

    def test_an_unreadable_claim_is_a_failure_not_a_pass(self):
        self.assertNotEqual(self.run_check("", 'termlens = "0.10"').returncode, 0)
        self.assertNotEqual(self.run_check("0.10.1", "insta = \"1\"").returncode, 0)

    def test_stale_documented_dependency_version_fails(self):
        run = self.run_check("0.11.0", 'termlens = "0.11"')
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("docs/SPEC.md does not name termlens 0.11", run.stdout)

    def test_the_committed_skill_matches_the_committed_manifest(self):
        run = subprocess.run(["/bin/bash", str(SCRIPTS / "check-skill-version.sh")],
                             text=True, capture_output=True, cwd=ROOT)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


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


class SemverBaselineTests(unittest.TestCase):
    """release.yml never skips cargo-semver-checks on a crates.io error (#130).

    The step this replaced read every answer but a 200 as "first release" and
    skipped the check, so a 429 or a 503 switched the gate off before publish.
    """

    def lookup(self, url, attempts=3):
        return subprocess.run(
            ["bash", str(SCRIPTS / "crates-io-baseline.sh"), "oxidelake-core"],
            env={**os.environ, "CRATES_IO_API": url, "BASELINE_ATTEMPTS": str(attempts),
                 "BASELINE_PAUSE": "0"},
            capture_output=True, text=True,
        )

    def test_a_published_baseline_passes(self):
        with stand_in([200]) as (url, asked):
            run = self.lookup(url)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(asked, ["/crates/oxidelake-core"])

    def test_an_unavailable_registry_fails_after_retrying(self):
        for status in (503, 429, 500):
            with self.subTest(status=status), stand_in([status]) as (url, asked):
                run = self.lookup(url, attempts=3)
                self.assertNotEqual(run.returncode, 0, run.stdout)
                self.assertIn("::error::", run.stdout)
                self.assertIn(f"HTTP {status}", run.stdout)
                self.assertEqual(len(asked), 3, "a 429 or a 5xx is retried before failing")

    def test_a_blip_is_retried_and_then_passes(self):
        with stand_in([503, 429, 200]) as (url, asked):
            run = self.lookup(url, attempts=4)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(len(asked), 3)

    def test_a_missing_crate_is_an_error_not_a_first_release(self):
        for status in (404, 301, 403):
            with self.subTest(status=status), stand_in([status]) as (url, asked):
                run = self.lookup(url)
                self.assertNotEqual(run.returncode, 0, run.stdout)
                self.assertIn("::error::", run.stdout)
                self.assertEqual(len(asked), 1, "only an overloaded registry is worth asking again")

    def test_no_answer_at_all_fails(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        run = self.lookup(f"http://127.0.0.1:{port}", attempts=2)
        self.assertNotEqual(run.returncode, 0, run.stdout)
        self.assertIn("HTTP 000", run.stdout)

    def test_the_release_workflow_runs_the_check_unconditionally(self):
        job = workflow_job("release.yml", "semver")
        self.assertIn("        run: .github/scripts/crates-io-baseline.sh oxidelake-core", job)
        self.assertTrue(any("obi1kenobi/cargo-semver-checks-action@" in line for line in job))
        conditions = [line.strip() for line in job if line.strip().startswith("if:")]
        self.assertEqual(conditions, [], "no step of the semver job may be skipped")


FAKE_GH = r'''#!/usr/bin/env python3
"""The three `gh release` calls binaries.yml makes, against a directory."""
import os, shutil, sys
from pathlib import Path

release = Path(os.environ["FAKE_RELEASE"])
args = sys.argv[1:]
with open(os.environ["FAKE_GH_LOG"], "a") as log:
    log.write(" ".join(args) + "\n")


def option(name, default=None):
    return args[args.index(name) + 1] if name in args else default


if args[:2] == ["release", "view"]:
    for asset in sorted(release.iterdir()):
        print(asset.name)
elif args[:2] == ["release", "upload"]:
    upload = Path(args[3])
    if (release / upload.name).exists():
        sys.exit(f"asset under the same name already exists: {upload.name}")
    shutil.copy(upload, release / upload.name)
elif args[:2] == ["release", "download"]:
    name = option("--pattern")
    if not (release / name).exists():
        sys.exit(f"no assets match the file pattern {name}")
    shutil.copy(release / name, Path(option("--dir", ".")) / name)
else:
    sys.exit(f"fake gh: unexpected call {args}")
'''


class BinariesRepairTests(unittest.TestCase):
    """binaries.yml completes a partly attached target from what is attached (#131).

    The steps run verbatim out of the workflow, against a fake `gh` whose
    release is a directory. The job's `if:` conditions are mirrored in
    `run_job`, and `test_the_build_runs_only_when_nothing_is_attached`
    holds the workflow to that mirror.
    """

    TAG = "v0.3.0"
    TARGET = "x86_64-unknown-linux-musl"
    ARCHIVE = "oxidelake-0.3.0-x86_64-unknown-linux-musl.tar.gz"
    ALIAS = "oxidelake-x86_64-unknown-linux-musl.tar.gz"
    STATUS = "Already on the release?"
    RECOVER = "Recover the missing files from the attached archive"
    ATTACH = "Attach to the release"

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for name in ("bin", "release", "work"):
            (self.root / name).mkdir()
        gh = self.root / "bin" / "gh"
        gh.write_text(FAKE_GH)
        gh.chmod(0o755)
        self.release = self.root / "release"
        self.log = self.root / "gh.log"
        self.log.touch()

    def attach(self, name, data):
        (self.release / name).write_bytes(data)

    def step(self, name):
        output = self.root / "github_output"
        output.write_text("")
        run = subprocess.run(
            ["bash", "-e", "-c", step_script("binaries.yml", name)], cwd=self.root / "work",
            text=True, capture_output=True,
            env={**os.environ, "PATH": str(self.root / "bin") + os.pathsep + os.environ["PATH"],
                 "TAG": self.TAG, "TARGET": self.TARGET, "GITHUB_REPOSITORY": "vyncint/oxidelake",
                 "GITHUB_OUTPUT": str(output), "GH_TOKEN": "unused", "FAKE_RELEASE": str(self.release),
                 "FAKE_GH_LOG": str(self.log)},
        )
        outputs = dict(line.split("=", 1) for line in output.read_text().splitlines() if "=" in line)
        return run, outputs

    def run_job(self):
        """The matrix job for one target, with nothing to build."""
        run, outputs = self.step(self.STATUS)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        state = outputs.get("state")
        self.assertIn(state, {"complete", "repair", "build"}, run.stdout)
        if state == "build":
            self.fail("this target needs a build, which a test cannot run")
        for name in ((self.RECOVER, self.ATTACH) if state == "repair" else ()):
            run, _ = self.step(name)
            self.assertEqual(run.returncode, 0, f"{name}:\n{run.stdout}{run.stderr}")
        return state

    def sha256_line(self, name):
        digest = hashlib.sha256((self.release / name).read_bytes()).hexdigest()
        return f"{digest}  {name}\n"

    def test_an_archive_without_its_checksum_and_alias_is_completed(self):
        # The run the issue describes: the archive uploaded, then the step died.
        self.attach(self.ARCHIVE, b"the archive that was published")
        self.assertEqual(self.run_job(), "repair")
        names = sorted(p.name for p in self.release.iterdir())
        self.assertEqual(names, sorted([self.ARCHIVE, self.ARCHIVE + ".sha256",
                                        self.ALIAS, self.ALIAS + ".sha256"]))
        self.assertEqual((self.release / self.ARCHIVE).read_bytes(), b"the archive that was published")
        self.assertEqual((self.release / self.ALIAS).read_bytes(), b"the archive that was published",
                         "the alias is the published archive, not a rebuild")
        self.assertEqual((self.release / f"{self.ARCHIVE}.sha256").read_text(), self.sha256_line(self.ARCHIVE))
        self.assertEqual((self.release / f"{self.ALIAS}.sha256").read_text(), self.sha256_line(self.ALIAS))
        # And a second re-run finds nothing left to do.
        self.assertEqual(self.run_job(), "complete")

    def test_each_missing_file_alone_is_restored_without_touching_the_rest(self):
        published = b"published bytes"
        everything = {self.ARCHIVE: published, self.ALIAS: published,
                      f"{self.ARCHIVE}.sha256": None, f"{self.ALIAS}.sha256": None}
        for missing in everything:
            with self.subTest(missing=missing):
                for asset in self.release.iterdir():
                    asset.unlink()
                for name, data in everything.items():
                    if name != missing:
                        self.attach(name, data if data is not None else
                                    f"{hashlib.sha256(published).hexdigest()}  {name[:-len('.sha256')]}\n".encode())
                before = {p.name: p.read_bytes() for p in self.release.iterdir()}
                self.assertEqual(self.run_job(), "repair")
                after = {p.name: p.read_bytes() for p in self.release.iterdir()}
                self.assertEqual(set(after), set(everything))
                for name, data in before.items():
                    self.assertEqual(after[name], data, f"{name} was attached already and must not change")

    def test_a_complete_target_is_left_alone(self):
        for name in (self.ARCHIVE, self.ALIAS):
            self.attach(name, b"bytes")
            self.attach(f"{name}.sha256", self.sha256_line(name).encode())
        self.assertEqual(self.run_job(), "complete")
        self.assertNotIn("release upload", self.log.read_text())

    def test_an_empty_release_is_built(self):
        _, outputs = self.step(self.STATUS)
        self.assertEqual(outputs.get("state"), "build")

    def test_a_checksum_without_any_archive_is_refused(self):
        # A rebuilt archive cannot match a checksum somebody already downloaded.
        self.attach(f"{self.ARCHIVE}.sha256", b"0" * 64 + b"  " + self.ARCHIVE.encode() + b"\n")
        run, outputs = self.step(self.STATUS)
        self.assertNotEqual(run.returncode, 0, run.stdout)
        self.assertIn("::error::", run.stdout)
        self.assertNotIn("state", outputs)

    def test_the_build_runs_only_when_nothing_is_attached(self):
        for name in ("Add the target's standard library", "Build", "Package"):
            self.assertEqual(step_condition("binaries.yml", name), "steps.published.outputs.state == 'build'", name)
        self.assertEqual(step_condition("binaries.yml", "musl linker (Linux)"),
                         "steps.published.outputs.state == 'build' && contains(matrix.target, 'musl')")
        self.assertEqual(step_condition("binaries.yml", self.RECOVER), "steps.published.outputs.state == 'repair'")
        self.assertEqual(step_condition("binaries.yml", self.ATTACH), "steps.published.outputs.state != 'complete'")


if __name__ == "__main__":
    unittest.main()
