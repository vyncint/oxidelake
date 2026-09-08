# OxideLake developer entry points. Every target is a plain cargo invocation;
# `make gate` is the quality gate from docs/verification.md (what CI runs).
#
#   make gate        full gate: fmt, clippy (default + cuda + metal on macOS), tests, docs, coherence, deny
#   make metal       the Metal lane, including the on-device conformance suite (macOS)
#   make msrv        verify the declared rust-version actually builds the workspace
#   make quickstart  gen-data + a query with the release binary

.DEFAULT_GOAL := help
SHELL := /bin/sh
CARGO ?= cargo
MSRV := $(shell sed -n 's/^rust-version = "\(.*\)"/\1/p' Cargo.toml)
UNAME := $(shell uname -s)
# The normal and ignored passes must resolve identical dependency features;
# selecting compute alone rebuilt Arrow/DataFusion for the second pass in CI.
METAL_TEST_ARGS := -p oxidelake-memory -p oxidelake-device -p oxidelake-compute -p oxidelake-runtime -p oxidelake-tui --features oxidelake-runtime/metal --locked --timings

.PHONY: help fmt fmt-check lint lint-cuda lint-metal lint-predict test test-predict test-io-uring test-termlens-cli test-metal test-metal-device check-cuda check-predict-no-second-cuda doc coherence deny gate metal msrv quickstart clean-data release-scripts crate-metadata skill-version zizmor ci-scripts stress-tui

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

fmt: ## format the workspace
	$(CARGO) fmt --all

fmt-check: ## fail on unformatted code
	$(CARGO) fmt --all -- --check

lint: ## clippy, default features (pure CPU), warnings are errors
	$(CARGO) clippy --workspace --all-targets --locked -- -D warnings

lint-cuda: ## clippy with the cuda feature (builds without CUDA installed)
	$(CARGO) clippy -p oxidelake-runtime --all-targets --features cuda --locked -- -D warnings

lint-metal: ## clippy with the metal feature (macOS)
	$(CARGO) clippy -p oxidelake-runtime --all-targets --features metal --locked -- -D warnings

lint-predict: ## clippy with the predict feature (in-database inference)
	$(CARGO) clippy -p oxidelake-runtime --all-targets --features predict --locked -- -D warnings

test-predict: ## the `predict` UDF against the model it loaded
	$(CARGO) test -p oxidelake-compute --features predict --test predict --locked --timings

# `cargo tree` prints an already-shown node again as `name ver (*)`, so a
# bare `grep -c` counts tree references rather than crates and reports two
# cudarcs where there is one. Strip the marker and count distinct versions:
# what matters is that oxmera adds no CUDA of its own, not how many edges
# reach OxideLake's.
#
# `--color never` is not cosmetic. CI sets CARGO_TERM_COLOR=always, which
# wraps the `(*)` in escape codes so the sed below cannot strip it and the
# two lines sort as distinct versions — this check passed locally and failed
# on the runner for exactly that reason. Pin the format rather than teach the
# parser about colour.
check-predict-no-second-cuda: ## oxmera must never bring a CUDA stack of its own
	@set -e; \
	tree=$$($(CARGO) tree --color never -p oxidelake-runtime --features predict,cuda -e normal --prefix none); \
	for crate in oxmera-cuda ctor; do \
	  if echo "$$tree" | grep -qE "^$$crate "; then \
	    echo "ERROR: $$crate is in the graph — oxmera is pulling its own CUDA backend"; \
	    echo "       (it must be depended on with default-features = false; ADR-0015)"; \
	    exit 1; \
	  fi; \
	done; \
	versions=$$(echo "$$tree" | sed 's/ (\*)$$//' | grep -E '^cudarc ' | sort -u | wc -l); \
	test "$$versions" = "1" || { \
	  echo "ERROR: $$versions distinct cudarc versions in one process"; exit 1; }; \
	echo "one cudarc (OxideLake's own), no oxmera-cuda, no pre-main ctor"

test: ## the full default-feature test suite
	$(CARGO) test --workspace --locked --timings

test-io-uring: ## io_uring object store tests (Linux; skips with a reason where io_uring is denied)
	$(CARGO) test -p oxidelake-storage --features io-uring --locked --timings

# Not part of `make gate`: it installs termlens-cli from crates.io at the
# version Cargo.lock names, and the gate has to run on a machine with no
# network. CI runs it in the default Linux test lane (docs/verification.md).
# The tests are #[ignore]d for the same reason — a `cargo test` on a
# published crate must not install anything behind a contributor's back.
test-termlens-cli: ## the termlens-cli suite against the committed screens (installs termlens-cli)
	$(CARGO) test -p oxidelake-tui --test termlens_cli --locked -- --ignored

test-metal: ## Metal package tests and macOS PTY tests with one dependency graph
	$(CARGO) test $(METAL_TEST_ARGS)

# `--skip termlens_cli` keeps this pass off the network: the graph includes
# -p oxidelake-tui, whose `termlens_cli` tests are #[ignore]d (so `--ignored`
# selects them) and install a tool from crates.io. Every test in that file is
# named with the prefix for exactly this. The conformance suite is what this
# target is for.
test-metal-device: ## reuse the Metal test graph for on-device conformance
	@if swift -e 'import Metal; exit(MTLCreateSystemDefaultDevice() == nil ? 1 : 0)' 2>/dev/null; then \
	  OXIDE_BACKEND=metal $(CARGO) test $(METAL_TEST_ARGS) -- --ignored --skip termlens_cli; \
	else \
	  echo "No Metal device on this runner — on-device conformance skipped."; \
	fi

check-cuda: ## the GPU compile gate: cuda code builds with no CUDA installed
	$(CARGO) check -p oxidelake-runtime --features cuda --locked

doc: ## rustdoc for the workspace, warnings are errors
	RUSTDOCFLAGS="-D warnings" $(CARGO) doc --workspace --no-deps --all-features --locked

coherence: ## no duplicate arrow / parquet / datafusion / object_store / tonic / prost majors
	@dups=$$($(CARGO) tree --workspace -d -e normal | grep -E '^(arrow|parquet|datafusion|object_store|tonic|prost)' || true); \
	if [ -n "$$dups" ]; then echo "duplicate majors:"; echo "$$dups"; exit 1; else echo "dependency chain coherent"; fi

release-scripts: ## the changelog extractor's own tests (release.yml depends on it)
	./.github/scripts/test-extract-changelog.sh

crate-metadata: ## every published crate carries README, repository, keywords, categories, docs.rs link
	./.github/scripts/check-crate-metadata.sh

ci-scripts: ## CI change detection, required results, Metal build reuse and stress coverage
	python3 -B -m unittest discover -s .github/scripts -p test_ci.py -v

stress-tui: ## build once and stress all three thread counts (ITERS defaults to 100)
	python3 .github/scripts/stress-tui.py

skill-version: ## the vendored termlens skill names the version the workspace depends on
	./.github/scripts/check-skill-version.sh

zizmor: ## workflow security audit at the level CI enforces (cargo install --locked zizmor --version 1.29.0)
	zizmor --persona=pedantic --offline .github/workflows/

deny: ## advisories, licenses, bans and sources (cargo-deny)
	$(CARGO) deny --all-features check

gate: fmt-check lint lint-cuda lint-predict test test-predict check-cuda check-predict-no-second-cuda doc coherence deny release-scripts crate-metadata ci-scripts skill-version zizmor msrv ## the whole quality gate — the same list CI runs
ifeq ($(UNAME),Linux)
	$(MAKE) test-io-uring
endif
ifeq ($(UNAME),Darwin)
	$(MAKE) metal
endif
	@echo "gate green"

metal: lint-metal test-metal test-metal-device ## Metal lane: lint, tests and conformance when a device is present

msrv: ## build the workspace with the declared rust-version
	rustup toolchain install $(MSRV) --profile minimal --no-self-update
	CARGO_TARGET_DIR=target/msrv rustup run $(MSRV) $(CARGO) check --workspace --all-targets --locked

quickstart: ## release build, 1M-row demo table, one query
	$(CARGO) build --release -p oxidelake-runtime
	./target/release/oxide gen-data --rows 1000000 --out data/
	./target/release/oxide sql -q "SELECT k, SUM(v) FROM t GROUP BY k ORDER BY k" --table t=data/

clean-data: ## remove generated datasets and spill files
	rm -rf data spill
