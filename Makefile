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

.PHONY: help fmt fmt-check lint lint-cuda lint-metal test test-io-uring check-cuda doc coherence deny gate metal msrv quickstart clean-data

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

fmt: ## format the workspace
	$(CARGO) fmt --all

fmt-check: ## fail on unformatted code
	$(CARGO) fmt --all -- --check

lint: ## clippy, default features (pure CPU), warnings are errors
	$(CARGO) clippy --workspace --all-targets -- -D warnings

lint-cuda: ## clippy with the cuda feature (builds without CUDA installed)
	$(CARGO) clippy -p oxidelake-runtime --all-targets --features cuda -- -D warnings

lint-metal: ## clippy with the metal feature (macOS)
	$(CARGO) clippy -p oxidelake-runtime --all-targets --features metal -- -D warnings

lint-predict: ## clippy with the predict feature (in-database inference)
	$(CARGO) clippy -p oxidelake-runtime --all-targets --features predict -- -D warnings

test-predict: ## the `predict` UDF against the model it loaded
	$(CARGO) test -p oxidelake-compute --features predict --test predict

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
	$(CARGO) test --workspace

test-io-uring: ## io_uring object store tests (Linux; skips with a reason where io_uring is denied)
	$(CARGO) test -p oxidelake-storage --features io-uring

check-cuda: ## the GPU compile gate: cuda code builds with no CUDA installed
	$(CARGO) check -p oxidelake-runtime --features cuda

doc: ## rustdoc for the workspace, warnings are errors
	RUSTDOCFLAGS="-D warnings" $(CARGO) doc --workspace --no-deps

coherence: ## no duplicate arrow / parquet / datafusion / object_store / tonic / prost majors
	@dups=$$($(CARGO) tree --workspace -d -e normal | grep -E '^(arrow|parquet|datafusion|object_store|tonic|prost)' || true); \
	if [ -n "$$dups" ]; then echo "duplicate majors:"; echo "$$dups"; exit 1; else echo "dependency chain coherent"; fi

release-scripts: ## the changelog extractor's own tests (release.yml depends on it)
	./.github/scripts/test-extract-changelog.sh

zizmor: ## workflow security audit at the level CI enforces (cargo install --locked zizmor --version 1.29.0)
	zizmor --persona=pedantic --offline .github/workflows/

deny: ## advisories, licenses, bans and sources (cargo-deny)
	$(CARGO) deny check

gate: fmt-check lint lint-cuda lint-predict test test-predict check-cuda check-predict-no-second-cuda doc coherence deny release-scripts zizmor ## the whole quality gate — the same list CI runs
ifeq ($(UNAME),Linux)
	$(MAKE) test-io-uring
endif
ifeq ($(UNAME),Darwin)
	$(MAKE) lint-metal
endif
	@echo "gate green"

metal: ## Metal lane: lint, device-independent tests, conformance ON the device (macOS)
	$(CARGO) clippy -p oxidelake-runtime --all-targets --features metal -- -D warnings
	$(CARGO) test -p oxidelake-memory -p oxidelake-device -p oxidelake-compute -p oxidelake-runtime --features metal
	$(CARGO) test -p oxidelake-compute --features metal -- --ignored

msrv: ## build the workspace with the declared rust-version
	rustup toolchain install $(MSRV) --profile minimal
	CARGO_TARGET_DIR=target/msrv $(CARGO) +$(MSRV) check --workspace --all-targets

quickstart: ## release build, 1M-row demo table, one query
	$(CARGO) build --release -p oxidelake-runtime
	./target/release/oxide gen-data --rows 1000000 --out data/
	./target/release/oxide sql -q "SELECT k, SUM(v) FROM t GROUP BY k ORDER BY k" --table t=data/

clean-data: ## remove generated datasets and spill files
	rm -rf data spill
