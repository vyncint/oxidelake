#!/usr/bin/env bash
# Fails unless crates.io confirms that CRATE is published, so the release's
# cargo-semver-checks job has a baseline to compare against (#130).
#
#   .github/scripts/crates-io-baseline.sh oxidelake-core
#
# This used to be a branch in release.yml: HTTP 200 ran the semver check and
# any other answer meant "first release, skip it". A 429 or a 503 therefore
# switched the gate off, silently, minutes before `cargo publish`. Every
# OxideLake crate has been on crates.io since 0.1.0, so there is no first
# release left to skip for, and nothing but a 200 is an answer to act on:
#
#   200            the baseline exists — the semver check runs
#   404            an error: the name is wrong or crates.io is, and either
#                  way skipping the check is not the remedy
#   429, 5xx, 000  retried (000 is curl's code for no answer at all), then
#                  an error: a blip is worth a minute, an outage is a re-run
#   anything else  an error
#
# CRATES_IO_API, BASELINE_ATTEMPTS and BASELINE_PAUSE exist for the stand-in
# server in .github/scripts/test_ci.py.
set -euo pipefail

crate="${1:?usage: crates-io-baseline.sh CRATE}"
api="${CRATES_IO_API:-https://crates.io/api/v1}"
attempts="${BASELINE_ATTEMPTS:-4}"
pause="${BASELINE_PAUSE:-20}"
# crates.io refuses requests without a user agent that names the client.
agent="oxidelake-release (github.com/${GITHUB_REPOSITORY:-vyncint/oxidelake})"

attempt=1
while :; do
  # `|| true`: curl exits non-zero when nothing answers, and that case gets
  # the same retry as a 503 rather than ending the script under `-e`.
  status="$(curl -sS -o /dev/null -w '%{http_code}' -A "$agent" "$api/crates/$crate" || true)"
  case "$status" in
    200)
      echo "$crate is on crates.io (HTTP 200): cargo-semver-checks has a baseline"
      exit 0
      ;;
    404)
      echo "::error::crates.io answered HTTP 404 for $crate. Every OxideLake crate has been published since 0.1.0, so this is a wrong crate name or a crates.io fault, not a first release; the semver check is not skipped for it."
      exit 1
      ;;
    429 | 5?? | 000) ;;
    *)
      echo "::error::crates.io answered HTTP $status for $crate; only a 200 lets the semver check run, and it is never skipped."
      exit 1
      ;;
  esac
  if [ "$attempt" -ge "$attempts" ]; then
    echo "::error::crates.io answered HTTP $status for $crate on all $attempts attempts. Nothing has been published: re-run this job once crates.io answers."
    exit 1
  fi
  echo "crates.io answered HTTP $status for $crate; retrying in ${pause}s (attempt $attempt of $attempts)"
  sleep "$pause"
  attempt=$((attempt + 1))
done
