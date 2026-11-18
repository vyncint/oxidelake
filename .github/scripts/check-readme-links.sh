#!/usr/bin/env bash
# README.md is the README of all nine published crates (`readme.workspace =
# true`), and crates.io resolves a relative link in it against that crate's
# own directory in the repository, not the root: `docs/SPEC.md` on the
# oxidelake-api page became …/blob/HEAD/crates/oxidelake-api/docs/SPEC.md, a
# 404. Every relative link in the 0.2.0 README was one, on every page (#129).
# GitHub renders the same file from the root, where each of them works, so
# nothing there could show it. termlens 0.10.3 and launchbound 2.2.1 fixed
# the same thing the same way.
#
# So no relative link at all, and a link into this repository's main branch
# must name a path that is in this checkout. That half is checked against the
# tree, never over the network, which is slow, flaky, and can answer 200 for
# the wrong reason. A link is every target a renderer follows: an inline
# `](…)`, a reference definition (`[x]: path`), and an HTML `src` or `href`.
#
# Runs from check-crate-metadata.sh (`make crate-metadata`, and CI's
# release-scripts job). Bash 3.2 compatible: macOS runs it too.
set -euo pipefail
cd "$(dirname "$0")/../.."

readme=README.md
repo="https://github.com/vyncint/oxidelake"
[ -f "$readme" ] || { echo "::error::no $readme at the repository root"; exit 1; }

# Every link target, once each. Inline links lose a title and angle brackets;
# a reference definition starts a line (a footnote, `[^1]:`, is not one); HTML
# attributes may use either quote.
links() {
  {
    grep -oE '\]\([^)]+\)' "$readme" \
      | sed -E 's/^\]\([[:space:]]*//; s/\)$//; s/^<//; s/[[:space:]].*$//; s/>$//' || true
    grep -E '^ {0,3}\[[^]^][^]]*\]:' "$readme" \
      | sed -E 's/^ {0,3}\[[^]]+\]:[[:space:]]*//; s/^<//; s/[[:space:]].*$//; s/>$//' || true
    grep -oE "(src|href)=(\"[^\"]*\"|'[^']*')" "$readme" | sed -E 's/^(src|href)=.//; s/.$//' || true
  } | sort -u
}

fail=0
checked=0
while IFS= read -r link; do
  [ -n "$link" ] || continue
  checked=$((checked + 1))
  case "$link" in
    '#'* | mailto:*) ;;
    "$repo/blob/main/"* | "$repo/tree/main/"*)
      path=${link#"$repo/blob/main/"}
      path=${path#"$repo/tree/main/"}
      path=${path%%#*}
      if [ ! -e "$path" ]; then
        echo "::error file=$readme::$link names $path, which is not in this repository"
        fail=1
      fi
      ;;
    http://* | https://*) ;;
    *)
      echo "::error file=$readme::relative link '$link': crates.io resolves it against each crate's directory, where it is a 404; write $repo/blob/main/${link#./}"
      fail=1
      ;;
  esac
done < <(links)

# Finding nothing means the extraction broke, not that the README is clean.
[ "$checked" -gt 0 ] || { echo "::error file=$readme::found no links at all; the link extraction is broken"; exit 1; }
[ "$fail" -eq 0 ] || { echo "readme links: a link above is relative or names a path this repository does not have"; exit 1; }
echo "readme links: all $checked absolute, every one into this repository names a path that is here"
