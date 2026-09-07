#!/usr/bin/env bash
# Every published crate must carry the metadata crates.io shows — README,
# repository, homepage, documentation, keywords, categories, description.
#
# `[workspace.package]` declaring a field does NOT put it in a crate: each
# manifest has to opt in with `field.workspace = true`. The nine 0.1.0 crates
# inherited version/edition/license/authors and nothing else, so crates.io
# showed them with no README, no repository link and no keywords. Metadata on
# a published version is immutable, which is why this is a gate and not a
# reminder.
set -euo pipefail
cd "$(dirname "$0")/../.."

fail=0
say() { printf '  %s\n' "$*"; }

meta="$(cargo metadata --no-deps --format-version 1 --locked)"
for pkg in $(printf '%s' "$meta" | jq -r '.packages[] | select(.publish == null) | .name' | sort); do
  # One field per line: a tab-separated read would collapse empty fields,
  # and an empty field is exactly what this script is looking for.
  mapfile -t f < <(printf '%s' "$meta" | jq -r --arg n "$pkg" '
    .packages[] | select(.name == $n) |
    (.readme // ""), (.repository // ""), (.homepage // ""), (.documentation // ""),
    (.keywords | length), (.categories | length), (.description // "")')
  readme=${f[0]}; repo=${f[1]}; home=${f[2]}; docs=${f[3]}; nkw=${f[4]}; ncat=${f[5]}; desc=${f[6]}
  missing=()
  [ -n "$readme" ] || missing+=(readme)
  [ -n "$repo" ]   || missing+=(repository)
  [ -n "$home" ]   || missing+=(homepage)
  [ -n "$docs" ]   || missing+=(documentation)
  [ "$nkw" -gt 0 ] || missing+=(keywords)
  [ "$ncat" -gt 0 ] || missing+=(categories)
  [ -n "$desc" ]   || missing+=(description)
  if [ "$docs" != "https://docs.rs/$pkg" ]; then
    missing+=("documentation should be https://docs.rs/$pkg, is '$docs'")
  fi
  # The README must actually be in the package, not merely named in it.
  if ! cargo package --list -p "$pkg" --allow-dirty 2>/dev/null | grep -qx 'README.md'; then
    missing+=("README.md not in the packaged file list")
  fi
  if [ "${#missing[@]}" -eq 0 ]; then
    say "ok   $pkg"
  else
    say "FAIL $pkg: ${missing[*]}"; fail=1
  fi
done
[ "$fail" -eq 0 ] || { echo "crate metadata: at least one published crate is missing what crates.io shows (see above)"; exit 1; }
echo "crate metadata: every published crate carries README, repository, homepage, docs.rs link, keywords, categories"
