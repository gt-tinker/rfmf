#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# test_chain_isolation.sh -- "C chains at once" means "C independent runs",
# proven byte-for-byte; plus determinism of a repeated command.
#
# CHAIN ISOLATION: chain i's stdout block inside a multi-chain launch must be
# byte-identical to the block of a C=1 run of that same (seed, t0). This is the
# central contract of the multi-chain kernel. It holds because the Gumbel handed
# to vertex v in sweep k of chain i is a pure function of (seeds[i], k, v) --
# counter-based Philox, not a stateful stream -- so a chain cannot observe how
# many neighbours it was launched with. The SELECT argmax is an order-free
# packed atomicMax, and each chain keeps its own spins, running cut and
# best-so-far snapshot.
#
# It is also the sharpest cross-chain-interference detector in the artifact: a
# chain whose slice arithmetic overlapped a neighbour's would score the wrong
# spins, and the very first byte of its partition string would move. The C=33
# case below deliberately exceeds the SM count so that wave scheduling is
# exercised -- chains that do not all start at once must still not see each
# other.
#
# DETERMINISM: the same command re-run must produce the same bytes. No seed is
# drawn at run time; every seed is either given on the command line or resolved
# once and printed to stderr.
#
# ---------------------------------------------------------------------------
# WHAT THIS TEST DELIBERATELY DOES NOT COVER, AND WHY
#
# The development tree's tests/test_invariance.sh additionally proves that
# output is invariant across the whole launch-geometry knob cross -- --tpb,
# --delta-width, --corr, --reduce, --scatter, --vmap and --state. None of that
# is reproduced here, because THIS ARTIFACT'S BINARY EXPOSES NO SUCH KNOBS.
# `./main` accepts exactly --seeds, --t0s, --phases, --iters and --device; the
# tuning knobs exist only in the development build, where they make each
# throughput optimisation verifiable as a pure schedule change.
#
# So "output is invariant across schedules" is not a claim this artifact makes
# or needs: the shipped binary has exactly one schedule, chosen by
# src/tuning.hpp. Porting the knob cross here would not be a stricter test, it
# would simply fail on unrecognised flags. What remains -- isolation and
# determinism -- is exactly the part that is meaningful for a single-schedule
# binary, and it is ported unchanged.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."

fail=0
n=0

# stdout only (stderr carries the load/geometry banner, which is expected to
# differ between a C=33 launch and a C=1 launch), with the wall-clock field
# normalized away.
#
# time-sec MUST be excluded: it is the one field that is legitimately different
# on every run. Everything else -- cut values, both partition strings per
# chain, and the summary's best-chain -- has to be invariant.
run() {
    ./main "$@" 2>/dev/null \
        | sed -e 's/time-sec: [0-9.]*/time-sec: <normalized>/'
}

# check_isolation LABEL INSTANCE SEEDS T0S PHASES ITERS CHAIN_INDEX...
#
# Gset weights all have scale 1, so a chain block is exactly 3 lines: the
# `device:` line plus the two Solution- lines.
check_isolation() {
    local label="$1" instance="$2" seeds="$3" t0s="$4" phases="$5" iters="$6"
    shift 6
    local multi
    multi=$(run "$instance" --seeds "$seeds" --t0s "$t0s" --phases "$phases" --iters "$iters")
    if [[ -z "$multi" ]]; then
        echo "  FAIL $label: multi-chain run produced no output"
        fail=$((fail + 1))
        return
    fi
    IFS=',' read -ra sArr <<<"$seeds"
    IFS=',' read -ra tArr <<<"$t0s"
    for i in "$@"; do
        local block single
        block=$(awk -v want="$((i + 1))" \
            '/^device:/{c++} c==want && (/^device:/ || /^Solution-/)' <<<"$multi")
        single=$(run "$instance" --seeds "${sArr[$i]}" --t0s "${tArr[$i]}" \
                     --phases "$phases" --iters "$iters" | grep -v '^summary:')
        n=$((n + 1))
        if [[ -z "$block" || "$block" != "$single" ]]; then
            echo "  FAIL $label: chain $i inside C=${#sArr[@]} differs from its C=1 run"
            fail=$((fail + 1))
        fi
    done
}

# check_determinism LABEL INSTANCE ARGS...
check_determinism() {
    local label="$1"; shift
    local ref again
    ref=$(run "$@")
    if [[ -z "$ref" ]]; then
        echo "  FAIL $label: reference run produced no output"
        fail=$((fail + 1))
        return
    fi
    again=$(run "$@")
    n=$((n + 1))
    if [[ "$again" != "$ref" ]]; then
        echo "  FAIL $label: re-run is not byte-identical"
        fail=$((fail + 1))
    fi
}

echo "chain isolation: G1 C=4 vs four C=1 runs"
check_isolation "G1-iso" Gset/G1 "42,7,42,9" "105,105,90,120" "8,4,2,1" \
    "150,150,150,150" 0 1 2 3

echo "chain isolation: G11 C=33 (wave scheduling) vs C=1 runs of chains 0, 15, 32"
seeds33=$(seq -s, 501 533)
t033=$(printf '105,%.0s' {1..32})105
check_isolation "G11-iso33" Gset/G11 "$seeds33" "$t033" "8,1" "100,100" 0 15 32

echo "chain isolation: G81 C=2 vs its C=1 runs (largest instance)"
check_isolation "G81-iso" Gset/G81 "3,4" "105,90" "64" "100" 0 1

echo "chain isolation: G70 C=2 (sparsest, mean deg 2)"
check_isolation "G70-iso" Gset/G70 "13,14" "105,105" "8,2,1" "150,150,150" 0 1

echo "chain isolation: G64 C=2 (heaviest degree tail, max deg 589)"
check_isolation "G64-iso" Gset/G64 "9,10" "105,95" "64,16" "150,150" 0 1

# t0=0.01 in the mix pushes every chain onto the int64 two-pass score path
# (the path is gated on min(t0s)), so isolation is checked on both score paths.
echo "chain isolation: G1 C=2 with t0=0.01 (int64 two-pass score path)"
check_isolation "G1-iso-int64" Gset/G1 "42,42" "0.01,105" "8,4" "150,150" 0 1

echo "determinism: identical command, identical bytes"
check_determinism "G1-det" Gset/G1 --seeds 42,7,42 --t0s 105,105,90 \
    --phases 64,32,16,8,4,2,1 --iters 150,150,150,150,150,150,150
check_determinism "G81-det" Gset/G81 --seeds 3,4 --t0s 105,90 --phases 64 --iters 100

echo "chain-isolation: $n comparisons, $fail failures"
[[ $fail -eq 0 ]]
