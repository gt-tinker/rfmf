#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_all.sh -- the in-image smoke suite.  Everything this artifact can check
# about itself in about twenty seconds, ordered so that each item takes strictly
# less on faith than the one before it.
#
# In one line: it proves CUT-VALUE CORRECTNESS, CHAIN ISOLATION and
# DETERMINISM.  It does not prove schedule invariance, and it does not prove
# bit-identity with the paper's run.  Both exclusions are spelled out below.
#
#   1. test_cut_consistency.sh -- the kernel's INCREMENTAL cut value against a
#      from-scratch CSR recompute, per chain, on all 71 Gset instances.  This
#      is the pair that catches the quiet failures: a wrong winner-winner
#      corr[] coupling, or a chain whose slice arithmetic overlapped a
#      neighbour's, leaves the spins and the printed partition looking entirely
#      plausible while the tracked value drifts away from them.
#
#   2. test_chain_isolation.sh -- a chain's block inside a C-chain run is
#      byte-identical to running that (seed, t0) on its own, including at
#      C=33 where wave scheduling means the chains do not all start together;
#      and the same command re-run gives the same bytes.  Chain isolation and
#      determinism.
#
#      NOTE: isolation and determinism are ALL this item checks.  In the
#      development tree the same comparisons sit alongside a sweep of the
#      launch-geometry knobs (--tpb, --corr, --reduce, --scatter, --vmap,
#      --state, --delta-width), which proves the output is invariant across
#      schedules.  That half is absent here and cannot be ported: this
#      artifact's ./main exposes only --seeds, --t0s, --phases, --iters and
#      --device, so the shipped binary has exactly one schedule, chosen at
#      compile time by src/tuning.hpp.  Schedule invariance is therefore not a
#      claim this package makes or can test.  See the header of
#      test_chain_isolation.sh.
#
#   3. verify_committed_cuts.sh -- every cut value committed under results/ is
#      recomputed from the partition bit string printed next to it, by
#      eval_cut.py: dependency-free Python that walks the original Gset edge
#      list and shares no code with the solver at all.  This is the one item a
#      reviewer who distrusts the entire C++/CUDA tree can still believe.
#
# ------------------------------ WHAT IT DOES NOT PROVE ---------------------
#
# It does not prove the annealing DYNAMICS.  This artifact deliberately ships
# no CPU reference implementation, so nothing here compares the kernel's
# tie-break order, corr[] coupling, cooling schedule or best-tracking against
# an independent serial implementation of the same algorithm.  Those are
# internal design claims and they are checked elsewhere, not in this package.
# What the three items above establish is narrower and, for an artifact, more
# to the point: that the value the kernel reports is internally consistent (1),
# that a chain's output is unchanged by the chains it was launched beside and
# that re-running the same command reproduces it byte-for-byte (2), and that it
# is genuinely achieved by the partitions the program printed (3).
#
# It does not prove SCHEDULE INVARIANCE.  Nothing here varies how the work is
# mapped onto the GPU, because nothing can: the shipped ./main has no
# launch-geometry knobs, so there is exactly one schedule to run.  Item 2 shows
# that C chains in one launch behave as C independent runs -- isolation, not
# independence from the launch geometry.  Any claim that this package
# demonstrates schedule invariance or schedule independence is wrong.
#
# It does not prove BIT-IDENTITY with the paper's run.  A green suite says the
# build on this machine is sane and self-consistent; it says nothing about
# whether this machine reproduces the committed logs byte-for-byte.  That is a
# separate and far longer check, and it is the one the reproducibility claim
# actually rests on:
#
#     ./scripts/bench_large_gset.sh --instances G60 --step 10000000  # ~15 min
#     ./scripts/bench_large_gset.sh --step 10000000                  # ~3.0 h
#
# then compare against README.md section 5.
#
# Run this suite first anyway.  Twenty seconds is a cheap way to discover that
# the GPU never got passed into the container, or that Gset/ did not make it
# into the image, before committing three hours to a paper-budget sweep.
#
# ------------------------------ ON MISSING TESTS ---------------------------
#
# A test script that is absent, or present but not executable, is reported as a
# FAILURE and not as a skip.  A suite that quietly skips what it cannot run is
# worse than no suite: it prints a passing verdict for coverage that never
# happened, which is exactly the thing an artifact is supposed to rule out.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."

usage() {
    cat <<'EOF'
Usage: ./tests/run_all.sh [--fast]

The in-image smoke suite.  Runs, in this order:

  1. tests/test_cut_consistency.sh    ~15 s     all 71 Gset instances
  2. tests/test_chain_isolation.sh    ~5 s      17 comparisons
  3. tests/verify_committed_cuts.sh   ~2 s      in its default mode
                                      -------
                              total   ~20-25 s  measured on a V100: 20 s in the
                                                container, 22 s on a host build

Proves cut-value correctness, chain isolation and determinism.  Prints PASS or
FAIL per item plus an overall verdict, and exits non-zero if any item failed.
A test that is missing or not executable counts as a FAILURE.

Items 1 and 2 need a GPU.  Item 3 is pure Python and needs none.

Options:
  --fast       run the cut-consistency test over a 6-instance sample instead of
               all 71, by setting GSET_SAMPLE=1.  Takes item 1 from ~15 s to
               ~1 s, so the suite finishes in ~8 s, and changes nothing else.
  -h, --help   print this message and exit.

This suite does NOT prove schedule invariance: the shipped ./main has no
launch-geometry knobs, so it has exactly one schedule and there is nothing to
vary.  See the header of this script.

It does NOT reproduce the paper's cut values either.  That is a separate, much
longer run, compared against README.md section 5:

  ./scripts/bench_large_gset.sh --instances G60 --step 10000000  # ~15 min
  ./scripts/bench_large_gset.sh --step 10000000                  # ~3.0 h
EOF
}

fast=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast)     fast=1; shift ;;
        -h|--help)  usage; exit 0 ;;
        *)
            printf 'run_all.sh: unrecognized argument: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

TOTAL=3
LABELS=()
STATUSES=()
ELAPSED=()

rule() { printf '%s\n' '==============================================================================='; }
thin() { printf '%s\n' '-------------------------------------------------------------------------------'; }

# Whole seconds as 19s / 2m14s.  Wall time is the number a reviewer budgets
# against, so it is printed per item as well as for the suite.
fmt_secs() {
    if (( $1 < 60 )); then
        printf '%ds' "$1"
    else
        printf '%dm%02ds' "$(($1 / 60))" "$(($1 % 60))"
    fi
}

# run_test <label> <script> [VAR=VALUE ...]
#
# Runs the script with its output flowing straight through -- these tests
# already print their own per-check FAIL lines, and swallowing them into a
# buffer would hide exactly the detail someone runs the suite to see.  Every
# item is attempted even after an earlier one fails; a partial verdict is not
# worth the seconds it would save.
run_test() {
    local label="$1" script="$2"; shift 2
    local -a envs=("$@")
    local start=$SECONDS status shown

    shown="$script"
    (( ${#envs[@]} )) && shown="${envs[*]} $script"

    printf '\n'
    rule
    printf '[%d/%d] %s\n' "$(( ${#LABELS[@]} + 1 ))" "$TOTAL" "$label"
    printf '       $ %s\n' "$shown"
    thin

    if [[ ! -e "$script" ]]; then
        printf 'FAILURE: %s does not exist.\n' "$script"
        printf 'This counts as a failure, not a skip -- see the header of this script.\n'
        status=127
    elif [[ ! -f "$script" || ! -r "$script" || ! -x "$script" ]]; then
        printf 'FAILURE: %s is not a readable executable file.\n' "$script"
        printf 'Fix with: chmod +x %s\n' "$script"
        status=126
    elif (( ${#envs[@]} )); then
        env "${envs[@]}" "$script"
        status=$?
    else
        "$script"
        status=$?
    fi

    local secs=$(( SECONDS - start ))
    LABELS+=("$label")
    STATUSES+=("$status")
    ELAPSED+=("$secs")

    thin
    if (( status == 0 )); then
        printf 'PASS  %s  (%s)\n' "$label" "$(fmt_secs "$secs")"
    else
        printf 'FAIL  %s  (%s, exit %d)\n' "$label" "$(fmt_secs "$secs")" "$status"
    fi
}

suite_start=$SECONDS

rule
printf 'annealer artifact -- in-image smoke suite (%d items)\n' "$TOTAL"
if (( fast )); then
    printf 'mode: --fast (cut consistency over a 6-instance sample)\n'
else
    printf 'mode: full (cut consistency over all 71 Gset instances)\n'
fi
printf 'proves: cut-value correctness, chain isolation, determinism.\n'
printf 'does not prove: schedule invariance (no knobs to vary), or that this\n'
printf 'machine reaches the paper cut values -- that is a paper-budget sweep.\n'
printf 'See ./tests/run_all.sh --help.\n'
rule

if (( fast )); then
    run_test "cut consistency" ./tests/test_cut_consistency.sh GSET_SAMPLE=1
else
    run_test "cut consistency" ./tests/test_cut_consistency.sh
fi
run_test "chain isolation"      ./tests/test_chain_isolation.sh
run_test "committed cut values" ./tests/verify_committed_cuts.sh

total=$(( SECONDS - suite_start ))
failed=0
for s in "${STATUSES[@]}"; do
    (( s == 0 )) || failed=$(( failed + 1 ))
done

printf '\n'
rule
printf 'SUMMARY\n'
thin
for i in "${!LABELS[@]}"; do
    if (( STATUSES[i] == 0 )); then
        printf '  PASS  %-24s %8s\n' "${LABELS[$i]}" "$(fmt_secs "${ELAPSED[$i]}")"
    else
        printf '  FAIL  %-24s %8s   exit %d\n' "${LABELS[$i]}" "$(fmt_secs "${ELAPSED[$i]}")" "${STATUSES[$i]}"
    fi
done
thin
printf 'total wall time: %s\n' "$(fmt_secs "$total")"
if (( failed == 0 )); then
    printf 'RESULT: PASS -- %d of %d items passed.\n' "$TOTAL" "$TOTAL"
    printf 'Reminder: this does not reproduce the paper cut values.\n'
    printf 'For that, run ./scripts/bench_large_gset.sh --step 10000000 (~3.0 h)\n'
    printf 'and compare against README.md section 5.\n'
    rule
    exit 0
fi
printf 'RESULT: FAIL -- %d of %d items failed.  Details are in the sections above.\n' \
    "$failed" "$TOTAL"
rule
exit 1
