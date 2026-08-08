#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# verify_committed_cuts.sh -- every cut value this artifact reports is one that
# a partition it PRINTED actually achieves.
#
# This is the check a skeptical reviewer should run first, because it is the
# only one in the package that does not ask them to trust the program under
# test.  Each committed log makes two separate claims per chain: "this chain
# reached a cut of 14187", and, on the very next line, "here is the partition
# that did it".  Nothing inside the solver forces those two claims to agree
# with each other.  The value is tracked incrementally on the device and
# reduced there; the bit string comes back through a different buffer, through
# a different host path, and is printed by different code.  The paper's table
# quotes the value and never shows the partition at all.  So the failure this
# script exists to catch is neither a crash nor an obviously wrong answer: it
# is a log that quotes a cut value no printed partition supports.  That log
# would sail through every other gate here -- it is self-consistent under
# test_cut_consistency, chain-isolated and deterministic under
# test_chain_isolation, and replays byte-for-byte under reproduce.sh, because
# all three of those compare the program against itself.
#
# The way out of that circle is to recompute the number from first principles.
# For each claimed cut we take the bit string printed beside it, hand it to
# scripts/eval_cut.py, and demand the exact integer back.  eval_cut.py shares
# nothing with the solver: no CUDA, no C++, no CSR, not even a helper module.
# It re-reads the original Gset edge list and adds up the weight of every edge
# whose endpoints landed on opposite sides.  If the kernel, the incremental
# value tracking, the device reduction and the host-side printer were all
# wrong in the same direction, this is the check that would still notice --
# and it is the reason eval_cut.py is invoked as a separate process, once per
# partition, rather than having its arithmetic inlined here where it would
# quietly become part of the thing being tested.
#
# The default mode audits exactly what the paper prints.  For each instance it
# takes the summary line's best-cut-value, finds the chain that summary line
# names, and scores THAT chain's Solution-best-partition.  It also compares the
# value against results.csv's best_cut column, because results.csv is what the
# table is generated from while the log is what results.csv is supposed to be
# summarizing.  If those two drift apart the table is reporting a number no log
# supports, which is a different bug from a bad cut value and a worse one, so
# it is called out separately and just as loudly.
#
# --chains all widens the audit from the 12 headline numbers to every number in
# the directory: for all 80 chains, both the best-cut-value/Solution-best-
# partition pair and the final-cut-value/Solution-final-partition pair, on all
# 12 instances.  1920 independent rescorings.  The final-value pairs are worth
# the extra time precisely because nothing in the paper depends on them: a
# best-tracking bug that copied the wrong spins into the best buffer leaves the
# final pair intact and the best pair broken, and seeing exactly one of the two
# families fail is what tells you which half of the code to go read.
#
# COST, measured on results/V100-SXM2-10M on a PACE V100 node:
#   --chains best   12 evaluations, 1.8 s total.
#   --chains all    1920 evaluations, 2 m 59 s total; 11-21 s per instance,
#                   scaling with edge count (G70 11 s, G63 and G81 21 s).
# That is about 0.09 s per evaluation, nearly all of it Python interpreter
# startup plus re-parsing the Gset edge list.  eval_cut.py scores one partition
# per process and takes no batch input, and it is not this script's place to
# change that: the separate process IS the independence being claimed, and
# folding the arithmetic in here to save two and a half minutes would make the
# check circular.  Three minutes to re-derive every number in a results
# directory from scratch is a bargain next to a 3-hour paper-budget sweep, and
# unlike such a sweep it needs no GPU and no build.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."

SELF="./tests/$(basename "$0")"

# --- defaults ---------------------------------------------------------------

# The reference the paper table was built from.                    [--ref DIR]
REF="results/V100-SXM2-10M"

# "best" = the 12 headline numbers; "all" = every chain.      [--chains best|all]
CHAINS="best"

# Unset means "every instance in <ref>/results.csv".         [--instances LIST]
# Tracked with a separate "was it given" flag rather than by testing the string
# for emptiness: "--instances ''" is a request that selects nothing, and it has
# to be distinguishable from not passing the flag at all so it can be refused
# instead of quietly turning into a full sweep.
INSTANCES_ARG=""
INSTANCES_GIVEN=0

# The independent scorer.  Deliberately the only arithmetic in this test.
EVAL="scripts/eval_cut.py"

die() { echo "Error: $*" >&2; exit 1; }

# A value that is itself a flag means the value was omitted; catch it here
# rather than silently consuming the next option.  Same contract as
# scripts/bench_large_gset.sh.
need_val() {
    [[ $2 -ge 2 ]]  || die "$1 requires a value"
    [[ $3 != --* ]] || die "$1 requires a value (found the flag '$3' instead)"
}

# --instances G60,G61 and --instances "G60 G61" are the same thing.
split_into() {
    local __name=$1
    mapfile -t "$__name" < <(tr ',' '\n' <<< "$2" | tr -s ' \t' '\n' | grep -v '^$')
}

usage() {
    cat <<EOF
Usage: $SELF [options]

Proves that every cut value committed under a results directory is actually
achieved by a partition the program printed, by recomputing each one from the
Gset edge list with scripts/eval_cut.py -- code that has nothing in common with
the solver.  Prints PASS or FAIL per instance and exits non-zero if any claimed
value fails to reproduce.

Needs no GPU, no build and no network: it reads committed logs and runs Python.

Options:
  --ref DIR         reference results directory to audit
                    default: $REF
  --instances LIST  comma- or space-separated instances to check
                    default: every instance listed in <ref>/results.csv
  --chains best|all which claims to rescore
                      best  (default) the headline number only: each instance's
                            summary best-cut-value, scored against the
                            Solution-best-partition of the chain that summary
                            line names, and cross-checked against results.csv's
                            best_cut column.  12 evaluations, 1.8 s measured.
                      all   additionally every chain's own best-cut-value and
                            final-cut-value against its own two partitions.
                            80 x 2 x 12 = 1920 evaluations, 2 m 59 s measured
                            (11-21 s per instance, scaling with edge count).
  -h, --help        this message

Cost note: eval_cut.py scores one partition per process and re-reads the edge
list every time, so an evaluation costs about 0.09 s, almost entirely startup.
That is not worked around on purpose -- calling the independent scorer as a
separate program is the whole point of this test, and reimplementing its
arithmetic inline to save two and a half minutes would make the check circular.

Examples:
  # the 12 headline cuts, the numbers the paper prints -- 1.8 s
  $SELF

  # the same audit against a different card's committed run
  $SELF --ref results/V100-PCIE-10M

  # every chain of one instance: 160 rescorings, 13 s
  $SELF --chains all --instances G60

  # every chain of all 12 instances: 1920 rescorings, 3 min
  $SELF --chains all
EOF
}

# --- parse args -------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref)       need_val "$1" $# "${2-}"; REF=$2;           shift 2 ;;
        --instances) need_val "$1" $# "${2-}"; INSTANCES_ARG=$2; INSTANCES_GIVEN=1; shift 2 ;;
        --chains)    need_val "$1" $# "${2-}"; CHAINS=$2;        shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        *)           echo "Error: unknown option '$1'" >&2; echo >&2; usage >&2; exit 1 ;;
    esac
done

case "$CHAINS" in
    best|all) ;;
    *) die "--chains must be 'best' or 'all', got '$CHAINS'" ;;
esac

REF=${REF%/}
CSV="$REF/results.csv"

[[ -d $REF ]]      || die "reference directory '$REF' not found (run from the artifact root, or pass --ref)"
[[ -d $REF/logs ]] || die "'$REF/logs' not found -- there are no committed logs to score"
[[ -f $CSV ]]      || die "'$CSV' not found -- without it there is no table column to cross-check the logs against"
[[ -f $EVAL ]]     || die "'$EVAL' not found -- this test has no arithmetic of its own and cannot run without it"
command -v python3 >/dev/null 2>&1 || die "python3 not found, and $EVAL needs it"

# --- locate the results.csv columns by NAME ---------------------------------
# By name rather than by position, so a column inserted upstream turns into a
# clean failure here instead of a silent comparison against the wrong field.
# Splitting the header on commas is safe: the header line has no quoted fields.
# Splitting DATA rows on commas is only safe up to field 10 -- t0s and seeds
# are comma-bearing quoted strings and start at field 11 -- so the resolved
# indices are checked against that boundary before any row is read.

read -r csv_hdr < "$CSV"
col_of() {
    awk -F, -v want="$1" 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == want) { print i; exit } }' <<<"$csv_hdr"
}
COL_INST=$(col_of instance)
COL_BEST=$(col_of best_cut)
COL_CHAIN=$(col_of best_chain)
[[ -n $COL_INST && -n $COL_BEST && -n $COL_CHAIN ]] \
    || die "$CSV is missing one of the instance/best_cut/best_chain columns (header: $csv_hdr)"
for c in "$COL_INST" "$COL_BEST" "$COL_CHAIN"; do
    (( c <= 10 )) || die "$CSV column $c sits past the quoted t0s/seeds fields; this parser cannot read it safely"
done

# --- resolve the instance selection -----------------------------------------

if (( INSTANCES_GIVEN )); then
    split_into SELECTED "$INSTANCES_ARG"
    SOURCE="--instances '$INSTANCES_ARG'"
else
    mapfile -t SELECTED < <(awk -F, -v c="$COL_INST" 'NR > 1 && $c != "" { print $c }' "$CSV")
    SOURCE="$CSV"
fi

# An empty sweep must FAIL, not vacuously pass: a truncated results.csv, a
# --ref pointing at a directory whose table has no rows, or an --instances list
# that selects nothing would otherwise report "0 instances, 0 failures" and
# exit 0 -- a green light for an audit that never audited anything.  Same
# defensive stance as tests/test_cut_consistency.sh.
if [[ ${#SELECTED[@]} -eq 0 ]]; then
    echo "FAIL: no instances to check -- $SOURCE selected none"
    exit 1
fi

# Instance names are interpolated into file paths and passed to eval_cut.py, so
# they are validated as data rather than trusted.
for g in "${SELECTED[@]}"; do
    [[ $g =~ ^G[0-9]+$ ]] || die "not a Gset instance name: '$g'"
    [[ -f $REF/logs/$g.out ]] \
        || die "'$REF/logs/$g.out' not found -- cannot score cuts for an instance with no log"
    [[ -f Gset/$g ]] \
        || die "'Gset/$g' not found -- eval_cut.py has no edge list to score against"
done

# --- the independent scorer -------------------------------------------------
# One process per partition, fed on stdin -- exactly the pipeline eval_cut.py's
# own usage message documents.  The bit string is never held in a shell
# variable and never goes through argv: partitions run to 20000 characters, and
# bash reads a process substitution one byte at a time when the producer is a
# pipe, so slurping this directory's 1920 partitions into arrays costs about
# 8 seconds of pure copying before a single cut is recomputed.  Addressing the
# line and letting sed hand it straight to python skips all of that, and it
# keeps this script's contact with the evidence down to "line N of the log,
# verbatim".
#
# stderr is folded into the captured output so a rejection inside eval_cut.py
# (a bit string of the wrong length, an unreadable instance) is reported rather
# than vanishing behind an empty result.  Those rejections are plain non-zero
# exits, not asserts, so they survive a python3 invoked with -O.
#
# The addressed line must carry the family we asked for, "best" or "final" --
# the prefix is named in the sed rather than wildcarded so that a line number
# landing on the wrong kind of line produces empty input and a loud non-zero
# exit, instead of a perfectly valid rescoring of the wrong partition.
score() {
    local inst=$1 log=$2 line=$3 kind=$4
    sed -n "${line}s/^Solution-${kind}-partition: //p" "$log" | python3 "$EVAL" "$inst" - 2>&1
}

fail=0
nevals=0

# What eval_cut.py returned on the last expect() call.  A global because
# expect() has already spent its exit status on pass/fail and its stdout on the
# FAIL lines, and the PASS line has to be able to print the number the scorer
# actually computed rather than re-printing the claim it was checking.  The two
# are equal exactly when the test passes, which is precisely why printing the
# claim there would be indistinguishable from a scorer that returned nothing.
EXPECT_GOT=""

# expect <instance> <chain> <kind> <claimed> <logfile> <lineno>
# Rescores the partition on one line and compares.  Prints a FAIL line naming
# the instance, the chain and the exact claim that did not reproduce; returns
# non-zero on mismatch so callers can keep their own counts.  Leaves the
# recomputed value in EXPECT_GOT.
expect() {
    local inst=$1 chain=$2 kind=$3 claimed=$4 log=$5 line=$6 got st
    got=$(score "$inst" "$log" "$line" "$kind"); st=$?
    EXPECT_GOT=$got
    nevals=$((nevals + 1))
    if (( st != 0 )); then
        echo "  FAIL $inst chain $chain $kind: $EVAL exited $st: $got"
        return 1
    fi
    if [[ $got != "$claimed" ]]; then
        echo "  FAIL $inst chain $chain $kind: log claims cut $claimed," \
             "$EVAL scores its Solution-$kind-partition at $got"
        return 1
    fi
    return 0
}

# --- per-instance verification ----------------------------------------------

echo "verify-committed-cuts: ref=$REF  mode=--chains $CHAINS  instances: ${#SELECTED[@]}"

for inst in "${SELECTED[@]}"; do
    log="$REF/logs/$inst.out"
    ibad=0
    start=$SECONDS

    # The summary line: the artifact's own statement of the headline number and
    # of which chain produced it.
    schains=""; sbestchain=""; sbestval=""
    read -r schains sbestchain sbestval < <(sed -n \
        's/^summary: chains: \([0-9]*\) best-chain: \([0-9]*\) best-cut-value: \([-0-9]*\).*/\1 \2 \3/p' "$log")
    if [[ -z $sbestval ]]; then
        echo "  FAIL $inst: no parsable 'summary:' line in $log"
        fail=$((fail + 1))
        continue
    fi

    # Per-chain claims, and the LINE NUMBERS of the partitions that are supposed
    # to support them.  Line numbers rather than the partitions themselves: see
    # score() above for why the bit strings never enter the shell.  Deriving
    # them by grep instead of from the documented 3-lines-per-chain layout means
    # a log whose block shape ever changed still gets scored against the right
    # lines instead of being silently misaligned.
    mapfile -t dbest  < <(sed -n 's/^device: .*best-cut-value: \([-0-9]*\).*/\1/p' "$log")
    mapfile -t dfinal < <(sed -n 's/^device: .*final-cut-value: \([-0-9]*\).*/\1/p' "$log")
    mapfile -t lbest  < <(grep -n '^Solution-best-partition: '  "$log" | cut -d: -f1)
    mapfile -t lfinal < <(grep -n '^Solution-final-partition: ' "$log" | cut -d: -f1)

    # Shape first.  A log with fewer partitions than claims would otherwise let
    # a missing chain slip past as "nothing to compare".
    if [[ ${#dbest[@]} -ne $schains || ${#dfinal[@]} -ne $schains \
          || ${#lbest[@]} -ne $schains || ${#lfinal[@]} -ne $schains ]]; then
        echo "  FAIL $inst: summary says $schains chains, but the log has ${#dbest[@]} best /" \
             "${#dfinal[@]} final cut values and ${#lbest[@]} best / ${#lfinal[@]} final partitions"
        fail=$((fail + 1))
        continue
    fi
    if (( sbestchain < 0 || sbestchain >= schains )); then
        echo "  FAIL $inst: summary names best-chain $sbestchain, out of range for $schains chains"
        fail=$((fail + 1))
        continue
    fi

    # results.csv versus the log.  Pure integer comparison, no evaluation: the
    # CSV carries no partition, so all it can be held to is agreeing with the
    # log that does.  A disagreement here means the generated table and the
    # committed evidence describe different runs, which is reported separately
    # from an arithmetic failure because it is a different bug.
    #
    # EVERY matching row is collected, not just the first.  Stopping at the
    # first match would make a results.csv carrying a second, contradictory row
    # for the same instance pass silently, and it would do so in the
    # unfavourable direction: the table generator and any human reader would be
    # free to pick up the OTHER row, so the number that gets published is
    # exactly the one this test never looked at.  A duplicated instance has no
    # single table value to hold the log to, so it is a failure in itself.
    mapfile -t csv_rows < <(awk -F, -v c="$COL_INST" -v g="$inst" 'NR > 1 && $c == g { print }' "$CSV")
    if [[ ${#csv_rows[@]} -eq 0 ]]; then
        echo "  FAIL $inst: no row for this instance in $CSV, so the table's number cannot be checked"
        ibad=$((ibad + 1))
    elif [[ ${#csv_rows[@]} -gt 1 ]]; then
        dup_bests=$(printf '%s\n' "${csv_rows[@]}" \
                    | awk -F, -v c="$COL_BEST" '{ printf "%s%s", sep, $c; sep = ", " }')
        echo "  FAIL $inst: $CSV carries ${#csv_rows[@]} rows for this instance," \
             "with best_cut $dup_bests -- there is no single table number to check $log against"
        ibad=$((ibad + 1))
    else
        csv_row=${csv_rows[0]}
        csv_best=$(awk -F, -v c="$COL_BEST"  '{ print $c }' <<<"$csv_row")
        csv_chain=$(awk -F, -v c="$COL_CHAIN" '{ print $c }' <<<"$csv_row")
        if [[ $csv_best != "$sbestval" ]]; then
            echo "  FAIL $inst: $CSV says best_cut=$csv_best but $log's summary says $sbestval" \
                 "-- the paper table and the committed log disagree"
            ibad=$((ibad + 1))
        fi
        if [[ $csv_chain != "$sbestchain" ]]; then
            echo "  FAIL $inst: $CSV says best_chain=$csv_chain but $log's summary says $sbestchain" \
                 "-- the table credits a different chain than the log does"
            ibad=$((ibad + 1))
        fi
    fi

    # The summary's headline value must be the value the named chain itself
    # reported.  One integer comparison, and it is what makes scoring that one
    # chain's partition sufficient: without it the summary could quote a number
    # no chain ever claimed and still pass every evaluation below.
    if [[ ${dbest[$sbestchain]} != "$sbestval" ]]; then
        echo "  FAIL $inst: summary claims best-cut-value $sbestval from chain $sbestchain," \
             "but that chain's own device line says ${dbest[$sbestchain]}"
        ibad=$((ibad + 1))
    fi

    if [[ $CHAINS == best ]]; then
        expect "$inst" "$sbestchain" best "$sbestval" "$log" "${lbest[$sbestchain]}" || ibad=$((ibad + 1))
        if (( ibad == 0 )); then
            echo "  PASS $inst  best-chain $sbestchain  claimed $sbestval  recomputed $EXPECT_GOT"
        fi
    else
        for ((i = 0; i < schains; i++)); do
            expect "$inst" "$i" best  "${dbest[$i]}"  "$log" "${lbest[$i]}"  || ibad=$((ibad + 1))
            expect "$inst" "$i" final "${dfinal[$i]}" "$log" "${lfinal[$i]}" || ibad=$((ibad + 1))
        done
        if (( ibad == 0 )); then
            echo "  PASS $inst  $schains chains x (best, final) = $((schains * 2)) partitions rescored," \
                 "headline $sbestval from chain $sbestchain  ($((SECONDS - start))s)"
        fi
    fi

    if (( ibad != 0 )); then
        echo "  FAIL $inst  ($ibad problem(s))"
        fail=$((fail + ibad))
    fi
done

echo "verify-committed-cuts: ${#SELECTED[@]} instances, $nevals partitions rescored by $EVAL, $fail failures"
if (( fail != 0 )); then
    echo "verify-committed-cuts: FAIL -- a committed cut value is not achieved by the partition printed with it,"
    echo "                       or the table and the logs disagree.  Details above."
fi
[[ $fail -eq 0 ]]
