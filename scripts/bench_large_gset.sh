#!/usr/bin/env bash
set -uo pipefail

SELF="./scripts/$(basename "$0")"
# Quote the args back, but keep a no-arg run reading as a bare command line
# rather than "$SELF '' " -- printf applies its format once even with no args.
if [[ $# -gt 0 ]]; then
    INVOCATION="$SELF $(printf '%q ' "$@")"
    INVOCATION="${INVOCATION% }"
else
    INVOCATION="$SELF"
fi
cd "$(dirname "$0")/.."

# ======================= CONFIG -- DEFAULTS, EDIT HERE =======================
# Each value here can also be overridden for a single run by the flag named in
# its comment.  List flags accept commas, spaces, or both:  --t0s 35,50,70
# and --t0s "35 50 70" are the same thing.

# Temperature ladder.                                          [--t0s LIST]
BASE_T0S=(35.0 50.0 70.0 100.0 140.0 200.0 280.0 400.0)

# Replicas per temperature -> 8 * 10 = 80 chains.              [--repeats N]
REPEATS_PER_T0=10

# The S Value
STEP=10000                                                   # [--step N]

# P0 = 128 default value
PHASES=(128 64 32 16 8 4 2 1)                                # [--phases LIST]

# Always 0: each chain resolves its own seed from random_device.
SEED_VALUE=0

# Where results land.  One directory per invocation.           [--outdir DIR]
OUTDIR="results/full_gset_$(date +%Y%m%d_%H%M%S)"

# Directory holding the graph files, relative to the artifact root.
GSET_DIR="Gset"

# The instances to run, listed explicitly (no directory scan).  [--instances LIST]
# Comment lines out to change the default set.
INSTANCES=(
  G60 G61 G62 G63 G64 G65 G66 G67 G70 G72
  G77 G81
)

# ===================== END CONFIG =====================

BIN=./main
DRY_RUN=0

die() { echo "Error: $*" >&2; exit 1; }

# Split a comma- and/or whitespace-separated list into a named array:
#   split_into PHASES "128, 64 32"   ->   PHASES=(128 64 32)
split_into() {
    local __name=$1
    mapfile -t "$__name" < <(tr ',' '\n' <<< "$2" | tr -s ' \t' '\n' | grep -v '^$')
}

# Validators.  With flags in play a bad value can arrive from anywhere, so
# reject it here rather than 40 graphs into the sweep.
is_pos_int()  { [[ $1 =~ ^[0-9]+$ ]] && [[ $1 != 0 ]]; }
is_uint()     { [[ $1 =~ ^[0-9]+$ ]]; }
is_pos_num()  { [[ $1 =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v v="$1" 'BEGIN{exit !(v+0>0)}'; }

# Usage text prints the LIVE defaults straight out of the CONFIG block, so it
# can never drift from what the script actually does.
usage() {
    cat <<EOF
Usage: $SELF [options]

Runs one ./main launch per Gset instance, sequentially, with C concurrent
chains (C = number of temperatures x replicas), and records best-cut,
kernel time and wall time for each.

Options (all optional; anything omitted keeps its CONFIG-block default).
List values accept commas, spaces, or both.

  --t0s LIST        temperature ladder
                    default: ${BASE_T0S[*]}
  --repeats N       replicas per temperature
                    default: $REPEATS_PER_T0   (=> ${#BASE_T0S[@]} x $REPEATS_PER_T0 = $(( ${#BASE_T0S[@]} * REPEATS_PER_T0 )) chains)
  --step N          sweeps per phase, applied to every phase
                    default: $STEP   (=> $(( STEP * ${#PHASES[@]} )) sweeps/chain)
  --phases LIST     group size per phase
                    default: ${PHASES[*]}
  --instances LIST  graphs to run, from $GSET_DIR/
                    default: ${#INSTANCES[@]} instances (${INSTANCES[0]} .. ${INSTANCES[${#INSTANCES[@]}-1]})
  --outdir DIR      results directory
                    default: results/full_gset_<timestamp>
  --dry-run         print the resolved config and the commands, run nothing
  -h, --help        this message

Examples:
  $SELF
  $SELF --repeats 20 --outdir results/replicas20
  $SELF --instances G1,G22,G81 --step 100000
  $SELF --t0s "20 40 80 160" --repeats 40 --dry-run
EOF
}

# --- parse args: overrides the CONFIG defaults, nothing else ---------------
# Strict "--flag value" pairs, unknown flag is fatal -- same contract as
# ./main's own parser.

# need_val <flag> <remaining argc> <candidate value>.  A value that is itself
# a flag means the value was omitted -- catch it here rather than silently
# consuming the next option.
need_val() {
    [[ $2 -ge 2 ]]  || die "$1 requires a value"
    [[ $3 != --* ]] || die "$1 requires a value (found the flag '$3' instead)"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --t0s)       need_val "$1" $# "${2-}"; split_into BASE_T0S "$2";  shift 2 ;;
        --repeats)   need_val "$1" $# "${2-}"; REPEATS_PER_T0=$2;         shift 2 ;;
        --step)      need_val "$1" $# "${2-}"; STEP=$2;                   shift 2 ;;
        --phases)    need_val "$1" $# "${2-}"; split_into PHASES "$2";    shift 2 ;;
        --instances) need_val "$1" $# "${2-}"; split_into INSTANCES "$2"; shift 2 ;;
        --outdir)    need_val "$1" $# "${2-}"; OUTDIR=$2;                 shift 2 ;;
        --dry-run)   DRY_RUN=1;                                           shift   ;;
        -h|--help)   usage; exit 0 ;;
        *)           echo "Error: unknown option '$1'" >&2; echo >&2; usage >&2; exit 1 ;;
    esac
done

# --- validate the resolved values -----------------------------------------

[[ ${#BASE_T0S[@]} -gt 0 ]]  || die "--t0s / BASE_T0S is empty"
[[ ${#PHASES[@]} -gt 0 ]]    || die "--phases / PHASES is empty"
[[ ${#INSTANCES[@]} -gt 0 ]] || die "--instances / INSTANCES is empty"
is_pos_int "$REPEATS_PER_T0" || die "--repeats must be a positive integer, got '$REPEATS_PER_T0'"
is_pos_int "$STEP"           || die "--step must be a positive integer, got '$STEP'"
for t in "${BASE_T0S[@]}"; do
    is_pos_num "$t" || die "every temperature must be a number > 0, got '$t'"
done
for p in "${PHASES[@]}"; do
    is_pos_int "$p" || die "every phase must be a positive integer, got '$p'"
done

# --- derive the schedule strings ------------------------------------------

# join <sep> <elem...>
join() { local IFS="$1"; shift; echo "$*"; }

# RFC 4180 field: wrap in double quotes, doubling any embedded quote.  The list
# columns (t0s, seeds, phases, iters) contain commas, so they must be quoted.
# Today's values are numeric and the doubling never fires; it is there so the
# helper stays correct if a future column ever carries free text.
csv_quote() { local s=${1//\"/\"\"}; printf '"%s"' "$s"; }

T0_ARR=()
for ((r = 0; r < REPEATS_PER_T0; ++r)); do
    T0_ARR+=("${BASE_T0S[@]}")
done
C=${#T0_ARR[@]}

SEED_ARR=()
for ((c = 0; c < C; ++c)); do SEED_ARR+=("$SEED_VALUE"); done

ITER_ARR=()
for ((p = 0; p < ${#PHASES[@]}; ++p)); do ITER_ARR+=("$STEP"); done

T0S=$(join , "${T0_ARR[@]}")
SEEDS=$(join , "${SEED_ARR[@]}")
PHASES_CSV=$(join , "${PHASES[@]}")
ITERS=$(join , "${ITER_ARR[@]}")
SWEEPS_PER_CHAIN=$((STEP * ${#PHASES[@]}))

# CSV forms of the three parameters that are the same for every instance.  The
# fourth, seeds, is per-instance (each ./main run draws its own) and is quoted
# inside the loop.
Q_T0S=$(csv_quote "$T0S")
Q_PHASES=$(csv_quote "$PHASES_CSV")
Q_ITERS=$(csv_quote "$ITERS")

# --- preflight: fail before burning GPU time, not halfway through ----------

[[ -x $BIN ]] || die "$BIN not found or not executable -- run 'make' in $(pwd)"
[[ -d $GSET_DIR ]] || die "graph directory $GSET_DIR not found"

missing=()
for g in "${INSTANCES[@]}"; do
    [[ -f "$GSET_DIR/$g" ]] || missing+=("$g")
done
if [[ ${#missing[@]} -gt 0 ]]; then
    die "missing graph file(s) under $GSET_DIR: ${missing[*]}"
fi

CSV="$OUTDIR/results.csv"
REPORT="$OUTDIR/report.txt"
REPRO="$OUTDIR/reproduce.sh"

# --- banner: everything needed to reproduce or spot a mis-edit -------------

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
gpu_drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)

print_banner() {
    echo "# full-Gset benchmark"
    echo "# invocation: $INVOCATION"
    echo "# date:      $(date -Is)"
    echo "# gpu:       ${gpu_name:-unknown} (driver ${gpu_drv:-unknown})"
    echo "# outdir:    $OUTDIR"
    echo "#"
    echo "# chains:              $C  (${#BASE_T0S[@]} temperatures x $REPEATS_PER_T0 replicas)"
    echo "# sweeps per chain:    $SWEEPS_PER_CHAIN  ($STEP x ${#PHASES[@]} phases)"
    echo "# chain-sweeps/graph:  $((C * SWEEPS_PER_CHAIN))"
    echo "# instances:           ${#INSTANCES[@]}"
    echo "#"
    echo "# --seeds  $SEEDS"
    echo "# --t0s    $T0S"
    echo "# --phases $PHASES_CSV"
    echo "# --iters  $ITERS"
    echo "#"
    echo "# kernel_s is the binary's time-sec: the cudaEvent bracket around the single"
    echo "# kernel launch, i.e. first-kernel-start to last-kernel-end.  wall_s is the"
    echo "# whole process, including graph load, H2D and D2H."
    echo "#"
}

# main_cmd <instance> -- the literal command run for that graph
main_cmd() {
    printf '%s %s/%s --seeds %s --t0s %s --phases %s --iters %s\n' \
           "$BIN" "$GSET_DIR" "$1" "$SEEDS" "$T0S" "$PHASES_CSV" "$ITERS"
}

if [[ $DRY_RUN -eq 1 ]]; then
    print_banner
    echo "# DRY RUN -- nothing executed, no output directory created."
    echo
    for g in "${INSTANCES[@]}"; do main_cmd "$g"; done
    exit 0
fi

mkdir -p "$OUTDIR/logs" || die "cannot create $OUTDIR"
print_banner | tee "$REPORT"

printf 'instance,n,m,chains,best_cut,hits_at_best,best_chain,best_t0,kernel_s,wall_s,t0s,seeds,phases,iters\n' > "$CSV"

# reproduce.sh: only the header now.  One ./main line per instance is APPENDED
# as that instance finishes, carrying the seeds it actually resolved to -- which
# are not known until the run has started, so they cannot be written up front.
#
# The `cd` walks back to the artifact root (where ./main and Gset/ live) from
# wherever OUTDIR ended up, so a --outdir at any depth -- or an absolute one --
# still replays correctly, and the pair stays relocatable.
REPRO_CD=$(realpath --relative-to="$(cd "$OUTDIR" && pwd)" "$PWD")
{
    echo '#!/usr/bin/env bash'
    echo '# Regenerated by bench_large_gset.sh -- the exact commands that produced this'
    echo '# directory, with the seeds each chain actually resolved to.'
    echo 'set -euo pipefail'
    echo "cd \"\$(dirname \"\$0\")/$REPRO_CD\""
} > "$REPRO"
chmod +x "$REPRO"

# hits_at_best: how many of the C chains reached best_cut.  1 means one chain
# carried the instance (the answer is a fluke of one seed); C means every chain
# found it (the operating point is converged there).
printf '%-8s %7s %8s %10s %7s %9s %11s %10s  %s\n' \
       instance n m best-cut hits best-t0 kernel_s wall_s status | tee -a "$REPORT"

# --- the sweep -------------------------------------------------------------

sweep_start=$(date +%s.%N)
ok_count=0
fail_count=0

for g in "${INSTANCES[@]}"; do
    out="$OUTDIR/logs/$g.out"
    err="$OUTDIR/logs/$g.err"

    t_start=$(date +%s.%N)
    "$BIN" "$GSET_DIR/$g" --seeds "$SEEDS" --t0s "$T0S" \
           --phases "$PHASES_CSV" --iters "$ITERS" > "$out" 2> "$err"
    rc=$?
    t_end=$(date +%s.%N)
    wall_s=$(awk -v a="$t_start" -v b="$t_end" 'BEGIN{printf "%.4f", b-a}')

    # stderr: the graph geometry (cheap, small file)
    n=$(sed -n 's/^Loaded: n=\([0-9]*\) .*/\1/p' "$err" | head -1)
    m=$(sed -n 's/^Loaded: n=[0-9]* m=\([0-9]*\) .*/\1/p' "$err" | head -1)

    # The seeds ACTUALLY USED: with SEED_VALUE=0 each chain draws its own from
    # random_device, so this differs per instance and is the only form that
    # reproduces the run.  ./main echoes it before the kernel starts, so it
    # survives a later failure.
    seeds_used=$(sed -n 's/^Resolved seeds: \(.*\)$/\1/p' "$err" | head -1)

    # Append this instance's reproduce.sh line while the seeds are in hand.
    # A crashed instance may never have echoed them; fall back to the literal
    # --seeds that was passed and say so, rather than emit a wrong command.
    if [[ -z $seeds_used ]]; then
        echo "# WARNING: seeds unresolved for $g" >> "$REPRO"
    fi
    printf '%s %s/%s --seeds %s --t0s %s --phases %s --iters %s\n' \
           "$BIN" "$GSET_DIR" "$g" "${seeds_used:-$SEEDS}" \
           "$T0S" "$PHASES_CSV" "$ITERS" >> "$REPRO"

    # stdout: the summary is always the last line (the file is large -- up to
    # ~3 MB of partition bitstrings -- so read the tail, not the whole thing).
    summary=$(tail -n 1 "$out")
    case "$summary" in
        summary:*) ;;
        *) summary=$(grep -a '^summary:' "$out" | tail -1) ;;
    esac

    status=OK
    best_cut=""; best_chain=""; kernel_s=""; best_t0=""; hits_at_best=""
    if [[ $rc -ne 0 ]]; then
        status=FAIL
    elif [[ -z $summary ]]; then
        status=NOSUMMARY
    else
        best_cut=$(sed -n 's/.*best-cut-value: \(-*[0-9]*\).*/\1/p' <<<"$summary")
        best_chain=$(sed -n 's/.*best-chain: \([0-9]*\).*/\1/p' <<<"$summary")
        kernel_s=$(sed -n 's/.*time-sec: \([0-9.]*\).*/\1/p' <<<"$summary")
        hits_at_best=$(sed -n 's/.*chains-at-best: \([0-9]*\).*/\1/p' <<<"$summary")
        if [[ -n $best_chain && $best_chain -lt $C ]]; then
            best_t0=${T0_ARR[$best_chain]}
        fi
        [[ -n $best_cut && -n $kernel_s && -n $hits_at_best ]] || status=NOSUMMARY
    fi

    if [[ $status == OK ]]; then
        ok_count=$((ok_count + 1))
    else
        fail_count=$((fail_count + 1))
        echo "  !! $g: $status (rc=$rc) -- $(tail -n 1 "$err")" | tee -a "$REPORT"
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$g" "${n:-}" "${m:-}" "$C" "${best_cut:-}" "${hits_at_best:-}" \
        "${best_chain:-}" "${best_t0:-}" \
        "${kernel_s:-}" "$wall_s" \
        "$Q_T0S" "$(csv_quote "${seeds_used:-}")" "$Q_PHASES" "$Q_ITERS" >> "$CSV"

    printf '%-8s %7s %8s %10s %7s %9s %11s %10s  %s\n' \
        "$g" "${n:-?}" "${m:-?}" "${best_cut:-?}" "${hits_at_best:-?}/$C" "${best_t0:-?}" \
        "${kernel_s:-?}" "$wall_s" "$status" | tee -a "$REPORT"
done

sweep_end=$(date +%s.%N)
total=$(awk -v a="$sweep_start" -v b="$sweep_end" 'BEGIN{printf "%.1f", b-a}')

{
    echo
    echo "# instances: ${#INSTANCES[@]}   ok: $ok_count   failed: $fail_count"
    echo "# total wall: ${total}s"
    echo "# csv:   $CSV"
    echo "# repro: $REPRO  (re-runs every instance with the seeds it drew)"
    echo "# logs:  $OUTDIR/logs/  (full stdout incl. partitions, and stderr)"
} | tee -a "$REPORT"
