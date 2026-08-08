#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# test_cut_consistency.sh -- two independent routes to the same cut value must
# agree, PER CHAIN, on every Gset instance.
#
#   route A: the kernel's INCREMENTAL bestVal/currentVal -- seeded from each
#            chain's initial cut, then advanced per flip by mDelta + mSpin*corr.
#            Printed to stderr as "chain i: kernel-best-val: ... kernel-final-val: ...".
#   route B: printChainResults' full RECOMPUTE, cutValueCsr(bestSpin) over the
#            CSR from scratch, sharing no code with the kernel.
#
# This is the check that catches the failure mode corr[] exists to prevent: if
# the winner-winner coupling were wrong then S, the spins, and the printed
# partition all stay plausible while the incremental value silently drifts.
# In GPU3 it earns a second job: a chain whose slice arithmetic overlapped a
# neighbour's would score the wrong spins, and this pair is where that shows
# first.  Runs THREE chains per instance -- two at one t0 (shared beta table)
# plus one at another -- so both the shared- and distinct-table paths are on
# every instance.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."

GSET=Gset
PHASES="64,32,16,8,4,2,1"
ITERS="120,120,120,120,120,120,120"
SEEDS="42,43,42"
T0S="105,105,90"
C=3

if [[ -n "${GSET_SAMPLE:-}" ]]; then
    INSTANCES="G1 G11 G22 G63 G70 G81"
    echo "cut-consistency: 6-instance sample (GSET_SAMPLE set), $C chains each"
else
    # Every G* instance present. Gset is NOT a contiguous G1..G81: it is
    # G1..G67 plus G70, G72, G77, G81 -- 71 files.
    INSTANCES=$(ls "$GSET" | grep -E '^G[0-9]+$' | sort -V)
    # An empty sweep must FAIL, not vacuously pass: a missing/empty ../Gset
    # would otherwise report "0 instances, 0 failures" and exit 0.
    if [[ -z "$INSTANCES" ]]; then
        echo "FAIL: no G* instances found in $GSET"
        exit 1
    fi
    echo "cut-consistency: all $(echo "$INSTANCES" | wc -w) Gset instances, $C chains each"
fi

fail=0
n=0
for inst in $INSTANCES; do
    out=$(./main "$GSET/$inst" --seeds "$SEEDS" --t0s "$T0S" \
              --phases "$PHASES" --iters "$ITERS" 2>&1)
    # Route A: per-chain incremental values (stderr, in chain order).
    mapfile -t kbest  < <(sed -n 's/^chain [0-9]*: kernel-best-val: \([-0-9]*\).*/\1/p' <<<"$out")
    mapfile -t kfinal < <(sed -n 's/.*kernel-final-val: \([-0-9]*\)$/\1/p' <<<"$out")
    # Route B: per-chain recomputed values (stdout blocks, in chain order).
    mapfile -t rbest  < <(sed -n 's/^device: .*best-cut-value: \([-0-9]*\).*/\1/p' <<<"$out")
    mapfile -t rfinal < <(sed -n 's/^device: .*final-cut-value: \([-0-9]*\).*/\1/p' <<<"$out")

    n=$((n + 1))
    if [[ ${#kbest[@]} -ne $C || ${#rbest[@]} -ne $C ]]; then
        echo "  FAIL $inst: expected $C chains, parsed ${#kbest[@]} incremental / ${#rbest[@]} recomputed"
        fail=$((fail + 1))
        continue
    fi
    for ((i = 0; i < C; i++)); do
        if [[ "${kbest[$i]}" != "${rbest[$i]}" ]]; then
            echo "  FAIL $inst chain $i: kernel bestVal=${kbest[$i]} but recomputed cut=${rbest[$i]}"
            fail=$((fail + 1))
        fi
        if [[ "${kfinal[$i]}" != "${rfinal[$i]}" ]]; then
            echo "  FAIL $inst chain $i: kernel currentVal=${kfinal[$i]} but recomputed cut=${rfinal[$i]}"
            fail=$((fail + 1))
        fi
    done

    # The summary line's semantics -- no other gate looks at it (every other
    # parser filters on ^device:/^Solution-), so a printSummary regression
    # would otherwise pass the whole suite. Asserts: chains == C, best-cut ==
    # max over the recomputed per-chain bests, best-chain == the SMALLEST
    # index achieving it (the documented tie-break), and chains-at-best ==
    # the SIZE of that tie group.
    read -r schains sbestchain sbestval satbest < <(sed -n \
        's/^summary: chains: \([0-9]*\) best-chain: \([0-9]*\) best-cut-value: \([-0-9]*\).*chains-at-best: \([0-9]*\).*/\1 \2 \3 \4/p' <<<"$out")
    maxv=${rbest[0]}
    maxi=0
    for ((i = 1; i < C; i++)); do
        if (( rbest[i] > maxv )); then maxv=${rbest[$i]}; maxi=$i; fi
    done
    nat=0
    for ((i = 0; i < C; i++)); do
        if (( rbest[i] == maxv )); then nat=$((nat + 1)); fi
    done
    if [[ "$schains" != "$C" || "$sbestval" != "$maxv" || "$sbestchain" != "$maxi" \
          || "$satbest" != "$nat" ]]; then
        echo "  FAIL $inst summary: got chains=$schains best-chain=$sbestchain best-cut=$sbestval" \
             "chains-at-best=$satbest, expected chains=$C best-chain=$maxi best-cut=$maxv" \
             "chains-at-best=$nat"
        fail=$((fail + 1))
    fi
done

echo "cut-consistency: $n instances x $C chains, $fail failures"
[[ $fail -eq 0 ]]
