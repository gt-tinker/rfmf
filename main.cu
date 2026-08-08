// ---------------------------------------------------------------------------
// Multi-chain CUDA Gumbel-competition annealer for Max-Cut.
//
// Runs C INDEPENDENT chains in ONE kernel launch -- one persistent block per
// chain, one (t0, seed) pair per chain from the zipped --t0s/--seeds lists.
// This binary optimizes THROUGHPUT (chain-sweeps/sec) rather than the latency
// of any single chain.
//
// Tuned for, and built only for, the Tesla V100 (sm_70); see src/tuning.hpp.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <iostream>
#include <vector>

#include "src/anneal_gpu.cuh"
#include "src/host_common.h"
#include "src/tuning.hpp"

int main(int argc, char* argv[]) {
    SolverContext ctx;
    if (setupSolver(argc, argv, ctx) != 0)
        return 1;

    const size_t C = numChains(ctx.args);

    std::vector<uint64_t> solverSeeds(C);
    for (size_t c = 0; c < C; ++c)
        solverSeeds[c] = ctx.args.seeds[c] + kSolverSeedOffset;

    // B is a graph property, shared by every chain.  Scanned ONCE here: the
    // banner below, cfg.sWidth and the solver's int32 score-path gate all key
    // on this same value (annealGpu takes it as a parameter).
    const int64_t B = maxWeightedDegree(ctx.csr);

    gpu3::LaunchConfig cfg;
    cfg.device = ctx.args.device;

    // corr: warp-per-winner needs rows wide enough to feed 32 lanes, so the
    // split is by MEAN DEGREE.
    const double meanDeg = ctx.csr.numVertices() > 0
        ? 2.0 * static_cast<double>(ctx.csr.numEdges()) / static_cast<double>(ctx.csr.numVertices())
        : 0.0;
    cfg.corr = (meanDeg >= gpu3::tuning::corrWarpwalkMinDeg) ? gpu3::CorrMode::Warpwalk
                                                             : gpu3::CorrMode::Walk;

    // tpb resolves in runDispatch, where the live SM count is known.

    // Width of the S cache.
    // if the bound fits in a 32-bit int, store S as int32_t; otherwise fall back to int64_t.
    cfg.sWidth = (B <= 2147483647LL) ? 32 : 64;

    const int64_t totalSteps = annealTotalSteps(ctx.args.iters);
    // The score-path gate is evaluated at min(t0s): the smallest t0 has the
    // largest betaMax, and ONE verdict picks the instantiation for the whole
    // grid (host_common.h).
    const double minT0 = *std::min_element(ctx.args.t0s.begin(), ctx.args.t0s.end());
    size_t distinctT0s = 0;
    {
        std::vector<double> seen;
        for (double t : ctx.args.t0s)
            if (std::find(seen.begin(), seen.end(), t) == seen.end()) seen.push_back(t);
        distinctT0s = seen.size();
    }

    std::cerr << "Chains: " << C
              << " distinct-t0: " << distinctT0s
              << " corr=" << (cfg.corr == gpu3::CorrMode::Warpwalk ? "warpwalk" : "walk")
              << " s-width=" << cfg.sWidth
              << " B=" << B
              << " total-sweeps-per-chain=" << totalSteps
              << " score-path=" << (annealScoreSafeInt32(totalSteps, minT0, B)
                                        ? "int32-packed" : "int64-twopass")
              << "\n";

    std::vector<std::vector<int8_t>> endSpins, bestSpins;
    float kernelMs = 0.0f;
    std::vector<int64_t> kernelBestVals, kernelFinalVals;

    // Timing is the kernel only (cudaEvents) -- ONE bracket around the ONE
    // launch that runs all C chains concurrently. No parsing, graph I/O,
    // setup, or transfer. Every chain's stdout line carries this same number.
    gpu3::annealGpu(ctx.csr, B, ctx.initialSpins, ctx.args.t0s, ctx.args.phases,
                    ctx.args.iters, solverSeeds, cfg, endSpins, bestSpins,
                    &kernelMs, &kernelBestVals, &kernelFinalVals);

    // Each chain's incrementally-tracked cut values, on stderr so stdout stays
    // one clean block per chain.  printChainResults independently RECOMPUTES
    // both from the returned spins, so the two routes can be cross-checked --
    // that is what would catch a corr[] / double-counting bug, or a
    // cross-chain slice overlap.
    for (size_t c = 0; c < C; ++c)
        std::cerr << "chain " << c << ": kernel-best-val: " << kernelBestVals[c]
                  << " kernel-final-val: " << kernelFinalVals[c] << "\n";

    const double sec = static_cast<double>(kernelMs) / 1000.0;
    std::vector<int64_t> bestVals(C);
    for (size_t c = 0; c < C; ++c)
        bestVals[c] = printChainResults(ctx, c, bestSpins[c], endSpins[c], sec);
    printSummary(bestVals, sec);

    return 0;
}
