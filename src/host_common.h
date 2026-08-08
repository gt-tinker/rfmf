#ifndef HOST_COMMON_H
#define HOST_COMMON_H

// Host-side driver surface.  The solver runs C independent annealing chains
// in one kernel launch, one per (t0, seed) pair from the zipped --t0s/--seeds
// lists.  Correctness contract: chain i's stdout block is bit-identical to a
// single-chain CPU run with --seed seeds[i] --t0 t0s[i] (the counter-based RNG
// in philox.cuh makes each chain's stream independent of geometry and of
// every other chain).

#include <string>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cmath>

#include "graph.hpp"
#include "immutable_graph.hpp"
#include "max_cut.hpp"
#include "read_gset.hpp"

// Offset separating a chain's solver Rng seed from its initial-spin Rng seed,
// verbatim from the CPU reference.  CAREFUL: it applies ONLY to the solver
// stream -- the initial spins are drawn from the RAW resolved seed.
constexpr uint64_t kSolverSeedOffset = 0xC2B2AE3D27D4EB4FULL;

struct AnnealArgs {
    std::string filename;
    std::vector<uint64_t> seeds;  // --seeds, zipped with t0s (one chain per
                                  // pair); 0 entries resolved from
                                  // random_device in setupSolver
    std::vector<double> t0s;      // --t0s, initial temperature per chain
    std::vector<size_t> phases;   // shared by every chain
    std::vector<uint64_t> iters;  // per-phase sweep counts, shared by every chain

    int device = 0;               // --device N

    // No tuning knobs on the command line: every performance choice is
    // resolved from tuning.hpp and is output-invariant by construction.
};

// All pre-computed data the solver needs.  The mutable Graph is a
// setupSolver-local transient (File => Graph => CSR), freed before the kernel.
struct SolverContext {
    AnnealArgs args;
    anneallib::AnnealCsr csr;         // frozen snapshot every chain runs on
    std::vector<std::vector<int8_t>> initialSpins;  // per chain: dense +/-1
    anneallib::ReadStats readStats;
};

// Number of chains (the zipped --t0s/--seeds length).
inline size_t numChains(const AnnealArgs& a) { return a.t0s.size(); }

// Stream chain `chain`'s argument echo, matching the CPU line format exactly:
//   instance: <base> --seed S --phases p,p --t0 T --iters i,i
std::ostream& streamChainArgs(std::ostream& os, const AnnealArgs& a, size_t chain);

void printUsage(const char* prog);

AnnealArgs parseArgs(int argc, char* argv[]);

// Parse args, load graph, remove degree-zero vertices, freeze CSR, resolve
// per-chain seeds (0 => random_device), generate each chain's initial spins
// from its RAW seed.  Returns 0 on success (errors already on stderr).
int setupSolver(int argc, char* argv[], SolverContext& ctx);

// Total planned sweeps over all phases (per chain) = sum(iters).
int64_t annealTotalSteps(const std::vector<uint64_t>& iters);

// beta_fixed (Q20) for a 1-based sweep id, verbatim from the serial kernel.
// Always computed ON THE HOST: device-side log() can differ by ~2 ulp, enough
// to flip beta_fixed on a boundary sweep.  `step` is continuous across phases.
inline int32_t betaFixedForStep(uint64_t step, double t0) {
    const double beta = std::log(1.0 + static_cast<double>(step)) / t0;
    return static_cast<int32_t>(std::round(beta * (1 << 20)));  // kGumbelFracBits
}

// Sanity ceiling on total sweeps per chain: keeps annealTotalSteps' unsigned
// sum far from wrap and rejects requests that could not finish anyway.
constexpr int64_t kMaxTotalSteps = 1'000'000'000'000;

// Run-length beta tables.  beta_fixed = round(log(1+step)/t0 * 2^20) is
// monotone non-decreasing in step for t0 > 0, so each DISTINCT t0's schedule
// is stored as (exclusive 0-based end step, value) runs the kernel walks with
// a per-thread cursor.  The encoding is exact, so bit-exactness is untouched.
struct BetaRunTables {
    // Within a table, run r covers 0-based steps [runEnd[r-1], runEnd[r]);
    // the last runEnd == totalSteps.  Every table holds >= 1 run even at
    // totalSteps == 0 (the kernel prologue reads entry 0 unconditionally).
    std::vector<int64_t> runEnd;
    std::vector<int32_t> runVal;      // beta_fixed over that run
    std::vector<int64_t> chainRunOff; // chain -> its table's first run index
    int64_t nTables = 0;              // distinct(t0s)
};

// Cap on total run count (~400 MB device-side at 12 B/run).  Scales with the
// beta value range, not with sweeps; buildBetaRunTables fails loudly past it,
// before any cudaMalloc.
constexpr int64_t kMaxBetaRuns = 33'000'000;

// Build the run tables: one per DISTINCT t0 (exact double equality, so
// temperature replicas share), chainRunOff mapping each chain to its table.
// O(runs log), never O(totalSteps).  Precondition: every t0 > 0.
BetaRunTables buildBetaRunTables(const std::vector<double>& t0s, int64_t totalSteps);

// int32-score overflow gate, copied from the CPU's annealSimdSafeInt32; here
// it gates the packed-(score,~id) atomicMax argmax.  Evaluated at min(t0s);
// the one verdict picks the score path for every chain.
bool annealScoreSafeInt32(int64_t totalSteps, double t0, int64_t deltaBound);

// Max weighted degree B = max_v sum_nb |w(v,nb)|; bounds both |delta| and |S|.
int64_t maxWeightedDegree(const anneallib::AnnealCsr& ig);

// Remove all degree-zero vertices in place.  MUST run before AnnealCsr(g):
// it changes n, hence the dense id mapping and the printed partition length.
void removeDegreeZeroVertices(anneallib::Graph& g);

// Print chain `chain`'s results to stdout, byte-identical in shape to a CPU
// run of the same (seed, t0) except the leading device field.  elapsedSec is
// the TOTAL kernel time (all chains run concurrently in one launch).
// Returns the independently RECOMPUTED best cut value.
int64_t printChainResults(const SolverContext& ctx, size_t chain,
                          const std::vector<int8_t>& bestSpin,
                          const std::vector<int8_t>& endSpin,
                          double elapsedSec);

// End-of-run summary:
//   summary: chains: C best-chain: i best-cut-value: V time-sec: T chains-at-best: K
// best-chain: 0-based index of the largest recomputed best cut (ties ->
// smallest index); K: how many chains reached V.
void printSummary(const std::vector<int64_t>& bestVals, double elapsedSec);

#endif // HOST_COMMON_H
