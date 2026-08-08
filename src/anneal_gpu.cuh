#ifndef ANNEAL_GPU_CUH
#define ANNEAL_GPU_CUH

// Kernel entry point and its knobs.  The solver optimizes THROUGHPUT: it runs
// C independent chains -- one (t0, seed) pair each -- in ONE kernel launch,
// one persistent block per chain.  Every knob here is output-invariant, and
// chains are isolated: chain i's output is byte-identical whether it runs
// alone or among any number of others.

#include <cstdint>
#include <vector>

#include "host_common.h"
#include "immutable_graph.hpp"

namespace gpu3 {

// Device CSR view of the frozen AnnealCsr, shared read-only by every chain.
// Neighbour slices are sorted ascending by dense id: the kernel never binary
// searches them, but the order fixes the summation order of corr[] and of the
// initial S recompute, which is what makes the result independent of the host
// standard library's hash iteration order.
struct DevCsr {
    const uint32_t* offsets;     // n + 1
    const int32_t*  edgeTo;      // 2m
    const int32_t*  edgeWeight;  // 2m
    int64_t         n;
};

// Live scalars a chain's sweep loop carries; one instance per chain, block b
// touches only entry b.
struct AnnealState {
    int64_t currentVal;
    int64_t bestVal;
    int32_t improved;   // did the sweep just finished raise bestVal?
    int32_t pad;
};

// How FLIP computes the winner-winner coupling corr[] (the `corr` knob).
// Both modes are bit-identical; they differ only in how the same neighbour
// list is partitioned across threads.
enum class CorrMode {
    Walk,      // one thread per winner walks its neighbour list
    Warpwalk,  // the walk with one warp per winner; gb=0 skipped (corr[0]==0)
};

// SELECT always maps vertices to threads blocked8: v in [8b, 8b+8), matching
// the Philox counter layout.  State always lives in global memory.

// Resolved in main.cu from tuning.hpp.  The knob that needs a live device
// query is elsewhere: tpb in runDispatch.
struct LaunchConfig {
    CorrMode corr = CorrMode::Walk;
    int device = 0;
    int sWidth = 32;    // storage width of the S cache (32 or 64).  Not a
                        // fidelity knob: |S| <= B, so any width holding B is
                        // identical
};

// Run C independent anneals on the GPU in one launch, C = t0s.size() =
// solverSeeds.size() = initialSpins.size().  `phases`/`iters` are parallel,
// shared by every chain.  solverSeeds[i] is seeds[i] + kSolverSeedOffset
// (initial spins come from the RAW seed, passed in already generated).
// *kernelMs receives the cudaEvent-measured time of the one launch -- no
// setup, no I/O, no transfers -- matching the CPU's kernel-only bracket.
// kernelBestVals/kernelFinalVals (if non-null) receive the values the kernel
// tracked incrementally, exposed for cross-checking against an independent
// cutValueCsr recompute.
//
// maxWeightedDeg is B = maxWeightedDegree(ig) (host_common.h), passed in
// rather than recomputed: the caller already needs it to pick cfg.sWidth, and
// it is what the int32 score-path gate keys on.  Taking it as a parameter
// keeps the O(m) scan to one pass AND makes the caller's reported score-path
// necessarily the one this function instantiates -- a defaulted field could
// silently arrive as 0, which certifies the packed path unconditionally.
void annealGpu(const anneallib::AnnealCsr& ig,
               int64_t maxWeightedDeg,
               const std::vector<std::vector<int8_t>>& initialSpins,
               const std::vector<double>& t0s,
               const std::vector<size_t>& phases,
               const std::vector<uint64_t>& iters,
               const std::vector<uint64_t>& solverSeeds,
               const LaunchConfig& cfg,
               std::vector<std::vector<int8_t>>& endSpins,
               std::vector<std::vector<int8_t>>& bestSpins,
               float* kernelMs,
               std::vector<int64_t>* kernelBestVals = nullptr,
               std::vector<int64_t>* kernelFinalVals = nullptr);

}  // namespace gpu3

#endif  // ANNEAL_GPU_CUH
