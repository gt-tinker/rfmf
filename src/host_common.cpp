#include "host_common.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>

#include "philox.cuh"
#include "rng.hpp"

namespace {

// Comma-separated unsigned list, shared by --phases, --iters and --seeds.
std::vector<uint64_t> parseU64List(const std::string& s) {
    std::vector<uint64_t> result;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ',')) {
        result.push_back(std::stoull(token));
    }
    return result;
}

// Comma-separated double list, for --t0s.
std::vector<double> parseF64List(const std::string& s) {
    std::vector<double> result;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ',')) {
        result.push_back(std::stod(token));
    }
    return result;
}

// Verbatim from the CPU's generateRandomSpins, so chain i starts from the
// same partition as a CPU run with --seed seeds[i].
std::vector<int8_t> generateRandomSpins(uint64_t n, anneallib::Rng& rng) {
    std::vector<int8_t> spin(n);
    for (uint64_t i = 0; i < n; ++i)
        spin[i] = static_cast<int8_t>(rng.bits(1) ? 1 : -1);
    return spin;
}

}  // namespace

// Field order and spacing match the CPU reference exactly, with this chain's
// seed/t0 in the singular positions.  GPU knobs are deliberately not printed,
// so a chain block differs from a CPU line only in "device:" and can be
// diffed directly against a single-chain run.
std::ostream& streamChainArgs(std::ostream& os, const AnnealArgs& a, size_t chain) {
    std::string instance = a.filename;
    auto pos = instance.find_last_of('/');
    if (pos != std::string::npos) instance = instance.substr(pos + 1);
    os << "instance: " << instance
       << " --seed " << a.seeds[chain]
       << " --phases ";
    for (size_t i = 0; i < a.phases.size(); ++i) {
        if (i) os << ',';
        os << a.phases[i];
    }
    os << " --t0 " << a.t0s[chain]
       << " --iters ";
    for (size_t i = 0; i < a.iters.size(); ++i) {
        if (i) os << ',';
        os << a.iters[i];
    }
    return os;
}

void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " <filename> --seeds <N,N,...> --t0s <F,F,...>\n"
              << "       --phases <N,N,...> --iters <N,N,...>\n"
              << "\nMulti-chain CUDA Gumbel-competition annealer for Max-Cut.\n"
              << "Runs one INDEPENDENT chain per (seed, t0) pair -- the two lists are\n"
              << "zipped and must have the same length -- in a single kernel launch,\n"
              << "one persistent block per chain.  Optimizes THROUGHPUT rather\n"
              << "than the latency of any single chain.\n"
              << "\nRequired options:\n"
              << "  --seeds <N,N,...>   RNG seed per chain; a 0 entry draws from\n"
              << "                      random_device (independently per entry)\n"
              << "  --t0s <F,F,...>     initial temperature per chain (each > 0)\n"
              << "  --phases <N,N,...>  group sizes per phase (e.g. 8,4,2,1); shared\n"
              << "                      by every chain\n"
              << "  --iters <N,N,...>   total Gumbel iterations per phase (one per\n"
              << "                      phase); shared by every chain\n"
              << "\nChains cannot see each other: chain i's output is identical\n"
              << "whether it runs alone or alongside any number of others.\n"
              << "\nOptional options:\n"
              << "  --device <N>        CUDA device ordinal (default 0)\n"
              << "\nThere are no tuning flags.  Every performance choice (corr,\n"
              << "threads per block, S-cache width) is resolved\n"
              << "from the hard-coded V100 measurements in src/tuning.hpp.  All of them\n"
              << "are output-invariant: they change the schedule, never a result.\n"
              << "\n--phases and --iters must have the same length; so must --seeds\n"
              << "and --t0s (one chain per entry).\n"
              << "\nExample (3 temperatures x 2 seeds = 6 chains, written out):\n"
              << "  " << prog << " Gset/G1 --seeds 1,2,1,2,1,2 --t0s 90,90,105,105,120,120"
              << " --phases 8,4,2,1 --iters 2000,2000,2000,2000\n";
}

AnnealArgs parseArgs(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        std::exit(1);
    }

    AnnealArgs args{};
    args.filename = argv[1];

    bool hasSeeds = false, hasPhases = false, hasT0s = false, hasIters = false;

    for (int i = 2; i < argc;) {
        std::string flag = argv[i];

        if (i + 1 >= argc) {
            std::cerr << "Error: missing value for " << flag << "\n";
            std::exit(1);
        }
        std::string val = argv[i + 1];

        // Single-chain spellings are rejected with a pointer, not aliased.
        if (flag == "--seed") {
            std::cerr << "Error: one chain per (seed, t0) pair; use "
                         "--seeds <N,N,...> (zipped with --t0s).\n";
            std::exit(1);
        }
        if (flag == "--t0") {
            std::cerr << "Error: one chain per (seed, t0) pair; use "
                         "--t0s <F,F,...> (zipped with --seeds).\n";
            std::exit(1);
        }

        if      (flag == "--seeds") { args.seeds = parseU64List(val); hasSeeds = true; }
        else if (flag == "--phases") { const auto v = parseU64List(val);
                                       args.phases.assign(v.begin(), v.end()); hasPhases = true; }
        else if (flag == "--t0s")    { args.t0s = parseF64List(val); hasT0s = true; }
        else if (flag == "--iters")  { args.iters = parseU64List(val); hasIters = true; }
        else if (flag == "--device") { args.device = std::stoi(val); }
        else {
            std::cerr << "Error: unknown option '" << flag << "'\n";
            printUsage(argv[0]);
            std::exit(1);
        }
        i += 2;
    }

    bool ok = true;
    if (!hasSeeds)  { std::cerr << "Error: --seeds is required\n"; ok = false; }
    if (!hasPhases) { std::cerr << "Error: --phases is required\n"; ok = false; }
    if (!hasT0s)    { std::cerr << "Error: --t0s is required\n"; ok = false; }
    if (!hasIters)  { std::cerr << "Error: --iters is required\n"; ok = false; }
    if (hasT0s && args.t0s.empty()) {
        std::cerr << "Error: --t0s must have at least one entry\n"; ok = false;
    }
    if (hasT0s) {
        for (double t : args.t0s)
            if (!(t > 0.0)) {
                std::cerr << "Error: every --t0s entry must be > 0\n"; ok = false;
                break;
            }
    }
    if (hasSeeds && hasT0s && args.seeds.size() != args.t0s.size()) {
        std::cerr << "Error: --seeds must have the same number of entries as --t0s ("
                  << args.t0s.size() << "); the lists are zipped, one chain per pair\n";
        ok = false;
    }
    if (hasPhases && hasIters && args.iters.size() != args.phases.size()) {
        std::cerr << "Error: --iters must have the same number of elements as --phases ("
                  << args.phases.size() << ")\n";
        ok = false;
    }
    if (!ok) {
        printUsage(argv[0]);
        std::exit(1);
    }

    return args;
}

// Two passes because removeVertex mutates the set getVertices() returns.
void removeDegreeZeroVertices(anneallib::Graph& g) {
    std::vector<int64_t> toRemove;
    for (int64_t v : g.getVertices()) {
        if (g.degree(v) == 0) {
            toRemove.push_back(v);
        }
    }
    for (int64_t v : toRemove) {
        g.removeVertex(v);
    }
}

int64_t annealTotalSteps(const std::vector<uint64_t>& iters) {
    // Summed and range-checked in UNSIGNED arithmetic, per entry: a naive
    // int64 sum wraps on legal CLI input and slips under kMaxTotalSteps' cap.
    uint64_t total = 0;
    for (uint64_t it : iters) {
        total += it;
        if (it > static_cast<uint64_t>(kMaxTotalSteps) ||
            total > static_cast<uint64_t>(kMaxTotalSteps)) {
            std::cerr << "Error: --iters totals more than " << kMaxTotalSteps
                      << " sweeps (the total-sweep sanity cap; see host_common.h).\n";
            std::exit(1);
        }
    }
    return static_cast<int64_t>(total);
}

namespace {

// First 1-based step in (lo, hiMax] with beta_fixed > v, or 0 if none.
// Precondition: betaFixedForStep(lo, t0) <= v.  `pred` only seeds the upper
// bracket; correctness never depends on it.
int64_t firstStepAbove(double t0, int32_t v, int64_t lo, int64_t hiMax,
                       int64_t pred) {
    int64_t hi = std::min(std::max(pred, lo + 1), hiMax);
    while (betaFixedForStep(static_cast<uint64_t>(hi), t0) <= v) {
        if (hi == hiMax) return 0;
        const int64_t span = hi - lo;  // >= 1; doubles each round
        lo = hi;
        hi = (span * 2 > hiMax - lo) ? hiMax : lo + span * 2;
    }
    // Invariant: beta(lo) <= v < beta(hi).  Bisect to the first crossing.
    while (hi - lo > 1) {
        const int64_t mid = lo + (hi - lo) / 2;
        if (betaFixedForStep(static_cast<uint64_t>(mid), t0) <= v) lo = mid;
        else hi = mid;
    }
    return hi;
}

// Append one t0's run-length encoding of betaFixedForStep over 1-based steps
// [1, totalSteps] (stored ends are 0-based exclusive).  O(runs) boundary
// searches, each seeded by inverting the value midpoint.
void appendBetaRuns(double t0, int64_t totalSteps,
                    std::vector<int64_t>& runEnd, std::vector<int32_t>& runVal) {
    if (totalSteps <= 0) {  // >= 1 run always: the kernel prologue reads entry 0
        runEnd.push_back(0);
        runVal.push_back(0);
        return;
    }
    int64_t s = 1;  // 1-based first step of the current run
    while (true) {
        if (static_cast<int64_t>(runEnd.size()) >= kMaxBetaRuns) {
            std::cerr << "Error: run-length beta tables exceed " << kMaxBetaRuns
                      << " runs (kMaxBetaRuns, ~400 MB; hit encoding t0=" << t0
                      << ").\nRuns scale with beta_fixed's value range "
                         "(~log(totalSteps)/t0 * 2^20), so this takes an\n"
                         "extreme t0 and/or sweep count; raise kMaxBetaRuns in "
                         "host_common.h if intentional.\n";
            std::exit(1);
        }
        const int32_t v = betaFixedForStep(static_cast<uint64_t>(s), t0);
        const double g =
            std::exp((static_cast<double>(v) + 0.5) * t0 /
                     (1 << gpu3::kGumbelFracBits)) - 1.0;
        const int64_t pred =
            (g >= static_cast<double>(totalSteps)) ? totalSteps
                                                   : static_cast<int64_t>(g) + 2;
        const int64_t next = firstStepAbove(t0, v, s, totalSteps, pred);
        if (next == 0) {  // v holds through totalSteps: close the table
            runEnd.push_back(totalSteps);
            runVal.push_back(v);
            return;
        }
        runEnd.push_back(next - 1);  // 1-based `next` == 0-based exclusive end + 1
        runVal.push_back(v);
        s = next;
    }
}

}  // namespace

BetaRunTables buildBetaRunTables(const std::vector<double>& t0s, int64_t totalSteps) {
    BetaRunTables t;
    t.chainRunOff.resize(t0s.size());
    std::vector<double> distinct;
    std::vector<int64_t> firstRun;  // distinct k -> its table's first run index
    for (size_t c = 0; c < t0s.size(); ++c) {
        size_t k = 0;
        while (k < distinct.size() && distinct[k] != t0s[c]) ++k;
        if (k == distinct.size()) {
            distinct.push_back(t0s[c]);
            firstRun.push_back(static_cast<int64_t>(t.runEnd.size()));
            appendBetaRuns(t0s[c], totalSteps, t.runEnd, t.runVal);
        }
        t.chainRunOff[c] = firstRun[k];
    }
    t.nTables = static_cast<int64_t>(distinct.size());
    return t;
}

bool annealScoreSafeInt32(int64_t totalSteps, double t0, int64_t deltaBound) {
    // The Q20 scale and the Gumbel magnitude bound come from philox.cuh, which
    // is plain C++ outside an nvcc device pass -- it pulls in no CUDA header.
    // Taking them from there rather than respelling the numbers is what puts
    // this gate under philox.cuh's static_assert against the LUT.
    const double betaMax = std::log(1.0 + static_cast<double>(totalSteps)) / t0;
    const int64_t betaMaxFixed =
        static_cast<int64_t>(std::llround(betaMax * (1 << gpu3::kGumbelFracBits)));
    // Overflow guard BEFORE the product: betaMaxFixed * deltaBound can wrap
    // past 2^63 and falsely certify the packed path where it is least safe.
    const int64_t limit = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) -
                          gpu3::kGumbelMaxMag;
    if (betaMaxFixed > 0 && deltaBound > 0 && deltaBound > limit / betaMaxFixed)
        return false;
    return (betaMaxFixed * deltaBound + gpu3::kGumbelMaxMag) <
           static_cast<int64_t>(std::numeric_limits<int32_t>::max());
}

int64_t maxWeightedDegree(const anneallib::AnnealCsr& ig) {
    const uint64_t n = ig.numVertices();
    int64_t B = 0;
    for (uint64_t v = 0; v < n; ++v) {
        auto nb = ig.neighbors(static_cast<int64_t>(v));
        int64_t s = 0;
        for (uint32_t i = 0; i < nb.size(); ++i)
            s += std::llabs(static_cast<long long>(nb.weight[i]));
        if (s > B) B = s;
    }
    return B;
}

int setupSolver(int argc, char* argv[], SolverContext& ctx) {
    ctx.args = parseArgs(argc, argv);

    // File => Graph => CSR.  Degree-zero removal MUST happen between the two:
    // it changes n and hence the dense id mapping and the printed partition
    // bits.  Weight scale 0.0 = auto (Gset files always resolve to scale 1).
    {
        anneallib::Graph g;
        if (!anneallib::read_instance_from_file(ctx.args.filename, g,
                                                &ctx.readStats, 0.0)) {
            std::cerr << "Error: failed to read graph: " << ctx.args.filename << "\n";
            return 1;
        }
        removeDegreeZeroVertices(g);
        ctx.csr = anneallib::AnnealCsr(g);
    }
    std::cerr << "Loaded: n=" << ctx.readStats.numVertices
              << " m=" << ctx.readStats.numEdges
              << " weight-scale=" << ctx.readStats.scale
              << " max-weighted-degree=" << ctx.readStats.maxAbsWeightedDegree
              << (ctx.readStats.floatWeights ? " (float weights)" : "")
              << (ctx.readStats.zeroDropped
                      ? (" zero-dropped=" + std::to_string(ctx.readStats.zeroDropped))
                      : std::string())
              << "\n";

    // Each 0 entry gets its own random_device draw; the resolved list is
    // logged so any run can be reproduced.
    for (uint64_t& s : ctx.args.seeds) {
        if (s == 0) {
            std::random_device rd;
            s = rd();
        }
    }
    std::cerr << "Resolved seeds:";
    for (size_t i = 0; i < ctx.args.seeds.size(); ++i)
        std::cerr << (i ? "," : " ") << ctx.args.seeds[i];
    std::cerr << "\n";

    // Duplicate (t0, seed) pairs are legal but produce bit-identical chains;
    // warn rather than let a typo silently burn a chain.
    const size_t C = numChains(ctx.args);
    for (size_t i = 0; i < C; ++i)
        for (size_t j = i + 1; j < C; ++j)
            if (ctx.args.seeds[i] == ctx.args.seeds[j] &&
                ctx.args.t0s[i] == ctx.args.t0s[j]) {
                std::cerr << "Warning: chains " << i << " and " << j
                          << " have identical (seed, t0) = (" << ctx.args.seeds[i]
                          << ", " << ctx.args.t0s[i]
                          << ") and will produce identical output.\n";
            }

    // RAW seed per chain -- NOT seed + kSolverSeedOffset (see host_common.h).
    ctx.initialSpins.resize(C);
    for (size_t i = 0; i < C; ++i) {
        anneallib::Rng rng_initial(ctx.args.seeds[i]);
        ctx.initialSpins[i] = generateRandomSpins(ctx.csr.numVertices(), rng_initial);
    }

    return 0;
}

int64_t printChainResults(const SolverContext& ctx, size_t chain,
                          const std::vector<int8_t>& bestSpin,
                          const std::vector<int8_t>& endSpin,
                          double elapsedSec) {
    int64_t best_val = anneallib::solver::cutValueCsr(ctx.csr, bestSpin);
    int64_t end_val  = anneallib::solver::cutValueCsr(ctx.csr, endSpin);

    // setprecision(4) must be applied BEFORE the args are streamed, so --t0
    // prints as the CPU does.  Do not reorder.
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "device: GPU ";
    streamChainArgs(std::cout, ctx.args, chain)
              << " best-cut-value: " << best_val
              << " final-cut-value: " << end_val << " time-sec: " << elapsedSec << "\n";

    if (ctx.readStats.scale != 1.0) {
        std::cout << "weight-scale: " << ctx.readStats.scale
                  << " descaled-best-cut-value: "
                  << static_cast<double>(best_val) / ctx.readStats.scale
                  << " descaled-final-cut-value: "
                  << static_cast<double>(end_val) / ctx.readStats.scale << "\n";
    }

    const uint64_t n = ctx.csr.numVertices();
    std::cout << "Solution-best-partition: ";
    for (uint64_t i = 0; i < n; ++i)
        std::cout << (bestSpin[i] < 0 ? "0" : "1");
    std::cout << "\n";

    std::cout << "Solution-final-partition: ";
    for (uint64_t i = 0; i < n; ++i)
        std::cout << (endSpin[i] < 0 ? "0" : "1");
    std::cout << "\n";

    return best_val;
}

void printSummary(const std::vector<int64_t>& bestVals, double elapsedSec) {
    const size_t C = bestVals.size();
    size_t best = 0;
    for (size_t i = 1; i < C; ++i)
        if (bestVals[i] > bestVals[best]) best = i;   // ties -> smallest index

    size_t atBest = 0;
    for (size_t i = 0; i < C; ++i)
        if (bestVals[i] == bestVals[best]) ++atBest;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "summary: chains: " << C
              << " best-chain: " << best
              << " best-cut-value: " << bestVals[best]
              << " time-sec: " << elapsedSec
              << " chains-at-best: " << atBest
              << "\n";
}
