#include "read_gset.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace anneallib {

namespace {

// Advance to the next data line: skip blanks and '#' comment lines.
bool nextDataLine(std::istream& in, std::string& line) {
    while (std::getline(in, line)) {
        size_t i = line.find_first_not_of(" \t\r");
        if (i == std::string::npos) continue;
        if (line[i] == '#') continue;
        return true;
    }
    return false;
}

// "Integral" = exactly representable as int64 (2^53 = double's exact limit).
bool isIntegralValue(double w) {
    return std::floor(w) == w && std::fabs(w) < 9007199254740992.0;
}

// Budgets for the automatic scale choice: scaled B under INT32_MAX/4 keeps
// deltas at int32; individual weights must fit the CSR's WeightT (int32).
constexpr double kBudgetB32 = static_cast<double>(INT32_MAX) / 4;   // preferred
constexpr double kBudgetB64 = static_cast<double>(INT64_MAX) / 8;   // fallback
constexpr double kBudgetW32 = static_cast<double>(INT32_MAX) / 2;

// Largest scale = 10^k (k in [-12, 12]) satisfying both budgets; maxDeg/2
// covers worst-case per-edge rounding.  Returns 0 if nothing fits.
double chooseScale(double maxAbsB, double maxAbsW, double maxDeg, double budgetB) {
    for (int k = 12; k >= -12; --k) {
        const double s = std::pow(10.0, k);
        if (maxAbsB * s + maxDeg / 2 <= budgetB && maxAbsW * s <= kBudgetW32)
            return s;
    }
    return 0.0;
}

// Scale choice + rounding shared by the loaders.  Produces surviving
// (non-zero) edges as parallel {u, v, w} arrays in file order, un-deduped.
struct ParsedInstance {
    int64_t numVertices = 0;      // header N
    int64_t numEdges = 0;         // header M
    double scale = 1.0;           // w_int = llround(scale * w)
    bool floatWeights = false;    // any non-integral weight value seen
    int64_t zeroDropped = 0;      // edges whose scaled weight rounded to 0
    std::vector<int64_t> u, v;    // endpoints of surviving edges
    std::vector<int64_t> w;       // rounded integer weights (same indexing)
};

bool parseAndScale(std::istream& in, double forcedScale, ParsedInstance& out) {
    std::string line;
    if (!nextDataLine(in, line)) {
        std::cerr << "Error: no header line (numVertices numEdges)\n";
        return false;
    }
    int64_t numVertices = 0, numEdges = 0;
    {
        std::istringstream hs(line);
        if (!(hs >> numVertices >> numEdges) || numVertices < 0 || numEdges < 0) {
            std::cerr << "Error: failed to read header (numVertices numEdges)\n";
            return false;
        }
    }

    // Pass 1: buffer all edges so the scale is chosen from the whole instance
    // before any weight is rounded.
    std::vector<int64_t> eu, ev;
    std::vector<double> ew;
    eu.reserve(numEdges); ev.reserve(numEdges); ew.reserve(numEdges);
    bool floatWeights = false;

    for (int64_t i = 0; i < numEdges; ++i) {
        if (!nextDataLine(in, line)) {
            std::cerr << "Error: failed to read edge " << i << "\n";
            return false;
        }
        const char* p = line.c_str();
        char* end = nullptr;
        const long long u = std::strtoll(p, &end, 10);
        if (end == p) { std::cerr << "Error: bad edge line " << i << "\n"; return false; }
        p = end;
        const long long v = std::strtoll(p, &end, 10);
        if (end == p) { std::cerr << "Error: bad edge line " << i << "\n"; return false; }
        p = end;
        const double w = std::strtod(p, &end);
        if (end == p) { std::cerr << "Error: bad edge line " << i << "\n"; return false; }
        if (!isIntegralValue(w)) floatWeights = true;
        eu.push_back(u); ev.push_back(v); ew.push_back(w);
    }

    // Unscaled per-vertex bounds for the scale choice.  Vertex ids are
    // usually 1..N; fall back to a hash map otherwise.
    double maxAbsB = 0.0, maxAbsW = 0.0, maxDeg = 0.0;
    {
        std::vector<double> absB;
        std::vector<int64_t> deg;
        std::unordered_map<int64_t, double> absBMap;
        std::unordered_map<int64_t, int64_t> degMap;
        bool dense = true;
        for (size_t i = 0; i < eu.size() && dense; ++i)
            dense = eu[i] >= 0 && eu[i] <= numVertices &&
                    ev[i] >= 0 && ev[i] <= numVertices;
        if (dense) { absB.assign(numVertices + 1, 0.0); deg.assign(numVertices + 1, 0); }
        for (size_t i = 0; i < eu.size(); ++i) {
            const double aw = std::fabs(ew[i]);
            maxAbsW = std::max(maxAbsW, aw);
            if (dense) {
                absB[eu[i]] += aw; absB[ev[i]] += aw;
                ++deg[eu[i]]; ++deg[ev[i]];
            } else {
                absBMap[eu[i]] += aw; absBMap[ev[i]] += aw;
                ++degMap[eu[i]]; ++degMap[ev[i]];
            }
        }
        if (dense) {
            for (double b : absB) maxAbsB = std::max(maxAbsB, b);
            for (int64_t d : deg) maxDeg = std::max(maxDeg, static_cast<double>(d));
        } else {
            for (const auto& [id, b] : absBMap) { (void)id; maxAbsB = std::max(maxAbsB, b); }
            for (const auto& [id, d] : degMap) { (void)id; maxDeg = std::max(maxDeg, static_cast<double>(d)); }
        }
    }

    // All-integral weights keep scale 1 (identical to the historical reader).
    double scale = 1.0;
    if (forcedScale > 0.0) {
        scale = forcedScale;
    } else if (floatWeights) {
        scale = chooseScale(maxAbsB, maxAbsW, maxDeg, kBudgetB32);
        if (scale == 0.0) scale = chooseScale(maxAbsB, maxAbsW, maxDeg, kBudgetB64);
        if (scale == 0.0) {
            std::cerr << "Error: no power-of-10 scale fits these weights into "
                         "int32 weights / int64 deltas (max|w|=" << maxAbsW
                      << ", maxB=" << maxAbsB << ")\n";
            return false;
        }
        if (maxAbsW * scale < 0.5) {
            std::cerr << "Error: chosen scale " << scale
                      << " rounds every weight to 0 (max|w|=" << maxAbsW << ")\n";
            return false;
        }
    }

    // Pass 2: round, drop zeros, emit surviving edges (no dedup here).
    out.u.clear(); out.v.clear(); out.w.clear();
    out.u.reserve(eu.size()); out.v.reserve(eu.size()); out.w.reserve(eu.size());
    int64_t zeroDropped = 0;
    for (size_t i = 0; i < eu.size(); ++i) {
        const int64_t wi = (scale == 1.0 && isIntegralValue(ew[i]))
                               ? static_cast<int64_t>(ew[i])
                               : llround(ew[i] * scale);
        if (wi == 0) { ++zeroDropped; continue; }
        out.u.push_back(eu[i]); out.v.push_back(ev[i]); out.w.push_back(wi);
    }

    out.numVertices = numVertices;
    out.numEdges = numEdges;
    out.scale = scale;
    out.floatWeights = floatWeights;
    out.zeroDropped = zeroDropped;
    return true;
}

} // namespace

bool read_instance_from_stream(std::istream& in, Graph& g,
                               ReadStats* stats, double forcedScale) {
    ParsedInstance pi;
    if (!parseAndScale(in, forcedScale, pi)) return false;

    // Graph::addEdge dedups (first weight wins); B is re-derived below over
    // the deduped adjacency.
    for (size_t i = 0; i < pi.u.size(); ++i)
        g.addEdge(pi.u[i], pi.v[i], pi.w[i]);

    int64_t maxAbsBInt = 0;
    for (int64_t vtx : g.getVertices()) {
        int64_t s = 0;
        for (const auto& [nb, w] : g.getEdges(vtx))
            s += std::llabs(static_cast<long long>(w));
        if (s > maxAbsBInt) maxAbsBInt = s;
    }

    if (stats) {
        stats->numVertices = pi.numVertices;
        stats->numEdges = pi.numEdges;
        stats->zeroDropped = pi.zeroDropped;
        stats->floatWeights = pi.floatWeights;
        stats->scale = pi.scale;
        stats->maxAbsWeightedDegree = maxAbsBInt;
    }
    return true;
}

bool read_instance_from_file(const std::string& filename, Graph& g,
                             ReadStats* stats, double forcedScale) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: could not open file: " << filename << "\n";
        return false;
    }
    return read_instance_from_stream(file, g, stats, forcedScale);
}

} // namespace anneallib
