#ifndef READ_GSET_HPP
#define READ_GSET_HPP

#include "graph.hpp"
#include <iostream>
#include <string>

namespace anneallib {
    // What the instance loader did, for logging/reproducibility.  Every field
    // here is echoed on the "Loaded:" stderr line.
    struct ReadStats {
        int64_t numVertices = 0;          // header N
        int64_t numEdges = 0;             // header M
        int64_t zeroDropped = 0;          // edges whose scaled weight rounded to 0
        bool floatWeights = false;        // any non-integral weight value seen
        double scale = 1.0;               // w_int = llround(scale * w)
        int64_t maxAbsWeightedDegree = 0; // B = max_v sum_nb |w_int| (scaled)
    };

    // Parse a graph in Gset or MQLib format: blank lines and '#' lines are
    // skipped; header line "<numVertices> <numEdges>"; then one
    // "<u> <v> <weight>" per line.  Vertices can be any int64_t.  Non-integral
    // weights are multiplied by the largest power of 10 that keeps the max
    // weighted degree B inside an int32-friendly budget, then rounded;
    // all-integral files keep scale 1 (identical to the historical reader).
    // forcedScale > 0 overrides the automatic choice.  Edges whose scaled
    // weight rounds to 0 are dropped and counted in stats.
    // Returns true on success; on failure prints to stderr and returns false.
    bool read_instance_from_stream(std::istream& in, Graph& g,
                                   ReadStats* stats = nullptr,
                                   double forcedScale = 0.0);
    bool read_instance_from_file(const std::string& filename, Graph& g,
                                 ReadStats* stats = nullptr,
                                 double forcedScale = 0.0);
}

#endif // READ_GSET_HPP
