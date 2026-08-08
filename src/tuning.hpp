#ifndef GPU3_TUNING_HPP
#define GPU3_TUNING_HPP


#include <cstdint>

namespace gpu3 {
namespace tuning {

inline constexpr double corrWarpwalkMinDeg = 8.0;
inline constexpr int     tpbCoResidentBlocked8     = 256;
inline constexpr int     tpbCoResidentBlocked8BigN = 512;
inline constexpr int64_t blocked8CoResBigMinN      = 16384;
inline constexpr int     tpbWarpwalk               = 1024;
inline constexpr int64_t tpbWalkSmallMaxN          = 1024;
inline constexpr int     tpbWalkSmall              = 512;
inline constexpr int     tpbWalkLarge              = 896;
}  // namespace tuning
}  // namespace gpu3

#endif  // GPU3_TUNING_HPP
