// Copyright 2021 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef XLS_NOC_SIMULATION_UNITS_H_
#define XLS_NOC_SIMULATION_UNITS_H_

#include <cmath>
#include <cstdint>
#include <queue>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"

// This file contains classes used define units and their conversions.

namespace xls::noc {

// Describes the unit different measurements are stored in.
//
// It is desired that the measurement stored within the simulator are
// loss-less and precise.
//
// In lieu of storing in decimal floating point, user specifications in decimal
// strings can be converted to an integer  annotated with a corresponding
// Unit structure.  For example, a latency (time)
// value specified as 0.9ns can be expressed as 900ps without loss.  Basic
// operations such as addition can be performed and round-tripped to the user
// without loss.
//
// Conversion to more user-friendly units or to doubles can be done during
// presentation or when needed for a particular algorithm.  The idea here is to
// delay that conversion for as long as possible.
struct Unit {
  // Time is measured in 10^{time_unit} seconds (-12 for ps).
  int32_t time_unit;

  // Data volume is measured in 10^{data_volume_unit}*data_volume_base.
  int16_t data_volume_unit;

  // Data volume is measured in multiples of {data_volume_base} bits
  // (8 for bytes).
  int16_t data_volume_base;
};

constexpr Unit kUnitPsBits = Unit{-12, 0, 1};
constexpr Unit kUnitSecBytes = Unit{0, 0, 8};

// Convert a time measure from the "from" unit to the "to" unit.
//
// For example, 10ps (ps being 10^-12 seconds) converting to seconds would be
//  ConvertTime( time = 10, from = Unit { .time_unit=-12 },
//                            to = Unit { .time_unit=0 })
//  which will return 10.0e-12 seconds as a double.
inline double ConvertTime(int64_t time, Unit from, Unit to) {
  // TODO(tedhong) : 2021-06-01 Create fast/slow paths to
  //                 improve precision and performance.
  int64_t exponent = from.time_unit - to.time_unit;

  double significand = static_cast<double>(time);
  double power = std::pow(10.0, exponent);

  return significand * power;
}

// Convert a data volume measure from the "from" unit to the "to" unit.
//
// For example, 10MB (MB being 10^6 bytes) converting to bits would be
//  ConvertDataVolume( space = 10,
//                     from = Unit { .data_volume_unit=6, .data_volume_base=8 },
//                     to = Unit { .data_volume_unit=0, .data_volume_base=0 })
//  which will return 80.0e6 bits.
inline double ConvertDataVolume(int64_t space, Unit from, Unit to) {
  int64_t exponent = from.data_volume_unit - to.data_volume_unit;

  double significand = (static_cast<double>(space) *
                        static_cast<double>(from.data_volume_base)) /
                       static_cast<double>(to.data_volume_base);
  double power = std::pow(10.0, exponent);

  return significand * power;
}

// Convert a data rate (ex. bits/sec) from one unit to the next.
inline double ConvertDataRate(int64_t space, int64_t time, Unit from, Unit to) {
  // TODO(tedhong): 2021-06-27 Don't rely on ConvertDataVolume and
  //  ConvertTime to improve precision.
  double space_converted = ConvertDataVolume(space, from, to);
  double time_converted = ConvertTime(time, from, to);
  return space_converted / time_converted;
}

}  // namespace xls::noc

#endif  // XLS_NOC_SIMULATION_UNITS_H_
