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

#include "xls/noc/simulation/units.h"

#include "gtest/gtest.h"

namespace xls::noc {
namespace {

TEST(SimUnitsTest, CommonConversions) {
  Unit unit_ps_bits = kUnitPsBits;
  Unit unit_sec_bytes = kUnitSecBytes;
  Unit unit_ms_kbytes = {
      .time_unit = -3, .data_volume_unit = 3, .data_volume_base = 8};

  double ps_100_as_sec = ConvertTime(100, unit_ps_bits, unit_sec_bytes);
  double ps_100_as_msec = ConvertTime(100, unit_ps_bits, unit_ms_kbytes);
  double sec_3_as_ps = ConvertTime(3, kUnitSecBytes, kUnitPsBits);

  EXPECT_DOUBLE_EQ(ps_100_as_sec, 100.0e-12);
  EXPECT_DOUBLE_EQ(ps_100_as_msec, 100.0e-9);
  EXPECT_DOUBLE_EQ(sec_3_as_ps, 3.0e12);

  double bytes_3_as_bits = ConvertDataVolume(3, kUnitSecBytes, kUnitPsBits);
  double kbytes_2_as_bits = ConvertDataVolume(2, unit_ms_kbytes, kUnitPsBits);
  double bits_16_as_bytes = ConvertDataVolume(16, kUnitPsBits, kUnitSecBytes);
  double bits_16_as_kbytes = ConvertDataVolume(16, kUnitPsBits, unit_ms_kbytes);

  EXPECT_DOUBLE_EQ(bytes_3_as_bits, 3.0 * 8.0);
  EXPECT_DOUBLE_EQ(kbytes_2_as_bits, 2000.0 * 8.0);
  EXPECT_DOUBLE_EQ(bits_16_as_bytes, 2.0);
  EXPECT_DOUBLE_EQ(bits_16_as_kbytes, 2.0e-3);

  double bytes_per_sec_707_as_bits_per_ps =
      ConvertDataRate(707, 1, kUnitSecBytes, kUnitPsBits);
  double bits_707_per_100_ps_as_bytes_per_sec =
      ConvertDataRate(707, 100, kUnitPsBits, kUnitSecBytes);

  EXPECT_DOUBLE_EQ(bytes_per_sec_707_as_bits_per_ps, 707.0 * 8.0 / 1.0e12);
  EXPECT_DOUBLE_EQ(bits_707_per_100_ps_as_bytes_per_sec,
                   707.0 / 100.0e-12 / 8.0);
}

}  // namespace
}  // namespace xls::noc
