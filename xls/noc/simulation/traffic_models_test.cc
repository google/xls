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

#include "xls/noc/simulation/traffic_models.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"

namespace xls::noc {
namespace {

TEST(TrafficModelsTest, GeneralizedGeometricModelTest) {
  double lambda = 0.2;
  double burst_prob = 0.1;
  int64_t packet_size_bits = 128;

  RandomNumberInterface rnd;
  GeneralizedGeometricTrafficModel model(lambda, burst_prob, packet_size_bits,
                                         rnd);
  TrafficModelMonitor monitor;

  int64_t cycle = 0;
  int64_t num_packets = 0;
  for (cycle = 0; cycle < 10'000'000; ++cycle) {
    XLS_VLOG(2) << "Cycle " << cycle << ":\n";

    std::vector<DataPacket> packets = model.GetNewCyclePackets(cycle);

    for (DataPacket& p : packets) {
      XLS_VLOG(2) << "  -  " << p.ToString() << "\n";
    }

    num_packets += packets.size();
    monitor.AcceptNewPackets(absl::MakeSpan(packets), cycle);
  }

  double expected_traffic = model.ExpectedTrafficRateInMiBps(500);
  double measured_traffic = monitor.MeasuredTrafficRateInMiBps(500);

  XLS_VLOG(1) << "Packet " << num_packets << "\n";
  XLS_VLOG(1) << "Cycles " << cycle << "\n";
  XLS_VLOG(1) << "Expected Traffic " << expected_traffic << "\n";
  XLS_VLOG(1) << "Measured Traffic " << measured_traffic << std::endl;

  EXPECT_EQ(num_packets / 1000, static_cast<int64_t>(lambda * cycle) / 1000);
  EXPECT_EQ(static_cast<int64_t>(expected_traffic / 100),
            static_cast<int64_t>(measured_traffic / 100));
}

}  // namespace
}  // namespace xls::noc
