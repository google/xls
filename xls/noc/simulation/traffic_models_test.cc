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

#include <memory>
#include <vector>

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

  model.SetVCIndex(1);
  model.SetSourceIndex(10);
  model.SetDestinationIndex(3);

  EXPECT_EQ(model.GetVCIndex(), 1);
  EXPECT_EQ(model.GetSourceIndex(), 10);
  EXPECT_EQ(model.GetDestinationIndex(), 3);
  EXPECT_EQ(model.GetPacketSizeInBits(), packet_size_bits);
  EXPECT_DOUBLE_EQ(model.GetLambda(), lambda);
  EXPECT_DOUBLE_EQ(model.GetBurstProb(), 0.1);

  TrafficModelMonitor monitor;

  int64_t cycle = 0;
  int64_t num_packets = 0;
  int64_t bits_sent = 0;

  for (cycle = 0; cycle < 10'000'000; ++cycle) {
    XLS_VLOG(2) << "Cycle " << cycle << ":";

    std::vector<DataPacket> packets = model.GetNewCyclePackets(cycle);

    for (DataPacket& p : packets) {
      EXPECT_EQ(p.vc, 1);
      EXPECT_EQ(p.source_index, 10);
      EXPECT_EQ(p.destination_index, 3);
      XLS_VLOG(2) << "  -  " << p.ToString();

      bits_sent += p.data.bit_count();
    }

    num_packets += packets.size();
    monitor.AcceptNewPackets(absl::MakeSpan(packets), cycle);
  }

  double expected_traffic = model.ExpectedTrafficRateInMiBps(500);
  double measured_traffic = monitor.MeasuredTrafficRateInMiBps(500);

  XLS_VLOG(1) << "Packet " << num_packets;
  XLS_VLOG(1) << "Cycles " << cycle;
  XLS_VLOG(1) << "Expected Traffic " << expected_traffic;
  XLS_VLOG(1) << "Measured Traffic " << measured_traffic;

  EXPECT_EQ(bits_sent, monitor.MeasuredBitsSent());
  EXPECT_EQ(num_packets, monitor.MeasuredPacketCount());
  EXPECT_NEAR(num_packets, lambda * cycle, 2e3);
  EXPECT_NEAR(measured_traffic, expected_traffic, 1e1);
  EXPECT_DOUBLE_EQ(expected_traffic,
                   lambda * 128.0 / 500.0e-12 / 1024.0 / 1024.0 / 8.0);
}

TEST(TrafficModelsTest, GeneralizedGeometricModelBuilderTest) {
  double lambda = 0.2;
  double burst_prob = 0.1;
  int64_t packet_size_bits = 128;

  RandomNumberInterface rnd;
  GeneralizedGeometricTrafficModelBuilder builder(lambda, burst_prob,
                                                  packet_size_bits, rnd);

  builder.SetVCIndex(1).SetSourceIndex(10).SetDestinationIndex(3);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GeneralizedGeometricTrafficModel> model, builder.Build());

  EXPECT_EQ(lambda, model->GetLambda());
  EXPECT_EQ(burst_prob, model->GetBurstProb());
  EXPECT_EQ(packet_size_bits, model->GetPacketSizeInBits());
  EXPECT_EQ(&rnd, model->GetRandomNumberInterface());
  EXPECT_EQ(model->GetVCIndex(), 1);
  EXPECT_EQ(model->GetSourceIndex(), 10);
  EXPECT_EQ(model->GetDestinationIndex(), 3);
}

TEST(TrafficModelsTest, ReplayModelTest) {
  int64_t packet_size_bits = 128;

  ReplayTrafficModel model(packet_size_bits, {0, 1, 2, 3, 4});

  model.SetVCIndex(1);
  model.SetSourceIndex(10);
  model.SetDestinationIndex(3);

  EXPECT_EQ(model.GetVCIndex(), 1);
  EXPECT_EQ(model.GetSourceIndex(), 10);
  EXPECT_EQ(model.GetDestinationIndex(), 3);
  EXPECT_EQ(model.GetPacketSizeInBits(), packet_size_bits);

  TrafficModelMonitor monitor;

  int64_t cycle = 0;
  int64_t num_packets = 0;
  int64_t bits_sent = 0;

  for (cycle = 0; cycle < 5; ++cycle) {
    XLS_VLOG(2) << "Cycle " << cycle << ":";

    std::vector<DataPacket> packets = model.GetNewCyclePackets(cycle);

    for (DataPacket& p : packets) {
      EXPECT_EQ(p.vc, 1);
      EXPECT_EQ(p.source_index, 10);
      EXPECT_EQ(p.destination_index, 3);
      XLS_VLOG(2) << "  -  " << p.ToString();

      bits_sent += p.data.bit_count();
    }

    num_packets += packets.size();
    monitor.AcceptNewPackets(absl::MakeSpan(packets), cycle);
  }

  double expected_traffic = model.ExpectedTrafficRateInMiBps(500);
  double measured_traffic = monitor.MeasuredTrafficRateInMiBps(500);

  XLS_VLOG(1) << "Packet " << num_packets;
  XLS_VLOG(1) << "Cycles " << cycle;
  XLS_VLOG(1) << "Expected Traffic " << expected_traffic;
  XLS_VLOG(1) << "Measured Traffic " << measured_traffic;

  EXPECT_EQ(bits_sent, monitor.MeasuredBitsSent());
  EXPECT_EQ(num_packets, monitor.MeasuredPacketCount());
  EXPECT_DOUBLE_EQ(measured_traffic, expected_traffic);
}

TEST(TrafficModelsTest, ReplayModelBuilderTest) {
  int64_t packet_size_bits = 128;

  ReplayTrafficModelBuilder builder(packet_size_bits, {0, 1, 2, 3, 4});

  builder.SetVCIndex(1).SetSourceIndex(10).SetDestinationIndex(3);

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ReplayTrafficModel> model,
                           builder.Build());

  EXPECT_EQ(packet_size_bits, model->GetPacketSizeInBits());
  EXPECT_EQ(model->GetVCIndex(), 1);
  EXPECT_EQ(model->GetSourceIndex(), 10);
  EXPECT_EQ(model->GetDestinationIndex(), 3);
}

TEST(TrafficModelsTest, ReplayModelClockCycleSortTest) {
  ReplayTrafficModel model(128);
  model.SetClockCycles({5, 4, 3, 2, 1, 0});

  EXPECT_EQ(model.GetClockCycles(), std::vector<int64_t>({0, 1, 2, 3, 4, 5}));
}

TEST(TrafficModelsTest, ReplayModelClockCycleClearTest) {
  ReplayTrafficModel model(128);
  model.SetClockCycles({5, 4, 3, 2, 1, 0});
  model.SetClockCycles({6, 7, 8, 9, 10});

  EXPECT_EQ(model.GetClockCycles(), std::vector<int64_t>({6, 7, 8, 9, 10}));
}

}  // namespace
}  // namespace xls::noc
