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

#include "xls/noc/simulation/packetizer.h"

#include <string>

#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/noc/simulation/flit.h"

namespace xls::noc {
namespace {

TEST(PacketizerTest, BuildPacket) {
  XLS_ASSERT_OK_AND_ASSIGN(DataPacket packet, DataPacketBuilder()
                                                  .Valid(true)
                                                  .SourceIndex(5)
                                                  .DestinationIndex(1)
                                                  .VirtualChannel(10)
                                                  .ZeroedData(16)
                                                  .Build());

  EXPECT_TRUE(packet.valid);
  EXPECT_EQ(packet.destination_index, 1);
  EXPECT_EQ(packet.vc, 10);
  EXPECT_EQ(packet.data, Bits(16));

  std::string str_rep = absl::StrFormat("%s", packet);
  EXPECT_EQ(str_rep,
            "{valid: 1, source_index: 5, dest_index: 1, vc: 10, data: 0}");
}

TEST(PacketizerTest, SendSingleFlit) {
  XLS_ASSERT_OK_AND_ASSIGN(DataPacket packet,
                           DataPacketBuilder()
                               .Valid(true)
                               .SourceIndex(5)
                               .DestinationIndex(1)
                               .VirtualChannel(10)
                               .Data(UBits(0b1010'0110'1100'0011, 16))
                               .Build());

  // Test packet to flit behavior.
  DePacketizer depacketizer(16, 3, 16);

  EXPECT_TRUE(depacketizer.IsIdle());

  XLS_ASSERT_OK(depacketizer.AcceptNewPacket(packet));
  EXPECT_FALSE(depacketizer.IsIdle());
  EXPECT_EQ(depacketizer.GetFlitPayloadBitCount(), 16);
  EXPECT_EQ(depacketizer.GetSourceIndexBitCount(), 3);
  EXPECT_EQ(depacketizer.GetMaxPacketBitCount(), 16);

  XLS_ASSERT_OK_AND_ASSIGN(DataFlit flit0, depacketizer.ComputeNextFlit());
  EXPECT_TRUE(depacketizer.IsIdle());

  EXPECT_EQ(flit0.destination_index, packet.destination_index);
  EXPECT_EQ(flit0.source_index, packet.source_index);
  EXPECT_EQ(flit0.vc, packet.vc);
  EXPECT_EQ(flit0.type, FlitType::kTail);

  EXPECT_EQ(flit0.data, UBits(0b1010'0110'1100'0011, 16));

  // Test flit to packet behavior.
  Packetizer packetizer(16, 3, 16, 1);
  XLS_ASSERT_OK(packetizer.AcceptNewFlit(flit0));

  absl::Span<const DataPacket> received_packets = packetizer.GetPackets();
  ASSERT_EQ(received_packets.size(), 1);

  DataPacket recv_packet = received_packets.at(0);

  EXPECT_TRUE(recv_packet.valid);
  EXPECT_EQ(recv_packet.destination_index, packet.destination_index);
  EXPECT_EQ(recv_packet.source_index, packet.source_index);
  EXPECT_EQ(recv_packet.vc, packet.vc);
  EXPECT_EQ(recv_packet.data, packet.data);

  XLS_ASSERT_OK(packetizer.AcceptNewFlit(flit0));
  ASSERT_EQ(packetizer.GetPackets().size(), 2);
}

TEST(PacketizerTest, SendMultiplePhit) {
  XLS_ASSERT_OK_AND_ASSIGN(DataPacket packet,
                           DataPacketBuilder()
                               .Valid(true)
                               .SourceIndex(5)
                               .DestinationIndex(1)
                               .VirtualChannel(10)
                               .Data(UBits(0b1010'0110'1100'0011, 16))
                               .Build());

  // Test packet to flit behavior.
  DePacketizer depacketizer(16, 3, 128);

  EXPECT_TRUE(depacketizer.IsIdle());
  EXPECT_EQ(depacketizer.GetFlitPayloadBitCount(), 16);
  EXPECT_EQ(depacketizer.GetSourceIndexBitCount(), 3);
  EXPECT_EQ(depacketizer.GetMaxPacketBitCount(), 128);

  XLS_ASSERT_OK(depacketizer.AcceptNewPacket(packet));
  EXPECT_FALSE(depacketizer.IsIdle());

  XLS_ASSERT_OK_AND_ASSIGN(DataFlit flit0, depacketizer.ComputeNextFlit());
  EXPECT_FALSE(depacketizer.IsIdle());

  EXPECT_EQ(flit0.destination_index, packet.destination_index);
  EXPECT_EQ(flit0.source_index, packet.source_index);
  EXPECT_EQ(flit0.vc, packet.vc);
  EXPECT_EQ(flit0.type, FlitType::kHead);

  EXPECT_EQ(flit0.data.Slice(0, flit0.data_bit_count),
            UBits(0b0'0110'1100'0011, 13));

  XLS_ASSERT_OK_AND_ASSIGN(DataFlit flit1, depacketizer.ComputeNextFlit());
  EXPECT_TRUE(depacketizer.IsIdle());

  EXPECT_EQ(flit1.destination_index, packet.destination_index);
  EXPECT_EQ(flit1.source_index, packet.source_index);
  EXPECT_EQ(flit1.vc, packet.vc);
  EXPECT_EQ(flit1.type, FlitType::kTail);

  EXPECT_EQ(flit1.data.Slice(0, flit1.data_bit_count), UBits(0b101, 3));

  // Test flit to packet behavior.
  Packetizer packetizer(16, 3, 128, 1);
  XLS_ASSERT_OK(packetizer.AcceptNewFlit(flit0));

  EXPECT_EQ(packetizer.GetPackets().size(), 0);
  EXPECT_EQ(packetizer.PartialPacketCount(), 1);

  XLS_ASSERT_OK(packetizer.AcceptNewFlit(flit1));
  EXPECT_EQ(packetizer.GetPackets().size(), 1);
  EXPECT_EQ(packetizer.PartialPacketCount(), 0);

  absl::Span<const DataPacket> received_packets = packetizer.GetPackets();
  DataPacket recv_packet = received_packets.at(0);

  EXPECT_TRUE(recv_packet.valid);
  EXPECT_EQ(recv_packet.destination_index, packet.destination_index);
  EXPECT_EQ(recv_packet.source_index, packet.source_index);
  EXPECT_EQ(recv_packet.vc, packet.vc);
  EXPECT_EQ(recv_packet.data, packet.data);
}

}  // namespace
}  // namespace xls::noc
