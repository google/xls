// Copyright 2020 The XLS Authors
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
#include <filesystem>  // NOLINT
#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/interpreter_proc_runtime.h"
#include "xls/interpreter/serial_proc_runtime.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

constexpr const char kIrPath[] = "xls/examples/matmul_4x4/matmul_4x4.ir";

using testing::Optional;

Value GetX0Value() {
  static int iter = 0;
  return {Value(UBits(111 + iter++, 32))};
}

Value GetX1Value() {
  static int iter = 0;
  return Value(UBits(2222 + iter++, 32));
}

Value GetX2Value() {
  static int iter = 0;
  return Value(UBits(33333 + iter++, 32));
}

Value GetX3Value() {
  static int iter = 0;
  return Value(UBits(444444 + iter++, 32));
}

// Straightforward test - can we correctly multiply a matrix against the
// identity matrix baked into the IR?
TEST(Matmul4x4Test, Works) {
  XLS_ASSERT_OK_AND_ASSIGN(std::filesystem::path ir_path,
                           GetXlsRunfilePath(kIrPath));
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir_text, GetFileContents(ir_path));
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));

  XLS_ASSERT_OK_AND_ASSIGN(auto interpreter,
                           CreateInterpreterSerialProcRuntime(package.get()));

  // Four input queues: (0,0)x, (1,0)x, (2,0)x, (3,0)x
  std::vector<std::unique_ptr<ChannelQueue>> rx_queues;
  XLS_ASSERT_OK_AND_ASSIGN(Channel * channel, package->GetChannel("c_0_0_x"));
  XLS_ASSERT_OK(interpreter->queue_manager().GetQueue(channel).AttachGenerator(
      []() { return GetX0Value(); }));
  XLS_ASSERT_OK_AND_ASSIGN(channel, package->GetChannel("c_1_0_x"));
  XLS_ASSERT_OK(interpreter->queue_manager().GetQueue(channel).AttachGenerator(
      []() { return GetX1Value(); }));
  XLS_ASSERT_OK_AND_ASSIGN(channel, package->GetChannel("c_2_0_x"));
  XLS_ASSERT_OK(interpreter->queue_manager().GetQueue(channel).AttachGenerator(
      []() { return GetX2Value(); }));
  XLS_ASSERT_OK_AND_ASSIGN(channel, package->GetChannel("c_3_0_x"));
  XLS_ASSERT_OK(interpreter->queue_manager().GetQueue(channel).AttachGenerator(
      []() { return GetX3Value(); }));

  // Now get the output queues.
  ChannelQueueManager& qm = interpreter->queue_manager();
  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * c30, qm.GetQueueByName("c_3_0_o"));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * c31, qm.GetQueueByName("c_3_1_o"));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * c32, qm.GetQueueByName("c_3_2_o"));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * c33, qm.GetQueueByName("c_3_3_o"));

  // Prop time: (0,0) -> (0,1) -> (0,2) -> (0,3) -> (1,3) -> (2,3) -> (3,3) = 7
  for (int i = 0; i < 7; i++) {
    XLS_EXPECT_OK(interpreter->Tick());
  }

  EXPECT_THAT(c30->Read(), Optional(Value(UBits(222, 32))));
  EXPECT_THAT(c31->Read(), Optional(Value(UBits(4444, 32))));
  EXPECT_THAT(c32->Read(), Optional(Value(UBits(66666, 32))));
  EXPECT_THAT(c33->Read(), Optional(Value(UBits(888888, 32))));
}

}  // namespace
}  // namespace xls
