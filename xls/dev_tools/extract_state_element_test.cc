// Copyright 2025 The XLS Authors
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

#include "xls/dev_tools/extract_state_element.h"

#include <filesystem>
#include <string_view>

#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "xls/common/golden_files.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

constexpr std::string_view kTestName = "extract_state_element_test";
constexpr std::string_view kTestdataPath = "xls/dev_tools/testdata";

static std::filesystem::path TestFilePath(std::string_view test_name) {
  return absl::StrFormat("%s/%s_%s.ir", kTestdataPath, kTestName, test_name);
}

class ExtractStateElementTest : public IrTestBase {};
TEST_F(ExtractStateElementTest, NoSendState) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * chan,
      p->CreateStreamingChannel("inp_chan", ChannelOps::kReceiveOnly,
                                p->GetBitsType(32)));
  ProcBuilder pb(TestName(), p.get());
  BValue a = pb.StateElement("a", UBits(1, 32));
  BValue b = pb.StateElement("b", UBits(1, 32));
  BValue c = pb.StateElement("c", UBits(1, 32));
  BValue d = pb.StateElement("d", UBits(1, 32));
  pb.Next(a, pb.Add(a, b));
  pb.Next(b, pb.Add(c, a));
  pb.Next(c, pb.Add(a, d));
  pb.Next(d,
          pb.Add(pb.TupleIndex(pb.Receive(chan, pb.Literal(Value::Token())), 1),
                 c));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto new_pkg,
                           ExtractStateElementsInNewPackage(
                               proc,
                               {a.node()->As<StateRead>()->state_element(),
                                b.node()->As<StateRead>()->state_element()},
                               /*send_state_values=*/false));
  ExpectEqualToGoldenFile(TestFilePath(TestName()), new_pkg->DumpIr());
}

TEST_F(ExtractStateElementTest, SendState) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * chan,
      p->CreateStreamingChannel("inp_chan", ChannelOps::kReceiveOnly,
                                p->GetBitsType(32)));
  ProcBuilder pb(TestName(), p.get());
  BValue a = pb.StateElement("a", UBits(1, 32));
  BValue b = pb.StateElement("b", UBits(1, 32));
  BValue c = pb.StateElement("c", UBits(1, 32));
  BValue d = pb.StateElement("d", UBits(1, 32));
  pb.Next(a, pb.Add(a, b));
  pb.Next(b, pb.Add(c, a));
  pb.Next(c, pb.Add(a, d));
  pb.Next(d,
          pb.Add(pb.TupleIndex(pb.Receive(chan, pb.Literal(Value::Token())), 1),
                 c));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto new_pkg,
                           ExtractStateElementsInNewPackage(
                               proc,
                               {a.node()->As<StateRead>()->state_element(),
                                b.node()->As<StateRead>()->state_element()},
                               /*send_state_values=*/true));
  RecordProperty("ir", new_pkg->DumpIr());
  ExpectEqualToGoldenFile(TestFilePath(TestName()), new_pkg->DumpIr());
}

}  // namespace
}  // namespace xls
