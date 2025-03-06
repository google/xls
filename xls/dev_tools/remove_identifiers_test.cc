// Copyright 2024 The XLS Authors
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

#include "xls/dev_tools/remove_identifiers.h"

#include <array>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

static constexpr std::string_view kUninteresting = "uninteresting";
class RemoveIdentifersTest : public IrTestBase {};
TEST_F(RemoveIdentifersTest, BasicProc) {
  static constexpr std::array<std::string_view, 10> kSecrets{
      "the_answer_proc", "secret_handshake", "secret_tunnel", "astounding",
      "Unbelievable",    "surprising",       "revealing",     "nxt_val",
      "nxt_tok",         "nxt_tok_val"};
  auto p = CreatePackage();
  ProcBuilder pb(NewStyleProc{}, "the_answer_proc", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto orig_chan,
                           pb.AddChannel("secret_tunnel", p->GetBitsType(32)));
  auto tok = pb.StateElement("secret_handshake", Value::Token());
  auto start_param = pb.StateElement("astounding", Value(UBits(42, 32)));
  auto recv = pb.Receive(orig_chan.receive_interface, tok, SourceInfo(),
                         "Unbelievable");
  auto next_param =
      pb.Add(start_param, pb.TupleIndex(recv, 1, SourceInfo(), "nxt_val"));
  auto out_tok = pb.Send(orig_chan.send_interface,
                         pb.TupleIndex(recv, 0, SourceInfo(), "nxt_tok"),
                         next_param, SourceInfo(), "surprising");
  pb.Next(start_param, next_param, /*pred=*/std::nullopt, SourceInfo(),
          "Revealing");
  pb.Next(tok, out_tok, /*pred=*/std::nullopt, SourceInfo(), "nxt_tok_val");
  XLS_ASSERT_OK_AND_ASSIGN(auto* orig_proc, pb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      auto stripped,
      StripPackage(p.get(), StripOptions{
                                .new_package_name = std::string(kUninteresting),
                            }));
  EXPECT_EQ(stripped->name(), kUninteresting);
  EXPECT_THAT(stripped->procs(), testing::SizeIs(1));
  EXPECT_NE(stripped->procs()[0]->name(), orig_proc->name());
  EXPECT_THAT(stripped->procs()[0]->StateElements(),
              testing::AllOf(testing::SizeIs(orig_proc->StateElements().size()),
                             testing::Each(testing::Not(testing::AnyOfArray(
                                 orig_proc->StateElements())))));
  ASSERT_THAT(stripped->procs()[0]->channels(), testing::SizeIs(1));
  EXPECT_NE(stripped->procs()[0]->channels()[0]->name(),
            orig_proc->channels()[0]->name());
  std::vector<std::string> orig_names;
  absl::c_transform(orig_proc->nodes(), std::back_inserter(orig_names),
                    [](Node* n) { return n->GetName(); });
  absl::c_transform(orig_proc->channels(), std::back_inserter(orig_names),
                    [](Channel* c) { return std::string(c->name()); });
  EXPECT_THAT(stripped->procs()[0]->channels(),
              testing::Each(testing::ResultOf(
                  [](Channel* c) { return std::string(c->name()); },
                  testing::Not(testing::AnyOfArray(orig_names)))));
  EXPECT_THAT(stripped->procs()[0]->nodes(),
              testing::Each(testing::ResultOf(
                  [](Node* n) { return n->GetName(); },
                  testing::Not(testing::AnyOfArray(orig_names)))));
  for (auto secret : kSecrets) {
    EXPECT_THAT(stripped->DumpIr(),
                testing::Not(testing::ContainsRegex(secret)));
  }
}

TEST_F(RemoveIdentifersTest, BasicFunction) {
  static constexpr std::array<std::string_view, 4> kSecrets{
      "the_answer_func", "foo", "bar", "ultimate_question"};
  auto p = CreatePackage();
  FunctionBuilder fb("the_answer_func", p.get());
  fb.Add(fb.Param("foo", p->GetBitsType(32)),
         fb.Param("bar", p->GetBitsType(32)), SourceInfo(),
         "ultimate_question");
  XLS_ASSERT_OK_AND_ASSIGN(auto* orig_func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      auto stripped,
      StripPackage(p.get(), StripOptions{
                                .new_package_name = std::string(kUninteresting),
                            }));

  EXPECT_EQ(stripped->name(), kUninteresting);
  EXPECT_THAT(stripped->functions(), testing::SizeIs(1));
  EXPECT_NE(stripped->functions()[0]->name(), orig_func->name());
  EXPECT_THAT(stripped->functions()[0]->params(),
              testing::AllOf(testing::SizeIs(orig_func->params().size()),
                             testing::Each(testing::Not(
                                 testing::AnyOfArray(orig_func->params())))));
  std::vector<std::string> orig_names;
  absl::c_transform(orig_func->nodes(), std::back_inserter(orig_names),
                    [](Node* n) { return n->GetName(); });
  EXPECT_THAT(stripped->functions()[0]->nodes(),
              testing::Each(testing::ResultOf(
                  [](Node* n) { return n->GetName(); },
                  testing::Not(testing::AnyOfArray(orig_names)))));
  for (auto secret : kSecrets) {
    EXPECT_THAT(stripped->DumpIr(),
                testing::Not(testing::ContainsRegex(secret)));
  }
}

TEST_F(RemoveIdentifersTest, BasicBlock) {
  static constexpr std::array<std::string_view, 8> kSecrets{
      "the_secret_recipe", "kitchen_timer", "flour", "water", "Mix",
      "start_rest",        "start_cutting", "pasta"};
  auto p = CreatePackage();
  BlockBuilder bb("the_secret_recipe", p.get());
  XLS_ASSERT_OK(bb.AddClockPort("kitchen_timer"));
  auto i1 = bb.InputPort("flour", p->GetBitsType(10));
  auto i2 = bb.InputPort("water", p->GetBitsType(10));
  auto mix = bb.Add(i1, i2, SourceInfo(), "Mix");
  auto rest = bb.InsertRegister("start_rest", mix);
  auto to_cut = bb.InsertRegister("start_cutting", rest);
  bb.OutputPort("pasta", bb.Tuple({
                             bb.BitSlice(to_cut, 0, 2, SourceInfo(), "S1"),
                             bb.BitSlice(to_cut, 2, 2, SourceInfo(), "S2"),
                             bb.BitSlice(to_cut, 4, 2, SourceInfo(), "S3"),
                             bb.BitSlice(to_cut, 6, 2, SourceInfo(), "S4"),
                             bb.BitSlice(to_cut, 8, 2, SourceInfo(), "S5"),
                         }));
  XLS_ASSERT_OK_AND_ASSIGN(auto* orig_block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      auto stripped,
      StripPackage(p.get(), StripOptions{
                                .new_package_name = std::string(kUninteresting),
                            }));

  EXPECT_EQ(stripped->name(), kUninteresting);
  EXPECT_THAT(stripped->blocks(), testing::SizeIs(1));
  EXPECT_NE(stripped->blocks()[0]->name(), orig_block->name());
  EXPECT_THAT(
      stripped->blocks()[0]->GetInputPorts(),
      testing::AllOf(testing::SizeIs(orig_block->GetInputPorts().size()),
                     testing::Each(testing::Not(
                         testing::AnyOfArray(orig_block->GetInputPorts())))));
  std::vector<std::string> orig_names;
  absl::c_transform(orig_block->nodes(), std::back_inserter(orig_names),
                    [](Node* n) { return n->GetName(); });
  absl::c_transform(orig_block->GetRegisters(), std::back_inserter(orig_names),
                    [](Register* n) { return n->name(); });
  orig_names.push_back(orig_block->GetClockPort()->name);
  absl::c_transform(orig_block->GetInputPorts(), std::back_inserter(orig_names),
                    [](InputPort* n) { return std::string(n->name()); });
  absl::c_transform(orig_block->GetOutputPorts(),
                    std::back_inserter(orig_names),
                    [](OutputPort* n) { return std::string(n->name()); });
  EXPECT_THAT(stripped->blocks()[0]->nodes(),
              testing::Each(testing::ResultOf(
                  [](Node* n) { return n->GetName(); },
                  testing::Not(testing::AnyOfArray(orig_names)))));
  EXPECT_THAT(stripped->blocks()[0]->GetInputPorts(),
              testing::Each(testing::ResultOf(
                  [](InputPort* n) { return n->name(); },
                  testing::Not(testing::AnyOfArray(orig_names)))));
  EXPECT_THAT(stripped->blocks()[0]->GetOutputPorts(),
              testing::Each(testing::ResultOf(
                  [](OutputPort* n) { return n->name(); },
                  testing::Not(testing::AnyOfArray(orig_names)))));
  EXPECT_THAT(stripped->blocks()[0]->GetRegisters(),
              testing::Each(testing::ResultOf(
                  [](Register* n) { return n->name(); },
                  testing::Not(testing::AnyOfArray(orig_names)))));
  for (auto secret : kSecrets) {
    EXPECT_THAT(stripped->DumpIr(),
                testing::Not(testing::ContainsRegex(secret)));
  }
}

}  // namespace
}  // namespace xls
