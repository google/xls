// Copyright 2023 The XLS Authors
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

#include "xls/ir/elaboration.h"

#include <cstdint>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_instantiation.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::HasSubstr;

class ElaborationTest : public IrTestBase {};

static InstantiationPath MakeInstantiationPath(
    Proc* top, absl::Span<const std::string_view> instantiations) {
  InstantiationPath path;
  path.top = top;
  Proc* proc = top;
  for (std::string_view instantiation_name : instantiations) {
    ProcInstantiation* instantiation =
        proc->GetProcInstantiation(instantiation_name).value();
    path.path.push_back(instantiation);
    proc = instantiation->proc();
  }
  return path;
}

absl::StatusOr<Proc*> CreateLeafProc(std::string_view name,
                                     int64_t input_channel_count,
                                     Package* package) {
  TokenlessProcBuilder pb(NewStyleProc(), name, "tkn", package);
  for (int64_t i = 0; i < input_channel_count; ++i) {
    XLS_RETURN_IF_ERROR(
        pb.AddInputChannel(absl::StrFormat("ch%d", i), package->GetBitsType(32))
            .status());
  }
  return pb.Build({});
}

absl::StatusOr<Proc*> CreatePassThroughProc(std::string_view name,
                                            int64_t input_channel_count,
                                            Proc* proc_to_instantiate,
                                            Package* package) {
  TokenlessProcBuilder pb(NewStyleProc(), name, "tkn", package);
  std::vector<ChannelReference*> channels;
  for (int64_t i = 0; i < input_channel_count; ++i) {
    XLS_ASSIGN_OR_RETURN(ChannelReference * channel_ref,
                         pb.AddInputChannel(absl::StrFormat("ch%d", i),
                                            package->GetBitsType(32)));
    channels.push_back(channel_ref);
  }
  XLS_RETURN_IF_ERROR(pb.InstantiateProc(
      absl::StrFormat("inst_%s", proc_to_instantiate->name()),
      proc_to_instantiate, channels));
  return pb.Build({});
}

absl::StatusOr<Proc*> CreateMultipleInstantiationProc(
    std::string_view name, int64_t input_channel_count,
    int64_t instantiated_channel_count,
    absl::Span<Proc* const> procs_to_instantiate, Package* package) {
  TokenlessProcBuilder pb(NewStyleProc(), name, "tkn", package);
  for (int64_t i = 0; i < input_channel_count; ++i) {
    XLS_RETURN_IF_ERROR(pb.AddInputChannel(absl::StrFormat("input%d", i),
                                           package->GetBitsType(32))
                            .status());
  }
  std::vector<ChannelReference*> channels;
  for (int64_t i = 0; i < instantiated_channel_count; ++i) {
    XLS_ASSIGN_OR_RETURN(
        ChannelReferences channel_refs,
        pb.AddChannel(absl::StrFormat("ch%d", i), package->GetBitsType(32)));
    channels.push_back(channel_refs.receive_ref);
  }
  for (int64_t i = 0; i < procs_to_instantiate.size(); ++i) {
    XLS_RETURN_IF_ERROR(pb.InstantiateProc(absl::StrFormat("inst%d", i),
                                           procs_to_instantiate[i], channels));
  }
  return pb.Build({});
}

TEST_F(ElaborationTest, SingleProcNoChannels) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * proc, CreateLeafProc("foo", /*input_channel_count=*/0, p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(Elaboration elab, Elaboration::Elaborate(proc));

  EXPECT_EQ(elab.top().proc(), proc);
  EXPECT_EQ(elab.top().path().ToString(), "foo");
  EXPECT_TRUE(elab.top().interface().empty());

  EXPECT_EQ(elab.ToString(), "foo<>");
}

TEST_F(ElaborationTest, SingleProcMultipleChannels) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * proc, CreateLeafProc("foo", /*input_channel_count=*/3, p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(Elaboration elab, Elaboration::Elaborate(proc));

  EXPECT_EQ(elab.top().proc(), proc);
  EXPECT_FALSE(elab.top().proc_instantiation().has_value());
  EXPECT_EQ(elab.top().path().ToString(), "foo");
  EXPECT_EQ(elab.top().interface().size(), 3);
  EXPECT_EQ(elab.top().interface()[0]->channel->name(), "ch0");
  EXPECT_EQ(elab.top().interface()[0]->path.ToString(), "foo");
  EXPECT_EQ(elab.top().interface()[1]->channel->name(), "ch1");
  EXPECT_EQ(elab.top().interface()[1]->path.ToString(), "foo");
  EXPECT_EQ(elab.top().interface()[2]->channel->name(), "ch2");
  EXPECT_EQ(elab.top().interface()[2]->path.ToString(), "foo");
  EXPECT_EQ(elab.ToString(), "foo<ch0, ch1, ch2>");
}

TEST_F(ElaborationTest, ProcInstantiatingProc) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * leaf_proc,
      CreateLeafProc("leaf", /*input_channel_count=*/2, p.get()));
  TokenlessProcBuilder pb(NewStyleProc(), "top_proc", "tkn", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(ReceiveChannelReference * in_ch,
                           pb.AddInputChannel("in_ch", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelReferences channel_refs,
                           pb.AddChannel("the_ch", p->GetBitsType(32)));
  XLS_ASSERT_OK(pb.InstantiateProc("leaf_inst", leaf_proc,
                                   {channel_refs.receive_ref, in_ch}));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * top, pb.Build({}));

  XLS_ASSERT_OK_AND_ASSIGN(Elaboration elab, Elaboration::Elaborate(top));

  EXPECT_EQ(elab.top().proc(), top);
  EXPECT_EQ(elab.top().path().ToString(), "top_proc");
  EXPECT_EQ(elab.top().instantiated_procs().size(), 1);
  EXPECT_EQ(elab.top().channels().size(), 1);

  EXPECT_THAT(elab.GetProcInstance(MakeInstantiationPath(top, {})),
              IsOkAndHolds(&elab.top()));
  EXPECT_THAT(elab.GetChannelInstance("the_ch", MakeInstantiationPath(top, {})),
              IsOkAndHolds(elab.top().channels().front().get()));

  ProcInstance* leaf_instance = elab.top().instantiated_procs().front().get();
  EXPECT_THAT(elab.GetProcInstance(MakeInstantiationPath(top, {"leaf_inst"})),
              IsOkAndHolds(leaf_instance));

  EXPECT_EQ(elab.ToString(), R"(top_proc<in_ch>
  chan the_ch
  leaf<ch0, ch1> [leaf_inst])");
}

TEST_F(ElaborationTest, ProcInstantiatingProcInstantiatedProcEtc) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * leaf_proc,
      CreateLeafProc("foo", /*input_channel_count=*/2, p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * proc0, CreatePassThroughProc("proc0", /*input_channel_count=*/2,
                                          leaf_proc, p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * proc1, CreatePassThroughProc("proc1", /*input_channel_count=*/2,
                                          proc0, p.get()));

  TokenlessProcBuilder pb(NewStyleProc(), "top_proc", "tkn", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(ReceiveChannelReference * in_ch0,
                           pb.AddInputChannel("in_ch0", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(ReceiveChannelReference * in_ch1,
                           pb.AddInputChannel("in_ch1", p->GetBitsType(32)));
  XLS_ASSERT_OK(pb.InstantiateProc("inst1", proc1, {in_ch0, in_ch1}));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * top, pb.Build({}));

  XLS_ASSERT_OK_AND_ASSIGN(Elaboration elab, Elaboration::Elaborate(top));

  XLS_ASSERT_OK_AND_ASSIGN(ProcInstance * leaf_inst,
                           elab.GetProcInstance(MakeInstantiationPath(
                               top, {"inst1", "inst_proc0", "inst_foo"})));
  EXPECT_EQ(leaf_inst->proc(), leaf_proc);

  EXPECT_EQ(elab.top().proc(), top);
  EXPECT_EQ(elab.top().path().ToString(), "top_proc");
  EXPECT_EQ(elab.ToString(), R"(top_proc<in_ch0, in_ch1>
  proc1<ch0, ch1> [inst1]
    proc0<ch0, ch1> [inst_proc0]
      foo<ch0, ch1> [inst_foo])");
}

TEST_F(ElaborationTest, MultipleInstantiations) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * leaf_proc,
      CreateLeafProc("leaf", /*input_channel_count=*/2, p.get()));

  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * middle_proc,
      CreateMultipleInstantiationProc(
          "middle", /*input_channel_count=*/2, /*instantiated_channel_count=*/2,
          {leaf_proc, leaf_proc, leaf_proc}, p.get()));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * top,
                           CreateMultipleInstantiationProc(
                               "top_proc", /*input_channel_count=*/2,
                               /*instantiated_channel_count=*/2,
                               {middle_proc, middle_proc, leaf_proc}, p.get()));

  XLS_ASSERT_OK_AND_ASSIGN(Elaboration elab, Elaboration::Elaborate(top));

  EXPECT_EQ(elab.top().proc(), top);
  EXPECT_EQ(elab.top().path().ToString(), "top_proc");
  EXPECT_EQ(elab.ToString(), R"(top_proc<input0, input1>
  chan ch0
  chan ch1
  middle<input0, input1> [inst0]
    chan ch0
    chan ch1
    leaf<ch0, ch1> [inst0]
    leaf<ch0, ch1> [inst1]
    leaf<ch0, ch1> [inst2]
  middle<input0, input1> [inst1]
    chan ch0
    chan ch1
    leaf<ch0, ch1> [inst0]
    leaf<ch0, ch1> [inst1]
    leaf<ch0, ch1> [inst2]
  leaf<ch0, ch1> [inst2])");
}

TEST_F(ElaborationTest, ProcInstantiatingProcWithNoChannels) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * leaf_proc,
      CreateLeafProc("foo", /*input_channel_count=*/0, p.get()));
  TokenlessProcBuilder pb(NewStyleProc(), "top_proc", "tkn", p.get());
  XLS_ASSERT_OK(pb.InstantiateProc("foo_inst", leaf_proc, {}));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * top, pb.Build({}));

  XLS_ASSERT_OK_AND_ASSIGN(Elaboration elab, Elaboration::Elaborate(top));

  EXPECT_EQ(elab.top().proc(), top);
  EXPECT_EQ(elab.top().path().ToString(), "top_proc");
  EXPECT_EQ(elab.ToString(), R"(top_proc<>
  foo<> [foo_inst])");
}

TEST_F(ElaborationTest, ElaborateOldStyleProc) {
  auto p = CreatePackage();
  TokenlessProcBuilder pb("old_style_proc", "tkn", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Proc * top, pb.Build({}));

  EXPECT_THAT(
      Elaboration::Elaborate(top),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("Cannot elaborate old-style proc `old_style_proc`")));
}

}  // namespace
}  // namespace xls
