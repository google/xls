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

#include "xls/ir/proc_elaboration.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
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
using ::testing::UnorderedElementsAre;

using ElaborationTest = IrTestBase;

MATCHER_P(ProcInstanceFor, value, "") { return arg->proc() == value; }

absl::StatusOr<Proc*> CreateLeafProc(std::string_view name,
                                     int64_t input_channel_count,
                                     Package* package) {
  TokenlessProcBuilder pb(NewStyleProc(), name, "tkn", package);
  for (int64_t i = 0; i < input_channel_count; ++i) {
    XLS_RETURN_IF_ERROR(pb.AddInputChannel(absl::StrFormat("leaf_ch%d", i),
                                           package->GetBitsType(32))
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
                         pb.AddInputChannel(absl::StrFormat("pass_ch%d", i),
                                            package->GetBitsType(32)));
    channels.push_back(channel_ref);
  }
  XLS_RETURN_IF_ERROR(pb.InstantiateProc(
      absl::StrFormat("%s_inst_%s", name, proc_to_instantiate->name()),
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
    XLS_RETURN_IF_ERROR(
        pb.InstantiateProc(absl::StrFormat("%s_inst%d", name, i),
                           procs_to_instantiate[i], channels));
  }
  return pb.Build({});
}

TEST_F(ElaborationTest, SingleProcNoChannels) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * proc, CreateLeafProc("foo", /*input_channel_count=*/0, p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elab,
                           ProcElaboration::Elaborate(proc));

  EXPECT_THAT(elab.top(), ProcInstanceFor(proc));
  EXPECT_TRUE(elab.top()->path().has_value());
  EXPECT_EQ(elab.top()->path()->ToString(), "foo");

  ASSERT_EQ(elab.proc_instances().size(), 1);
  EXPECT_THAT(elab.proc_instances().front(), ProcInstanceFor(proc));

  EXPECT_TRUE(elab.channel_instances().empty());

  EXPECT_EQ(elab.GetInstances(proc).size(), 1);
  EXPECT_EQ(elab.GetInstances(proc).front(), elab.proc_instances().front());

  EXPECT_EQ(elab.ToString(), "foo<>");
}

TEST_F(ElaborationTest, SingleProcMultipleChannels) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * proc, CreateLeafProc("foo", /*input_channel_count=*/3, p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elab,
                           ProcElaboration::Elaborate(proc));

  EXPECT_THAT(elab.top(), ProcInstanceFor(proc));
  EXPECT_FALSE(elab.top()->proc_instantiation().has_value());
  EXPECT_TRUE(elab.top()->path().has_value());
  EXPECT_EQ(elab.top()->path()->ToString(), "foo");

  ASSERT_EQ(elab.proc_instances().size(), 1);
  EXPECT_THAT(elab.proc_instances().front(), ProcInstanceFor(proc));

  EXPECT_EQ(elab.channel_instances().size(), 3);
  EXPECT_EQ(elab.channel_instances()[0]->channel->name(), "leaf_ch0");
  EXPECT_THAT(elab.top()->GetChannelInstance("leaf_ch0"),
              IsOkAndHolds(elab.channel_instances()[0]));
  EXPECT_EQ(elab.channel_instances()[1]->channel->name(), "leaf_ch1");
  EXPECT_THAT(elab.top()->GetChannelInstance("leaf_ch1"),
              IsOkAndHolds(elab.channel_instances()[1]));
  EXPECT_EQ(elab.channel_instances()[2]->channel->name(), "leaf_ch2");
  EXPECT_THAT(elab.top()->GetChannelInstance("leaf_ch2"),
              IsOkAndHolds(elab.channel_instances()[2]));

  EXPECT_EQ(elab.ToString(), "foo<leaf_ch0, leaf_ch1, leaf_ch2>");
}

TEST_F(ElaborationTest, ProcInstantiatingProc) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * leaf_proc,
      CreateLeafProc("leaf", /*input_channel_count=*/2, p.get()));
  TokenlessProcBuilder pb(NewStyleProc(), "top_proc", "tkn", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(ReceiveChannelReference * in_ch,
                           pb.AddInputChannel("in_ch", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelReferences the_channel_refs,
                           pb.AddChannel("the_ch", p->GetBitsType(32)));
  XLS_ASSERT_OK(pb.InstantiateProc("leaf_inst", leaf_proc,
                                   {the_channel_refs.receive_ref, in_ch}));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * top, pb.Build({}));

  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elab,
                           ProcElaboration::Elaborate(top));

  EXPECT_THAT(elab.top(), ProcInstanceFor(top));
  EXPECT_EQ(elab.top()->path()->ToString(), "top_proc");
  EXPECT_EQ(elab.top()->instantiated_procs().size(), 1);
  EXPECT_EQ(elab.top()->channels().size(), 1);

  EXPECT_THAT(elab.GetProcInstance("top_proc"), IsOkAndHolds(elab.top()));
  EXPECT_THAT(elab.GetChannelInstance("the_ch", "top_proc"),
              IsOkAndHolds(elab.top()->channels().front().get()));

  ProcInstance* leaf_instance = elab.top()->instantiated_procs().front().get();
  EXPECT_THAT(elab.GetProcInstance("top_proc::leaf_inst->leaf"),
              IsOkAndHolds(leaf_instance));

  XLS_ASSERT_OK_AND_ASSIGN(ChannelInstance * leaf_ch0_instance,
                           leaf_instance->GetChannelInstance("leaf_ch0"));
  EXPECT_EQ(leaf_ch0_instance->channel->name(), "the_ch");
  XLS_ASSERT_OK_AND_ASSIGN(ChannelInstance * leaf_ch1_instance,
                           leaf_instance->GetChannelInstance("leaf_ch1"));
  EXPECT_EQ(leaf_ch1_instance->channel->name(), "in_ch");

  EXPECT_THAT(
      elab.GetChannelInstance("leaf_ch0", leaf_instance->path().value()),
      IsOkAndHolds(leaf_instance->GetChannelInstance("leaf_ch0").value()));

  EXPECT_EQ(elab.ToString(), R"(top_proc<in_ch>
  chan the_ch
  leaf<leaf_ch0=the_ch, leaf_ch1=in_ch> [leaf_inst])");
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
  XLS_ASSERT_OK(pb.InstantiateProc("top_inst_1", proc1, {in_ch0, in_ch1}));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * top, pb.Build({}));

  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elab,
                           ProcElaboration::Elaborate(top));

  EXPECT_THAT(elab.procs(), UnorderedElementsAre(top, proc0, proc1, leaf_proc));

  XLS_ASSERT_OK_AND_ASSIGN(
      ProcInstance * leaf_inst,
      elab.GetProcInstance("top_proc::top_inst_1->proc1::proc1_inst_proc0->"
                           "proc0::proc0_inst_foo->foo"));
  EXPECT_THAT(leaf_inst, ProcInstanceFor(leaf_proc));

  EXPECT_THAT(elab.top(), ProcInstanceFor(top));
  EXPECT_EQ(elab.top()->path()->ToString(), "top_proc");
  EXPECT_EQ(elab.ToString(), R"(top_proc<in_ch0, in_ch1>
  proc1<pass_ch0=in_ch0, pass_ch1=in_ch1> [top_inst_1]
    proc0<pass_ch0=pass_ch0, pass_ch1=pass_ch1> [proc1_inst_proc0]
      foo<leaf_ch0=pass_ch0, leaf_ch1=pass_ch1> [proc0_inst_foo])");
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

  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elab,
                           ProcElaboration::Elaborate(top));

  EXPECT_THAT(elab.procs(), UnorderedElementsAre(top, middle_proc, leaf_proc));

  EXPECT_EQ(elab.GetInstances(leaf_proc).size(), 7);

  EXPECT_EQ(elab.GetInstances(leaf_proc)[0]->proc(), leaf_proc);
  EXPECT_EQ(elab.GetInstances(leaf_proc)[0]->path()->ToString(),
            "top_proc::top_proc_inst0->middle::middle_inst0->leaf");
  EXPECT_EQ(elab.GetInstances(leaf_proc)[1]->proc(), leaf_proc);
  EXPECT_EQ(elab.GetInstances(leaf_proc)[1]->path()->ToString(),
            "top_proc::top_proc_inst0->middle::middle_inst1->leaf");

  EXPECT_EQ(elab.GetInstances(leaf_proc)[2]->proc(), leaf_proc);
  EXPECT_EQ(elab.GetInstances(leaf_proc)[2]->path()->ToString(),
            "top_proc::top_proc_inst0->middle::middle_inst2->leaf");

  EXPECT_EQ(elab.GetInstances(leaf_proc)[3]->proc(), leaf_proc);
  EXPECT_EQ(elab.GetInstances(leaf_proc)[3]->path()->ToString(),
            "top_proc::top_proc_inst1->middle::middle_inst0->leaf");

  EXPECT_EQ(elab.GetInstances(leaf_proc)[6]->proc(), leaf_proc);
  EXPECT_EQ(elab.GetInstances(leaf_proc)[6]->path()->ToString(),
            "top_proc::top_proc_inst2->leaf");

  EXPECT_EQ(elab.GetInstances(middle_proc->channels()[0]).size(), 2);
  EXPECT_EQ(elab.GetInstances(middle_proc->channels()[1]).size(), 2);

  EXPECT_EQ(
      elab.GetInstancesOfChannelReference(leaf_proc->interface()[0]).size(), 7);
  EXPECT_EQ(
      elab.GetInstancesOfChannelReference(leaf_proc->interface()[1]).size(), 7);

  EXPECT_EQ(elab.GetInstancesOfChannelReference(leaf_proc->interface()[0])[0]
                ->path->ToString(),
            "top_proc::top_proc_inst0->middle");
  EXPECT_EQ(elab.GetInstancesOfChannelReference(leaf_proc->interface()[0])[1]
                ->path->ToString(),
            "top_proc::top_proc_inst0->middle");
  EXPECT_EQ(elab.GetInstancesOfChannelReference(leaf_proc->interface()[0])[6]
                ->path->ToString(),
            "top_proc");

  EXPECT_THAT(elab.top(), ProcInstanceFor(top));
  EXPECT_EQ(elab.top()->path()->ToString(), "top_proc");
  EXPECT_EQ(elab.ToString(), R"(top_proc<input0, input1>
  chan ch0
  chan ch1
  middle<input0=ch0, input1=ch1> [top_proc_inst0]
    chan ch0
    chan ch1
    leaf<leaf_ch0=ch0, leaf_ch1=ch1> [middle_inst0]
    leaf<leaf_ch0=ch0, leaf_ch1=ch1> [middle_inst1]
    leaf<leaf_ch0=ch0, leaf_ch1=ch1> [middle_inst2]
  middle<input0=ch0, input1=ch1> [top_proc_inst1]
    chan ch0
    chan ch1
    leaf<leaf_ch0=ch0, leaf_ch1=ch1> [middle_inst0]
    leaf<leaf_ch0=ch0, leaf_ch1=ch1> [middle_inst1]
    leaf<leaf_ch0=ch0, leaf_ch1=ch1> [middle_inst2]
  leaf<leaf_ch0=ch0, leaf_ch1=ch1> [top_proc_inst2])");
}

TEST_F(ElaborationTest, ProcInstantiatingProcWithNoChannels) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * leaf_proc,
      CreateLeafProc("foo", /*input_channel_count=*/0, p.get()));
  TokenlessProcBuilder pb(NewStyleProc(), "top_proc", "tkn", p.get());
  XLS_ASSERT_OK(pb.InstantiateProc("foo_inst", leaf_proc, {}));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * top, pb.Build({}));

  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elab,
                           ProcElaboration::Elaborate(top));

  EXPECT_THAT(elab.top(), ProcInstanceFor(top));
  EXPECT_EQ(elab.top()->path()->ToString(), "top_proc");
  EXPECT_EQ(elab.ToString(), R"(top_proc<>
  foo<> [foo_inst])");
}

TEST_F(ElaborationTest, ElaborateOldStyleProcWithWrongMethod) {
  auto p = CreatePackage();
  TokenlessProcBuilder pb("old_style_proc", "tkn", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Proc * top, pb.Build({}));

  EXPECT_THAT(
      ProcElaboration::Elaborate(top),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("Cannot elaborate old-style proc `old_style_proc`")));
}

TEST_F(ElaborationTest, ElaborateOldStyleProc) {
  auto p = CreatePackage();
  TokenlessProcBuilder pb("old_style_proc", "tkn", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Proc * top, pb.Build({}));

  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elab,
                           ProcElaboration::ElaborateOldStylePackage(p.get()));

  EXPECT_THAT(elab.procs(), UnorderedElementsAre(top));

  ASSERT_EQ(elab.proc_instances().size(), 1);
  absl::Span<ProcInstance* const> proc_instances = elab.GetInstances(top);
  EXPECT_EQ(proc_instances, elab.proc_instances());
  EXPECT_EQ(top, proc_instances.front()->proc());

  EXPECT_TRUE(elab.channel_instances().empty());
}

TEST_F(ElaborationTest, ElaborateOldStyleMultiprocNetwork) {
  Package p("package");

  Type* u32 = p.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch1,
      p.CreateStreamingChannel("ch1", ChannelOps::kSendReceive, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch2,
      p.CreateStreamingChannel("ch2", ChannelOps::kSendReceive, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch3,
      p.CreateStreamingChannel("ch3", ChannelOps::kSendReceive, u32));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc1,
                           TokenlessProcBuilder("proc1", "tkn", &p).Build({}));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc2,
                           TokenlessProcBuilder("proc2", "tkn", &p).Build({}));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc3,
                           TokenlessProcBuilder("proc3", "tkn", &p).Build({}));

  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elab,
                           ProcElaboration::ElaborateOldStylePackage(&p));

  EXPECT_THAT(elab.procs(), UnorderedElementsAre(proc1, proc2, proc3));

  ASSERT_EQ(elab.proc_instances().size(), 3);
  EXPECT_EQ(elab.GetInstances(proc1).size(), 1);
  EXPECT_EQ(elab.GetInstances(proc1).front()->proc(), proc1);
  EXPECT_EQ(elab.GetInstances(proc2).size(), 1);
  EXPECT_EQ(elab.GetInstances(proc2).front()->proc(), proc2);
  EXPECT_EQ(elab.GetInstances(proc3).size(), 1);
  EXPECT_EQ(elab.GetInstances(proc3).front()->proc(), proc3);

  EXPECT_EQ(elab.channel_instances().size(), 3);
  EXPECT_EQ(elab.GetInstances(ch1).size(), 1);
  EXPECT_EQ(elab.GetInstances(ch1).front()->channel, ch1);
  EXPECT_EQ(elab.GetInstances(ch2).size(), 1);
  EXPECT_EQ(elab.GetInstances(ch2).front()->channel, ch2);
  EXPECT_EQ(elab.GetInstances(ch3).size(), 1);
  EXPECT_EQ(elab.GetInstances(ch3).front()->channel, ch3);

  EXPECT_EQ(elab.ToString(), R"(proc1
proc2
proc3)");
}

}  // namespace
}  // namespace xls
