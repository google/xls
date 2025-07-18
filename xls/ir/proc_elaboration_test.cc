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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_instantiation.h"
#include "ortools/graph/graph_io.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

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
  std::vector<ChannelInterface*> channels;
  for (int64_t i = 0; i < input_channel_count; ++i) {
    XLS_ASSIGN_OR_RETURN(ChannelInterface * channel_ref,
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
  std::vector<ChannelInterface*> channels;
  for (int64_t i = 0; i < instantiated_channel_count; ++i) {
    XLS_ASSIGN_OR_RETURN(
        ChannelWithInterfaces channel_refs,
        pb.AddChannel(absl::StrFormat("ch%d", i), package->GetBitsType(32)));
    channels.push_back(channel_refs.receive_interface);
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
  XLS_ASSERT_OK_AND_ASSIGN(ReceiveChannelInterface * in_ch,
                           pb.AddInputChannel("in_ch", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelWithInterfaces the_channel_refs,
                           pb.AddChannel("the_ch", p->GetBitsType(32)));
  XLS_ASSERT_OK(pb.InstantiateProc(
      "leaf_inst", leaf_proc, {the_channel_refs.receive_interface, in_ch}));
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

  EXPECT_TRUE(elab.IsTopInterfaceChannel(
      elab.GetChannelInstance("in_ch", "top_proc").value()));
  EXPECT_FALSE(elab.IsTopInterfaceChannel(
      elab.GetChannelInstance("the_ch", "top_proc").value()));

  ProcInstance* leaf_instance = elab.top()->instantiated_procs().front().get();
  EXPECT_THAT(elab.GetProcInstance("top_proc::leaf_inst->leaf"),
              IsOkAndHolds(leaf_instance));

  XLS_ASSERT_OK_AND_ASSIGN(ChannelInstance * leaf_ch0_instance,
                           leaf_instance->GetChannelInstance("leaf_ch0"));
  EXPECT_EQ(leaf_ch0_instance->channel->name(), "the_ch");
  EXPECT_FALSE(elab.IsTopInterfaceChannel(leaf_ch0_instance));

  XLS_ASSERT_OK_AND_ASSIGN(ChannelInstance * leaf_ch1_instance,
                           leaf_instance->GetChannelInstance("leaf_ch1"));
  EXPECT_EQ(leaf_ch1_instance->channel->name(), "in_ch");
  EXPECT_TRUE(elab.IsTopInterfaceChannel(leaf_ch1_instance));

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
  XLS_ASSERT_OK_AND_ASSIGN(ReceiveChannelInterface * in_ch0,
                           pb.AddInputChannel("in_ch0", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(ReceiveChannelInterface * in_ch1,
                           pb.AddInputChannel("in_ch1", p->GetBitsType(32)));
  XLS_ASSERT_OK(pb.InstantiateProc("top_inst_1", proc1, {in_ch0, in_ch1}));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * top, pb.Build({}));

  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elab,
                           ProcElaboration::Elaborate(top));

  EXPECT_THAT(elab.procs(), ElementsAre(top, proc1, proc0, leaf_proc));

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

  EXPECT_THAT(elab.procs(), ElementsAre(top, middle_proc, leaf_proc));

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
      elab.GetInstancesOfChannelInterface(leaf_proc->interface()[0]).size(), 7);
  EXPECT_EQ(
      elab.GetInstancesOfChannelInterface(leaf_proc->interface()[1]).size(), 7);

  EXPECT_EQ(elab.GetInstancesOfChannelInterface(leaf_proc->interface()[0])[0]
                ->path->ToString(),
            "top_proc::top_proc_inst0->middle");
  EXPECT_EQ(elab.GetInstancesOfChannelInterface(leaf_proc->interface()[0])[1]
                ->path->ToString(),
            "top_proc::top_proc_inst0->middle");
  EXPECT_EQ(elab.GetInstancesOfChannelInterface(leaf_proc->interface()[0])[6]
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

  EXPECT_THAT(elab.procs(), ElementsAre(top));

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

  EXPECT_THAT(elab.procs(), ElementsAre(proc1, proc2, proc3));

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

// TODO(allight): Use proc_network.x once ir_converter is capable of making
// new-style procs.
TEST_F(ElaborationTest, GraphNewStyle) {
  // Proc graph.
  // A -> B1 -> B2 -> B3 -> C -> A
  // EXT -> I -> A
  // C -> I -> EXT
  // NB The actual ir is in examples/proc_network.x
  auto p = CreatePackage();
  Proc* initiator;
  Proc* a_proc;
  Proc* b_proc;
  Proc* c_proc;
  Type* s32 = p->GetBitsType(32);
  {
    TokenlessProcBuilder ab(NewStyleProc{}, "A_proc", "tok", p.get());
    // A channels
    XLS_ASSERT_OK_AND_ASSIGN(auto a_inp, ab.AddInputChannel("inp", s32));
    XLS_ASSERT_OK_AND_ASSIGN(auto a_output, ab.AddOutputChannel("output", s32));
    XLS_ASSERT_OK_AND_ASSIGN(auto a_ext, ab.AddInputChannel("ext", s32));
    // A Next
    auto recv_ext = ab.Receive(a_ext, ab.AfterAll({}));
    auto recv_inp = ab.ReceiveNonBlocking(a_inp, ab.TupleIndex(recv_ext, 0));
    ab.Send(
        a_output, ab.TupleIndex(recv_inp, 0),
        ab.Add(ab.Add(ab.TupleIndex(recv_ext, 1), ab.TupleIndex(recv_inp, 1)),
               ab.Literal(UBits(1, 32))));
    XLS_ASSERT_OK_AND_ASSIGN(a_proc, ab.Build({}));
  }
  {
    TokenlessProcBuilder bb(NewStyleProc{}, "B_proc", "tok", p.get());
    // B Channels
    XLS_ASSERT_OK_AND_ASSIGN(auto b_inp, bb.AddInputChannel("inp", s32));
    XLS_ASSERT_OK_AND_ASSIGN(auto b_output, bb.AddOutputChannel("output", s32));
    // B Next
    auto recv_inp = bb.Receive(b_inp, bb.AfterAll({}));
    bb.Send(b_output, bb.TupleIndex(recv_inp, 0),
            bb.Add(bb.TupleIndex(recv_inp, 1), bb.Literal(UBits(100, 32))));
    XLS_ASSERT_OK_AND_ASSIGN(b_proc, bb.Build({}));
  }
  {
    TokenlessProcBuilder cb(NewStyleProc{}, "C_proc", "tok", p.get());
    // C Channels
    XLS_ASSERT_OK_AND_ASSIGN(auto c_inp, cb.AddInputChannel("inp", s32));
    XLS_ASSERT_OK_AND_ASSIGN(auto c_output, cb.AddOutputChannel("output", s32));
    XLS_ASSERT_OK_AND_ASSIGN(auto c_ext, cb.AddOutputChannel("ext", s32));
    // C Next
    auto recv = cb.Receive(c_inp, cb.AfterAll({}));
    auto c_data = cb.Add(cb.TupleIndex(recv, 1), cb.Literal(UBits(10000, 32)));
    auto c_tok = cb.TupleIndex(recv, 0);
    cb.Send(c_output, c_tok, c_data);
    cb.Send(c_ext, c_tok, c_data);
    XLS_ASSERT_OK_AND_ASSIGN(c_proc, cb.Build({}));
  }

  {
    TokenlessProcBuilder ib(NewStyleProc{}, "initiator", "tok", p.get());
    // initiator channels
    XLS_ASSERT_OK_AND_ASSIGN(auto ext_in_ch, ib.AddInputChannel("ext_in", s32));
    XLS_ASSERT_OK_AND_ASSIGN(auto ext_out_ch,
                             ib.AddOutputChannel("ext_out", s32));
    // config
    XLS_ASSERT_OK_AND_ASSIGN((auto [c1, c_ext, init_rcv]),
                             ib.AddChannel("init_in_chans", s32));
    XLS_ASSERT_OK_AND_ASSIGN((auto [c2, init_snd, a_ext]),
                             ib.AddChannel("init_out_chans", s32));
    XLS_ASSERT_OK_AND_ASSIGN((auto [c3, a_to_b1_out, a_to_b1_in]),
                             ib.AddChannel("a_to_b1", s32));
    XLS_ASSERT_OK_AND_ASSIGN((auto [c4, b1_to_b2_out, b1_to_b2_in]),
                             ib.AddChannel("b1_to_b2", s32));
    XLS_ASSERT_OK_AND_ASSIGN((auto [c5, b2_to_b3_out, b2_to_b3_in]),
                             ib.AddChannel("b2_to_b3", s32));
    XLS_ASSERT_OK_AND_ASSIGN((auto [c6, b3_to_c_out, b3_to_c_in]),
                             ib.AddChannel("b3_to_c", s32));
    XLS_ASSERT_OK_AND_ASSIGN((auto [c7, c_to_a_out, c_to_a_in]),
                             ib.AddChannel("c_to_a", s32));
    XLS_ASSERT_OK(
        ib.InstantiateProc("a_inst", a_proc, {c_to_a_in, a_to_b1_out, a_ext}));
    XLS_ASSERT_OK(
        ib.InstantiateProc("B1_inst", b_proc, {a_to_b1_in, b1_to_b2_out}));
    XLS_ASSERT_OK(
        ib.InstantiateProc("B2_inst", b_proc, {b1_to_b2_in, b2_to_b3_out}));
    XLS_ASSERT_OK(
        ib.InstantiateProc("B3_inst", b_proc, {b2_to_b3_in, b3_to_c_out}));
    XLS_ASSERT_OK(
        ib.InstantiateProc("C_inst", c_proc, {b3_to_c_in, c_to_a_out, c_ext}));
    // next
    auto ext_recv = ib.Receive(ext_in_ch, ib.AfterAll({}));
    auto init_in = ib.Send(init_snd, ib.TupleIndex(ext_recv, 0),
                           ib.TupleIndex(ext_recv, 1));
    auto init_out = ib.Receive(init_rcv, init_in);
    ib.Send(ext_out_ch, ib.TupleIndex(init_out, 0), ib.TupleIndex(init_out, 1));
    XLS_ASSERT_OK_AND_ASSIGN(initiator, ib.Build({}));
  }
  // TODO(allight): Once ir_convert can create proc-scoped channels use the
  // proc_network.x design instead.
  // XLS_ASSERT_OK_AND_ASSIGN(
  //     auto path, GetXlsRunfilePath("xls/examples/proc_network.psc.ir"));
  // XLS_ASSERT_OK_AND_ASSIGN(auto ir, GetFileContents(path));
  // XLS_ASSERT_OK_AND_ASSIGN(auto pkg, ParsePackage(ir));
  RecordProperty("ir", p->DumpIr());
  XLS_ASSERT_OK_AND_ASSIGN(auto elab, ProcElaboration::Elaborate(initiator));
  XLS_ASSERT_OK_AND_ASSIGN(const ProcElaboration::ChannelGraph& graph,
                           elab.GetChannelGraph());

  // 2 nodes should have 2 outputs (C and I). 2 nodes (A & I) should have
  // 2 inputs.
  // TODO(allight): Some way to identify the proc associated with each id
  // would be nice.
  RecordProperty(
      "adjacency",
      util::GraphToString(graph, util::PRINT_GRAPH_ADJACENCY_LISTS_SORTED));
  int64_t cnt_2s_out = 0;
  int64_t cnt_2s_in = 0;
  for (int64_t i = 0; i < elab.proc_instances().size(); ++i) {
    if (graph.InDegree(i) == 2) {
      cnt_2s_in++;
    } else {
      EXPECT_EQ(graph.InDegree(i), 1);
    }
    if (graph.OutDegree(i) == 2) {
      cnt_2s_out++;
    } else {
      EXPECT_EQ(graph.OutDegree(i), 1);
    }
  }
  EXPECT_EQ(cnt_2s_out, 2);
  EXPECT_EQ(cnt_2s_in, 2);
  EXPECT_EQ(graph.OutDegree(elab.proc_instances().size()), 1);
  EXPECT_EQ(graph.InDegree(elab.proc_instances().size()), 1);
}

MATCHER_P(ChannelInterfaceNameIs, name,
          absl::StrCat("ChannelInterface arg names (",
                       testing::DescribeMatcher<std::string>(name), ")")) {
  *result_listener << arg->name() << " does not match the expectation.";
  return ExplainMatchResult(name, arg->name(), result_listener);
}

TEST_F(ElaborationTest, GraphMultipleEdgesNewStyle) {
  // I -> A
  // I -> A
  // I -> A
  // A -> I
  // A -> I
  auto p = CreatePackage();
  Proc* initiator;
  Proc* subproc;
  Type* s32 = p->GetBitsType(32);
  {
    TokenlessProcBuilder sb(NewStyleProc{}, "subproc", "tok", p.get());
    // subproc channels
    XLS_ASSERT_OK_AND_ASSIGN(auto sub_inp1, sb.AddInputChannel("inp1", s32));
    XLS_ASSERT_OK_AND_ASSIGN(auto sub_inp2, sb.AddInputChannel("inp2", s32));
    XLS_ASSERT_OK_AND_ASSIGN(auto sub_inp3, sb.AddInputChannel("inp3", s32));
    XLS_ASSERT_OK_AND_ASSIGN(auto sub_output1,
                             sb.AddOutputChannel("output1", s32));
    XLS_ASSERT_OK_AND_ASSIGN(auto sub_output2,
                             sb.AddOutputChannel("output2", s32));
    // subproc Next
    auto r1 = sb.Receive(sub_inp1);
    auto r2 = sb.Receive(sub_inp2);
    auto r3 = sb.Receive(sub_inp3);
    sb.Send(sub_output1, sb.Add(r1, sb.Literal(UBits(100, 32))));
    sb.Send(sub_output2, sb.Add(r2, r3));
    XLS_ASSERT_OK_AND_ASSIGN(subproc, sb.Build({}));
  }
  {
    TokenlessProcBuilder ib(NewStyleProc{}, "initiator", "tok", p.get());
    // initiator channels
    // config
    XLS_ASSERT_OK_AND_ASSIGN((auto [c1, i_snd_1, a_rcv_1]),
                             ib.AddChannel("send_1", s32));
    XLS_ASSERT_OK_AND_ASSIGN((auto [c2, i_snd_2, a_rcv_2]),
                             ib.AddChannel("send_2", s32));
    XLS_ASSERT_OK_AND_ASSIGN((auto [c3, i_snd_3, a_rcv_3]),
                             ib.AddChannel("send_3", s32));
    XLS_ASSERT_OK_AND_ASSIGN((auto [c4, a_snd_1, i_rcv_1]),
                             ib.AddChannel("recv_1", s32));
    XLS_ASSERT_OK_AND_ASSIGN((auto [c5, a_snd_2, i_rcv_2]),
                             ib.AddChannel("recv_2", s32));
    XLS_ASSERT_OK(ib.InstantiateProc(
        "subproc", subproc, {a_rcv_1, a_rcv_2, a_rcv_3, a_snd_1, a_snd_2}));
    // Next
    ib.Send(i_snd_1, ib.Literal(UBits(1, 32)));
    ib.Send(i_snd_2, ib.Literal(UBits(2, 32)));
    ib.Send(i_snd_3, ib.Literal(UBits(3, 32)));
    ib.Receive(i_rcv_1);
    ib.Receive(i_rcv_2);
    XLS_ASSERT_OK_AND_ASSIGN(initiator, ib.Build({}));
  }
  RecordProperty("ir", p->DumpIr());
  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elab,
                           ProcElaboration::Elaborate(initiator));
  XLS_ASSERT_OK_AND_ASSIGN(const ProcElaboration::ChannelGraph& graph,
                           elab.GetChannelGraph());
  RecordProperty(
      "adjacency",
      util::GraphToString(graph, util::PRINT_GRAPH_ADJACENCY_LISTS_SORTED));
  ProcElaboration::ProcInstanceId subproc_id;
  ProcElaboration::ProcInstanceId initiator_id;
  if (elab.proc_instances().front()->proc() == subproc) {
    subproc_id = 0;
    initiator_id = 1;
  } else {
    subproc_id = 1;
    initiator_id = 0;
  }

  EXPECT_EQ(graph.OutDegree(initiator_id), 3);
  EXPECT_EQ(graph.InDegree(initiator_id), 2);
  EXPECT_EQ(graph.OutDegree(subproc_id), 2);
  EXPECT_EQ(graph.InDegree(subproc_id), 3);

  std::vector<ChannelInterface*> subproc_outs;
  for (auto arc : graph.OutgoingArcs(subproc_id)) {
    XLS_ASSERT_OK_AND_ASSIGN(auto edge, elab.GetChannelRefs(arc));
    subproc_outs.push_back(std::get<SendChannelInterface*>(edge.send));
  }
  EXPECT_THAT(subproc_outs,
              testing::UnorderedElementsAre(ChannelInterfaceNameIs("output1"),
                                            ChannelInterfaceNameIs("output2")));
  std::vector<ChannelInterface*> subproc_ins;
  for (auto arc : graph.IncomingArcs(subproc_id)) {
    XLS_ASSERT_OK_AND_ASSIGN(auto edge, elab.GetChannelRefs(arc));
    subproc_ins.push_back(std::get<ReceiveChannelInterface*>(edge.recv));
  }
  EXPECT_THAT(subproc_ins,
              testing::UnorderedElementsAre(ChannelInterfaceNameIs("inp1"),
                                            ChannelInterfaceNameIs("inp2"),
                                            ChannelInterfaceNameIs("inp3")));

  std::vector<ChannelInterface*> initiator_outs;
  for (auto arc : graph.OutgoingArcs(initiator_id)) {
    XLS_ASSERT_OK_AND_ASSIGN(auto edge, elab.GetChannelRefs(arc));
    initiator_outs.push_back(std::get<SendChannelInterface*>(edge.send));
  }
  EXPECT_THAT(initiator_outs,
              testing::UnorderedElementsAre(ChannelInterfaceNameIs("send_1"),
                                            ChannelInterfaceNameIs("send_2"),
                                            ChannelInterfaceNameIs("send_3")));
  std::vector<ChannelInterface*> initiator_ins;
  for (auto arc : graph.IncomingArcs(initiator_id)) {
    XLS_ASSERT_OK_AND_ASSIGN(auto edge, elab.GetChannelRefs(arc));
    initiator_ins.push_back(std::get<ReceiveChannelInterface*>(edge.recv));
  }
  EXPECT_THAT(initiator_ins,
              testing::UnorderedElementsAre(ChannelInterfaceNameIs("recv_1"),
                                            ChannelInterfaceNameIs("recv_2")));
}

TEST_F(ElaborationTest, GraphOldStyle) {
  // Proc graph.
  // A -> B1 -> B2 -> B3 -> C -> A
  // EXT -> I -> A
  // C -> I -> EXT
  // NB The actual ir is in examples/proc_network.x
  XLS_ASSERT_OK_AND_ASSIGN(auto path,
                           GetXlsRunfilePath("xls/examples/proc_network.ir"));
  XLS_ASSERT_OK_AND_ASSIGN(auto ir, GetFileContents(path));
  XLS_ASSERT_OK_AND_ASSIGN(auto pkg, ParsePackage(ir));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto elab, ProcElaboration::ElaborateOldStylePackage(pkg.get()));
  XLS_ASSERT_OK_AND_ASSIGN(const ProcElaboration::ChannelGraph& graph,
                           elab.GetChannelGraph());

  // 2 nodes should have 2 outputs (C and I). 2 nodes (A & I) should have 2
  // inputs.
  // TODO(allight): Some way to identify the proc associated with each id would
  // be nice.
  RecordProperty(
      "adjacency",
      util::GraphToString(graph, util::PRINT_GRAPH_ADJACENCY_LISTS_SORTED));
  int64_t cnt_2s_out = 0;
  int64_t cnt_2s_in = 0;
  for (int64_t i = 0; i < elab.proc_instances().size(); ++i) {
    if (graph.InDegree(i) == 2) {
      cnt_2s_in++;
    } else {
      EXPECT_EQ(graph.InDegree(i), 1);
    }
    if (graph.OutDegree(i) == 2) {
      cnt_2s_out++;
    } else {
      EXPECT_EQ(graph.OutDegree(i), 1);
    }
  }
  EXPECT_EQ(cnt_2s_out, 2);
  EXPECT_EQ(cnt_2s_in, 2);
  EXPECT_EQ(graph.OutDegree(elab.proc_instances().size()), 1);
  EXPECT_EQ(graph.InDegree(elab.proc_instances().size()), 1);
}

TEST_F(ElaborationTest, GraphMultipleEdgesOldStyle) {
  // I -> A
  // I -> A
  // I -> A
  // A -> I
  // A -> I
  auto p = CreatePackage();
  Proc* initiator;
  Proc* subproc;
  Type* s32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * c1,
      p->CreateStreamingChannel("chan1", ChannelOps::kSendReceive, s32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * c2,
      p->CreateStreamingChannel("chan2", ChannelOps::kSendReceive, s32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * c3,
      p->CreateStreamingChannel("chan3", ChannelOps::kSendReceive, s32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * c4,
      p->CreateStreamingChannel("chan4", ChannelOps::kSendReceive, s32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * c5,
      p->CreateStreamingChannel("chan5", ChannelOps::kSendReceive, s32));
  {
    TokenlessProcBuilder sb("subproc", "tok", p.get());
    // subproc channels
    // subproc Next
    auto r1 = sb.Receive(c1);
    auto r2 = sb.Receive(c2);
    auto r3 = sb.Receive(c3);
    sb.Send(c4, sb.Add(r1, sb.Literal(UBits(100, 32))));
    sb.Send(c5, sb.Add(r2, r3));
    XLS_ASSERT_OK_AND_ASSIGN(subproc, sb.Build({}));
  }
  {
    TokenlessProcBuilder ib("initiator", "tok", p.get());
    // Next
    ib.Send(c1, ib.Literal(UBits(1, 32)));
    ib.Send(c2, ib.Literal(UBits(2, 32)));
    ib.Send(c3, ib.Literal(UBits(3, 32)));
    ib.Receive(c4);
    ib.Receive(c5);
    XLS_ASSERT_OK_AND_ASSIGN(initiator, ib.Build({}));
  }
  RecordProperty("ir", p->DumpIr());
  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elab,
                           ProcElaboration::ElaborateOldStylePackage(p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(const ProcElaboration::ChannelGraph& graph,
                           elab.GetChannelGraph());
  RecordProperty(
      "adjacency",
      util::GraphToString(graph, util::PRINT_GRAPH_ADJACENCY_LISTS_SORTED));
  ProcElaboration::ProcInstanceId subproc_id;
  ProcElaboration::ProcInstanceId initiator_id;
  if (elab.proc_instances().front()->proc() == subproc) {
    subproc_id = 0;
    initiator_id = 1;
  } else {
    subproc_id = 1;
    initiator_id = 0;
  }

  EXPECT_EQ(graph.OutDegree(initiator_id), 3);
  EXPECT_EQ(graph.InDegree(initiator_id), 2);
  EXPECT_EQ(graph.OutDegree(subproc_id), 2);
  EXPECT_EQ(graph.InDegree(subproc_id), 3);

  std::vector<Channel*> subproc_outs;
  for (auto arc : graph.OutgoingArcs(subproc_id)) {
    XLS_ASSERT_OK_AND_ASSIGN(auto edge, elab.GetChannelRefs(arc));
    subproc_outs.push_back(std::get<Channel*>(edge.send));
  }
  EXPECT_THAT(subproc_outs,
              testing::UnorderedElementsAre(ChannelInterfaceNameIs("chan4"),
                                            ChannelInterfaceNameIs("chan5")));
  std::vector<Channel*> subproc_ins;
  for (auto arc : graph.IncomingArcs(subproc_id)) {
    XLS_ASSERT_OK_AND_ASSIGN(auto edge, elab.GetChannelRefs(arc));
    subproc_ins.push_back(std::get<Channel*>(edge.recv));
  }
  EXPECT_THAT(subproc_ins,
              testing::UnorderedElementsAre(ChannelInterfaceNameIs("chan1"),
                                            ChannelInterfaceNameIs("chan2"),
                                            ChannelInterfaceNameIs("chan3")));

  std::vector<Channel*> initiator_outs;
  for (auto arc : graph.OutgoingArcs(initiator_id)) {
    XLS_ASSERT_OK_AND_ASSIGN(auto edge, elab.GetChannelRefs(arc));
    initiator_outs.push_back(std::get<Channel*>(edge.send));
  }
  EXPECT_THAT(initiator_outs,
              testing::UnorderedElementsAre(ChannelInterfaceNameIs("chan1"),
                                            ChannelInterfaceNameIs("chan2"),
                                            ChannelInterfaceNameIs("chan3")));
  std::vector<Channel*> initiator_ins;
  for (auto arc : graph.IncomingArcs(initiator_id)) {
    XLS_ASSERT_OK_AND_ASSIGN(auto edge, elab.GetChannelRefs(arc));
    initiator_ins.push_back(std::get<Channel*>(edge.recv));
  }
  EXPECT_THAT(initiator_ins,
              testing::UnorderedElementsAre(ChannelInterfaceNameIs("chan4"),
                                            ChannelInterfaceNameIs("chan5")));
}

}  // namespace
}  // namespace xls
