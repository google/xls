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

#include "xls/ir/proc_conversion.h"

#include <memory>

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/elaboration.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"

namespace xls {
namespace {

class ProcConversionTest : public IrTestBase {};

// Create proc network with the following topology:
//
//               +----+
//               |    |
//               V    |
// IN ---> A <-> B <--+
//         |
//         + -----> OUT
//
// IN ---> C -----> OUT
//
absl::Status CreateProcTopology(Package* p) {
  Type* u32 = p->GetBitsType(32);
  XLS_ASSIGN_OR_RETURN(
      Channel * ch_in_a,
      p->CreateStreamingChannel("in_a", ChannelOps::kReceiveOnly, u32));
  XLS_ASSIGN_OR_RETURN(
      Channel * ch_out_a,
      p->CreateStreamingChannel("out_a", ChannelOps::kSendOnly, u32));
  XLS_ASSIGN_OR_RETURN(
      Channel * ch_a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32));
  XLS_ASSIGN_OR_RETURN(
      Channel * ch_b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32));
  XLS_ASSIGN_OR_RETURN(
      Channel * ch_b_loopback,
      p->CreateStreamingChannel("b_loopback", ChannelOps::kSendReceive, u32));
  XLS_ASSIGN_OR_RETURN(
      Channel * ch_in_c,
      p->CreateStreamingChannel("in_c", ChannelOps::kReceiveOnly, u32));
  XLS_ASSIGN_OR_RETURN(
      Channel * ch_out_c,
      p->CreateStreamingChannel("out_c", ChannelOps::kSendOnly, u32));

  {
    TokenlessProcBuilder b("A", "tkn", p);
    b.Send(ch_a_to_b, b.Receive(ch_in_a));
    b.Send(ch_out_a, b.Receive(ch_b_to_a));
    XLS_RETURN_IF_ERROR(b.Build({}).status());
  }
  {
    TokenlessProcBuilder b("B", "tkn", p);
    b.Send(ch_b_to_a, b.Receive(ch_a_to_b));
    b.Send(ch_b_loopback, b.Receive(ch_b_loopback));
    XLS_RETURN_IF_ERROR(b.Build({}).status());
  }
  {
    TokenlessProcBuilder b("C", "tkn", p);
    b.Send(ch_out_c, b.Receive(ch_in_c));
    XLS_RETURN_IF_ERROR(b.Build({}).status());
  }
  return absl::OkStatus();
}

TEST_F(ProcConversionTest, ProcPipeline) {
  {
    // Topology with proc `A` as top.
    auto p = CreatePackage();
    XLS_ASSERT_OK(CreateProcTopology(p.get()));
    XLS_ASSERT_OK_AND_ASSIGN(Proc * a, p->GetProc("A"));
    XLS_ASSERT_OK(p->SetTop(a));

    XLS_ASSERT_OK(ConvertPackageToNewStyleProcs(p.get()));

    XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elab,
                             ProcElaboration::Elaborate(a));
    EXPECT_EQ(elab.ToString(), R"(A<in_a, out_a, in_c, out_c>
  chan a_to_b
  chan b_to_a
  B<a_to_b=a_to_b, b_to_a=b_to_a> [B_inst]
    chan b_loopback
  C<in_c=in_c, out_c=out_c> [C_inst])");
  }

  {
    // Topology with proc `B` as top.
    auto p = CreatePackage();
    XLS_ASSERT_OK(CreateProcTopology(p.get()));
    XLS_ASSERT_OK_AND_ASSIGN(Proc * b, p->GetProc("B"));
    XLS_ASSERT_OK(p->SetTop(b));

    XLS_ASSERT_OK(ConvertPackageToNewStyleProcs(p.get()));

    XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elab,
                             ProcElaboration::Elaborate(b));
    EXPECT_EQ(elab.ToString(), R"(B<in_a, out_a, in_c, out_c>
  chan a_to_b
  chan b_to_a
  chan b_loopback
  A<a_to_b=a_to_b, b_to_a=b_to_a, in_a=in_a, out_a=out_a> [A_inst]
  C<in_c=in_c, out_c=out_c> [C_inst])");
  }

  {
    // Topology with proc `C` as top.
    auto p = CreatePackage();
    XLS_ASSERT_OK(CreateProcTopology(p.get()));
    XLS_ASSERT_OK_AND_ASSIGN(Proc * c, p->GetProc("C"));
    XLS_ASSERT_OK(p->SetTop(c));

    XLS_ASSERT_OK(ConvertPackageToNewStyleProcs(p.get()));

    XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elab,
                             ProcElaboration::Elaborate(c));
    EXPECT_EQ(elab.ToString(), R"(C<in_a, out_a, in_c, out_c>
  chan a_to_b
  chan b_to_a
  A<a_to_b=a_to_b, b_to_a=b_to_a, in_a=in_a, out_a=out_a> [A_inst]
  B<a_to_b=a_to_b, b_to_a=b_to_a> [B_inst]
    chan b_loopback)");
  }
}

}  // namespace
}  // namespace xls
