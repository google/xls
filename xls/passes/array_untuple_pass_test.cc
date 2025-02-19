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

#include "xls/passes/array_untuple_pass.h"

#include <cstdint>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/ir/value_builder.h"
#include "xls/passes/dataflow_simplification_pass.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/solvers/z3_ir_equivalence_testutils.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {
using ::absl_testing::IsOkAndHolds;
using ::testing::_;
using ::testing::IsSupersetOf;
using ::testing::UnorderedElementsAre;

class ArrayUntuplePassTest : public IrTestBase {
 public:
  absl::StatusOr<bool> RunPass(Package* p) {
    // Do a little pipeline with dataflow to get rid of any (tuple-index (tuple
    // ...)) constructs.
    OptimizationCompoundPass pass("test", "test passes");
    pass.Add<ArrayUntuplePass>();
    pass.Add<DeadCodeEliminationPass>();
    pass.Add<DataflowSimplificationPass>();
    pass.Add<DeadCodeEliminationPass>();
    PassResults res;
    OptimizationContext context;
    return pass.Run(p, {}, &res, &context);
  }
};

TEST_F(ArrayUntuplePassTest, BasicIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue lit = fb.Literal(ValueBuilder::ArrayB(
      {ValueBuilder::TupleB({ValueBuilder::Bits(UBits(1, 8)),
                             ValueBuilder::Bits(UBits(2, 8)),
                             ValueBuilder::Bits(UBits(3, 8))}),
       ValueBuilder::TupleB({ValueBuilder::Bits(UBits(4, 8)),
                             ValueBuilder::Bits(UBits(5, 8)),
                             ValueBuilder::Bits(UBits(6, 8))}),
       ValueBuilder::TupleB({ValueBuilder::Bits(UBits(7, 8)),
                             ValueBuilder::Bits(UBits(8, 8)),
                             ValueBuilder::Bits(UBits(9, 8))}),
       ValueBuilder::TupleB({ValueBuilder::Bits(UBits(10, 8)),
                             ValueBuilder::Bits(UBits(11, 8)),
                             ValueBuilder::Bits(UBits(12, 8))}),
       ValueBuilder::TupleB({ValueBuilder::Bits(UBits(13, 8)),
                             ValueBuilder::Bits(UBits(14, 8)),
                             ValueBuilder::Bits(UBits(15, 8))})}));
  BValue i = fb.Param("i", p->GetBitsType(8));
  BValue tmp =
      fb.Add(fb.TupleIndex(fb.ArrayIndex(lit, {i}), 0),
             fb.TupleIndex(
                 fb.ArrayIndex(lit, {fb.Add(i, fb.Literal(UBits(1, 8)))}), 1));
  fb.Add(tmp,
         fb.TupleIndex(
             fb.ArrayIndex(lit, {fb.Subtract(i, fb.Literal(UBits(1, 8)))}), 2));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  solvers::z3::ScopedVerifyEquivalence sve(f);
  ScopedRecordIr sri(p.get());

  ASSERT_THAT(RunPass(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(
      f->return_value(),
      m::Add(
          m::Add(m::ArrayIndex(m::Literal(), {m::Param()}),
                 m::ArrayIndex(m::Literal(),
                               {m::Add(m::Param(), m::Literal(UBits(1, 8)))})),
          m::ArrayIndex(m::Literal(),
                        {m::Sub(m::Param(), m::Literal(UBits(1, 8)))})));
  EXPECT_THAT(
      f->nodes(),
      IsSupersetOf(
          {m::Literal(Value::UBitsArray({1, 4, 7, 10, 13}, 8).value()),
           m::Literal(Value::UBitsArray({2, 5, 8, 11, 14}, 8).value()),
           m::Literal(Value::UBitsArray({3, 6, 9, 12, 15}, 8).value())}));
}

TEST_F(ArrayUntuplePassTest, EmptyTupleArray) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue lit = fb.Literal(ValueBuilder::ArrayB(
      {ValueBuilder::TupleB({}), ValueBuilder::TupleB({}),
       ValueBuilder::TupleB({}), ValueBuilder::TupleB({}),
       ValueBuilder::TupleB({}), ValueBuilder::TupleB({})}));
  BValue i = fb.Param("i", p->GetBitsType(8));
  fb.Tuple({fb.ArrayIndex(lit, {i}), fb.ArrayIndex(lit, {i})});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  solvers::z3::ScopedVerifyEquivalence sve(f);
  ScopedRecordIr sri(p.get());

  // No point doing anything with empty tuple arrays.
  ArrayUntuplePass pass;
  PassResults res;
  OptimizationContext ctx;
  ASSERT_THAT(pass.Run(p.get(), {}, &res, &ctx), IsOkAndHolds(false));
}

TEST_F(ArrayUntuplePassTest, MaybeUpdate) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue do_up = fb.Param("do_up", p->GetBitsType(1));
  BValue up_val = fb.Param(
      "upval", p->GetTupleType({p->GetBitsType(1), p->GetBitsType(8)}));
  BValue wr_idx = fb.Param("wr_idx", p->GetBitsType(8));
  BValue rd_idx = fb.Param("rd_idx", p->GetBitsType(8));
  BValue init = fb.Literal(ValueBuilder::ArrayB({
      ValueBuilder::Tuple(
          {ValueBuilder::Bits(UBits(1, 1)), ValueBuilder::Bits((UBits(1, 8)))}),
      ValueBuilder::Tuple(
          {ValueBuilder::Bits(UBits(0, 1)), ValueBuilder::Bits((UBits(2, 8)))}),
      ValueBuilder::Tuple(
          {ValueBuilder::Bits(UBits(1, 1)), ValueBuilder::Bits((UBits(3, 8)))}),
      ValueBuilder::Tuple(
          {ValueBuilder::Bits(UBits(0, 1)), ValueBuilder::Bits((UBits(4, 8)))}),
  }));
  fb.ArrayIndex(
      fb.Select(do_up, {init, fb.ArrayUpdate(init, up_val, {wr_idx})}),
      {rd_idx});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedRecordIr sri(p.get());
  solvers::z3::ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(RunPass(p.get()), IsOkAndHolds(true));
  auto tup_0_lit_match = m::Literal(Value::UBitsArray({1, 0, 1, 0}, 1).value());
  auto tup_1_lit_match = m::Literal(Value::UBitsArray({1, 2, 3, 4}, 8).value());
  EXPECT_THAT(
      f->nodes(),
      IsSupersetOf({
          tup_0_lit_match,
          tup_1_lit_match,
          m::Select(m::Param(),
                    {tup_0_lit_match, m::ArrayUpdate(tup_0_lit_match, _, {_})}),
          m::Select(m::Param(),
                    {tup_1_lit_match, m::ArrayUpdate(tup_1_lit_match, _, {_})}),
      }));
}

TEST_F(ArrayUntuplePassTest, Compare) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue lhs = fb.Literal(ValueBuilder::ArrayB({
      ValueBuilder::Tuple(
          {ValueBuilder::Bits(UBits(1, 1)), ValueBuilder::Bits((UBits(1, 8)))}),
      ValueBuilder::Tuple(
          {ValueBuilder::Bits(UBits(0, 1)), ValueBuilder::Bits((UBits(2, 8)))}),
      ValueBuilder::Tuple(
          {ValueBuilder::Bits(UBits(1, 1)), ValueBuilder::Bits((UBits(3, 8)))}),
      ValueBuilder::Tuple(
          {ValueBuilder::Bits(UBits(0, 1)), ValueBuilder::Bits((UBits(4, 8)))}),
  }));
  Type* rhs_ty = p->GetTupleType({p->GetBitsType(1), p->GetBitsType(8)});
  BValue rhs1 = fb.Param("rhs1", rhs_ty);
  BValue rhs2 = fb.Param("rhs2", rhs_ty);
  BValue rhs3 = fb.Param("rhs3", rhs_ty);
  BValue rhs4 = fb.Param("rhs4", rhs_ty);
  BValue rhs = fb.Array({rhs1, rhs2, rhs3, rhs4}, rhs_ty);

  fb.Tuple({fb.Eq(lhs, rhs), fb.Ne(lhs, rhs)});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedRecordIr sri(p.get());
  solvers::z3::ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(RunPass(p.get()), IsOkAndHolds(true));

  auto tup_0_lit_match = m::Literal(Value::UBitsArray({1, 0, 1, 0}, 1).value());
  auto tup_1_lit_match = m::Literal(Value::UBitsArray({1, 2, 3, 4}, 8).value());
  EXPECT_THAT(
      f->return_value(),
      m::Tuple(m::And(m::Eq(tup_0_lit_match, _), m::Eq(tup_1_lit_match, _)),
               m::Or(m::Ne(tup_0_lit_match, _), m::Ne(tup_1_lit_match, _))));
}

TEST_F(ArrayUntuplePassTest, Gate) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue do_gate = fb.Param("DoGate", p->GetBitsType(1));
  Type* rhs_ty = p->GetTupleType({p->GetBitsType(1), p->GetBitsType(8)});
  BValue rhs1 = fb.Param("rhs1", rhs_ty);
  BValue rhs2 = fb.Param("rhs2", rhs_ty);
  BValue rhs3 = fb.Param("rhs3", rhs_ty);
  BValue rhs4 = fb.Param("rhs4", rhs_ty);
  BValue rhs = fb.Array({rhs1, rhs2, rhs3, rhs4}, rhs_ty);

  fb.Eq(fb.Gate(do_gate, rhs), rhs);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedRecordIr sri(p.get());
  solvers::z3::ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(RunPass(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(),
              m::And(m::Eq(m::Gate(), _), m::Eq(m::Gate(), _)));
}

TEST_F(ArrayUntuplePassTest, ArrayConcat) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue arr = fb.ArrayConcat({
      fb.Literal(ValueBuilder::Array(
          {ValueBuilder::Tuple({Value(UBits(1, 3)), Value(UBits(2, 3))})})),
      fb.Literal(ValueBuilder::Array(
          {ValueBuilder::Tuple({Value(UBits(3, 3)), Value(UBits(4, 3))})})),
      fb.Literal(ValueBuilder::Array(
          {ValueBuilder::Tuple({Value(UBits(5, 3)), Value(UBits(6, 3))})})),
  });
  fb.ArrayIndex(arr, {fb.Param("idx", p->GetBitsType(32))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedRecordIr sri(p.get());
  solvers::z3::ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(RunPass(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(),
              m::Tuple(m::ArrayIndex(m::ArrayConcat(m::Literal(), m::Literal(),
                                                    m::Literal()),
                                     {m::Param()}),
                       m::ArrayIndex(m::ArrayConcat(m::Literal(), m::Literal(),
                                                    m::Literal()),
                                     {m::Param()})));
}

TEST_F(ArrayUntuplePassTest, ProcStateArrayWithNext) {
  // Really simple sram with written-bit type proc
  auto p = CreatePackage();
  // struct Request {
  //   bool do_write;
  //   bool clear_written;
  //   int addr;
  //   int data;
  // };
  constexpr int64_t kDoWriteOff = 0;
  constexpr int64_t kClearWrittenOff = 1;
  constexpr int64_t kAddrOff = 2;
  constexpr int64_t kDataOff = 3;
  XLS_ASSERT_OK_AND_ASSIGN(
      auto chan_req,
      p->CreateStreamingChannel(
          "cmd", ChannelOps::kReceiveOnly,
          p->GetTupleType({p->GetBitsType(1), p->GetBitsType(1),
                           p->GetBitsType(3), p->GetBitsType(3)})));
  // Old value of state index 'addr'
  // struct Response {
  //   bool ever_written;
  //   int data;
  // };
  XLS_ASSERT_OK_AND_ASSIGN(
      auto chan_resp,
      p->CreateStreamingChannel(
          "resp", ChannelOps::kSendOnly,
          p->GetTupleType({p->GetBitsType(1), p->GetBitsType(3)})));
  ProcBuilder pb(TestName(), p.get());
  // struct State {
  //   bool ever_written;
  //   int data;
  // };
  BValue state = pb.StateElement(
      "foo", ValueBuilder::ArrayB({
                 ValueBuilder::Tuple({ValueBuilder::Bits(UBits(0, 1)),
                                      ValueBuilder::Bits((UBits(1, 3)))}),
                 ValueBuilder::Tuple({ValueBuilder::Bits(UBits(0, 1)),
                                      ValueBuilder::Bits((UBits(2, 3)))}),
                 ValueBuilder::Tuple({ValueBuilder::Bits(UBits(0, 1)),
                                      ValueBuilder::Bits((UBits(3, 3)))}),
                 ValueBuilder::Tuple({ValueBuilder::Bits(UBits(0, 1)),
                                      ValueBuilder::Bits((UBits(4, 3)))}),
             }));
  BValue req_tup = pb.Receive(chan_req, pb.Literal(Value::Token()));
  BValue tok = pb.TupleIndex(req_tup, 0);
  BValue req = pb.TupleIndex(req_tup, 1);
  BValue req_addr = pb.TupleIndex(req, kAddrOff);
  BValue read_res = pb.ArrayIndex(state, {req_addr});
  pb.Send(chan_resp, tok, read_res);
  BValue request_clear_written = pb.TupleIndex(req, kClearWrittenOff);
  BValue next_written_val = pb.Not(request_clear_written);
  BValue write_requested = pb.TupleIndex(req, kDoWriteOff);
  BValue write_nxt = pb.ArrayUpdate(state,
                                    pb.Tuple({
                                        next_written_val,
                                        pb.TupleIndex(req, kDataOff),
                                    }),
                                    {req_addr});
  BValue clear_only_nxt = pb.ArrayUpdate(state,
                                         pb.Tuple({
                                             pb.Literal(UBits(0, 1)),
                                             pb.TupleIndex(read_res, 1),
                                         }),
                                         {req_addr});

  // Select next state.
  // Write actually happened
  pb.Next(state, write_nxt, write_requested);
  // Clear but not write.
  pb.Next(state, clear_only_nxt,
          pb.And(request_clear_written, pb.Not(write_requested)));
  // Nothing written or cleared.
  pb.Next(state, state,
          pb.And(pb.Not(request_clear_written), pb.Not(write_requested)));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * pr, pb.Build());
  ScopedRecordIr sri(p.get());
  solvers::z3::ScopedVerifyProcEquivalence svpe(pr, /*activation_count=*/4,
                                                /*include_state=*/false);
  ASSERT_THAT(RunPass(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(pr->StateElements(),
              IsSupersetOf({m::StateElement(_, m::Type("bits[1][4]")),
                            m::StateElement(_, m::Type("bits[3][4]"))}));
  auto state_read_of_type = [](std::string_view type) {
    return testing::AllOf(m::StateRead(), m::Type(type));
  };
  EXPECT_THAT(
      pr->next_values(),
      UnorderedElementsAre(
          m::Next(m::StateRead("foo"), m::StateRead("foo"), _),
          m::Next(m::StateRead("foo"), m::StateRead("foo"), _),
          m::Next(m::StateRead("foo"), m::StateRead("foo"), _),
          m::Next(state_read_of_type("bits[1][4]"),
                  state_read_of_type("bits[1][4]"), _),
          m::Next(state_read_of_type("bits[1][4]"), m::Type("bits[1][4]"), _),
          m::Next(state_read_of_type("bits[1][4]"), m::Type("bits[1][4]"), _),
          m::Next(state_read_of_type("bits[3][4]"),
                  state_read_of_type("bits[3][4]"), _),
          m::Next(state_read_of_type("bits[3][4]"), m::Type("bits[3][4]"), _),
          m::Next(state_read_of_type("bits[3][4]"), m::Type("bits[3][4]"), _)));
}

TEST_F(ArrayUntuplePassTest, ProcStateArrayImplicitNext) {
  // Really simple sram with written-bit type proc
  auto p = CreatePackage();
  // struct Request {
  //   bool do_write;
  //   bool clear_written;
  //   int addr;
  //   int data;
  // };
  constexpr int64_t kDoWriteOff = 0;
  constexpr int64_t kClearWrittenOff = 1;
  constexpr int64_t kAddrOff = 2;
  constexpr int64_t kDataOff = 3;
  XLS_ASSERT_OK_AND_ASSIGN(
      auto chan_req,
      p->CreateStreamingChannel(
          "cmd", ChannelOps::kReceiveOnly,
          p->GetTupleType({p->GetBitsType(1), p->GetBitsType(1),
                           p->GetBitsType(3), p->GetBitsType(3)})));
  // Old value of state index 'addr'
  // struct Response {
  //   bool ever_written;
  //   int data;
  // };
  XLS_ASSERT_OK_AND_ASSIGN(
      auto chan_resp,
      p->CreateStreamingChannel(
          "resp", ChannelOps::kSendOnly,
          p->GetTupleType({p->GetBitsType(1), p->GetBitsType(3)})));
  ProcBuilder pb(TestName(), p.get());
  // struct State {
  //   bool ever_written;
  //   int data;
  // };
  BValue state = pb.StateElement(
      "foo", ValueBuilder::ArrayB({
                 ValueBuilder::Tuple({ValueBuilder::Bits(UBits(0, 1)),
                                      ValueBuilder::Bits((UBits(1, 3)))}),
                 ValueBuilder::Tuple({ValueBuilder::Bits(UBits(0, 1)),
                                      ValueBuilder::Bits((UBits(2, 3)))}),
                 ValueBuilder::Tuple({ValueBuilder::Bits(UBits(0, 1)),
                                      ValueBuilder::Bits((UBits(3, 3)))}),
                 ValueBuilder::Tuple({ValueBuilder::Bits(UBits(0, 1)),
                                      ValueBuilder::Bits((UBits(4, 3)))}),
             }));
  BValue req_tup = pb.Receive(chan_req, pb.Literal(Value::Token()));
  BValue tok = pb.TupleIndex(req_tup, 0);
  BValue req = pb.TupleIndex(req_tup, 1);
  BValue req_addr = pb.TupleIndex(req, kAddrOff);
  BValue read_res = pb.ArrayIndex(state, {req_addr});
  pb.Send(chan_resp, tok, read_res);
  BValue request_clear_written = pb.TupleIndex(req, kClearWrittenOff);
  BValue next_written_val = pb.Not(request_clear_written);
  BValue write_requested = pb.TupleIndex(req, kDoWriteOff);
  BValue write_nxt = pb.ArrayUpdate(state,
                                    pb.Tuple({
                                        next_written_val,
                                        pb.TupleIndex(req, kDataOff),
                                    }),
                                    {req_addr});
  BValue clear_only_nxt = pb.ArrayUpdate(state,
                                         pb.Tuple({
                                             pb.Literal(UBits(0, 1)),
                                             pb.TupleIndex(read_res, 1),
                                         }),
                                         {req_addr});

  // Select next state.
  BValue do_clear_only_next =
      pb.And(request_clear_written, pb.Not(write_requested));
  BValue next_state = pb.PrioritySelect(pb.Concat({
                                            // Clear but not write.
                                            do_clear_only_next,
                                            // Write actually happened
                                            write_requested,
                                        }),
                                        {
                                            // Write actually happened
                                            write_nxt,
                                            // Clear but not write.
                                            clear_only_nxt,
                                            // Nothing written or cleared.
                                        },
                                        state);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * pr, pb.Build({next_state}));
  solvers::z3::ScopedVerifyProcEquivalence svpe(pr, /*activation_count=*/4,
                                                /*include_state=*/false);
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(RunPass(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(pr->StateElements(),
              IsSupersetOf({m::StateElement(_, m::Type("bits[1][4]")),
                            m::StateElement(_, m::Type("bits[3][4]"))}));
}

}  // namespace
}  // namespace xls
