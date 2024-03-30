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

#include "xls/passes/ternary_query_engine.h"

#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "benchmark/benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/ir/benchmark_support.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/ternary.h"
#include "xls/ir/value.h"
#include "xls/ir/value_builder.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

struct TernaryPair {
  Value known_bits;
  Value known_bit_values;
};
// Helper to create composite values with ternaries at their base.
class TValue {
 public:
  explicit TValue(std::string_view sv) : val_(sv) {}

  static TValue ArrayV(absl::Span<TValue const> vs) {
    return TValue(
        ArrayMarker{.vals_ = std::vector<TValue>(vs.begin(), vs.end())});
  }

  static TValue ArrayS(absl::Span<std::string_view const> vs) {
    std::vector<TValue> tvs;
    tvs.reserve(vs.size());
    absl::c_transform(vs, std::back_inserter(tvs),
                      [&](auto v) { return TValue(v); });
    return ArrayV(tvs);
  }

  static TValue Array(
      absl::Span<std::variant<TValue, std::string_view> const> vs) {
    std::vector<TValue> tvs;
    tvs.reserve(vs.size());
    absl::c_transform(vs, std::back_inserter(tvs), [&](auto v) {
      return absl::visit(Visitor{[](std::string_view v) { return TValue(v); },
                                 [](const TValue& v) { return v; }},
                         v);
    });
    return ArrayV(tvs);
  }

  static TValue TupleV(absl::Span<TValue const> vs) {
    return TValue(
        TupleMarker{.vals_ = std::vector<TValue>(vs.begin(), vs.end())});
  }

  static TValue TupleS(absl::Span<std::string_view const> vs) {
    std::vector<TValue> tvs;
    tvs.reserve(vs.size());
    absl::c_transform(vs, std::back_inserter(tvs),
                      [&](auto v) { return TValue(v); });
    return TupleV(tvs);
  }

  static TValue Tuple(
      absl::Span<std::variant<TValue, std::string_view> const> vs) {
    std::vector<TValue> tvs;
    tvs.reserve(vs.size());
    absl::c_transform(vs, std::back_inserter(tvs), [&](auto v) {
      return absl::visit(Visitor{[](std::string_view v) { return TValue(v); },
                                 [](const TValue& v) { return v; }},
                         v);
    });
    return TupleV(tvs);
  }

  absl::StatusOr<TernaryPair> Build() const {
    return absl::visit(
        Visitor{
            [&](std::string_view sv) -> absl::StatusOr<TernaryPair> {
              XLS_ASSIGN_OR_RETURN(auto tv, StringToTernaryVector(sv));
              return TernaryPair{
                  .known_bits = Value(ternary_ops::ToKnownBits(tv)),
                  .known_bit_values =
                      Value(ternary_ops::ToKnownBitsValues(tv))};
            },
            [&](const ArrayMarker& arr) -> absl::StatusOr<TernaryPair> {
              std::vector<ValueBuilder> kbs;
              std::vector<ValueBuilder> kbvs;
              kbs.reserve(arr.vals_.size());
              kbvs.reserve(arr.vals_.size());
              for (const auto& v : arr.vals_) {
                XLS_ASSIGN_OR_RETURN(auto tern, v.Build());
                kbs.push_back(ValueBuilder(tern.known_bits));
                kbvs.push_back(ValueBuilder(tern.known_bit_values));
              }
              XLS_ASSIGN_OR_RETURN(auto kb, ValueBuilder::ArrayB(kbs).Build());
              XLS_ASSIGN_OR_RETURN(auto kbv,
                                   ValueBuilder::ArrayB(kbvs).Build());
              return TernaryPair{.known_bits = kb, .known_bit_values = kbv};
            },
            [&](const TupleMarker& tup) -> absl::StatusOr<TernaryPair> {
              std::vector<ValueBuilder> kbs;
              std::vector<ValueBuilder> kbvs;
              kbs.reserve(tup.vals_.size());
              kbvs.reserve(tup.vals_.size());
              for (const auto& v : tup.vals_) {
                XLS_ASSIGN_OR_RETURN(auto tern, v.Build());
                kbs.push_back(ValueBuilder(tern.known_bits));
                kbvs.push_back(ValueBuilder(tern.known_bit_values));
              }
              XLS_ASSIGN_OR_RETURN(auto kb, ValueBuilder::TupleB(kbs).Build());
              XLS_ASSIGN_OR_RETURN(auto kbv,
                                   ValueBuilder::TupleB(kbvs).Build());
              return TernaryPair{.known_bits = kb, .known_bit_values = kbv};
            }},
        val_);
  }

 private:
  struct ArrayMarker {
    std::vector<TValue> vals_;
  };
  struct TupleMarker {
    std::vector<TValue> vals_;
  };
  explicit TValue(std::variant<std::string_view, ArrayMarker, TupleMarker> sv)
      : val_(std::move(sv)) {}
  std::variant<std::string_view, ArrayMarker, TupleMarker> val_;
};

class TernaryQueryEngineTest : public IrTestBase {
 protected:
  // Create a BValue with known bits equal to the given ternary vector. Created
  // using a param and AND/OR masks.
  BValue MakeValueWithKnownBits(std::string_view name,
                                std::string_view known_bits,
                                FunctionBuilder* fb) {
    return MakeValueWithKnownBits(
        name, StringToTernaryVector(known_bits).value(), fb);
  }
  // Create a BValue with known bits equal to the bits set in `known_bits` with
  // the values set by `known_bits_values`. It is an error (causing a test
  // failure) if the shape of known_bits and known_bits_values differ or if any
  // tokens are present in the values.
  absl::StatusOr<BValue> MakeValueWithKnownBits(
      std::string_view name, const ValueBuilder& known_bits,
      const ValueBuilder& known_bit_values, FunctionBuilder* fb) {
    XLS_ASSIGN_OR_RETURN(auto kb, known_bits.Build());
    XLS_ASSIGN_OR_RETURN(auto kbv, known_bit_values.Build());
    return MakeValueWithKnownBits(name, kb, kbv, fb);
  }
  absl::StatusOr<BValue> MakeValueWithKnownBits(std::string_view name,
                                                const TValue& tvalue,
                                                FunctionBuilder* fb) {
    XLS_ASSIGN_OR_RETURN(auto ternary, tvalue.Build());
    return MakeValueWithKnownBits(name, ternary.known_bits,
                                  ternary.known_bit_values, fb);
  }
  // Create a BValue with known bits equal to the bits set in `known_bits` with
  // the values set by `known_bits_values`. It is an error (causing a test
  // failure) if the shape of known_bits and known_bits_values differ or if any
  // tokens are present in the values.
  absl::StatusOr<BValue> MakeValueWithKnownBits(std::string_view name,
                                                const TernaryPair& ternary,
                                                FunctionBuilder* fb) {
    return MakeValueWithKnownBits(name, ternary.known_bits,
                                  ternary.known_bit_values, fb);
  }
  // Create a BValue with known bits equal to the bits set in `known_bits` with
  // the values set by `known_bits_values`. It is an error (causing a test
  // failure) if the shape of known_bits and known_bits_values differ or if any
  // tokens are present in the values.
  absl::StatusOr<BValue> MakeValueWithKnownBits(std::string_view name,
                                                const Value& known_bits,
                                                const Value& known_bit_values,
                                                FunctionBuilder* fb) {
    XLS_RET_CHECK(known_bits.SameTypeAs(known_bit_values));
    XLS_RET_CHECK(!known_bits.IsToken() && !known_bit_values.IsToken());
    if (known_bits.IsBits()) {
      return MakeValueWithKnownBits(
          name,
          ternary_ops::FromKnownBits(known_bits.bits(),
                                     known_bit_values.bits()),
          fb);
    }
    if (known_bits.IsArray()) {
      std::vector<BValue> elements;
      elements.reserve(known_bits.size());
      for (int64_t i = 0; i < known_bits.size(); ++i) {
        XLS_ASSIGN_OR_RETURN(auto v, MakeValueWithKnownBits(
                                         absl::StrFormat("%s_arr[%d]", name, i),
                                         known_bits.element(i),
                                         known_bit_values.element(i), fb));
        elements.push_back(v);
      }
      return fb->Array(elements, elements.front().GetType(), SourceInfo(),
                       name);
    }
    if (known_bits.IsTuple()) {
      std::vector<BValue> elements;
      elements.reserve(known_bits.size());
      for (int64_t i = 0; i < known_bits.size(); ++i) {
        XLS_ASSIGN_OR_RETURN(auto v, MakeValueWithKnownBits(
                                         absl::StrFormat("%s_tup[%d]", name, i),
                                         known_bits.element(i),
                                         known_bit_values.element(i), fb));
        elements.push_back(v);
      }
      return fb->Tuple(elements, SourceInfo(), name);
    }
    return absl::InternalError(
        absl::StrFormat("Unexpected type: %v", known_bits));
  }
  // Create a BValue with known bits equal to the given ternary vector. Created
  // using a param and AND/OR masks.
  BValue MakeValueWithKnownBits(std::string_view name,
                                const TernaryVector& known_bits,
                                FunctionBuilder* fb) {
    absl::InlinedVector<bool, 1> known_zeros;
    absl::InlinedVector<bool, 1> known_ones;
    for (TernaryValue value : known_bits) {
      known_zeros.push_back(value == TernaryValue::kKnownZero);
      known_ones.push_back(value == TernaryValue::kKnownOne);
    }
    BValue and_mask = fb->Literal(bits_ops::Not(Bits(known_zeros)));
    BValue or_mask = fb->Literal(Bits(known_ones));
    return fb->Or(or_mask,
                  fb->And(and_mask, fb->Param(name, fb->package()->GetBitsType(
                                                        known_bits.size()))));
  }

  // Runs QueryEngine on the op created with the passed in function. The
  // inputs to the op is crafted to have known bits equal to the given
  // TernaryVectors.
  absl::StatusOr<std::string> RunOnBinaryOp(
      std::string_view lhs_known_bits, std::string_view rhs_known_bits,
      std::function<void(BValue, BValue, FunctionBuilder*)> make_op) {
    Package p("test_package");
    FunctionBuilder fb("f", &p);
    BValue lhs = MakeValueWithKnownBits("lhs", lhs_known_bits, &fb);
    BValue rhs = MakeValueWithKnownBits("rhs", rhs_known_bits, &fb);
    make_op(lhs, rhs, &fb);
    XLS_ASSIGN_OR_RETURN(Function * f, fb.Build());
    VLOG(3) << f->DumpIr();
    TernaryQueryEngine query_engine;
    XLS_RETURN_IF_ERROR(query_engine.Populate(f).status());
    return query_engine.ToString(f->return_value());
  }
};

TEST_F(TernaryQueryEngineTest, Uge) {
  auto make_uge = [](BValue lhs, BValue rhs, FunctionBuilder* fb) {
    fb->UGe(lhs, rhs);
  };
  EXPECT_THAT(RunOnBinaryOp("0bXXX", "0b11X", make_uge), IsOkAndHolds("0bX"));
  EXPECT_THAT(RunOnBinaryOp("0b111", "0bX1X", make_uge), IsOkAndHolds("0b1"));
  EXPECT_THAT(RunOnBinaryOp("0b1XX", "0bX11", make_uge), IsOkAndHolds("0bX"));
  EXPECT_THAT(RunOnBinaryOp("0b0XX", "0b1XX", make_uge), IsOkAndHolds("0b0"));
}

TEST_F(TernaryQueryEngineTest, Ugt) {
  auto make_ugt = [](BValue lhs, BValue rhs, FunctionBuilder* fb) {
    fb->UGt(lhs, rhs);
  };
  EXPECT_THAT(RunOnBinaryOp("0bXXX", "0b111", make_ugt), IsOkAndHolds("0b0"));
  EXPECT_THAT(RunOnBinaryOp("0b111", "0bX1X", make_ugt), IsOkAndHolds("0bX"));
  EXPECT_THAT(RunOnBinaryOp("0b111", "0bX10", make_ugt), IsOkAndHolds("0b1"));
  EXPECT_THAT(RunOnBinaryOp("0b1XX", "0bX11", make_ugt), IsOkAndHolds("0bX"));
  EXPECT_THAT(RunOnBinaryOp("0b0XX", "0b1XX", make_ugt), IsOkAndHolds("0b0"));
}

TEST_F(TernaryQueryEngineTest, Ule) {
  auto make_ule = [](BValue lhs, BValue rhs, FunctionBuilder* fb) {
    fb->ULe(lhs, rhs);
  };
  EXPECT_THAT(RunOnBinaryOp("0bXXX", "0b111", make_ule), IsOkAndHolds("0b1"));
  EXPECT_THAT(RunOnBinaryOp("0b000", "0bX1X", make_ule), IsOkAndHolds("0b1"));
  EXPECT_THAT(RunOnBinaryOp("0b111", "0bX10", make_ule), IsOkAndHolds("0b0"));
  EXPECT_THAT(RunOnBinaryOp("0b1XX", "0bX11", make_ule), IsOkAndHolds("0bX"));
  EXPECT_THAT(RunOnBinaryOp("0b0XX", "0b1XX", make_ule), IsOkAndHolds("0b1"));
}

TEST_F(TernaryQueryEngineTest, Ult) {
  auto make_ult = [](BValue lhs, BValue rhs, FunctionBuilder* fb) {
    fb->ULt(lhs, rhs);
  };
  EXPECT_THAT(RunOnBinaryOp("0bXXX", "0b111", make_ult), IsOkAndHolds("0bX"));
  EXPECT_THAT(RunOnBinaryOp("0b000", "0bX1X", make_ult), IsOkAndHolds("0b1"));
  EXPECT_THAT(RunOnBinaryOp("0b111", "0bX10", make_ult), IsOkAndHolds("0b0"));
  EXPECT_THAT(RunOnBinaryOp("0b1XX", "0bX11", make_ult), IsOkAndHolds("0bX"));
  EXPECT_THAT(RunOnBinaryOp("0b0XX", "0b1XX", make_ult), IsOkAndHolds("0b1"));
}

TEST_F(TernaryQueryEngineTest, Ne) {
  auto make_ne = [](BValue lhs, BValue rhs, FunctionBuilder* fb) {
    fb->Ne(lhs, rhs);
  };
  EXPECT_THAT(RunOnBinaryOp("0bXX1", "0b110", make_ne), IsOkAndHolds("0b1"));
  EXPECT_THAT(RunOnBinaryOp("0bXX1", "0b111", make_ne), IsOkAndHolds("0bX"));
  EXPECT_THAT(RunOnBinaryOp("0b011", "0b111", make_ne), IsOkAndHolds("0b1"));
  EXPECT_THAT(RunOnBinaryOp("0b011", "0b011", make_ne), IsOkAndHolds("0b0"));
  auto run_compound = [&](const TValue& left,
                          const TValue& right) -> absl::StatusOr<std::string> {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    XLS_ASSIGN_OR_RETURN(auto l, MakeValueWithKnownBits("left", left, &fb));
    XLS_ASSIGN_OR_RETURN(auto r, MakeValueWithKnownBits("right", right, &fb));
    BValue result = fb.Ne(l, r);
    XLS_ASSIGN_OR_RETURN(Function * f, fb.Build());
    TernaryQueryEngine tqe;
    XLS_RETURN_IF_ERROR(tqe.Populate(f).status());
    XLS_RET_CHECK(tqe.IsTracked(result.node()));
    return tqe.ToString(result.node());
  };
  EXPECT_THAT(run_compound(TValue::Tuple({}), TValue::Tuple({})),
              IsOkAndHolds("0b0"));
  EXPECT_THAT(run_compound(TValue::Array({"0b110", "0b101"}),
                           TValue::Array({"0b110", "0b101"})),
              IsOkAndHolds("0b0"));
  EXPECT_THAT(run_compound(TValue::Array({"0bX10", "0b101"}),
                           TValue::Array({"0b110", "0b101"})),
              IsOkAndHolds("0bX"));
  EXPECT_THAT(run_compound(TValue::Array({"0bX10", "0b001"}),
                           TValue::Array({"0b110", "0b101"})),
              IsOkAndHolds("0b1"));
  EXPECT_THAT(run_compound(TValue::Array({"0b110", "0b0XX"}),
                           TValue::Array({"0b110", "0b1XX"})),
              IsOkAndHolds("0b1"));
  EXPECT_THAT(run_compound(TValue::Array({"0bX10", "0b0XX"}),
                           TValue::Array({"0b110", "0b1XX"})),
              IsOkAndHolds("0b1"));
  EXPECT_THAT(run_compound(TValue::Array({"0bX10", "0b0XX"}),
                           TValue::Array({"0bX10", "0bXXX"})),
              IsOkAndHolds("0bX"));
  EXPECT_THAT(run_compound(TValue::Tuple({"0b110", "0b101"}),
                           TValue::Tuple({"0b110", "0b101"})),
              IsOkAndHolds("0b0"));
  EXPECT_THAT(run_compound(TValue::Tuple({"0bX10", "0b101"}),
                           TValue::Tuple({"0b110", "0b101"})),
              IsOkAndHolds("0bX"));
  EXPECT_THAT(run_compound(TValue::Tuple({"0bX10", "0b001"}),
                           TValue::Tuple({"0b110", "0b101"})),
              IsOkAndHolds("0b1"));
  EXPECT_THAT(run_compound(TValue::Tuple({"0b110", "0b0XX"}),
                           TValue::Tuple({"0b110", "0b1XX"})),
              IsOkAndHolds("0b1"));
  EXPECT_THAT(run_compound(TValue::Tuple({"0bX10", "0b0XX"}),
                           TValue::Tuple({"0b110", "0b1XX"})),
              IsOkAndHolds("0b1"));
  EXPECT_THAT(run_compound(TValue::Tuple({"0bX10", "0b0XX"}),
                           TValue::Tuple({"0bX10", "0bXXX"})),
              IsOkAndHolds("0bX"));
}

TEST_F(TernaryQueryEngineTest, Eq) {
  auto make_ne = [](BValue lhs, BValue rhs, FunctionBuilder* fb) {
    fb->Eq(lhs, rhs);
  };
  EXPECT_THAT(RunOnBinaryOp("0bXX1", "0b110", make_ne), IsOkAndHolds("0b0"));
  EXPECT_THAT(RunOnBinaryOp("0bXX1", "0b111", make_ne), IsOkAndHolds("0bX"));
  EXPECT_THAT(RunOnBinaryOp("0b011", "0b111", make_ne), IsOkAndHolds("0b0"));
  EXPECT_THAT(RunOnBinaryOp("0b011", "0b011", make_ne), IsOkAndHolds("0b1"));
  auto run_compound = [&](const TValue& left,
                          const TValue& right) -> absl::StatusOr<std::string> {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    XLS_ASSIGN_OR_RETURN(auto l, MakeValueWithKnownBits("left", left, &fb));
    XLS_ASSIGN_OR_RETURN(auto r, MakeValueWithKnownBits("right", right, &fb));
    BValue result = fb.Eq(l, r);
    XLS_ASSIGN_OR_RETURN(Function * f, fb.Build());
    TernaryQueryEngine tqe;
    XLS_RETURN_IF_ERROR(tqe.Populate(f).status());
    XLS_RET_CHECK(tqe.IsTracked(result.node()));
    return tqe.ToString(result.node());
  };
  EXPECT_THAT(run_compound(TValue::Tuple({}), TValue::Tuple({})),
              IsOkAndHolds("0b1"));
  EXPECT_THAT(run_compound(TValue::Array({"0b110", "0b101"}),
                           TValue::Array({"0b110", "0b101"})),
              IsOkAndHolds("0b1"));
  EXPECT_THAT(run_compound(TValue::Array({"0bX10", "0b101"}),
                           TValue::Array({"0b110", "0b101"})),
              IsOkAndHolds("0bX"));
  EXPECT_THAT(run_compound(TValue::Array({"0bX10", "0b001"}),
                           TValue::Array({"0b110", "0b101"})),
              IsOkAndHolds("0b0"));
  EXPECT_THAT(run_compound(TValue::Array({"0b110", "0b0XX"}),
                           TValue::Array({"0b110", "0b1XX"})),
              IsOkAndHolds("0b0"));
  EXPECT_THAT(run_compound(TValue::Array({"0bX10", "0b0XX"}),
                           TValue::Array({"0b110", "0b1XX"})),
              IsOkAndHolds("0b0"));
  EXPECT_THAT(run_compound(TValue::Array({"0bX10", "0b0XX"}),
                           TValue::Array({"0bX10", "0bXXX"})),
              IsOkAndHolds("0bX"));
  EXPECT_THAT(run_compound(TValue::Tuple({"0b110", "0b101"}),
                           TValue::Tuple({"0b110", "0b101"})),
              IsOkAndHolds("0b1"));
  EXPECT_THAT(run_compound(TValue::Tuple({"0bX10", "0b101"}),
                           TValue::Tuple({"0b110", "0b101"})),
              IsOkAndHolds("0bX"));
  EXPECT_THAT(run_compound(TValue::Tuple({"0bX10", "0b001"}),
                           TValue::Tuple({"0b110", "0b101"})),
              IsOkAndHolds("0b0"));
  EXPECT_THAT(run_compound(TValue::Tuple({"0b110", "0b0XX"}),
                           TValue::Tuple({"0b110", "0b1XX"})),
              IsOkAndHolds("0b0"));
  EXPECT_THAT(run_compound(TValue::Tuple({"0bX10", "0b0XX"}),
                           TValue::Tuple({"0b110", "0b1XX"})),
              IsOkAndHolds("0b0"));
  EXPECT_THAT(run_compound(TValue::Tuple({"0bX10", "0b0XX"}),
                           TValue::Tuple({"0bX10", "0bXXX"})),
              IsOkAndHolds("0bX"));
}

TEST_F(TernaryQueryEngineTest, Gate) {
  auto make_gate = [](BValue lhs, BValue rhs, FunctionBuilder* fb) {
    fb->Gate(lhs, rhs);
  };
  EXPECT_THAT(RunOnBinaryOp("0bX", "0b110", make_gate), IsOkAndHolds("0bXX0"));
  EXPECT_THAT(RunOnBinaryOp("0bX", "0b111", make_gate), IsOkAndHolds("0bXXX"));
  EXPECT_THAT(RunOnBinaryOp("0bX", "0b0X1", make_gate), IsOkAndHolds("0b0XX"));
  EXPECT_THAT(RunOnBinaryOp("0b1", "0b110", make_gate), IsOkAndHolds("0b110"));
  EXPECT_THAT(RunOnBinaryOp("0b1", "0b111", make_gate), IsOkAndHolds("0b111"));
  EXPECT_THAT(RunOnBinaryOp("0b1", "0b0X1", make_gate), IsOkAndHolds("0b0X1"));
  EXPECT_THAT(RunOnBinaryOp("0b0", "0b110", make_gate), IsOkAndHolds("0b000"));
  EXPECT_THAT(RunOnBinaryOp("0b0", "0b111", make_gate), IsOkAndHolds("0b000"));
  EXPECT_THAT(RunOnBinaryOp("0b0", "0b0X1", make_gate), IsOkAndHolds("0b000"));
}

TEST_F(TernaryQueryEngineTest, GateCompound) {
  auto test_with_condition = [&](std::string_view gate_cond,
                                 std::string_view tup_v, std::string_view arr_1,
                                 std::string_view arr_2) {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    XLS_ASSERT_OK_AND_ASSIGN(
        BValue compound,
        MakeValueWithKnownBits(
            "compound",
            TValue::Tuple({"0bX10", TValue::Array({"0b0X", "0b10"})}), &fb));
    BValue gate =
        fb.Gate(MakeValueWithKnownBits("gate_cond", gate_cond, &fb), compound);
    BValue tup_out = fb.TupleIndex(gate, 0);
    BValue tup_arr = fb.TupleIndex(gate, 1);
    BValue arr_1_out = fb.ArrayIndex(tup_arr, {fb.Literal(UBits(0, 1))});
    BValue arr_2_out = fb.ArrayIndex(tup_arr, {fb.Literal(UBits(1, 1))});
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
    VLOG(3) << f->DumpIr();
    TernaryQueryEngine query_engine;
    XLS_ASSERT_OK(query_engine.Populate(f).status());
    EXPECT_EQ(query_engine.ToString(tup_out.node()), tup_v);
    EXPECT_EQ(query_engine.ToString(arr_1_out.node()), arr_1);
    EXPECT_EQ(query_engine.ToString(arr_2_out.node()), arr_2);
  };
  test_with_condition("0bX", "0bXX0", "0b0X", "0bX0");
  test_with_condition("0b1", "0bX10", "0b0X", "0b10");
  test_with_condition("0b0", "0b000", "0b00", "0b00");
}

TEST_F(TernaryQueryEngineTest, Sel) {
  // No default.
  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    auto out = fb.Select(MakeValueWithKnownBits("selector", "0b0X", &fb),
                         {
                             MakeValueWithKnownBits("p1", "0b000X", &fb),
                             MakeValueWithKnownBits("p2", "0b00X0", &fb),
                             MakeValueWithKnownBits("p3", "0b0X00", &fb),
                             MakeValueWithKnownBits("p4", "0bX000", &fb),
                         });
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
    TernaryQueryEngine tqe;
    XLS_ASSERT_OK(tqe.Populate(f).status());
    EXPECT_EQ(tqe.ToString(out.node()), "0b00XX");
  }
  // with default.
  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    auto out =
        fb.Select(MakeValueWithKnownBits("selector", "0bX0", &fb),
                  {
                      MakeValueWithKnownBits("p1", "0b00X", &fb),
                      MakeValueWithKnownBits("p2", "0b0X0", &fb),
                  },
                  /*default_value=*/MakeValueWithKnownBits("p3", "0bX00", &fb));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
    TernaryQueryEngine tqe;
    XLS_ASSERT_OK(tqe.Populate(f).status());
    EXPECT_EQ(tqe.ToString(out.node()), "0bX0X");
  }
  // compound
  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    XLS_ASSERT_OK_AND_ASSIGN(
        auto p1,
        MakeValueWithKnownBits(
            "p1", TValue::Tuple({"0b00X", TValue::Array({"0b00X", "0b00X"})}),
            &fb));
    XLS_ASSERT_OK_AND_ASSIGN(
        auto p2,
        MakeValueWithKnownBits(
            "p2", TValue::Tuple({"0b0X0", TValue::Array({"0b0X0", "0b0X0"})}),
            &fb));
    XLS_ASSERT_OK_AND_ASSIGN(
        auto p3,
        MakeValueWithKnownBits(
            "p3", TValue::Tuple({"0bX00", TValue::Array({"0bX00", "0bX00"})}),
            &fb));
    auto out =
        fb.Select(MakeValueWithKnownBits("selector", "0bX0", &fb), {p1, p2},
                  /*default_value=*/p3);
    auto tup_v = fb.TupleIndex(out, 0);
    auto tup_arr = fb.TupleIndex(out, 1);
    auto arr_0 = fb.ArrayIndex(tup_arr, {fb.Literal(UBits(0, 1))});
    auto arr_1 = fb.ArrayIndex(tup_arr, {fb.Literal(UBits(1, 1))});
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
    TernaryQueryEngine tqe;
    XLS_ASSERT_OK(tqe.Populate(f).status());
    EXPECT_EQ(tqe.ToString(tup_v.node()), "0bX0X");
    EXPECT_EQ(tqe.ToString(arr_0.node()), "0bX0X");
    EXPECT_EQ(tqe.ToString(arr_1.node()), "0bX0X");
  }
}

TEST_F(TernaryQueryEngineTest, OneHotSel) {
  // bits
  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    auto out =
        fb.OneHotSelect(MakeValueWithKnownBits("selector", "0b00XX", &fb),
                        {
                            MakeValueWithKnownBits("p1", "0b000X", &fb),
                            MakeValueWithKnownBits("p2", "0b00X0", &fb),
                            MakeValueWithKnownBits("p3", "0b0X00", &fb),
                            MakeValueWithKnownBits("p4", "0bX000", &fb),
                        });
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
    TernaryQueryEngine tqe;
    XLS_ASSERT_OK(tqe.Populate(f).status());
    EXPECT_EQ(tqe.ToString(out.node()), "0b00XX");
  }
  // compound
  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    XLS_ASSERT_OK_AND_ASSIGN(
        auto p1,
        MakeValueWithKnownBits(
            "p1",
            TValue::Tuple({"0b000X", TValue::Array({"0b000X", "0b000X"})}),
            &fb));
    XLS_ASSERT_OK_AND_ASSIGN(
        auto p2,
        MakeValueWithKnownBits(
            "p2",
            TValue::Tuple({"0b00X0", TValue::Array({"0b00X0", "0b00X0"})}),
            &fb));
    XLS_ASSERT_OK_AND_ASSIGN(
        auto p3,
        MakeValueWithKnownBits(
            "p3",
            TValue::Tuple({"0b0X00", TValue::Array({"0b0X00", "0b0X00"})}),
            &fb));
    XLS_ASSERT_OK_AND_ASSIGN(
        auto p4,
        MakeValueWithKnownBits(
            "p4",
            TValue::Tuple({"0bX000", TValue::Array({"0bX000", "0bX000"})}),
            &fb));
    auto out = fb.OneHotSelect(
        MakeValueWithKnownBits("selector", "0b00XX", &fb), {p1, p2, p3, p4});
    auto tup_v = fb.TupleIndex(out, 0);
    auto tup_arr = fb.TupleIndex(out, 1);
    auto arr_0 = fb.ArrayIndex(tup_arr, {fb.Literal(UBits(0, 1))});
    auto arr_1 = fb.ArrayIndex(tup_arr, {fb.Literal(UBits(1, 1))});
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
    TernaryQueryEngine tqe;
    XLS_ASSERT_OK(tqe.Populate(f).status());
    EXPECT_EQ(tqe.ToString(tup_v.node()), "0b00XX");
    EXPECT_EQ(tqe.ToString(arr_0.node()), "0b00XX");
    EXPECT_EQ(tqe.ToString(arr_1.node()), "0b00XX");
  }
}

TEST_F(TernaryQueryEngineTest, OneHotSelOfOneHot) {
  // bits
  constexpr std::string_view kP1Value = "0b0000001X";
  constexpr std::string_view kP2Value = "0b00001X10";
  constexpr std::string_view kP3Value = "0b001X1000";
  // NB The one-hot instruction means that p4 is always possible.
  constexpr std::string_view kP4Value = "0b1X101010";
  constexpr std::string_view kP1Or2Value = "0bXXX0_XX1X";
  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    auto out = fb.OneHotSelect(
        fb.OneHot(MakeValueWithKnownBits("selector", "0b0XX", &fb),
                  LsbOrMsb::kMsb),
        {
            MakeValueWithKnownBits("p1", kP1Value, &fb),
            MakeValueWithKnownBits("p2", kP2Value, &fb),
            MakeValueWithKnownBits("p3", kP3Value, &fb),
            MakeValueWithKnownBits("p4", kP4Value, &fb),
        });
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
    TernaryQueryEngine tqe;
    XLS_ASSERT_OK(tqe.Populate(f).status());
    EXPECT_EQ(tqe.ToString(out.node()), kP1Or2Value);
  }
  // compound
  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    XLS_ASSERT_OK_AND_ASSIGN(
        auto p1,
        MakeValueWithKnownBits(
            "p1",
            TValue::Tuple({kP1Value, TValue::Array({kP1Value, kP1Value})}),
            &fb));
    XLS_ASSERT_OK_AND_ASSIGN(
        auto p2,
        MakeValueWithKnownBits(
            "p2",
            TValue::Tuple({kP2Value, TValue::Array({kP2Value, kP2Value})}),
            &fb));
    XLS_ASSERT_OK_AND_ASSIGN(
        auto p3,
        MakeValueWithKnownBits(
            "p3",
            TValue::Tuple({kP3Value, TValue::Array({kP3Value, kP3Value})}),
            &fb));
    XLS_ASSERT_OK_AND_ASSIGN(
        auto p4,
        MakeValueWithKnownBits(
            "p4",
            TValue::Tuple({kP4Value, TValue::Array({kP4Value, kP4Value})}),
            &fb));
    auto out = fb.OneHotSelect(
        fb.OneHot(MakeValueWithKnownBits("selector", "0b0XX", &fb),
                  LsbOrMsb::kMsb),
        {p1, p2, p3, p4});
    auto tup_v = fb.TupleIndex(out, 0);
    auto tup_arr = fb.TupleIndex(out, 1);
    auto arr_0 = fb.ArrayIndex(tup_arr, {fb.Literal(UBits(0, 1))});
    auto arr_1 = fb.ArrayIndex(tup_arr, {fb.Literal(UBits(1, 1))});
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
    TernaryQueryEngine tqe;
    XLS_ASSERT_OK(tqe.Populate(f).status());
    EXPECT_EQ(tqe.ToString(tup_v.node()), kP1Or2Value);
    EXPECT_EQ(tqe.ToString(arr_0.node()), kP1Or2Value);
    EXPECT_EQ(tqe.ToString(arr_1.node()), kP1Or2Value);
  }
}

TEST_F(TernaryQueryEngineTest, PrioritySel) {
  // bits
  constexpr std::string_view kP1Value = "0b0000001X";
  constexpr std::string_view kP2Value = "0b00001X10";
  constexpr std::string_view kP3Value = "0b001X1000";
  constexpr std::string_view kP4Value = "0b1X100000";
  // NB Join with implicit 0 value leads to all 4 being Xs
  constexpr std::string_view kP1Or2Value = "0b0000_XXXX";
  // Bits
  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    auto out =
        fb.PrioritySelect(MakeValueWithKnownBits("selector", "0b00XX", &fb),
                          {
                              MakeValueWithKnownBits("p1", kP1Value, &fb),
                              MakeValueWithKnownBits("p2", kP2Value, &fb),
                              MakeValueWithKnownBits("p3", kP3Value, &fb),
                              MakeValueWithKnownBits("p4", kP4Value, &fb),
                          });
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
    TernaryQueryEngine tqe;
    XLS_ASSERT_OK(tqe.Populate(f).status());
    EXPECT_EQ(tqe.ToString(out.node()), kP1Or2Value);
  }
  // compound
  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    XLS_ASSERT_OK_AND_ASSIGN(
        auto p1,
        MakeValueWithKnownBits(
            "p1",
            TValue::Tuple({kP1Value, TValue::Array({kP1Value, kP1Value})}),
            &fb));
    XLS_ASSERT_OK_AND_ASSIGN(
        auto p2,
        MakeValueWithKnownBits(
            "p2",
            TValue::Tuple({kP2Value, TValue::Array({kP2Value, kP2Value})}),
            &fb));
    XLS_ASSERT_OK_AND_ASSIGN(
        auto p3,
        MakeValueWithKnownBits(
            "p3",
            TValue::Tuple({kP3Value, TValue::Array({kP3Value, kP3Value})}),
            &fb));
    XLS_ASSERT_OK_AND_ASSIGN(
        auto p4,
        MakeValueWithKnownBits(
            "p4",
            TValue::Tuple({kP4Value, TValue::Array({kP4Value, kP4Value})}),
            &fb));
    auto out = fb.PrioritySelect(
        MakeValueWithKnownBits("selector", "0b00XX", &fb), {p1, p2, p3, p4});
    auto tup_v = fb.TupleIndex(out, 0);
    auto tup_arr = fb.TupleIndex(out, 1);
    auto arr_0 = fb.ArrayIndex(tup_arr, {fb.Literal(UBits(0, 1))});
    auto arr_1 = fb.ArrayIndex(tup_arr, {fb.Literal(UBits(1, 1))});
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
    TernaryQueryEngine tqe;
    XLS_ASSERT_OK(tqe.Populate(f).status());
    EXPECT_EQ(tqe.ToString(tup_v.node()), kP1Or2Value);
    EXPECT_EQ(tqe.ToString(arr_0.node()), kP1Or2Value);
    EXPECT_EQ(tqe.ToString(arr_1.node()), kP1Or2Value);
  }
}
TEST_F(TernaryQueryEngineTest, PrioritySelWithOneHot) {
  // bits
  constexpr std::string_view kP1Value = "0b0000001X";
  constexpr std::string_view kP2Value = "0b00001X10";
  constexpr std::string_view kP3Value = "0b001X1000";
  constexpr std::string_view kP4Value = "0b1X101010";
  constexpr std::string_view kP1Or2Value = "0bXXX0_XX1X";
  // Bits
  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    auto out = fb.PrioritySelect(
        fb.OneHot(MakeValueWithKnownBits("selector", "0b0XX", &fb),
                  LsbOrMsb::kMsb),
        {
            MakeValueWithKnownBits("p1", kP1Value, &fb),
            MakeValueWithKnownBits("p2", kP2Value, &fb),
            MakeValueWithKnownBits("p3", kP3Value, &fb),
            MakeValueWithKnownBits("p4", kP4Value, &fb),
        });
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
    TernaryQueryEngine tqe;
    XLS_ASSERT_OK(tqe.Populate(f).status());
    EXPECT_EQ(tqe.ToString(out.node()), kP1Or2Value);
  }
  // compound
  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    XLS_ASSERT_OK_AND_ASSIGN(
        auto p1,
        MakeValueWithKnownBits(
            "p1",
            TValue::Tuple({kP1Value, TValue::Array({kP1Value, kP1Value})}),
            &fb));
    XLS_ASSERT_OK_AND_ASSIGN(
        auto p2,
        MakeValueWithKnownBits(
            "p2",
            TValue::Tuple({kP2Value, TValue::Array({kP2Value, kP2Value})}),
            &fb));
    XLS_ASSERT_OK_AND_ASSIGN(
        auto p3,
        MakeValueWithKnownBits(
            "p3",
            TValue::Tuple({kP3Value, TValue::Array({kP3Value, kP3Value})}),
            &fb));
    XLS_ASSERT_OK_AND_ASSIGN(
        auto p4,
        MakeValueWithKnownBits(
            "p4",
            TValue::Tuple({kP4Value, TValue::Array({kP4Value, kP4Value})}),
            &fb));
    auto out = fb.PrioritySelect(
        fb.OneHot(MakeValueWithKnownBits("selector", "0b0XX", &fb),
                  LsbOrMsb::kMsb),
        {p1, p2, p3, p4});
    auto tup_v = fb.TupleIndex(out, 0);
    auto tup_arr = fb.TupleIndex(out, 1);
    auto arr_0 = fb.ArrayIndex(tup_arr, {fb.Literal(UBits(0, 1))});
    auto arr_1 = fb.ArrayIndex(tup_arr, {fb.Literal(UBits(1, 1))});
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
    TernaryQueryEngine tqe;
    XLS_ASSERT_OK(tqe.Populate(f).status());
    EXPECT_EQ(tqe.ToString(tup_v.node()), kP1Or2Value);
    EXPECT_EQ(tqe.ToString(arr_0.node()), kP1Or2Value);
    EXPECT_EQ(tqe.ToString(arr_1.node()), kP1Or2Value);
  }
}

TEST_F(TernaryQueryEngineTest, PrioritySelWithKnownBit) {
  // bits
  constexpr std::string_view kP1Value = "0b0000_001X";
  constexpr std::string_view kP2Value = "0b0000_1X10";
  constexpr std::string_view kP3Value = "0b001X_1000";
  constexpr std::string_view kP4Value = "0b1X10_0000";
  constexpr std::string_view kP1Or2Value = "0b0000_XXXX";
  // Bits
  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    auto out =
        fb.PrioritySelect(MakeValueWithKnownBits("selector", "0b0X1X", &fb),
                          {
                              MakeValueWithKnownBits("p1", kP1Value, &fb),
                              MakeValueWithKnownBits("p2", kP2Value, &fb),
                              MakeValueWithKnownBits("p3", kP3Value, &fb),
                              MakeValueWithKnownBits("p4", kP4Value, &fb),
                          });
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
    TernaryQueryEngine tqe;
    XLS_ASSERT_OK(tqe.Populate(f).status());
    EXPECT_EQ(tqe.ToString(out.node()), kP1Or2Value);
  }
  // compound
  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    XLS_ASSERT_OK_AND_ASSIGN(
        auto p1,
        MakeValueWithKnownBits(
            "p1",
            TValue::Tuple({kP1Value, TValue::Array({kP1Value, kP1Value})}),
            &fb));
    XLS_ASSERT_OK_AND_ASSIGN(
        auto p2,
        MakeValueWithKnownBits(
            "p2",
            TValue::Tuple({kP2Value, TValue::Array({kP2Value, kP2Value})}),
            &fb));
    XLS_ASSERT_OK_AND_ASSIGN(
        auto p3,
        MakeValueWithKnownBits(
            "p3",
            TValue::Tuple({kP3Value, TValue::Array({kP3Value, kP3Value})}),
            &fb));
    XLS_ASSERT_OK_AND_ASSIGN(
        auto p4,
        MakeValueWithKnownBits(
            "p4",
            TValue::Tuple({kP4Value, TValue::Array({kP4Value, kP4Value})}),
            &fb));
    auto out = fb.PrioritySelect(
        MakeValueWithKnownBits("selector", "0b0X1X", &fb), {p1, p2, p3, p4});
    auto tup_v = fb.TupleIndex(out, 0);
    auto tup_arr = fb.TupleIndex(out, 1);
    auto arr_0 = fb.ArrayIndex(tup_arr, {fb.Literal(UBits(0, 1))});
    auto arr_1 = fb.ArrayIndex(tup_arr, {fb.Literal(UBits(1, 1))});
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
    TernaryQueryEngine tqe;
    XLS_ASSERT_OK(tqe.Populate(f).status());
    EXPECT_EQ(tqe.ToString(tup_v.node()), kP1Or2Value);
    EXPECT_EQ(tqe.ToString(arr_0.node()), kP1Or2Value);
    EXPECT_EQ(tqe.ToString(arr_1.node()), kP1Or2Value);
  }
}

TEST_F(TernaryQueryEngineTest, Identity) {
  {
    auto package = CreatePackage();
    FunctionBuilder fb(TestName(), package.get());
    auto out = fb.Identity(MakeValueWithKnownBits("inp", "0b01X", &fb));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
    TernaryQueryEngine query_engine;
    XLS_ASSERT_OK(query_engine.Populate(f).status());
    EXPECT_EQ(query_engine.ToString(out.node()), "0b01X");
  }
  {
    auto package = CreatePackage();
    FunctionBuilder fb(TestName(), package.get());
    XLS_ASSERT_OK_AND_ASSIGN(
        BValue compound,
        MakeValueWithKnownBits(
            "compound",
            TValue::Tuple({"0bX10", TValue::Array({"0b0X", "0b10"})}), &fb));
    BValue ident = fb.Identity(compound);
    BValue tup_out = fb.TupleIndex(ident, 0);
    BValue tup_arr = fb.TupleIndex(ident, 1);
    BValue arr_1_out = fb.ArrayIndex(tup_arr, {fb.Literal(UBits(0, 1))});
    BValue arr_2_out = fb.ArrayIndex(tup_arr, {fb.Literal(UBits(1, 1))});
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
    VLOG(3) << f->DumpIr();
    TernaryQueryEngine query_engine;
    XLS_ASSERT_OK(query_engine.Populate(f).status());
    EXPECT_EQ(query_engine.ToString(tup_out.node()), "0bX10");
    EXPECT_EQ(query_engine.ToString(arr_1_out.node()), "0b0X");
    EXPECT_EQ(query_engine.ToString(arr_2_out.node()), "0b10");
  }
}

TEST_F(TernaryQueryEngineTest, EmptyCompoundIdentity) {
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  fb.Identity(fb.Tuple({}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());
}

TEST_F(TernaryQueryEngineTest, ArrayIndexJoins) {
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      BValue array,
      MakeValueWithKnownBits("array", TValue::ArrayS({"0b000", "0b001"}), &fb));
  BValue arr_value_idx = MakeValueWithKnownBits("arr_value_idx", "0bX", &fb);
  BValue result = fb.ArrayIndex(array, {arr_value_idx});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  VLOG(3) << f->DumpIr();
  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  EXPECT_THAT(query_engine.ToString(result.node()), "0b00X");
}

TEST_F(TernaryQueryEngineTest, ArrayIndexTooLargeGetsBottom) {
  // How many layers to use to ensure the analysis gives up.
  static constexpr int64_t kTooManyLayers = 16;
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  ValueBuilder cur = ValueBuilder::Bits(UBits(0, 1));
  for (int64_t i = 0; i < kTooManyLayers; ++i) {
    cur = ValueBuilder::Array({cur, cur});
  }
  // array is:
  // [
  //   ...
  //   [
  //     [
  //       [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
  //       [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
  //     ], ...
  //   ], [ ... ]
  //   ...
  // ]
  // With 2**16 bottom entries every one of which is 0b0.
  XLS_ASSERT_OK_AND_ASSIGN(auto array_value, cur.Build());
  BValue array = fb.Literal(array_value);
  BValue arr_value_idx = MakeValueWithKnownBits("arr_value_idx", "0bX", &fb);
  // Try to get some unconstrained element. 16 unknown bits.
  BValue result =
      fb.ArrayIndex(array, std::vector<BValue>(kTooManyLayers, arr_value_idx));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  EXPECT_THAT(query_engine.ToString(result.node()), "0bX");
}

TEST_F(TernaryQueryEngineTest, ArrayIndexLiteralResult) {
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  BValue arr_value_idx = MakeValueWithKnownBits("arr_value_idx", "0b0X", &fb);
  XLS_ASSERT_OK_AND_ASSIGN(auto array_value,
                           ValueBuilder::UBitsArray(
                               {
                                   0,  // possible
                                   0,  // possible
                                   1,  // impossible
                                   1,  // impossible
                               },
                               3)
                               .Build());
  BValue array = fb.Literal(array_value);
  BValue result = fb.ArrayIndex(array, {arr_value_idx});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  VLOG(3) << f->DumpIr();
  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  EXPECT_THAT(query_engine.ToString(result.node()), "0b000");
}

TEST_F(TernaryQueryEngineTest, ArrayIndexExact) {
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  BValue arr_value_idx = MakeValueWithKnownBits("arr_value_idx", "0b000", &fb);
  XLS_ASSERT_OK_AND_ASSIGN(BValue array, MakeValueWithKnownBits("array",
                                                                TValue::ArrayS({
                                                                    "0b00X",
                                                                    "0b1X1",
                                                                    "0bX11",
                                                                    "0b1XX",
                                                                }),
                                                                &fb));
  BValue result = fb.ArrayIndex(array, {arr_value_idx});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  VLOG(3) << f->DumpIr();
  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  EXPECT_THAT(query_engine.ToString(result.node()), "0b00X");
}

TEST_F(TernaryQueryEngineTest, ArrayOperationResult) {
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  BValue arr_value_one = MakeValueWithKnownBits("arr_value_one", "0b000", &fb);
  BValue arr_value_two = MakeValueWithKnownBits("arr_value_two", "0b001", &fb);
  BValue arr_value_three =
      MakeValueWithKnownBits("arr_value_three", "0b110", &fb);
  BValue arr_value_idx = MakeValueWithKnownBits("arr_value_idx", "0b0X", &fb);
  BValue array = fb.Array(
      {
          arr_value_one,    // 0b00
          arr_value_two,    // 0b01
          arr_value_three,  // 0b10
          arr_value_three,  // 0b11
      },
      arr_value_one.GetType());
  BValue result = fb.ArrayIndex(array, {arr_value_idx});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  VLOG(3) << f->DumpIr();
  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  EXPECT_THAT(query_engine.ToString(result.node()), "0b00X");
}

TEST_F(TernaryQueryEngineTest, ArrayIndexOOB) {
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      BValue array,
      MakeValueWithKnownBits(TestName(),
                             TValue::Array({
                                 TValue::Array({"0b000", "0b000", "0b1X1"}),
                                 TValue::Array({"0b000", "0b000", "0bX11"}),
                                 TValue::Array({"0b000", "0b000", "0b11X"}),
                             }),
                             &fb));
  // Known to be min 4. OOB for both index.
  BValue off = MakeValueWithKnownBits("off", "0b1XX", &fb);
  BValue result = fb.ArrayIndex(array, {off, off});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RecordProperty("ir", f->DumpIr());
  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  EXPECT_THAT(query_engine.ToString(result.node()), "0b11X");
}

TEST_F(TernaryQueryEngineTest, ArrayIndexFullyKnown) {
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      BValue array,
      MakeValueWithKnownBits(
          "array", TValue::Array({"0bXXX", "0bXXX", "0b001", "0bXXX"}), &fb));
  BValue idx_2 = MakeValueWithKnownBits("idx_3", "0b10", &fb);
  BValue result = fb.ArrayIndex(array, {idx_2});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  VLOG(3) << f->DumpIr();
  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  EXPECT_THAT(query_engine.ToString(result.node()), "0b001");
}

TEST_F(TernaryQueryEngineTest, ArrayIndexFullyKnownIsNegOne63Bit) {
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      BValue array,
      MakeValueWithKnownBits(
          "array", TValue::Array({"0bXXX", "0bXXX", "0bXXX", "0b001"}), &fb));
  BValue idx_2 = MakeValueWithKnownBits("idx_3",
                                        "0b111_1111_1111_1111"
                                        "_1111_1111_1111_1111"
                                        "_1111_1111_1111_1111"
                                        "_1111_1111_1111_1111",
                                        &fb);
  ASSERT_EQ(idx_2.BitCountOrDie(), 63);
  BValue result = fb.ArrayIndex(array, {idx_2});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  VLOG(3) << f->DumpIr();
  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  EXPECT_THAT(query_engine.ToString(result.node()), "0b001");
}
TEST_F(TernaryQueryEngineTest, ArrayIndexFullyKnownIsNegOne64Bit) {
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      BValue array,
      MakeValueWithKnownBits(
          "array", TValue::Array({"0bXXX", "0bXXX", "0bXXX", "0b001"}), &fb));
  BValue idx_2 = MakeValueWithKnownBits("idx_3",
                                        "0b1111_1111_1111_1111"
                                        "_1111_1111_1111_1111"
                                        "_1111_1111_1111_1111"
                                        "_1111_1111_1111_1111",
                                        &fb);
  ASSERT_EQ(idx_2.BitCountOrDie(), 64);
  BValue result = fb.ArrayIndex(array, {idx_2});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  VLOG(3) << f->DumpIr();
  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  EXPECT_THAT(query_engine.ToString(result.node()), "0b001");
}
TEST_F(TernaryQueryEngineTest, ArrayIndexFullyKnownIsNegOne128Bit) {
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      BValue array,
      MakeValueWithKnownBits(
          "array", TValue::Array({"0bXXX", "0bXXX", "0bXXX", "0b001"}), &fb));
  BValue idx_2 = MakeValueWithKnownBits("idx_3",
                                        "0b1111_1111_1111_1111"
                                        "_1111_1111_1111_1111"
                                        "_1111_1111_1111_1111"
                                        "_1111_1111_1111_1111"
                                        "_1111_1111_1111_1111"
                                        "_1111_1111_1111_1111"
                                        "_1111_1111_1111_1111"
                                        "_1111_1111_1111_1111",
                                        &fb);
  ASSERT_EQ(idx_2.BitCountOrDie(), 128);
  BValue result = fb.ArrayIndex(array, {idx_2});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  VLOG(3) << f->DumpIr();
  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  EXPECT_THAT(query_engine.ToString(result.node()), "0b001");
}

TEST_F(TernaryQueryEngineTest, ArrayIndexClamps) {
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  BValue arr_value_idx = MakeValueWithKnownBits("arr_value_idx", "0bXX0", &fb);
  XLS_ASSERT_OK_AND_ASSIGN(Value array_const,
                           ValueBuilder::UBitsArray(
                               {
                                   0b001,  // possible
                                   0b000,
                                   0b001,  // possible
                                   0b011,  // possible through OOB.
                               },
                               3)
                               .Build());
  BValue array = fb.Literal(array_const);
  BValue result = fb.ArrayIndex(array, {arr_value_idx});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  VLOG(3) << f->DumpIr();
  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  EXPECT_THAT(query_engine.ToString(result.node()), "0b0X1");
}

TEST_F(TernaryQueryEngineTest, ArrayIndexMultiDimensionalConstantResult) {
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      BValue array,
      MakeValueWithKnownBits(
          "array",
          TValue::Array({
              TValue::Array({"0bXXXX", "0bXXXX", "0bXXXX", "0bXXXX"}),
              TValue::Array({"0bXXXX", "0bXXXX", "0bXXXX", "0bXXXX"}),
              TValue::Array({"0b0000", "0b0010", "0bXXXX", "0bXXXX"}),
              TValue::Array({"0b1010", "0b1000", "0bXXXX", "0bXXXX"}),
          }),  // only 4 fully known ones possible.
          &fb));
  BValue arr_value_idx_one =
      MakeValueWithKnownBits("arr_value_idx_one", "0b0X", &fb);
  BValue arr_value_idx_two =
      MakeValueWithKnownBits("arr_value_idx_two", "0b1X", &fb);
  BValue result = fb.ArrayIndex(array, {arr_value_idx_two, arr_value_idx_one});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  VLOG(3) << f->DumpIr();
  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  EXPECT_THAT(query_engine.ToString(result.node()), "0bX0X0");
}

TEST_F(TernaryQueryEngineTest, ArrayIndexMultiDimensionalSomeValuesKnown) {
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      BValue array,
      MakeValueWithKnownBits(
          "array",
          TValue::Array({
              TValue::Array({
                  TValue::Array({"0bX00", "0bX00"}),
                  TValue::Array({"0b000", "0b001"}),  // either possible
              }),
              TValue::Array({
                  TValue::Array({"0bX00", "0bX00"}),
                  TValue::Array({"0b010", "0b011"}),  // either possible
              }),
          }),
          &fb));
  BValue result =
      fb.ArrayIndex(array, {
                               MakeValueWithKnownBits("idx1", "0bX", &fb),
                               MakeValueWithKnownBits("idx2", "0b1", &fb),
                               MakeValueWithKnownBits("idx3", "0bX", &fb),
                           });
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RecordProperty("ir", f->DumpIr());
  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  EXPECT_THAT(query_engine.ToString(result.node()), "0b0XX");
}

TEST_F(TernaryQueryEngineTest, ArrayIndexMultiDimensionalSomeValuesKnown2) {
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      BValue array,
      MakeValueWithKnownBits("array",
                             TValue::Array({
                                 TValue::Array({
                                     TValue::Array({"0bX0", "0bX0"}),
                                     TValue::Array({"0bX0", "0bX0"}),
                                 }),
                                 TValue::Array({
                                     //        possibilities \/
                                     TValue::Array({"0bX0", "0b00"}),
                                     TValue::Array({"0bX0", "0b01"}),
                                 }),  //       possibilities ^
                             }),
                             &fb));
  BValue result =
      fb.ArrayIndex(array, {
                               MakeValueWithKnownBits("idx1", "0b1", &fb),
                               MakeValueWithKnownBits("idx2", "0bX", &fb),
                               MakeValueWithKnownBits("idx3", "0b1", &fb),
                           });
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RecordProperty("ir", f->DumpIr());
  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  EXPECT_THAT(query_engine.ToString(result.node()), "0b0X");
}

TEST_F(TernaryQueryEngineTest, ArrayIndexMultiDimensionalSomeValues3) {
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  BValue param = fb.Param("param", package->GetBitsType(2));
  ValueBuilder all_ones = ValueBuilder::UBitsArray(
      {
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
      },
      3);
  ValueBuilder zeros_and_ones = ValueBuilder::UBitsArray(
      {
          0,
          0,
          0,
          0,
          1,
          1,
          1,
          1,
      },
      3);
  XLS_ASSERT_OK_AND_ASSIGN(Value array_value,
                           // 4 0s 4 7s
                           ValueBuilder::Array({
                                                   zeros_and_ones,
                                                   zeros_and_ones,
                                                   zeros_and_ones,
                                                   zeros_and_ones,
                                                   all_ones,
                                                   all_ones,
                                                   all_ones,
                                                   all_ones,
                                               })
                               .Build());
  BValue arr_out = fb.Literal(array_value);
  BValue result = fb.ArrayIndex(arr_out, {param, param});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RecordProperty("ir", f->DumpIr());
  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  EXPECT_THAT(query_engine.ToString(result.node()), "0b000");
  EXPECT_TRUE(query_engine.IsTracked(result.node()));
}

TEST_F(TernaryQueryEngineTest, MultipleArrayIndex) {
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      BValue array,
      MakeValueWithKnownBits(
          "array",
          TValue::Array({
              TValue::Array({
                  TValue::Array({"0bX00", "0bX00"}),
                  TValue::Array({"0b000", "0b001"}),  // either possible
              }),
              TValue::Array({
                  TValue::Array({"0bX00", "0bX00"}),
                  TValue::Array({"0b010", "0b011"}),  // either possible
              }),
          }),
          &fb));
  BValue r1 =
      fb.ArrayIndex(array, {MakeValueWithKnownBits("idx1", "0bX", &fb)});
  BValue r2 = fb.ArrayIndex(r1, {MakeValueWithKnownBits("idx2", "0b1", &fb)});
  BValue result =
      fb.ArrayIndex(r2, {MakeValueWithKnownBits("idx3", "0bX", &fb)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RecordProperty("ir", f->DumpIr());
  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  EXPECT_THAT(query_engine.ToString(result.node()), "0b0XX");
}

TEST_F(TernaryQueryEngineTest, ArrayUpdate) {
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  // bits[12][3]
  XLS_ASSERT_OK_AND_ASSIGN(
      Value array_val,
      ValueBuilder::UBitsArray({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 3)
          .Build());
  BValue array = fb.Literal(array_val);
  // Either 9 or 11
  BValue idx = MakeValueWithKnownBits("idx", "0b10X1", &fb);
  BValue update = MakeValueWithKnownBits("update", "0b01X", &fb);
  BValue result = fb.ArrayUpdate(array, update, {idx});
  std::vector<BValue> reads;
  for (int64_t i = 0; i < array_val.size(); ++i) {
    reads.push_back(fb.ArrayIndex(result, {fb.Literal(UBits(i, 8))}));
  }
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RecordProperty("ir", f->DumpIr());
  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  ASSERT_THAT(reads, testing::SizeIs(14));
  EXPECT_THAT(query_engine.ToString(reads[0].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[1].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[2].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[3].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[4].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[5].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[6].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[7].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[8].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[9].node()), "0b0XX");
  EXPECT_THAT(query_engine.ToString(reads[10].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[11].node()), "0b0XX");
  EXPECT_THAT(query_engine.ToString(reads[12].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[13].node()), "0b000");
}

TEST_F(TernaryQueryEngineTest, ArrayUpdateExact) {
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      BValue array,
      MakeValueWithKnownBits("array",
                             TValue::Array({"0bXX", "0bXX", "0bXX", "0bXX",
                                            "0bXX", "0bXX", "0bXX", "0bXX"}),
                             &fb));
  BValue idx = fb.Literal(UBits(2, 3));
  BValue update = fb.Literal(UBits(0, 2));
  BValue result = fb.ArrayUpdate(array, update, {idx});
  std::vector<BValue> reads;
  for (int64_t i = 0; i < array.GetType()->AsArrayOrDie()->size(); ++i) {
    reads.push_back(fb.ArrayIndex(result, {fb.Literal(UBits(i, 8))}));
  }
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RecordProperty("ir", f->DumpIr());
  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  ASSERT_THAT(reads, testing::SizeIs(8));
  EXPECT_THAT(query_engine.ToString(reads[0].node()), "0bXX");
  EXPECT_THAT(query_engine.ToString(reads[1].node()), "0bXX");
  EXPECT_THAT(query_engine.ToString(reads[2].node()), "0b00");
  EXPECT_THAT(query_engine.ToString(reads[3].node()), "0bXX");
  EXPECT_THAT(query_engine.ToString(reads[4].node()), "0bXX");
  EXPECT_THAT(query_engine.ToString(reads[5].node()), "0bXX");
  EXPECT_THAT(query_engine.ToString(reads[6].node()), "0bXX");
  EXPECT_THAT(query_engine.ToString(reads[7].node()), "0bXX");
}

TEST_F(TernaryQueryEngineTest, ArrayUpdateExactOOB) {
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      BValue array,
      MakeValueWithKnownBits("array",
                             TValue::Array({"0b1X", "0b1X", "0b1X", "0b1X",
                                            "0b1X", "0b1X", "0b1X", "0b1X"}),
                             &fb));
  BValue idx = fb.Literal(UBits(20, 32));
  BValue update = fb.Literal(UBits(0, 2));
  BValue result = fb.ArrayUpdate(array, update, {idx});
  std::vector<BValue> reads;
  for (int64_t i = 0; i < array.GetType()->AsArrayOrDie()->size(); ++i) {
    reads.push_back(fb.ArrayIndex(result, {fb.Literal(UBits(i, 8))}));
  }
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RecordProperty("ir", f->DumpIr());
  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  ASSERT_THAT(reads, testing::SizeIs(8));
  EXPECT_THAT(query_engine.ToString(reads[0].node()), "0b1X");
  EXPECT_THAT(query_engine.ToString(reads[1].node()), "0b1X");
  EXPECT_THAT(query_engine.ToString(reads[2].node()), "0b1X");
  EXPECT_THAT(query_engine.ToString(reads[3].node()), "0b1X");
  EXPECT_THAT(query_engine.ToString(reads[4].node()), "0b1X");
  EXPECT_THAT(query_engine.ToString(reads[5].node()), "0b1X");
  EXPECT_THAT(query_engine.ToString(reads[6].node()), "0b1X");
  EXPECT_THAT(query_engine.ToString(reads[7].node()), "0b1X");
}

TEST_F(TernaryQueryEngineTest, ArrayUpdateSometimesOOB) {
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  // bits[12][3]
  XLS_ASSERT_OK_AND_ASSIGN(
      Value array_val,
      ValueBuilder::UBitsArray({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 3)
          .Build());
  BValue array = fb.Literal(array_val);
  // Either 9 or 11 in bounds
  BValue idx = MakeValueWithKnownBits("idx", "0bX10X1", &fb);
  BValue update = MakeValueWithKnownBits("update", "0b01X", &fb);
  BValue result = fb.ArrayUpdate(array, update, {idx});
  std::vector<BValue> reads;
  for (int64_t i = 0; i < array_val.size(); ++i) {
    reads.push_back(fb.ArrayIndex(result, {fb.Literal(UBits(i, 8))}));
  }
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RecordProperty("ir", f->DumpIr());
  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  ASSERT_THAT(reads, testing::SizeIs(14));
  EXPECT_THAT(query_engine.ToString(reads[0].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[1].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[2].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[3].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[4].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[5].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[6].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[7].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[8].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[9].node()), "0b0XX");
  EXPECT_THAT(query_engine.ToString(reads[10].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[11].node()), "0b0XX");
  EXPECT_THAT(query_engine.ToString(reads[12].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[13].node()), "0b000");
}

TEST_F(TernaryQueryEngineTest, ArrayUpdateOOB) {
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  // bits[12][3]
  XLS_ASSERT_OK_AND_ASSIGN(
      Value array_val,
      ValueBuilder::UBitsArray({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 3)
          .Build());
  BValue array = fb.Literal(array_val);
  // Always oob
  BValue idx = MakeValueWithKnownBits("idx", "0b1X10X1", &fb);
  BValue update = MakeValueWithKnownBits("update", "0b01X", &fb);
  BValue result = fb.ArrayUpdate(array, update, {idx});
  std::vector<BValue> reads;
  for (int64_t i = 0; i < array_val.size(); ++i) {
    reads.push_back(fb.ArrayIndex(result, {fb.Literal(UBits(i, 8))}));
  }
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RecordProperty("ir", f->DumpIr());
  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  ASSERT_THAT(reads, testing::SizeIs(14));
  EXPECT_THAT(query_engine.ToString(reads[0].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[1].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[2].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[3].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[4].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[5].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[6].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[7].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[8].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[9].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[10].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[11].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[12].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[13].node()), "0b000");
}

TEST_F(TernaryQueryEngineTest, ArrayUpdateMultiDimensional) {
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      Value array_val,
      ValueBuilder::UBits2DArray(
          {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}, 3)
          .Build());
  BValue array = fb.Literal(array_val);
  // Either 2 or 3
  BValue idx_1 = MakeValueWithKnownBits("idx1", "0b1X", &fb);
  // Either 0 or 2
  BValue idx_2 = MakeValueWithKnownBits("idx2", "0bX0", &fb);
  BValue update = MakeValueWithKnownBits("update", "0b01X", &fb);
  BValue result = fb.ArrayUpdate(array, update, {idx_1, idx_2});
  std::vector<std::vector<BValue>> reads;
  for (int64_t i = 0; i < array_val.size(); ++i) {
    auto& cur_reads = reads.emplace_back();
    for (int64_t j = 0; j < array_val.elements()[i].size(); ++j) {
      cur_reads.push_back(fb.ArrayIndex(
          result, {fb.Literal(UBits(i, 8)), fb.Literal(UBits(j, 8))}));
    }
  }
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RecordProperty("ir", f->DumpIr());
  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  ASSERT_THAT(reads, testing::SizeIs(4));
  ASSERT_THAT(reads, testing::Each(testing::SizeIs(4)));
  EXPECT_THAT(query_engine.ToString(reads[0][0].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[0][1].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[0][2].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[0][3].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[1][0].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[1][1].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[1][2].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[1][3].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[2][0].node()), "0b0XX");
  EXPECT_THAT(query_engine.ToString(reads[2][1].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[2][2].node()), "0b0XX");
  EXPECT_THAT(query_engine.ToString(reads[2][3].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[3][0].node()), "0b0XX");
  EXPECT_THAT(query_engine.ToString(reads[3][1].node()), "0b000");
  EXPECT_THAT(query_engine.ToString(reads[3][2].node()), "0b0XX");
  EXPECT_THAT(query_engine.ToString(reads[3][3].node()), "0b000");
}

TEST_F(TernaryQueryEngineTest, TupleIndex) {
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());
  XLS_ASSERT_OK_AND_ASSIGN(BValue tuple,
                           MakeValueWithKnownBits("tuple",
                                                  TValue::Tuple({
                                                      "0bX0",  // <- selected
                                                      "0b0X",
                                                  }),
                                                  &fb));
  BValue result = fb.TupleIndex(tuple, 0);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  VLOG(3) << f->DumpIr();
  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  EXPECT_THAT(query_engine.ToString(result.node()), "0bX0");
}

TEST_F(TernaryQueryEngineTest, TupleIndexIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      BValue tuple, MakeValueWithKnownBits("tuple",
                                           TValue::Tuple({
                                               TValue::Tuple({
                                                   "0b000X",
                                                   "0b00X0",
                                               }),
                                               TValue::Tuple({
                                                   "0b0X00",  // <- selected
                                                   "0bX000",
                                               }),
                                           }),
                                           &fb));
  BValue tup1 = fb.TupleIndex(tuple, 1);
  BValue tup2 = fb.TupleIndex(tup1, 0);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  VLOG(3) << f->DumpIr();
  TernaryQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  EXPECT_THAT(query_engine.ToString(tup2.node()), "0b0X00");
}

namespace {

class ArrayCreation : public benchmark_support::strategy::NaryNode {
 public:
  absl::StatusOr<BValue> GenerateInteriorPoint(
      FunctionBuilder& builder, absl::Span<BValue> inputs) const final {
    return builder.Array(inputs, inputs.front().GetType());
  }
};
}  // namespace

// Single level array
void BM_ArrayIndexExactShallow(benchmark::State& state) {
  auto p = std::make_unique<VerifiedPackage>("array_index_exact");
  FunctionBuilder fb("array_index_exact_shallow", p.get());
  BValue v = fb.Literal(UBits(0, 1));
  std::vector<BValue> values(int64_t{1} << state.range(0), v);
  BValue array = fb.Array(values, values.front().GetType());
  fb.ArrayIndex(array, {fb.Param("selector", p->GetBitsType(state.range(0)))});
  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  for (auto _ : state) {
    TernaryQueryEngine tqe;
    XLS_ASSERT_OK_AND_ASSIGN(auto r, tqe.Populate(f));
    benchmark::DoNotOptimize(r);
  }
}

// Deep tree, 1 bit per level, minimal number of instructions.
void BM_ArrayIndexExactDeep(benchmark::State& state) {
  auto p = std::make_unique<VerifiedPackage>("array_index_exact");
  FunctionBuilder fb("array_index_deep", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      BValue array,
      benchmark_support::GenerateFullyConnectedLayerGraph(
          fb, state.range(0), /*fan_out=*/2, ArrayCreation(),
          benchmark_support::strategy::SharedLiteral(UBits(0, 64))));
  std::vector<BValue> selectors;
  selectors.reserve(state.range(0));
  // NB extra layer for the top of the connected array layers.
  for (int i = 0; i < state.range(0) + 1; ++i) {
    selectors.push_back(
        fb.Param(absl::StrCat("selector_", i), p->GetBitsType(1)));
  }
  fb.ArrayIndex(array, selectors);
  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  for (auto _ : state) {
    TernaryQueryEngine tqe;
    XLS_ASSERT_OK_AND_ASSIGN(auto r, tqe.Populate(f));
    benchmark::DoNotOptimize(r);
  }
}

// Deep tree, 1 bit per level, Full tree.
void BM_ArrayIndexExactTree(benchmark::State& state) {
  auto p = std::make_unique<VerifiedPackage>("array_index_exact");
  FunctionBuilder fb("array_index_deep", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      BValue array,
      benchmark_support::GenerateBalancedTree(
          fb, state.range(0), /*fan_out=*/2, ArrayCreation(),
          benchmark_support::strategy::SharedLiteral(UBits(0, 64))));
  std::vector<BValue> selectors;
  selectors.reserve(state.range(0));
  for (int i = 0; i < state.range(0); ++i) {
    selectors.push_back(
        fb.Param(absl::StrCat("selector_", i), p->GetBitsType(1)));
  }
  fb.ArrayIndex(array, selectors);
  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  for (auto _ : state) {
    TernaryQueryEngine tqe;
    XLS_ASSERT_OK_AND_ASSIGN(auto r, tqe.Populate(f));
    benchmark::DoNotOptimize(r);
  }
}

BENCHMARK(BM_ArrayIndexExactDeep)->DenseRange(2, 14, 1);
BENCHMARK(BM_ArrayIndexExactShallow)->DenseRange(2, 14, 1);
BENCHMARK(BM_ArrayIndexExactTree)->DenseRange(2, 12, 1);

}  // namespace
}  // namespace xls
