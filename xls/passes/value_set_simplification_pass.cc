// Copyright 2026 The XLS Authors
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

#include "xls/passes/value_set_simplification_pass.h"

#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xls/common/math_util.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/area_model/area_estimator.h"
#include "xls/ir/bits.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/partial_info_query_engine.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {
namespace {

class TemporaryNodeScope {
 public:
  explicit TemporaryNodeScope(FunctionBase* f) : f_(f) {}
  ~TemporaryNodeScope() {
    for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
      CHECK_OK(f_->RemoveNode(*it));
    }
  }

  template <typename NodeT, typename... Args>
  absl::StatusOr<Node*> AddNode(Args&&... args) {
    XLS_ASSIGN_OR_RETURN(Node * n,
                         f_->MakeNode<NodeT>(std::forward<Args>(args)...));
    nodes_.push_back(n);
    return n;
  }

 private:
  FunctionBase* f_;
  std::vector<Node*> nodes_;
};

template <typename NodeT, typename... Args>
absl::StatusOr<double> GetAreaForNode(const AreaEstimator* area_estimator,
                                      FunctionBase* f, Args&&... args) {
  TemporaryNodeScope temp_nodes(f);
  XLS_ASSIGN_OR_RETURN(Node * temp,
                       temp_nodes.AddNode<NodeT>(std::forward<Args>(args)...));
  absl::StatusOr<double> area =
      area_estimator->GetOperationAreaInSquareMicrons(temp);
  return area;
}

absl::StatusOr<Node*> CreateDivZeroResult(Node* node, Node* dividend) {
  int64_t bit_width = dividend->BitCountOrDie();
  if (node->op() == Op::kUDiv) {
    return node->function_base()->MakeNode<Literal>(
        node->loc(), Value(Bits::AllOnes(bit_width)));
  }
  XLS_ASSIGN_OR_RETURN(Node * msb,
                       node->function_base()->MakeNode<BitSlice>(
                           node->loc(), dividend, bit_width - 1, 1));
  XLS_ASSIGN_OR_RETURN(Node * max_signed,
                       node->function_base()->MakeNode<Literal>(
                           node->loc(), Value(Bits::MaxSigned(bit_width))));
  XLS_ASSIGN_OR_RETURN(Node * min_signed,
                       node->function_base()->MakeNode<Literal>(
                           node->loc(), Value(Bits::MinSigned(bit_width))));
  return node->function_base()->MakeNode<Select>(
      node->loc(), msb, std::vector<Node*>{max_signed, min_signed},
      std::nullopt);
}

struct DivisionSimplificationSpec {
  IntervalSet intervals;
  int64_t non_pow2_constants_count;
  int64_t pow2_constants_count;
  int64_t max_pow2_shift;
  bool prefer_shift_by_encode;
};

static constexpr int64_t kMaxNonPowerOfTwoCasesToSplitDivision = 16;

absl::StatusOr<std::optional<DivisionSimplificationSpec>>
CheckMultipleConstantDivisionApplicability(Node* node,
                                           const QueryEngine& query_engine) {
  if (!node->OpIn({Op::kUDiv, Op::kSDiv})) {
    return std::nullopt;
  }

  IntervalSet intervals = query_engine.GetIntervals(node->operand(1)).Get({});
  if (!intervals.Size().has_value() || intervals.Size().value() <= 1) {
    return std::nullopt;
  }

  int64_t pow2_constants_count = 0;
  int64_t max_pow2_shift = 0;
  int64_t bit_width = node->BitCountOrDie();

  int64_t max_search = (node->op() == Op::kUDiv) ? bit_width : bit_width - 1;
  for (int64_t i = 0; i < max_search; ++i) {
    if (intervals.Covers(Bits::PowerOfTwo(i, bit_width))) {
      pow2_constants_count++;
      max_pow2_shift = i;
    }
  }

  int64_t non_pow2_constants_count = intervals.Size().value() -
                                     pow2_constants_count -
                                     (intervals.CoversZero() ? 1 : 0);

  return DivisionSimplificationSpec{
      .intervals = std::move(intervals),
      .non_pow2_constants_count = non_pow2_constants_count,
      .pow2_constants_count = pow2_constants_count,
      .max_pow2_shift = max_pow2_shift,
      .prefer_shift_by_encode = false,
  };
}

absl::StatusOr<std::optional<DivisionSimplificationSpec>>
IsMultipleConstantDivisionProfitable(Node* node,
                                     const DivisionSimplificationSpec& spec,
                                     const AreaEstimator* area_estimator) {
  FunctionBase* f = node->function_base();
  Node* dividend = node->operand(0);
  Node* divisor = node->operand(1);
  int64_t bit_width = node->BitCountOrDie();

  if (spec.non_pow2_constants_count > kMaxNonPowerOfTwoCasesToSplitDivision) {
    return std::nullopt;
  }

  bool prefer_shift_by_encode = false;
  bool profitable = false;

  if (area_estimator != nullptr) {
    double mul_area = 0;
    Op mul_op = (node->op() == Op::kUDiv) ? Op::kUMul : Op::kSMul;
    XLS_ASSIGN_OR_RETURN(mul_area, GetAreaForNode<ArithOp>(
                                       area_estimator, f, node->loc(), dividend,
                                       divisor, bit_width, mul_op));

    XLS_ASSIGN_OR_RETURN(double div_area,
                         area_estimator->GetOperationAreaInSquareMicrons(node));

    double all_constants_select_area = 0;
    int64_t all_constants_cases =
        spec.non_pow2_constants_count + spec.pow2_constants_count;
    if (all_constants_cases > 0) {
      TemporaryNodeScope temp_nodes_all(f);
      XLS_ASSIGN_OR_RETURN(
          Node * selector_all,
          temp_nodes_all.AddNode<Literal>(
              node->loc(), Value(UBits(0, all_constants_cases))));
      std::vector<Node*> cases_all(all_constants_cases, dividend);
      XLS_ASSIGN_OR_RETURN(Node * select_all,
                           temp_nodes_all.AddNode<PrioritySelect>(
                               node->loc(), selector_all, cases_all, dividend));
      XLS_ASSIGN_OR_RETURN(
          all_constants_select_area,
          area_estimator->GetOperationAreaInSquareMicrons(select_all));
    }

    double non_pow2_constants_select_area = 0;
    if (spec.non_pow2_constants_count > 0) {
      TemporaryNodeScope temp_nodes_non_pow2(f);
      XLS_ASSIGN_OR_RETURN(
          Node * selector_non_pow2,
          temp_nodes_non_pow2.AddNode<Literal>(
              node->loc(), Value(UBits(0, spec.non_pow2_constants_count))));
      std::vector<Node*> cases_non_pow2(spec.non_pow2_constants_count,
                                        dividend);
      XLS_ASSIGN_OR_RETURN(
          Node * select_non_pow2,
          temp_nodes_non_pow2.AddNode<PrioritySelect>(
              node->loc(), selector_non_pow2, cases_non_pow2, dividend));
      XLS_ASSIGN_OR_RETURN(
          non_pow2_constants_select_area,
          area_estimator->GetOperationAreaInSquareMicrons(select_non_pow2));
    }

    TemporaryNodeScope temp_nodes(f);
    Op shift_op = (node->op() == Op::kUDiv) ? Op::kShrl : Op::kShra;
    XLS_ASSIGN_OR_RETURN(Node * encode,
                         temp_nodes.AddNode<Encode>(node->loc(), divisor));
    XLS_ASSIGN_OR_RETURN(
        Node * shift,
        temp_nodes.AddNode<BinOp>(node->loc(), dividend, encode, shift_op));
    XLS_ASSIGN_OR_RETURN(
        double shift_area,
        area_estimator->GetOperationAreaInSquareMicrons(shift));
    XLS_ASSIGN_OR_RETURN(
        double encode_area,
        area_estimator->GetOperationAreaInSquareMicrons(encode));

    double extra_sdiv_area = 0;
    if (node->op() == Op::kSDiv) {
      XLS_ASSIGN_OR_RETURN(Node * msb,
                           temp_nodes.AddNode<BitSlice>(node->loc(), dividend,
                                                        bit_width - 1, 1));
      XLS_ASSIGN_OR_RETURN(
          Node * one,
          temp_nodes.AddNode<Literal>(node->loc(), Value(UBits(1, bit_width))));
      XLS_ASSIGN_OR_RETURN(
          Node * sub,
          temp_nodes.AddNode<BinOp>(node->loc(), divisor, one, Op::kSub));
      XLS_ASSIGN_OR_RETURN(
          Node * zero,
          temp_nodes.AddNode<Literal>(node->loc(), Value(UBits(0, bit_width))));
      XLS_ASSIGN_OR_RETURN(Node * bias_sel,
                           temp_nodes.AddNode<Select>(
                               node->loc(), msb, std::vector<Node*>{zero, sub},
                               /*default_value=*/std::nullopt));
      XLS_ASSIGN_OR_RETURN(
          Node * add,
          temp_nodes.AddNode<BinOp>(node->loc(), dividend, bias_sel, Op::kAdd));

      XLS_ASSIGN_OR_RETURN(
          double sub_area,
          area_estimator->GetOperationAreaInSquareMicrons(sub));
      XLS_ASSIGN_OR_RETURN(
          double sel_area,
          area_estimator->GetOperationAreaInSquareMicrons(bias_sel));
      XLS_ASSIGN_OR_RETURN(
          double add_area,
          area_estimator->GetOperationAreaInSquareMicrons(add));
      extra_sdiv_area = sub_area + sel_area + add_area;
    }

    prefer_shift_by_encode = all_constants_select_area >
                             non_pow2_constants_select_area + encode_area +
                                 shift_area + extra_sdiv_area;

    if (prefer_shift_by_encode) {
      profitable = spec.non_pow2_constants_count * mul_area +
                       non_pow2_constants_select_area + encode_area +
                       shift_area + extra_sdiv_area <
                   div_area;
    } else {
      profitable =
          spec.non_pow2_constants_count * mul_area + all_constants_select_area <
          div_area;
    }
  } else {
    prefer_shift_by_encode =
        spec.pow2_constants_count >
        FloorOfLog2(spec.max_pow2_shift) + (node->op() == Op::kSDiv ? 2 : 1);
    profitable = prefer_shift_by_encode ? (spec.non_pow2_constants_count <= 1)
                                        : (spec.non_pow2_constants_count <= 2);
  }

  if (profitable) {
    DivisionSimplificationSpec updated_spec = spec;
    updated_spec.prefer_shift_by_encode = prefer_shift_by_encode;
    return updated_spec;
  }
  return std::nullopt;
}

absl::StatusOr<Node*> ApplyMultipleConstantDivisionTransformation(
    Node* node, const DivisionSimplificationSpec& spec) {
  FunctionBase* f = node->function_base();
  Node* dividend = node->operand(0);
  Node* divisor = node->operand(1);
  int64_t bit_width = node->BitCountOrDie();

  std::vector<Node*> predicates;
  std::vector<Node*> cases;
  Node* default_case = nullptr;

  std::vector<Bits> values;
  for (const Bits& v : spec.intervals.Values()) {
    values.push_back(v);
  }

  if (spec.prefer_shift_by_encode) {
    for (const Bits& value : values) {
      if (value.IsZero() || value.IsPowerOfTwo()) {
        continue;
      }
      XLS_ASSIGN_OR_RETURN(Node * value_lit,
                           f->MakeNode<Literal>(node->loc(), Value(value)));
      XLS_ASSIGN_OR_RETURN(
          Node * eq,
          f->MakeNode<CompareOp>(node->loc(), divisor, value_lit, Op::kEq));
      predicates.push_back(eq);
      XLS_ASSIGN_OR_RETURN(
          Node * div_case,
          f->MakeNode<BinOp>(node->loc(), dividend, value_lit, node->op()));
      cases.push_back(div_case);
    }

    XLS_ASSIGN_OR_RETURN(Node * shift_amt,
                         f->MakeNode<Encode>(node->loc(), divisor));
    if (node->op() == Op::kUDiv) {
      XLS_ASSIGN_OR_RETURN(
          Node * shifted,
          f->MakeNode<BinOp>(node->loc(), dividend, shift_amt, Op::kShrl));
      default_case = shifted;
    } else {
      // SDiv requires a bias for negative numbers to truncate towards zero
      // rather than negative infinity (which is what Shra does).
      // Bias formula: (dividend < 0 ? divisor - 1 : 0).
      //
      // Note: we can also compute divisor - 1 as ~(all_ones << shift_amt).
      // We choose subtraction over shifting because in hardware, an adder (Sub)
      // is usually smaller than a barrel shifter (Shll) for large widths.
      XLS_ASSIGN_OR_RETURN(
          Node * msb,
          f->MakeNode<BitSlice>(node->loc(), dividend, bit_width - 1, 1));
      XLS_ASSIGN_OR_RETURN(
          Node * one,
          f->MakeNode<Literal>(node->loc(), Value(UBits(1, bit_width))));
      XLS_ASSIGN_OR_RETURN(
          Node * divisor_minus_one,
          f->MakeNode<BinOp>(node->loc(), divisor, one, Op::kSub));
      XLS_ASSIGN_OR_RETURN(
          Node * zero,
          f->MakeNode<Literal>(node->loc(), Value(UBits(0, bit_width))));
      XLS_ASSIGN_OR_RETURN(
          Node * bias,
          f->MakeNode<Select>(node->loc(), msb,
                              std::vector<Node*>{zero, divisor_minus_one},
                              std::nullopt));
      XLS_ASSIGN_OR_RETURN(
          Node * biased_dividend,
          f->MakeNode<BinOp>(node->loc(), dividend, bias, Op::kAdd));
      XLS_ASSIGN_OR_RETURN(Node * shifted,
                           f->MakeNode<BinOp>(node->loc(), biased_dividend,
                                              shift_amt, Op::kShra));
      default_case = shifted;
    }

    if (spec.intervals.CoversZero()) {
      XLS_ASSIGN_OR_RETURN(Node * div_zero_result,
                           CreateDivZeroResult(node, dividend));
      XLS_ASSIGN_OR_RETURN(
          Node * zero_lit,
          f->MakeNode<Literal>(node->loc(), Value(UBits(0, bit_width))));
      XLS_ASSIGN_OR_RETURN(
          Node * is_zero,
          f->MakeNode<CompareOp>(node->loc(), divisor, zero_lit, Op::kEq));
      XLS_ASSIGN_OR_RETURN(
          default_case,
          f->MakeNode<Select>(node->loc(), is_zero,
                              std::vector<Node*>{default_case, div_zero_result},
                              std::nullopt));
    }
  } else {
    for (const Bits& value : values) {
      if (value.IsZero()) {
        continue;
      }
      XLS_ASSIGN_OR_RETURN(Node * value_lit,
                           f->MakeNode<Literal>(node->loc(), Value(value)));
      XLS_ASSIGN_OR_RETURN(
          Node * eq,
          f->MakeNode<CompareOp>(node->loc(), divisor, value_lit, Op::kEq));
      predicates.push_back(eq);
      XLS_ASSIGN_OR_RETURN(
          Node * div_case,
          f->MakeNode<BinOp>(node->loc(), dividend, value_lit, node->op()));
      cases.push_back(div_case);
    }

    if (spec.intervals.CoversZero()) {
      XLS_ASSIGN_OR_RETURN(default_case, CreateDivZeroResult(node, dividend));
    } else {
      XLS_ASSIGN_OR_RETURN(
          default_case,
          f->MakeNode<Literal>(node->loc(), Value(UBits(0, bit_width))));
    }
  }

  if (predicates.empty()) {
    return default_case;
  }

  std::vector<Node*> reversed_predicates(predicates.rbegin(),
                                         predicates.rend());
  XLS_ASSIGN_OR_RETURN(Node * selector,
                       f->MakeNode<Concat>(node->loc(), reversed_predicates));

  return f->MakeNode<PrioritySelect>(node->loc(), selector, cases,
                                     default_case);
}

absl::StatusOr<bool> TrySimplifyDivisionWithMultipleConstants(
    Node* node, const QueryEngine& query_engine,
    const AreaEstimator* area_estimator) {
  XLS_ASSIGN_OR_RETURN(
      std::optional<DivisionSimplificationSpec> spec,
      CheckMultipleConstantDivisionApplicability(node, query_engine));
  if (!spec.has_value()) {
    return false;
  }

  XLS_ASSIGN_OR_RETURN(
      std::optional<DivisionSimplificationSpec> updated_spec,
      IsMultipleConstantDivisionProfitable(node, *spec, area_estimator));
  if (!updated_spec.has_value()) {
    return false;
  }

  XLS_ASSIGN_OR_RETURN(
      Node * result,
      ApplyMultipleConstantDivisionTransformation(node, *updated_spec));
  XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(result));
  return true;
}

}  // namespace

absl::StatusOr<bool> ValueSetSimplificationPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  auto query_engine = UnionQueryEngine::Of(
      StatelessQueryEngine(),
      GetSharedQueryEngine<PartialInfoQueryEngine>(context, f));

  XLS_RETURN_IF_ERROR(query_engine.Populate(f).status());

  bool modified = false;
  for (Node* node : context.TopoSort(f)) {
    if (node->op() == Op::kUDiv || node->op() == Op::kSDiv) {
      XLS_ASSIGN_OR_RETURN(bool node_modified,
                           TrySimplifyDivisionWithMultipleConstants(
                               node, query_engine, options.area_estimator));
      modified |= node_modified;
    }
  }
  return modified;
}

}  // namespace xls
