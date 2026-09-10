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

#include "xls/passes/select_to_bitwise_ops_pass.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/passes/lazy_ternary_query_engine.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {
namespace {

// Depth cap for the xor-delta fold below: together with the `visited` set,
// recursion stays bounded so that deeply nested (or adversarially shaped) IR
// cannot exhaust the stack. A fold that would exceed the cap is a conservative
// miss, never an incorrect answer.
constexpr int64_t kMaxXorFoldDepth = 1000;

// Flattens `node` into its non-concat pieces, most-significant first: the same
// order in which `concat` lists its operands and `bits_ops::Concat`
// reassembles them.
void FlattenConcats(Node* node, std::vector<Node*>& pieces) {
  std::vector<Node*> stack;
  stack.push_back(node);
  while (!stack.empty()) {
    Node* current = stack.back();
    stack.pop_back();
    if (current->Is<Concat>()) {
      for (int64_t i = static_cast<int64_t>(current->operands().size()) - 1;
           i >= 0; --i) {
        stack.push_back(current->operand(i));
      }
    } else {
      pieces.push_back(current);
    }
  }
}

// Returns the fully-known constant `a ^ b` when the two equal-width bits-typed
// nodes differ only in bits whose values are known, and std::nullopt otherwise.
//
// The two inputs do not need to literally form `xor(other, c)`: earlier passes
// (most notably `concat_simp`) routinely split a wide xor limb apart, e.g.
//
//   sel(p, [concat(x, c1), concat(xor(x, c2), xor(c1, c3))])
//
// where the individual pieces of one arm are the corresponding pieces of the
// other arm xor'd with known constants. The fold walks the structure of the
// two arms (concatenations, xors against fully-known operands, nots), consults
// the query engine for fully-known pieces, and pieces together the full-width
// difference.
//
// The fold is recursive but bounded: `depth` is capped at kMaxXorFoldDepth, and
// `visited` guarantees that each distinct (a, b) pair is folded at most once
// per top-level call, with the first occurrence winning. Shared sub-structure
// or overly deep inputs therefore yield a conservative miss rather than
// unbounded work.
std::optional<Bits> TryFoldXorDelta(
    Node* a, Node* b, const QueryEngine& query_engine,
    absl::flat_hash_set<std::pair<Node*, Node*>>& visited, int64_t depth) {
  if (!a->GetType()->IsBits() || !b->GetType()->IsBits() ||
      a->BitCountOrDie() != b->BitCountOrDie()) {
    return std::nullopt;
  }
  if (depth > kMaxXorFoldDepth || !visited.insert({a, b}).second) {
    return std::nullopt;
  }

  // Identical arms fold to zero.
  if (a == b) {
    return Bits(a->BitCountOrDie());
  }
  // Fully-known arms fold to their xor.
  if (query_engine.IsFullyKnown(a) && query_engine.IsFullyKnown(b)) {
    return bits_ops::Xor(*query_engine.KnownValueAsBits(a),
                         *query_engine.KnownValueAsBits(b));
  }
  // `(u ^ c) ^ v == (u ^ v) ^ c`: pull a fully-known operand out of an xor arm
  // and keep folding the remaining operand.
  if (a->op() == Op::kXor && a->operands().size() == 2) {
    for (Node* operand : a->operands()) {
      std::optional<Bits> known = query_engine.KnownValueAsBits(operand);
      if (known.has_value()) {
        Node* rest = a->operand(0) == operand ? a->operand(1) : a->operand(0);
        std::optional<Bits> rest_delta =
            TryFoldXorDelta(rest, b, query_engine, visited, depth + 1);
        if (!rest_delta.has_value()) {
          return std::nullopt;
        }
        return bits_ops::Xor(*rest_delta, *known);
      }
    }
  }
  // Xor is symmetric, so treat `b` the same way.
  if (b->op() == Op::kXor && b->operands().size() == 2) {
    for (Node* operand : b->operands()) {
      std::optional<Bits> known = query_engine.KnownValueAsBits(operand);
      if (known.has_value()) {
        Node* rest = b->operand(0) == operand ? b->operand(1) : b->operand(0);
        std::optional<Bits> rest_delta =
            TryFoldXorDelta(a, rest, query_engine, visited, depth + 1);
        if (!rest_delta.has_value()) {
          return std::nullopt;
        }
        return bits_ops::Xor(*rest_delta, *known);
      }
    }
  }
  // `not` inverts the delta: `(~u) ^ v == ~(u ^ v)` and `(~u) ^ (~v) == u ^ v`.
  if (a->op() == Op::kNot && b->op() == Op::kNot) {
    return TryFoldXorDelta(a->operand(0), b->operand(0), query_engine, visited,
                           depth + 1);
  }
  if (a->op() == Op::kNot && a->operand(0) == b) {
    return Bits::AllOnes(a->BitCountOrDie());
  }
  if (b->op() == Op::kNot && b->operand(0) == a) {
    return Bits::AllOnes(b->BitCountOrDie());
  }
  // Decompose a concatenation on either side piece by piece so that a
  // per-piece xor-with-constant difference is recognized even when the pieces
  // themselves are not fully known. The pieces of both arms must line up
  // one-to-one; otherwise the difference is conservatively deemed too complex.
  if (a->Is<Concat>() || b->Is<Concat>()) {
    std::vector<Node*> pieces_a;
    std::vector<Node*> pieces_b;
    FlattenConcats(a, pieces_a);
    FlattenConcats(b, pieces_b);
    if (pieces_a.size() != pieces_b.size()) {
      return std::nullopt;
    }
    std::vector<Bits> deltas;
    deltas.reserve(pieces_a.size());
    for (int64_t i = 0; i < static_cast<int64_t>(pieces_a.size()); ++i) {
      std::optional<Bits> piece_delta = TryFoldXorDelta(
          pieces_a[i], pieces_b[i], query_engine, visited, depth + 1);
      if (!piece_delta.has_value()) {
        return std::nullopt;
      }
      deltas.push_back(*piece_delta);
    }
    return bits_ops::Concat(deltas);
  }
  // The two arms share no foldable structure and at least one is not fully
  // known, so the delta cannot be proven constant.
  return std::nullopt;
}

// Returns the (on_false, on_true) arm pair of a single-bit binary mux: either
// two explicit cases, or one explicit case plus a default value. Returns
// std::nullopt for any other select shape.
std::optional<std::pair<Node*, Node*>> GetBinaryMuxArms(Select* sel) {
  absl::Span<Node* const> cases = sel->cases();
  if (cases.size() == 2 && !sel->default_value().has_value()) {
    return std::make_pair(cases[0], cases[1]);
  }
  if (cases.size() == 1 && sel->default_value().has_value()) {
    return std::make_pair(cases[0], *sel->default_value());
  }
  return std::nullopt;
}

// A binary select arm that literally is `xor(other_arm, c)`.
struct XorCasePattern {
  Node* base;                // The non-xor arm.
  Node* constant;            // The xor operand that is not the base arm.
  bool xor_arm_is_case_one;  // Xor arm is selected on selector == 1.
};

// Matches `xor_arm` against `other_arm` for the literal `xor(other_arm, c)`
// pattern. Xor is symmetric, so `c` may be either operand.
std::optional<XorCasePattern> MatchXorCase(Node* xor_arm, Node* other_arm,
                                           bool xor_arm_is_case_one) {
  if (xor_arm->op() != Op::kXor || xor_arm->operands().size() != 2) {
    return std::nullopt;
  }
  if (xor_arm->operand(0) != other_arm && xor_arm->operand(1) != other_arm) {
    return std::nullopt;
  }
  return XorCasePattern{
      .base = other_arm,
      .constant = xor_arm->operand(1) == other_arm ? xor_arm->operand(0)
                                                   : xor_arm->operand(1),
      .xor_arm_is_case_one = xor_arm_is_case_one,
  };
}

// Converts a binary select whose two arms differ by a single XOR with a fully
// known constant into an and-masked xor:
//
//   sel(p, cases=[x, xor(x, c)]) => xor(x, and(sign_ext(p, W), c))
//   sel(p, cases=[xor(x, c), x]) => xor(x, and(sign_ext(not(p), W), c))
//
// The constant term `c` must be fully known (e.g., a literal) for the rewrite
// to apply, in which case it is materialized as a fresh literal. If the two
// arms do not literally form `xor(base, c)` (e.g. because the xor limb was
// already split into per-slice pieces by `concat_simp`), the rewrite still
// applies as long as they provably differ by a fully-known constant; see
// TryFoldXorDelta above.
//
absl::StatusOr<bool> MaybeConvertSelectToBitwiseOps(
    Node* node, const QueryEngine& query_engine) {
  // Defensive: this helper is currently only called on bits-typed selects, but
  // the guards keep it safe to reuse elsewhere.
  if (!node->Is<Select>() || !node->GetType()->IsBits()) {
    return false;
  }
  Select* sel = node->As<Select>();
  if (sel->selector()->BitCountOrDie() != 1) {
    return false;
  }

  // Determine the two arms of the binary mux.
  std::optional<std::pair<Node*, Node*>> mux_arms = GetBinaryMuxArms(sel);
  if (!mux_arms.has_value()) {
    return false;
  }
  auto [on_false, on_true] = *mux_arms;

  // Find a pair (base, c) such that one arm is `xor(base, c)` and the other
  // arm is `base`. `xor_arm_is_case_one` records whether the xor arm is
  // selected when the 1-bit selector is 1, which decides whether the mask
  // condition is the selector itself or its complement.
  std::optional<XorCasePattern> pattern =
      MatchXorCase(on_true, on_false, /*xor_arm_is_case_one=*/true);
  if (!pattern.has_value()) {
    pattern = MatchXorCase(on_false, on_true, /*xor_arm_is_case_one=*/false);
  }

  Node* base = nullptr;
  bool xor_arm_is_case_one = false;
  std::optional<Bits> delta;
  if (pattern.has_value()) {
    if (!query_engine.IsFullyKnown(pattern->constant)) {
      return false;
    }
    base = pattern->base;
    xor_arm_is_case_one = pattern->xor_arm_is_case_one;
    delta = query_engine.KnownValueAsBits(pattern->constant);
  } else {
    // The xor limb may no longer be a literal `xor(base, c)` because an
    // earlier pass (e.g. concat_simp) split it into per-slice pieces. The
    // masked-xor rewrite still applies as long as the two arms provably differ
    // by a fully-known constant; the delta is then `on_false ^ on_true`.
    absl::flat_hash_set<std::pair<Node*, Node*>> visited;
    delta = TryFoldXorDelta(on_false, on_true, query_engine, visited,
                            /*depth=*/0);
    if (!delta.has_value()) {
      return false;
    }
    // `sel = on_false ^ (sign_ext(p, W) & delta)` yields `on_false` for
    // p == 0 and `on_false ^ delta == on_true` for p == 1, so the mask
    // condition is the selector itself, with `on_false` as the base.
    base = on_false;
    xor_arm_is_case_one = true;
    VLOG(2) << absl::StrFormat(
        "Select with split xor case rewritten as masked xor: %s",
        node->ToString());
  }

  // Identical cases in a select are handled by a different simplification.
  if (delta->IsZero()) {
    return false;
  }

  FunctionBase* f = node->function_base();
  XLS_ASSIGN_OR_RETURN(Node * delta_literal,
                       f->MakeNode<Literal>(node->loc(), Value(*delta)));

  // The mask condition is the selector itself when the xor arm is selected on
  // p == 1, and its complement when the xor arm is the on-false case.
  Node* condition = sel->selector();
  if (!xor_arm_is_case_one) {
    XLS_ASSIGN_OR_RETURN(
        condition, f->MakeNode<UnOp>(node->loc(), sel->selector(), Op::kNot));
  }

  Node* mask = condition;
  if (node->BitCountOrDie() != 1) {
    XLS_ASSIGN_OR_RETURN(
        mask, f->MakeNode<ExtendOp>(node->loc(), condition,
                                    /*new_bit_count=*/node->BitCountOrDie(),
                                    Op::kSignExt));
  }

  XLS_ASSIGN_OR_RETURN(
      Node * masked_delta,
      f->MakeNode<NaryOp>(node->loc(), std::vector<Node*>{mask, delta_literal},
                          Op::kAnd));
  VLOG(2) << absl::StrFormat(
      "Select with xor-with-constant case rewritten as masked xor: %s",
      node->ToString());
  XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<NaryOp>(
                              std::vector<Node*>{base, masked_delta}, Op::kXor)
                          .status());
  return true;
}

}  // namespace

absl::StatusOr<bool> SelectToBitwiseOpsPass::RunOnFunctionBaseInternal(
    FunctionBase* func, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  if (!SplitsEnabled(options.opt_level)) {
    return false;
  }

  // The rewrite only needs to prove arms fully known; ternary analysis is
  // sufficient for that and is much cheaper than range analysis.
  auto query_engine = UnionQueryEngine::Of(
      StatelessQueryEngine(),
      context.SharedQueryEngine<LazyTernaryQueryEngine>(func));
  XLS_RETURN_IF_ERROR(query_engine.Populate(func).status());

  // The rewrite produces xor/and/sign-ext nodes and never another select, so a
  // single pass over the original node list is sufficient.
  XLS_ASSIGN_OR_RETURN(std::vector<Node*> topo_sort_nodes,
                       context.TopoSort(func));
  bool changed = false;
  for (Node* node : topo_sort_nodes) {
    XLS_ASSIGN_OR_RETURN(bool node_changed,
                         MaybeConvertSelectToBitwiseOps(node, query_engine));
    changed = changed || node_changed;
  }
  return changed;
}

}  // namespace xls
