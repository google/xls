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

#include "xls/passes/bitwise_recombination_pass.h"

#include <cstdint>
#include <deque>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

enum class SliceElementKind {
  kRaw,       // BitSlice(X, start, width) or X
  kInverted,  // Not(BitSlice(X, start, width)) or BitSlice(Not(X), ...)
  kZero,      // Literal(0, width)
  kOne,       // Literal(all_ones, width)
};

struct SliceComponent {
  Node* node;
  Node* source;  // nullptr for literals
  SliceElementKind kind;
  int64_t start;  // -1 if undetermined literal
  int64_t width;
};

// Classifies an operand of a Concat into a SliceComponent if possible.
std::optional<SliceComponent> ClassifyOperand(Node* node) {
  if (node->Is<BitSlice>()) {
    const BitSlice* slice = node->As<BitSlice>();
    Node* input = slice->operand(0);
    if (input->op() == Op::kNot) {
      return SliceComponent{
          .node = node,
          .source = input->operand(0),
          .kind = SliceElementKind::kInverted,
          .start = slice->start(),
          .width = slice->width(),
      };
    }
    return SliceComponent{
        .node = node,
        .source = input,
        .kind = SliceElementKind::kRaw,
        .start = slice->start(),
        .width = slice->width(),
    };
  }

  if (node->op() == Op::kNot) {
    Node* input = node->operand(0);
    if (input->Is<BitSlice>()) {
      const BitSlice* slice = input->As<BitSlice>();
      return SliceComponent{
          .node = node,
          .source = slice->operand(0),
          .kind = SliceElementKind::kInverted,
          .start = slice->start(),
          .width = slice->width(),
      };
    }
  }

  if (node->Is<Literal>()) {
    const Bits& bits = node->As<Literal>()->value().bits();
    if (bits.IsZero()) {
      return SliceComponent{
          .node = node,
          .source = nullptr,
          .kind = SliceElementKind::kZero,
          .start = -1,
          .width = bits.bit_count(),
      };
    }
    if (bits.IsAllOnes()) {
      return SliceComponent{
          .node = node,
          .source = nullptr,
          .kind = SliceElementKind::kOne,
          .start = -1,
          .width = bits.bit_count(),
      };
    }
    return std::nullopt;
  }

  // Handle unsliced whole nodes (negated and otherwise).
  if (node->op() == Op::kNot && node->GetType()->IsBits()) {
    Node* input = node->operand(0);
    return SliceComponent{
        .node = node,
        .source = input,
        .kind = SliceElementKind::kInverted,
        .start = 0,
        .width = node->BitCountOrDie(),
    };
  }
  if (node->GetType()->IsBits()) {
    return SliceComponent{
        .node = node,
        .source = node,
        .kind = SliceElementKind::kRaw,
        .start = 0,
        .width = node->BitCountOrDie(),
    };
  }

  return std::nullopt;
}

// Recombines a contiguous run of components for source X into a single node
// (slice, masked AND, masked OR, masked XOR, or a 2-gate combination).
absl::StatusOr<Node*> RecombineRun(FunctionBase* f, const SourceInfo& loc,
                                   absl::Span<const SliceComponent> run) {
  Node* source = nullptr;
  for (const auto& elem : run) {
    if (elem.source != nullptr) {
      source = elem.source;
      break;
    }
  }
  XLS_RET_CHECK(source != nullptr);

  int64_t total_width = 0;
  for (const auto& elem : run) {
    total_width += elem.width;
  }
  const int64_t run_start = run.back().start;

  Node* base_slice;
  if (run_start == 0 && total_width == source->BitCountOrDie()) {
    base_slice = source;
  } else {
    XLS_ASSIGN_OR_RETURN(
        base_slice, f->MakeNode<BitSlice>(loc, source, run_start, total_width));
  }

  bool has_raw = false;
  bool has_inv = false;
  bool has_zero = false;
  bool has_one = false;
  for (const auto& elem : run) {
    switch (elem.kind) {
      case SliceElementKind::kRaw:
        has_raw = true;
        break;
      case SliceElementKind::kInverted:
        has_inv = true;
        break;
      case SliceElementKind::kZero:
        has_zero = true;
        break;
      case SliceElementKind::kOne:
        has_one = true;
        break;
    }
  }

  // Pure raw slices fuse into base_slice directly.
  if (!has_inv && !has_zero && !has_one) {
    return base_slice;
  }

  // Pure inverted slices fuse into not(base_slice) directly.
  if (has_inv && !has_raw && !has_zero && !has_one) {
    return f->MakeNode<UnOp>(loc, base_slice, Op::kNot);
  }

  InlineBitmap mask_and(total_width, /*fill=*/true);
  InlineBitmap mask_or(total_width, /*fill=*/false);
  InlineBitmap mask_xor(total_width, /*fill=*/false);
  for (const auto& elem : run) {
    const int64_t bit_offset = elem.start - run_start;
    switch (elem.kind) {
      case SliceElementKind::kRaw:
        break;
      case SliceElementKind::kInverted:
        mask_xor.SetRange(bit_offset, bit_offset + elem.width, true);
        break;
      case SliceElementKind::kZero:
        mask_and.SetRange(bit_offset, bit_offset + elem.width, false);
        break;
      case SliceElementKind::kOne:
        mask_and.SetRange(bit_offset, bit_offset + elem.width, false);
        mask_or.SetRange(bit_offset, bit_offset + elem.width, true);
        break;
    }
  }

  // 1. Single operator: XOR (raw + inverted)
  if (has_inv && !has_zero && !has_one) {
    XLS_ASSIGN_OR_RETURN(
        Node * lit_xor_mask,
        f->MakeNode<Literal>(loc,
                             Value(Bits::FromBitmap(std::move(mask_xor)))));
    return f->MakeNode<NaryOp>(
        loc, absl::MakeConstSpan({base_slice, lit_xor_mask}), Op::kXor);
  }

  // 2. Single operator: AND (raw + 0s)
  if (has_zero && !has_inv && !has_one) {
    XLS_ASSIGN_OR_RETURN(
        Node * lit_and_mask,
        f->MakeNode<Literal>(loc,
                             Value(Bits::FromBitmap(std::move(mask_and)))));
    return f->MakeNode<NaryOp>(
        loc, absl::MakeConstSpan({base_slice, lit_and_mask}), Op::kAnd);
  }

  // 3. Single operator: OR (raw + 1s)
  if (has_one && !has_inv && !has_zero) {
    XLS_ASSIGN_OR_RETURN(
        Node * lit_or_mask,
        f->MakeNode<Literal>(loc, Value(Bits::FromBitmap(std::move(mask_or)))));
    return f->MakeNode<NaryOp>(
        loc, absl::MakeConstSpan({base_slice, lit_or_mask}), Op::kOr);
  }

  // 4. Two operators: (base & mask_and) | mask_or (classic bitfield deposit)
  if (has_zero && has_one && !has_inv) {
    XLS_ASSIGN_OR_RETURN(
        Node * lit_and_mask,
        f->MakeNode<Literal>(loc,
                             Value(Bits::FromBitmap(std::move(mask_and)))));
    XLS_ASSIGN_OR_RETURN(
        Node * and_masked,
        f->MakeNode<NaryOp>(
            loc, absl::MakeConstSpan({base_slice, lit_and_mask}), Op::kAnd));
    XLS_ASSIGN_OR_RETURN(
        Node * lit_or_mask,
        f->MakeNode<Literal>(loc, Value(Bits::FromBitmap(std::move(mask_or)))));
    return f->MakeNode<NaryOp>(
        loc, absl::MakeConstSpan({and_masked, lit_or_mask}), Op::kOr);
  }

  // 5. Two operators: (base ^ mask_xor) | mask_or (inverted + 1s)
  if (has_inv && has_one && !has_zero) {
    Node* inverted;
    if (!has_raw) {
      XLS_ASSIGN_OR_RETURN(inverted,
                           f->MakeNode<UnOp>(loc, base_slice, Op::kNot));
    } else {
      XLS_ASSIGN_OR_RETURN(
          Node * lit_xor_mask,
          f->MakeNode<Literal>(loc,
                               Value(Bits::FromBitmap(std::move(mask_xor)))));
      XLS_ASSIGN_OR_RETURN(
          inverted,
          f->MakeNode<NaryOp>(
              loc, absl::MakeConstSpan({base_slice, lit_xor_mask}), Op::kXor));
    }
    XLS_ASSIGN_OR_RETURN(
        Node * lit_or_mask,
        f->MakeNode<Literal>(loc, Value(Bits::FromBitmap(std::move(mask_or)))));
    return f->MakeNode<NaryOp>(
        loc, absl::MakeConstSpan({inverted, lit_or_mask}), Op::kOr);
  }

  // 6. Two operators: (base ^ mask_xor) & mask_and (inverted + 0s)
  if (has_inv && has_zero && !has_one) {
    Node* inverted;
    if (!has_raw) {
      XLS_ASSIGN_OR_RETURN(inverted,
                           f->MakeNode<UnOp>(loc, base_slice, Op::kNot));
    } else {
      XLS_ASSIGN_OR_RETURN(
          Node * lit_xor_mask,
          f->MakeNode<Literal>(loc,
                               Value(Bits::FromBitmap(std::move(mask_xor)))));
      XLS_ASSIGN_OR_RETURN(
          inverted,
          f->MakeNode<NaryOp>(
              loc, absl::MakeConstSpan({base_slice, lit_xor_mask}), Op::kXor));
    }
    XLS_ASSIGN_OR_RETURN(
        Node * lit_and_mask,
        f->MakeNode<Literal>(loc,
                             Value(Bits::FromBitmap(std::move(mask_and)))));
    return f->MakeNode<NaryOp>(
        loc, absl::MakeConstSpan({inverted, lit_and_mask}), Op::kAnd);
  }

  // 7. Inverted bitfield deposit: (not(base) & mask_and) | mask_or
  // (inverted + 0s + 1s, no raw slices)
  if (!has_raw && has_inv && has_zero && has_one) {
    XLS_ASSIGN_OR_RETURN(Node * not_base,
                         f->MakeNode<UnOp>(loc, base_slice, Op::kNot));
    XLS_ASSIGN_OR_RETURN(
        Node * lit_and_mask,
        f->MakeNode<Literal>(loc,
                             Value(Bits::FromBitmap(std::move(mask_and)))));
    XLS_ASSIGN_OR_RETURN(
        Node * and_masked,
        f->MakeNode<NaryOp>(loc, absl::MakeConstSpan({not_base, lit_and_mask}),
                            Op::kAnd));
    XLS_ASSIGN_OR_RETURN(
        Node * lit_or_mask,
        f->MakeNode<Literal>(loc, Value(Bits::FromBitmap(std::move(mask_or)))));
    return f->MakeNode<NaryOp>(
        loc, absl::MakeConstSpan({and_masked, lit_or_mask}), Op::kOr);
  }

  // 8. General fallback: (base & mask_and) ^ (mask_xor | mask_or)
  mask_xor.Union(mask_or);
  XLS_ASSIGN_OR_RETURN(
      Node * lit_and_mask,
      f->MakeNode<Literal>(loc, Value(Bits::FromBitmap(std::move(mask_and)))));
  XLS_ASSIGN_OR_RETURN(
      Node * and_masked,
      f->MakeNode<NaryOp>(loc, absl::MakeConstSpan({base_slice, lit_and_mask}),
                          Op::kAnd));
  XLS_ASSIGN_OR_RETURN(
      Node * lit_xor_mask,
      f->MakeNode<Literal>(loc, Value(Bits::FromBitmap(std::move(mask_xor)))));
  return f->MakeNode<NaryOp>(
      loc, absl::MakeConstSpan({and_masked, lit_xor_mask}), Op::kXor);
}

// Attempts to form a contiguous run of components from a single source node
// covering the given range of classified concat operands.
// Returns std::nullopt if the components cannot form a valid contiguous run.
std::optional<std::vector<SliceComponent>> TryFormRun(
    absl::Span<const std::optional<SliceComponent>> range) {
  if (range.size() <= 1) {
    return std::nullopt;
  }
  for (const auto& comp : range) {
    if (!comp.has_value()) {
      return std::nullopt;
    }
  }

  // Find unique non-null source and the first non-null slice index.
  Node* source = nullptr;
  int64_t first_slice_idx = -1;
  for (int64_t i = 0; i < range.size(); ++i) {
    if (range[i]->source != nullptr) {
      if (source == nullptr) {
        source = range[i]->source;
        first_slice_idx = i;
      } else if (range[i]->source != source) {
        return std::nullopt;
      }
    }
  }
  if (source == nullptr) {
    // Pure literals cannot form a bitwise run on a source node.
    return std::nullopt;
  }

  // Preceding literals before first_slice_idx.
  int64_t lit_start =
      range[first_slice_idx]->start + range[first_slice_idx]->width;
  std::vector<int64_t> preceding_starts(first_slice_idx);
  for (int64_t back = first_slice_idx - 1; back >= 0; --back) {
    preceding_starts[back] = lit_start;
    lit_start += range[back]->width;
  }
  if (lit_start > source->BitCountOrDie()) {
    return std::nullopt;
  }

  std::vector<SliceComponent> run;
  run.reserve(range.size());
  for (int64_t k = 0; k < first_slice_idx; ++k) {
    run.push_back(SliceComponent{
        .node = range[k]->node,
        .source = source,
        .kind = range[k]->kind,
        .start = preceding_starts[k],
        .width = range[k]->width,
    });
  }

  int64_t current_expected_start =
      run.empty() ? range[first_slice_idx]->start : run.back().start;
  for (int64_t k = first_slice_idx; k < range.size(); ++k) {
    const auto& comp = *range[k];
    int64_t elem_start;
    if (comp.source != nullptr) {
      elem_start = comp.start;
    } else {
      elem_start = current_expected_start - comp.width;
    }

    if (elem_start < 0) {
      return std::nullopt;
    }
    if (!run.empty()) {
      if (elem_start + comp.width != run.back().start) {
        return std::nullopt;
      }
    }

    run.push_back(SliceComponent{
        .node = comp.node,
        .source = source,
        .kind = comp.kind,
        .start = elem_start,
        .width = comp.width,
    });
    current_expected_start = elem_start;
  }

  return run;
}

enum class MaskPeriodicity {
  kClean,
  kAlmostClean,
  kUnclean,
};

// Checks if a mask exhibits periodic repetition.
MaskPeriodicity GetMaskPeriodicity(const InlineBitmap& bitmap) {
  if (bitmap.IsAllZeroes()) {
    return MaskPeriodicity::kClean;
  }
  const int64_t width = bitmap.bit_count();
  const int64_t hex_width = 4 * CeilOfRatio(width, int64_t{4});
  if (bitmap.IsAllOnes()) {
    return width == hex_width ? MaskPeriodicity::kClean
                              : MaskPeriodicity::kAlmostClean;
  }

  int64_t period = 16;
  while (period * 2 > hex_width) {
    period /= 2;
  }
  if (period < 2 || period >= width) {
    return MaskPeriodicity::kUnclean;
  }

  for (int64_t i = 0; i < width - period; ++i) {
    if (bitmap.Get(i) != bitmap.Get(i + period)) {
      return MaskPeriodicity::kUnclean;
    }
  }

  // When the width is not a multiple of 4, the top nibble in hex is padded with
  // zeros. The mask will look clean in hex if the virtual bits completing that
  // top nibble in the periodic continuation will be zeros.
  if (hex_width > width) {
    for (int64_t i = width; i < hex_width; ++i) {
      if (bitmap.Get(i % period)) {
        return MaskPeriodicity::kAlmostClean;
      }
    }
  }

  return MaskPeriodicity::kClean;
}

constexpr int64_t kInvalidCost = -1;

constexpr int64_t kUnaryExpressionCost = 1;
constexpr int64_t kMaskCost = 2;
constexpr int64_t kTwoMaskPenalty = 1;

int64_t GetMaskCost(const InlineBitmap& mask) {
  switch (GetMaskPeriodicity(mask)) {
    case MaskPeriodicity::kClean:
      return kMaskCost;
    case MaskPeriodicity::kAlmostClean:
      return kMaskCost + 1;
    case MaskPeriodicity::kUnclean:
      return kMaskCost + CeilOfRatio(mask.bit_count(), int64_t{4});
  }
}

// Evaluates the cognitive/readability cost of a candidate partition segment.
int64_t EvaluateSegmentCost(
    absl::Span<const std::optional<SliceComponent>> range) {
  if (range.size() == 1) {
    if (range[0].has_value() && range[0]->kind == SliceElementKind::kInverted) {
      return kUnaryExpressionCost;
    }
    return 0;
  }

  std::optional<std::vector<SliceComponent>> maybe_run = TryFormRun(range);
  if (!maybe_run.has_value()) {
    return kInvalidCost;
  }
  const auto& run = *maybe_run;
  const int64_t run_start = run.back().start;

  int64_t total_width = 0;
  for (const auto& elem : run) {
    total_width += elem.width;
  }

  bool has_raw = false;
  bool has_inv = false;
  bool has_zero = false;
  bool has_one = false;
  for (const auto& elem : run) {
    switch (elem.kind) {
      case SliceElementKind::kRaw:
        has_raw = true;
        break;
      case SliceElementKind::kInverted:
        has_inv = true;
        break;
      case SliceElementKind::kZero:
        has_zero = true;
        break;
      case SliceElementKind::kOne:
        has_one = true;
        break;
    }
  }

  // 0 gates or 1 unary gate:
  if (!has_inv && !has_zero && !has_one) {
    return 0;
  }
  if (has_inv && !has_raw && !has_zero && !has_one) {
    return kUnaryExpressionCost;
  }

  InlineBitmap mask_and(total_width, /*fill=*/true);
  InlineBitmap mask_or(total_width, /*fill=*/false);
  InlineBitmap mask_xor(total_width, /*fill=*/false);
  for (const auto& elem : run) {
    const int64_t bit_offset = elem.start - run_start;
    switch (elem.kind) {
      case SliceElementKind::kRaw:
        break;
      case SliceElementKind::kInverted:
        mask_xor.SetRange(bit_offset, bit_offset + elem.width, true);
        break;
      case SliceElementKind::kZero:
        mask_and.SetRange(bit_offset, bit_offset + elem.width, false);
        break;
      case SliceElementKind::kOne:
        mask_and.SetRange(bit_offset, bit_offset + elem.width, false);
        mask_or.SetRange(bit_offset, bit_offset + elem.width, true);
        break;
    }
  }

  // 1 binary gate cases:
  bool is_one_binary_gate = false;
  const InlineBitmap* active_mask = nullptr;
  if (has_inv && !has_zero && !has_one) {
    is_one_binary_gate = true;
    active_mask = &mask_xor;
  } else if (has_zero && !has_inv && !has_one) {
    is_one_binary_gate = true;
    active_mask = &mask_and;
  } else if (has_one && !has_inv && !has_zero) {
    is_one_binary_gate = true;
    active_mask = &mask_or;
  } else if (!has_raw && has_inv && has_zero && !has_one) {
    is_one_binary_gate = true;
    active_mask = &mask_and;
  } else if (!has_raw && has_inv && has_one && !has_zero) {
    is_one_binary_gate = true;
    active_mask = &mask_or;
  }

  if (is_one_binary_gate) {
    return GetMaskCost(*active_mask);
  }

  // 2 binary gate cases:
  if (has_zero && has_one && !has_inv) {
    return GetMaskCost(mask_and) + GetMaskCost(mask_or) + kTwoMaskPenalty;
  }

  if (!has_raw && has_inv && has_zero && has_one) {
    return kUnaryExpressionCost + GetMaskCost(mask_and) + GetMaskCost(mask_or) +
           kTwoMaskPenalty;
  }

  const InlineBitmap& secondary_mask = has_zero ? mask_and : mask_or;
  return kUnaryExpressionCost + GetMaskCost(mask_xor) +
         GetMaskCost(secondary_mask) + kTwoMaskPenalty;
}

// Scans a concat and recombines runs of contiguous components from a single
// source using 1D dynamic programming optimal partitioning.
absl::StatusOr<bool> SimplifyConcat(Concat* concat) {
  if (concat->operand_count() == 1) {
    XLS_RETURN_IF_ERROR(concat->ReplaceUsesWith(concat->operand(0)));
    return true;
  }

  const int64_t num_ops = concat->operand_count();
  std::vector<std::optional<SliceComponent>> components;
  components.reserve(num_ops);
  for (Node* op : concat->operands()) {
    components.push_back(ClassifyOperand(op));
  }

  std::vector<int64_t> dp(num_ops + 1, kInvalidCost);
  std::vector<int64_t> best_split(num_ops + 1, -1);
  dp[0] = 0;

  for (int64_t j = 1; j <= num_ops; ++j) {
    // Iterate from j-1 to 0; by checking the shortest segment first, in the
    // event of a tie, we favor less merging.
    for (int64_t i = j - 1; i >= 0; --i) {
      if (dp[i] == kInvalidCost) {
        continue;
      }
      int64_t segment_cost = EvaluateSegmentCost(
          absl::MakeConstSpan(components).subspan(i, j - i));
      if (segment_cost == kInvalidCost) {
        continue;
      }
      int64_t cost = 1 + segment_cost;
      if (dp[j] == kInvalidCost || dp[i] + cost < dp[j]) {
        dp[j] = dp[i] + cost;
        best_split[j] = i;
      }
    }
  }

  if (dp[num_ops] == kInvalidCost) {
    return false;
  }

  // Reconstruct partition:
  std::vector<std::pair<int64_t, int64_t>> segments;
  for (int64_t curr = num_ops; curr > 0; curr = best_split[curr]) {
    segments.push_back({best_split[curr], curr});
  }
  absl::c_reverse(segments);

  // Check if any segment merged multiple operands.
  bool any_merged = false;
  for (const auto& [start, end] : segments) {
    if (end - start > 1) {
      any_merged = true;
      break;
    }
  }
  if (!any_merged) {
    return false;
  }

  std::vector<Node*> new_operands;
  new_operands.reserve(segments.size());
  for (const auto& [start, end] : segments) {
    if (end - start == 1) {
      new_operands.push_back(concat->operand(start));
    } else {
      auto run = TryFormRun(
          absl::MakeConstSpan(components).subspan(start, end - start));
      XLS_RET_CHECK(run.has_value());
      XLS_ASSIGN_OR_RETURN(Node * recombined,
                           RecombineRun(concat->function_base(), concat->loc(),
                                        absl::MakeConstSpan(*run)));
      new_operands.push_back(recombined);
    }
  }

  if (new_operands.size() == 1) {
    XLS_RETURN_IF_ERROR(concat->ReplaceUsesWith(new_operands[0]));
  } else {
    XLS_RETURN_IF_ERROR(
        concat->ReplaceUsesWithNew<Concat>(new_operands).status());
  }
  return true;
}

}  // namespace

absl::StatusOr<bool> BitwiseRecombinationPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  bool changed = false;
  std::deque<Concat*> worklist;
  for (Node* node : f->nodes()) {
    if (node->Is<Concat>()) {
      worklist.push_back(node->As<Concat>());
    }
  }

  while (!worklist.empty()) {
    Concat* concat = worklist.front();
    worklist.pop_front();
    if (concat->IsDead()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(bool concat_changed, SimplifyConcat(concat));
    if (concat_changed) {
      changed = true;
    }
  }

  return changed;
}

}  // namespace xls
