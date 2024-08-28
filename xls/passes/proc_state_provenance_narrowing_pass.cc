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

#include "xls/passes/proc_state_provenance_narrowing_pass.h"

#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/proc.h"
#include "xls/ir/ternary.h"
#include "xls/ir/value.h"
#include "xls/passes/bit_provenance_analysis.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/ternary_query_engine.h"

namespace xls {

namespace {

struct BitSegment {
  int64_t low_bit;
  int64_t width;
  std::optional<Bits> value = std::nullopt;

  absl::StatusOr<Node*> SliceNode(Node* src) const {
    XLS_RET_CHECK(!value) << "This is a segment with a constant value";
    if (low_bit == 0 && width == src->BitCountOrDie()) {
      return src;
    }
    return src->function_base()->MakeNodeWithName<BitSlice>(
        src->loc(), src, low_bit, width,
        src->HasAssignedName() ? absl::StrFormat("sliced_%s", src->GetName())
                               : "");
  }

  Bits SliceBits(const Bits& v) const {
    CHECK(!value);
    return v.Slice(low_bit, width);
  }
};

Bits NarrowValue(const Bits& v, absl::Span<const BitSegment> segments) {
  std::vector<Bits> pieces;
  pieces.reserve(segments.size());
  for (const BitSegment& segment : segments) {
    if (!segment.value) {
      pieces.push_back(segment.SliceBits(v));
    }
  }
  return bits_ops::Concat(pieces);
}

// Get the bit segments which are not set in the mask
//
// Returned list is ordered from MSB to LSB (i.e. how a concat would be ordered)
std::vector<BitSegment> ExtractBitSegments(const Bits& mask,
                                           const Bits& value) {
  std::vector<BitSegment> segments;
  Bits remaining = mask;
  int64_t cur = 0;
  while (remaining.bit_count() != 0) {
    if (remaining.Get(0)) {
      // true segment which is ignored.
      int64_t segment_len = remaining.CountTrailingOnes();
      segments.push_back({.low_bit = cur,
                          .width = segment_len,
                          .value = value.Slice(cur, segment_len)});
      remaining =
          remaining.Slice(segment_len, remaining.bit_count() - segment_len);
      cur += segment_len;
    } else {
      int64_t segment_len = remaining.CountTrailingZeros();
      segments.push_back({.low_bit = cur, .width = segment_len});
      remaining =
          remaining.Slice(segment_len, remaining.bit_count() - segment_len);
      cur += segment_len;
    }
  }
  absl::c_reverse(segments);
  return segments;
}

// TODO(allight): It might be worthwhile to give each segment its own unique
// Param or turn it into a tuple param. It would make this simpler at least.
class NarrowTransform final : public Proc::StateElementTransformer {
 public:
  explicit NarrowTransform(std::vector<BitSegment>&& segments)
      : segments_(std::move(segments)) {}

  absl::StatusOr<Node*> TransformParamRead(Proc* proc, Param* new_param,
                                           Param* old_param) final {
    std::vector<Node*> concat_args;
    concat_args.reserve(segments_.size());
    int64_t bits_used = 0;
    for (const BitSegment& segment : segments_) {
      if (segment.value) {
        XLS_ASSIGN_OR_RETURN(
            Node * lit, proc->MakeNodeWithName<Literal>(
                            old_param->loc(), Value(*segment.value),
                            absl::StrFormat("%s_constant_bits_%d_width_%d",
                                            old_param->name(), segment.low_bit,
                                            segment.width)));
        concat_args.push_back(lit);
      } else {
        // NB segments are ordered MSB to LSB so bits_used is how many bits from
        // the end we are.
        int64_t slice_start =
            new_param->BitCountOrDie() - (bits_used + segment.width);
        Node* slice;
        if (segment.width == new_param->BitCountOrDie() && slice_start == 0) {
          slice = new_param;
        } else {
          XLS_ASSIGN_OR_RETURN(
              slice,
              proc->MakeNodeWithName<BitSlice>(
                  old_param->loc(), new_param, slice_start, segment.width,
                  absl::StrFormat("%s_variable_bits_%d_width_%d",
                                  old_param->name(), segment.low_bit,
                                  segment.width)));
        }
        concat_args.push_back(slice);
        bits_used += segment.width;
      }
    }
    if (concat_args.size() == 1) {
      return concat_args.front();
    }
    return proc->MakeNodeWithName<Concat>(
        old_param->loc(), concat_args,
        absl::StrFormat("%s_reconstructed", old_param->name()));
  }

  absl::StatusOr<Node*> TransformNextValue(Proc* proc, Param* new_param,
                                           Next* old_next) final {
    std::vector<Node*> concat_args;
    concat_args.reserve(segments_.size());
    for (const BitSegment& segment : segments_) {
      if (segment.value) {
        // Don't need to do anything with const portions.
        continue;
      }
      XLS_ASSIGN_OR_RETURN(Node * slice, segment.SliceNode(old_next->value()));
      concat_args.push_back(slice);
    }
    if (concat_args.size() == 1) {
      return concat_args.front();
    }
    return proc->MakeNodeWithName<Concat>(
        old_next->value()->loc(), concat_args,
        old_next->value()->HasAssignedName()
            ? absl::StrFormat("%s_slice_reduced", old_next->value()->GetName())
            : "");
  }

 private:
  std::vector<BitSegment> segments_;
};

absl::StatusOr<Bits> UnchangedBits(Proc* proc, Param* param,
                                   const Bits& initial_bits,
                                   const TernaryQueryEngine& tqe,
                                   const BitProvenanceAnalysis& provenance) {
  Bits unchanged_bits = Bits::AllOnes(initial_bits.bit_count());
  for (Next* next : proc->next_values(param)) {
    if (next->value() == param) {
      // Pass-through nexts are trivially unaffecting.
      continue;
    }
    if (unchanged_bits.IsZero()) {
      return unchanged_bits;
    }
    TernaryVector ternary = tqe.GetTernary(next->value()).Get({});
    Bits ternary_known_unchanged = bits_ops::Not(
        // One on every bit that is not known to be identical
        bits_ops::Or(
            // 1 on every bit that's unknown
            bits_ops::Not(ternary_ops::ToKnownBits(ternary)),
            // 1 on every bit that differs
            bits_ops::Xor(initial_bits,
                          ternary_ops::ToKnownBitsValues(ternary))));
    const TreeBitSources& sources =
        provenance.GetBitSources(next->value()).Get({});
    InlineBitmap provenance_unchanged_bm(initial_bits.bit_count());
    for (const auto& segment : sources.ranges()) {
      if (segment.source_node() == param &&
          segment.source_tree_index().empty() &&
          segment.source_bit_index_low() == segment.dest_bit_index_low() &&
          segment.source_bit_index_high() == segment.dest_bit_index_high()) {
        // unchanged segment
        provenance_unchanged_bm.SetRange(segment.dest_bit_index_low(),
                                         // NB Set range is exclusive on end.
                                         segment.dest_bit_index_high() + 1,
                                         /*value=*/true);
      }
    }
    Bits provenance_unchanged =
        Bits::FromBitmap(std::move(provenance_unchanged_bm));
    Bits general_unchanged =
        bits_ops::Or(provenance_unchanged, ternary_known_unchanged);
    unchanged_bits = bits_ops::And(unchanged_bits, general_unchanged);
  }
  return unchanged_bits;
}

}  // namespace

absl::StatusOr<bool> ProcStateProvenanceNarrowingPass::RunOnProcInternal(
    Proc* proc, const OptimizationPassOptions& options,
    PassResults* results) const {
  // Query engine to identify writes of (parts of) the initial value.
  TernaryQueryEngine tqe;
  XLS_ASSIGN_OR_RETURN(BitProvenanceAnalysis provenance,
                       BitProvenanceAnalysis::Create(proc));
  XLS_RETURN_IF_ERROR(tqe.Populate(proc).status());
  bool made_changes = false;

  std::vector<std::tuple<Param*, NarrowTransform, Bits>> transforms;

  for (Param* param : proc->params()) {
    if (!param->GetType()->IsBits()) {
      // TODO(allight): Narrowing arrays/exploding arrays and narrowing might be
      // worthwhile.
      continue;
    }
    XLS_ASSIGN_OR_RETURN(Value init, proc->GetInitValue(param));
    XLS_RET_CHECK(init.IsBits());
    const Bits& initial_bits = init.bits();
    XLS_ASSIGN_OR_RETURN(
        Bits unchanged_bits,
        UnchangedBits(proc, param, initial_bits, tqe, provenance));
    // Do the actual splitting
    if (unchanged_bits.IsZero()) {
      VLOG(3) << "Unable to narrow " << param
              << " no bits survive unconditionally.";
      continue;
    }
    std::vector<BitSegment> segments =
        ExtractBitSegments(unchanged_bits, initial_bits);
    VLOG(2) << "parameter '" << param->name() << ": " << param->GetType()
            << "' has bits which never change (unchanged bits: "
            << unchanged_bits.ToDebugString() << "). Can narrow from "
            << param->GetType()->GetFlatBitCount() << " to "
            << (unchanged_bits.bit_count() - unchanged_bits.PopCount());
    Bits narrowed_init = NarrowValue(initial_bits, segments);
    transforms.push_back(
        {param, NarrowTransform(std::move(segments)), narrowed_init});
  }

  for (auto& [param, transform, narrowed_init] : transforms) {
    made_changes = true;
    XLS_RETURN_IF_ERROR(
        proc->TransformStateElement(param, Value(narrowed_init), transform)
            .status());
  }

  return made_changes;
}

REGISTER_OPT_PASS(ProcStateProvenanceNarrowingPass);
}  // namespace xls
