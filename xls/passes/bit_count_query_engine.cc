// Copyright 2025 The XLS Authors
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

#include "xls/passes/bit_count_query_engine.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "cppitertools/zip.hpp"
#include "xls/common/source_location.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/dataflow_visitor.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"

namespace xls {

namespace {
using LeadingBitsTree = LeafTypeTree<internal::LeadingBits>;
class BitCountVisitor : public DataflowVisitor<internal::LeadingBits> {
 public:
  absl::Status InjectValue(Node* node, const LeadingBitsTree* value) {
    if (value == nullptr) {
      return DefaultHandler(node);
    }
    return SetValue(node, value->AsView());
  }

  absl::Status DefaultHandler(Node* node) override {
    XLS_ASSIGN_OR_RETURN(
        LeadingBitsTree unknown,
        LeadingBitsTree::CreateFromFunction(
            node->GetType(),
            [](Type* leaf_type) -> absl::StatusOr<internal::LeadingBits> {
              if (leaf_type->IsBits() && leaf_type->GetFlatBitCount() != 0) {
                return internal::LeadingBits::Unconstrained();
              }
              // Invalid bit-count since non-bits leaf type.
              return internal::LeadingBits::ZeroSize();
            }));
    return SetValue(node, std::move(unknown));
  }

  absl::Status HandleLiteral(Literal* lit) override {
    XLS_ASSIGN_OR_RETURN(LeafTypeTree<Value> ltt,
                         ValueToLeafTypeTree(lit->value(), lit->GetType()));
    LeafTypeTree<internal::LeadingBits> res =
        leaf_type_tree::Map<internal::LeadingBits, Value>(
            ltt.AsView(), [](const Value& v) -> internal::LeadingBits {
              if (!v.IsBits() || v.bits().bit_count() == 0) {
                return internal::LeadingBits::ZeroSize();
              }
              int64_t leading_zeros = v.bits().CountLeadingZeros();
              if (leading_zeros != 0) {
                return internal::LeadingBits::KnownZeros(leading_zeros);
              }
              return internal::LeadingBits::KnownOnes(
                  v.bits().CountLeadingOnes());
            });
    return SetValue(lit, std::move(res));
  }

  absl::Status HandleSignExtend(ExtendOp* sign_ext) override {
    Node* input = sign_ext->operand(0);
    internal::LeadingBits input_cnt = GetValue(input).Get({});
    return SetSingleValue(
        sign_ext,
        input_cnt.ExtendBy(sign_ext->new_bit_count() - input->BitCountOrDie()));
  }

  absl::Status HandleZeroExtend(ExtendOp* zero_ext) override {
    Node* input = zero_ext->operand(0);
    internal::LeadingBits input_cnt = GetValue(input).Get({});
    int64_t extend_by = zero_ext->new_bit_count() - input->BitCountOrDie();
    if (extend_by == 0) {
      return SetSingleValue(zero_ext, input_cnt);
    }
    return SetSingleValue(zero_ext, internal::LeadingBits::KnownZeros(
                                        input_cnt.leading_zeros() + extend_by));
  }

  absl::Status HandleConcat(Concat* concat) override {
    Node* zero_op = nullptr;
    int64_t start = 0;
    for (start = 0; start < concat->operand_count(); ++start) {
      if (concat->operand(start)->BitCountOrDie() != 0) {
        zero_op = concat->operand(start);
        break;
      }
    }
    if (start == concat->operand_count()) {
      return SetSingleValue(concat, internal::LeadingBits::ZeroSize());
    }

    internal::LeadingBits cnts = GetValue(zero_op).Get({});
    if (cnts.leading_signs() == zero_op->BitCountOrDie()) {
      // The MSB are either all ones or all zeros so we can get bits from the
      // later ones.
      bool known_zero = cnts.leading_zeros() != 0;
      bool known_one = cnts.leading_ones() != 0;
      for (int64_t i = start + 1; i < concat->operand_count(); ++i) {
        Node* op = concat->operand(i);
        if (op->BitCountOrDie() == 0) {
          continue;
        }
        internal::LeadingBits last_cnt = GetValue(op).Get({});
        if ((last_cnt.leading_zeros() != 0 && known_zero) ||
            (last_cnt.leading_ones() != 0 && known_one) ||
            (!known_zero && !known_one && concat->operand(i) == zero_op)) {
          cnts = cnts.ExtendBy(last_cnt.leading_signs());
        } else {
          break;
        }
        if (last_cnt.leading_signs() != op->BitCountOrDie()) {
          break;
        }
      }
    }
    return SetSingleValue(concat, cnts);
  }

  // Z3 correctness proof in bit_count_query_engine_proofs_z3.py
  absl::Status HandleNeg(UnOp* neg) override {
    if (neg->BitCountOrDie() == 0) {
      return SetSingleValue(neg, internal::LeadingBits::ZeroSize());
    }
    Node* input = neg->operand(0);
    internal::LeadingBits cnts = GetValue(input).Get({});
    // Since 0 and INT_MIN map to themselves only sign-bits stay the same unless
    // the value is exactly either 0 or int-min
    if (!(cnts.leading_ones() == input->BitCountOrDie() ||
          cnts.leading_zeros() == input->BitCountOrDie())) {
      // NB Need to reduce the number of sign bits by one since, for example
      // i16{-1} has 16 sign bits but i16{1} only has 15.
      // 0 might still be a possible value so we can't swap non-sign bits.
      bool known_positive = cnts.leading_zeros() != 0;
      cnts = cnts.ToSignBits();
      if (!known_positive) {
        // Positive -> negative doesn't lose any sign bits. Negative -> positive
        // can however.
        cnts = cnts.ShortenBy(1);
      }
    }
    return SetSingleValue(neg, cnts);
  }

  absl::Status HandleNot(UnOp* not_op) override {
    Node* input = not_op->operand(0);
    internal::LeadingBits input_cnt = GetValue(input).Get({});
    switch (input_cnt.value()) {
      case TernaryValue::kUnknown:
        return SetSingleValue(not_op, input_cnt);
      case TernaryValue::kKnownZero:
        return SetSingleValue(not_op, internal::LeadingBits::KnownOnes(
                                          input_cnt.leading_zeros()));
      case TernaryValue::kKnownOne:
        return SetSingleValue(not_op, internal::LeadingBits::KnownZeros(
                                          input_cnt.leading_ones()));
    }
  }

  absl::Status HandleBitSlice(BitSlice* slice) override {
    if (slice->BitCountOrDie() == 0) {
      return SetSingleValue(slice, internal::LeadingBits::ZeroSize());
    }
    Node* input = slice->operand(0);
    internal::LeadingBits input_cnt = GetValue(input).Get({});
    int64_t last_bit_before = slice->start() + slice->width();
    int64_t input_width = input->BitCountOrDie();
    int64_t leading_bits_removed = input_width - last_bit_before;
    return SetSingleValue(slice, input_cnt.ShortenBy(leading_bits_removed)
                                     .LimitSizeTo(slice->BitCountOrDie()));
  }

  absl::Status HandleShll(BinOp* op) override {
    Node* lhs = op->operand(0);
    Node* rhs = op->operand(1);
    if (lhs->BitCountOrDie() == 0) {
      return SetSingleValue(op, internal::LeadingBits::ZeroSize());
    }
    internal::LeadingBits lhs_cnt = GetValue(lhs).Get({});
    StatelessQueryEngine sqe;
    std::optional<Value> rhs_value = sqe.KnownValue(rhs);
    if (rhs_value) {
      // Exactly known.
      int64_t rhs_int = rhs_value->bits().FitsInNBitsUnsigned(63)
                            ? rhs_value->bits().ToUint64().value()
                            : std::numeric_limits<int64_t>::max();
      if (rhs_int >= lhs->BitCountOrDie()) {
        // Clear the value by shifting out everything.
        return SetSingleValue(
            op, internal::LeadingBits::KnownZeros(lhs->BitCountOrDie()));
      }
      return SetSingleValue(op, lhs_cnt.ShortenBy(rhs_int));
    }
    // TODO(allight): Technically we can tell a bit about the range of values
    // using known-sign but not really worth doing anything. Be conservative.
    return SetSingleValue(op, internal::LeadingBits::Unconstrained());
  }

  absl::Status HandleShrl(BinOp* op) override {
    Node* lhs = op->operand(0);
    Node* rhs = op->operand(1);
    if (lhs->BitCountOrDie() == 0) {
      return SetSingleValue(op, internal::LeadingBits::ZeroSize());
    }
    internal::LeadingBits lhs_cnt = GetValue(lhs).Get({});
    StatelessQueryEngine sqe;
    std::optional<Value> rhs_value = sqe.KnownValue(rhs);
    if (rhs_value) {
      // Exactly known.
      int64_t rhs_int = rhs_value->bits().FitsInNBitsUnsigned(63)
                            ? rhs_value->bits().ToUint64().value()
                            : std::numeric_limits<int64_t>::max();
      if (rhs_int == 0) {
        return SetSingleValue(op, lhs_cnt);
      }
      int64_t new_lead_zeros = std::min(lhs_cnt.leading_zeros() != 0
                                            ? lhs_cnt.leading_zeros() + rhs_int
                                            : rhs_int,
                                        op->BitCountOrDie());
      return SetSingleValue(op,
                            internal::LeadingBits::KnownZeros(new_lead_zeros));
    }
    // TODO(allight): Technically we can tell a bit more about the range of
    // values using known-sign but not really worth doing anything. Be
    // conservative.
    return SetSingleValue(
        op, internal::LeadingBits::KnownZeros(lhs_cnt.leading_zeros()));
  }

  absl::Status HandleShra(BinOp* op) override {
    Node* lhs = op->operand(0);
    Node* rhs = op->operand(1);
    if (lhs->BitCountOrDie() == 0) {
      return SetSingleValue(op, internal::LeadingBits::ZeroSize());
    }
    internal::LeadingBits lhs_cnt = GetValue(lhs).Get({});
    StatelessQueryEngine sqe;
    std::optional<Value> rhs_value = sqe.KnownValue(rhs);
    if (rhs_value) {
      // Exactly known.
      int64_t rhs_int = rhs_value->bits().FitsInNBitsUnsigned(63)
                            ? rhs_value->bits().ToUint64().value()
                            : std::numeric_limits<int64_t>::max();
      return SetSingleValue(
          op, lhs_cnt.ExtendBy(rhs_int).LimitSizeTo(op->BitCountOrDie()));
    }
    // TODO(allight): Technically we can tell a bit about the range of values
    // using known-sign but not really worth doing anything. Be conservative,
    // counts are at least the current lhs values.
    return SetSingleValue(op, lhs_cnt);
  }

  // Z3 correctness proof in bit_count_query_engine_proofs_z3.py
  absl::Status HandleAdd(BinOp* add) override {
    Node* lhs = add->operand(0);
    Node* rhs = add->operand(1);
    internal::LeadingBits lhs_cnt = GetValue(lhs).Get({});
    internal::LeadingBits rhs_cnt = GetValue(rhs).Get({});
    int64_t new_lead_zeros =
        std::min(lhs_cnt.leading_zeros(), rhs_cnt.leading_zeros()) - 1;
    if (new_lead_zeros > 0) {
      return SetSingleValue(add,
                            internal::LeadingBits::KnownZeros(new_lead_zeros));
    }
    int64_t new_lead_ones =
        std::min(lhs_cnt.leading_ones(), rhs_cnt.leading_ones()) - 1;
    if (new_lead_ones > 0) {
      return SetSingleValue(add,
                            internal::LeadingBits::KnownOnes(new_lead_ones));
    }
    int64_t new_lead_signs =
        std::max(std::min(lhs_cnt.leading_signs(), rhs_cnt.leading_signs()) - 1,
                 int64_t{1});
    return SetSingleValue(add,
                          internal::LeadingBits::SignValues(new_lead_signs));
  }

  // Z3 correctness proof in bit_count_query_engine_proofs_z3.py
  absl::Status HandleSub(BinOp* sub) override {
    Node* lhs = sub->operand(0);
    Node* rhs = sub->operand(1);
    auto res =
        // TODO(allight): We should be able to get the leading zeros and ones
        // constrained somewhat. It's more complicated than ideal though since
        // the results are different depending on if the sides have the same
        // sign. Also the fact that 0 and INT_MIN are their own negatives causes
        // issues.
        internal::LeadingBits::SignValues(
            std::max(std::min(GetValue(lhs).Get({}).leading_signs(),
                              GetValue(rhs).Get({}).leading_signs()) -
                         1,
                     int64_t{1}));
    return SetSingleValue(sub, res);
  }

  // TODO(allight): SMul/UMul/Div can be handled.

 protected:
  absl::StatusOr<internal::LeadingBits> JoinElements(
      Type* element_type,
      absl::Span<const internal::LeadingBits* const> data_sources,
      absl::Span<const LeafTypeTreeView<internal::LeadingBits>> control_sources,
      Node* node, absl::Span<const int64_t> index) override {
    if (element_type->GetFlatBitCount() == 0) {
      return internal::LeadingBits::ZeroSize();
    }
    internal::LeadingBits res = *data_sources.front();
    int64_t leading_zeros = res.leading_zeros();
    int64_t leading_ones = res.leading_ones();
    int64_t leading_signs = res.leading_signs();
    // TODO(allight): In the case of priority/OHS we can actually do better than
    // this using leading bits of the control to eliminate some possibilities.
    for (const internal::LeadingBits* const data_source :
         data_sources.subspan(1)) {
      leading_zeros = std::min(leading_zeros, data_source->leading_zeros());
      leading_ones = std::min(leading_ones, data_source->leading_ones());
      leading_signs = std::min(leading_signs, data_source->leading_signs());
    }
    XLS_RET_CHECK(leading_zeros == 0 || leading_ones == 0);
    XLS_RET_CHECK_LE(leading_signs, element_type->GetFlatBitCount());
    XLS_RET_CHECK_LE(leading_zeros, element_type->GetFlatBitCount());
    XLS_RET_CHECK_LE(leading_ones, element_type->GetFlatBitCount());
    if (leading_zeros != 0) {
      XLS_RET_CHECK_EQ(leading_zeros, leading_signs);
      XLS_RET_CHECK_EQ(leading_ones, 0);
      return internal::LeadingBits::KnownZeros(leading_zeros);
    }
    if (leading_ones != 0) {
      XLS_RET_CHECK_EQ(leading_ones, leading_signs);
      return internal::LeadingBits::KnownOnes(leading_ones);
    }
    return internal::LeadingBits::SignValues(leading_signs);
  }

 private:
  absl::Status SetSingleValue(
      Node* node, internal::LeadingBits value,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current()) {
    XLS_RET_CHECK_LE(value.leading_signs(), node->BitCountOrDie())
        << loc.file_name() << ":" << loc.line() << " -> " << node;
    return SetValue(
        node, LeadingBitsTree::CreateSingleElementTree(node->GetType(), value));
  }
};
}  // namespace

LeafTypeTree<internal::LeadingBits> BitCountQueryEngine::ComputeInfo(
    Node* node,
    absl::Span<const LeafTypeTree<internal::LeadingBits>* const> operand_infos)
    const {
  BitCountVisitor vis;
  for (const auto& [operand, operand_info] :
       iter::zip(node->operands(), operand_infos)) {
    CHECK_OK(vis.InjectValue(operand, operand_info));
  }
  CHECK_OK(node->VisitSingleNode(&vis));
  absl::flat_hash_map<
      Node*, std::unique_ptr<SharedLeafTypeTree<internal::LeadingBits>>>
      result = std::move(vis).ToStoredValues();
  return std::move(*result.at(node)).ToOwned();
}

absl::Status BitCountQueryEngine::MergeWithGiven(
    internal::LeadingBits& info, const internal::LeadingBits& given) const {
  int64_t leading_ones = std::max(info.leading_ones(), given.leading_ones());
  int64_t leading_zeros = std::max(info.leading_zeros(), given.leading_zeros());
  int64_t leading_signs = std::max(info.leading_signs(), given.leading_signs());
  XLS_RET_CHECK(leading_zeros == 0 || leading_ones == 0);
  if (leading_zeros != 0) {
    XLS_RET_CHECK_EQ(leading_zeros, leading_signs);
    XLS_RET_CHECK_EQ(leading_ones, 0);
    info = internal::LeadingBits::KnownZeros(leading_zeros);
  } else if (leading_ones != 0) {
    XLS_RET_CHECK_EQ(leading_ones, leading_signs);
    info = internal::LeadingBits::KnownOnes(leading_ones);
  } else {
    info = internal::LeadingBits::SignValues(leading_signs);
  }
  return absl::OkStatus();
}

bool BitCountQueryEngine::AtLeastOneTrue(
    absl::Span<TreeBitLocation const> bits) const {
  for (const TreeBitLocation& bit : bits) {
    std::optional<SharedLeafTypeTree<internal::LeadingBits>> info =
        GetInfo(bit.node());
    if (!info) {
      continue;
    }
    LeafTypeTreeView<internal::LeadingBits> view =
        info->AsView(bit.tree_index());
    const internal::LeadingBits& cnt = view.Get({});
    Type* ty = view.type();
    if (cnt.leading_ones() >= (ty->GetFlatBitCount() - bit.bit_index())) {
      return true;
    }
  }
  return false;
}

bool BitCountQueryEngine::KnownEquals(const TreeBitLocation& a,
                                      const TreeBitLocation& b) const {
  if (a.node() != b.node() || !absl::c_equal(a.tree_index(), b.tree_index())) {
    return false;
  }
  // NB nodes and index are the same.
  std::optional<SharedLeafTypeTree<internal::LeadingBits>> info =
      GetInfo(a.node());
  if (!info) {
    return false;
  }
  LeafTypeTreeView<internal::LeadingBits> view = info->AsView(a.tree_index());
  const internal::LeadingBits& cnt = view.Get({});
  // If both bits are in the same leaf and are in the sign-bits portion they are
  // definitely equal.
  int64_t leaf_bit_count = view.type()->GetFlatBitCount();
  return cnt.leading_signs() >= (leaf_bit_count - a.bit_index()) &&
         cnt.leading_signs() >= (leaf_bit_count - b.bit_index());
}

}  // namespace xls
