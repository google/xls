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

#include "xls/passes/resource_sharing_equivalence.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/area_model/area_estimator.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"

namespace xls {

// `IdentityEquivalenceMapping` handles nodes with identical ops and bit widths.
std::optional<std::unique_ptr<EquivalenceMapping>>
IdentityEquivalenceMapping::TryCreate(const Node* original,
                                      const Node* variant) {
  if (original->op() != variant->op()) {
    return std::nullopt;
  }
  if (!HasSameBitWidth(original, variant)) {
    return std::nullopt;
  }
  if (!HaveSameOperandBitWidths(original, variant)) {
    return std::nullopt;
  }
  return std::make_unique<IdentityEquivalenceMapping>(original, variant);
}

namespace {

// `BitwidthExtendingEquivalenceMapping` handles nodes with identical ops where
// the original node and its operands are narrower than or equal to the variant
// node's bit widths (with at least one bit width being strictly smaller).
class BitwidthExtendingEquivalenceMapping : public EquivalenceMapping {
 public:
  static std::optional<std::unique_ptr<EquivalenceMapping>> TryCreate(
      const Node* original, const Node* variant) {
    if (original->op() != variant->op()) {
      return std::nullopt;
    }
    if (!IsBitsTyped(original, variant)) {
      return std::nullopt;
    }
    if (original->BitCountOrDie() > variant->BitCountOrDie()) {
      return std::nullopt;
    }
    if (original->operand_count() != variant->operand_count()) {
      return std::nullopt;
    }
    bool strictly_smaller =
        original->BitCountOrDie() < variant->BitCountOrDie();
    for (int i = 0; i < original->operand_count(); ++i) {
      if (!IsBitsTyped(original->operand(i), variant->operand(i))) {
        return std::nullopt;
      }
      if (original->operand(i)->BitCountOrDie() >
          variant->operand(i)->BitCountOrDie()) {
        return std::nullopt;
      }
      if (original->operand(i)->BitCountOrDie() <
          variant->operand(i)->BitCountOrDie()) {
        strictly_smaller = true;
      }
    }
    if (!strictly_smaller) {
      return std::nullopt;
    }
    return std::make_unique<BitwidthExtendingEquivalenceMapping>(original,
                                                                 variant);
  }

  using EquivalenceMapping::EquivalenceMapping;

  absl::StatusOr<std::vector<Node*>> ApplyToOperands(
      FunctionBase* f,
      absl::Span<Node* const> original_operands) const override {
    XLS_RET_CHECK_EQ(original_operands.size(), variant_->operand_count());
    std::vector<Node*> result;
    result.reserve(original_operands.size());
    for (int i = 0; i < original_operands.size(); ++i) {
      Node* op = original_operands[i];
      XLS_RET_CHECK(op->GetType()->IsBits());
      XLS_RET_CHECK(variant_->operand(i)->GetType()->IsBits());
      int64_t target_width = variant_->operand(i)->BitCountOrDie();
      if (op->BitCountOrDie() < target_width) {
        Op ext_op =
            IsSigned(const_cast<Node*>(variant_)) ? Op::kSignExt : Op::kZeroExt;
        XLS_ASSIGN_OR_RETURN(
            Node * ext,
            f->MakeNode<ExtendOp>(op->loc(), op, target_width, ext_op));
        result.push_back(ext);
      } else {
        result.push_back(op);
      }
    }
    return result;
  }

  absl::StatusOr<Node*> ApplyToOutput(FunctionBase* f,
                                      Node* variant_output) const override {
    XLS_RET_CHECK(original_->GetType()->IsBits());
    XLS_RET_CHECK(variant_->GetType()->IsBits());
    if (original_->BitCountOrDie() < variant_->BitCountOrDie()) {
      return f->MakeNode<BitSlice>(variant_output->loc(), variant_output,
                                   /*start=*/0,
                                   /*width=*/original_->BitCountOrDie());
    }
    return variant_output;
  }

  bool RequiresOperandTransformation() const override {
    for (int i = 0; i < original_->operand_count(); ++i) {
      if (original_->operand(i)->BitCountOrDie() <
          variant_->operand(i)->BitCountOrDie()) {
        return true;
      }
    }
    return false;
  }

  bool RequiresOutputTransformation() const override {
    return original_->BitCountOrDie() < variant_->BitCountOrDie();
  }
};

absl::StatusOr<double> EstimateAreaForNegatingNode(
    Node* n, const AreaEstimator& area_estimator) {
  Package p("area_check");
  FunctionBuilder fb("area_check", &p);
  XLS_ASSIGN_OR_RETURN(Type * input_type,
                       p.MapTypeFromOtherPackage(n->GetType()));
  BValue value_to_negate = fb.Param("value_to_negate", input_type);
  fb.Negate(value_to_negate);
  XLS_ASSIGN_OR_RETURN(Function * f, fb.Build());
  XLS_ASSIGN_OR_RETURN(
      double area,
      area_estimator.GetOperationAreaInSquareMicrons(f->return_value()));
  return area;
}

// `AddSubEquivalenceMapping` handles folding between `add` and `sub` nodes.
class AddSubEquivalenceMapping : public EquivalenceMapping {
 public:
  static std::optional<std::unique_ptr<EquivalenceMapping>> TryCreate(
      const Node* original, const Node* variant) {
    if (!IsBitsTyped(original, variant)) {
      return std::nullopt;
    }
    if (!original->OpIn({Op::kAdd, Op::kSub}) ||
        !variant->OpIn({Op::kAdd, Op::kSub})) {
      return std::nullopt;
    }
    if (original->operand_count() != 2 || variant->operand_count() != 2) {
      return std::nullopt;
    }
    if (!IsBitsTyped(original->operand(0), variant->operand(0)) ||
        !IsBitsTyped(original->operand(1), variant->operand(1))) {
      return std::nullopt;
    }
    if (original->BitCountOrDie() > variant->BitCountOrDie()) {
      return std::nullopt;
    }
    if (original->operand(0)->BitCountOrDie() >
            variant->operand(0)->BitCountOrDie() ||
        original->operand(1)->BitCountOrDie() >
            variant->operand(1)->BitCountOrDie()) {
      return std::nullopt;
    }
    return std::make_unique<AddSubEquivalenceMapping>(original, variant);
  }

  using EquivalenceMapping::EquivalenceMapping;

  absl::StatusOr<std::vector<Node*>> ApplyToOperands(
      FunctionBase* f,
      absl::Span<Node* const> original_operands) const override {
    XLS_RET_CHECK_EQ(original_operands.size(), 2);
    Node* op0 = original_operands[0];
    Node* op1 = original_operands[1];
    XLS_RET_CHECK(op0->GetType()->IsBits());
    XLS_RET_CHECK(op1->GetType()->IsBits());
    XLS_RET_CHECK(variant_->operand(0)->GetType()->IsBits());
    XLS_RET_CHECK(variant_->operand(1)->GetType()->IsBits());

    if (original_->op() != variant_->op()) {
      if (op1->op() == Op::kNeg) {
        op1 = op1->operand(0);
      } else {
        XLS_ASSIGN_OR_RETURN(op1, f->MakeNode<UnOp>(op1->loc(), op1, Op::kNeg));
      }
    }

    if (op0->BitCountOrDie() < variant_->operand(0)->BitCountOrDie()) {
      XLS_ASSIGN_OR_RETURN(
          op0, f->MakeNode<ExtendOp>(op0->loc(), op0,
                                     variant_->operand(0)->BitCountOrDie(),
                                     Op::kZeroExt));
    }
    if (op1->BitCountOrDie() < variant_->operand(1)->BitCountOrDie()) {
      XLS_ASSIGN_OR_RETURN(
          op1, f->MakeNode<ExtendOp>(op1->loc(), op1,
                                     variant_->operand(1)->BitCountOrDie(),
                                     Op::kZeroExt));
    }

    return std::vector<Node*>{op0, op1};
  }

  absl::StatusOr<Node*> ApplyToOutput(FunctionBase* f,
                                      Node* variant_output) const override {
    XLS_RET_CHECK(original_->GetType()->IsBits());
    XLS_RET_CHECK(variant_->GetType()->IsBits());
    if (original_->BitCountOrDie() < variant_->BitCountOrDie()) {
      return f->MakeNode<BitSlice>(variant_output->loc(), variant_output,
                                   /*start=*/0,
                                   /*width=*/original_->BitCountOrDie());
    }
    return variant_output;
  }

  absl::StatusOr<double> EstimateAreaOverhead(
      const AreaEstimator& area_estimator, absl::Span<Node* const> operands,
      const Node* output) const override {
    if (original_->op() != variant_->op()) {
      if (operands[1]->op() != Op::kNeg) {
        return EstimateAreaForNegatingNode(operands[1], area_estimator);
      }
    }
    return 0.0;
  }

  bool RequiresOperandTransformation() const override { return true; }

  bool RequiresOutputTransformation() const override {
    return original_->BitCountOrDie() < variant_->BitCountOrDie();
  }
};

}  // namespace

std::optional<std::unique_ptr<EquivalenceMapping>>
ShiftEquivalenceMapping::TryCreate(const Node* original, const Node* variant) {
  bool is_shll_to_shrl =
      original->op() == Op::kShll && variant->op() == Op::kShrl;
  bool is_shrl_to_shll =
      original->op() == Op::kShrl && variant->op() == Op::kShll;
  if (!is_shll_to_shrl && !is_shrl_to_shll) {
    return std::nullopt;
  }
  if (!HasSameBitWidth(original, variant)) {
    return std::nullopt;
  }
  if (!HaveSameOperandBitWidths(original, variant)) {
    return std::nullopt;
  }
  bool same_shift_amount = original->operand(1) == variant->operand(1) ||
                           (original->operand(1)->Is<Literal>() &&
                            variant->operand(1)->Is<Literal>() &&
                            original->operand(1)->As<Literal>()->value() ==
                                variant->operand(1)->As<Literal>()->value());
  if (!same_shift_amount) {
    return std::nullopt;
  }
  return std::make_unique<ShiftEquivalenceMapping>(original, variant);
}

absl::StatusOr<std::vector<Node*>> ShiftEquivalenceMapping::ApplyToOperands(
    FunctionBase* f, absl::Span<Node* const> original_operands) const {
  XLS_RET_CHECK_EQ(original_operands.size(), 2);
  XLS_RET_CHECK(original_operands[0]->GetType()->IsBits());
  XLS_ASSIGN_OR_RETURN(Node * reversed_op0,
                       f->MakeNode<UnOp>(original_operands[0]->loc(),
                                         original_operands[0], Op::kReverse));
  return std::vector<Node*>{reversed_op0, original_operands[1]};
}

absl::StatusOr<Node*> ShiftEquivalenceMapping::ApplyToOutput(
    FunctionBase* f, Node* variant_output) const {
  XLS_RET_CHECK(variant_output->GetType()->IsBits());
  return f->MakeNode<UnOp>(variant_output->loc(), variant_output, Op::kReverse);
}

absl::StatusOr<double> ShiftEquivalenceMapping::EstimateAreaOverhead(
    const AreaEstimator& area_estimator,
    absl::Span<Node* const> original_operands,
    const Node* original_node) const {
  // kReverse is just a wire reordering in hardware (0 area overhead).
  return 0.0;
}

void NodeEquivalenceMapper::Register(Factory factory) {
  absl::MutexLock lock(mutex_);
  factories_.push_back(factory);
}

std::optional<std::unique_ptr<EquivalenceMapping>>
NodeEquivalenceMapper::ComputeMapping(const Node* original,
                                      const Node* variant) const {
  absl::MutexLock lock(mutex_);
  for (Factory factory : factories_) {
    if (auto mapping = factory(original, variant); mapping.has_value()) {
      return mapping;
    }
  }
  return std::nullopt;
}

NodeEquivalenceMapper& GetNodeEquivalenceMapper() {
  static auto* mapper = []() {
    auto* m = new NodeEquivalenceMapper();
    m->Register<IdentityEquivalenceMapping>();
    m->Register<BitwidthExtendingEquivalenceMapping>();
    m->Register<AddSubEquivalenceMapping>();
    m->Register<ShiftEquivalenceMapping>();
    return m;
  }();
  return *mapper;
}

}  // namespace xls
