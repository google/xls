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
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
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

namespace {

// Returns true if both nodes are bits-typed.
bool BothBitsTyped(const Node* a, const Node* b) {
  return a->GetType()->IsBits() && b->GetType()->IsBits();
}

// Returns true if `original` and `variant` are bit-typed and `original` is
// narrower or equal in width to `variant`.
bool LessThanOrEqualBitwidth(const Node* original, const Node* variant) {
  if (!BothBitsTyped(original, variant)) {
    return false;
  }
  if (original->BitCountOrDie() > variant->BitCountOrDie()) {
    return false;
  }
  for (int i = 0; i < original->operand_count(); ++i) {
    if (!BothBitsTyped(original->operand(i), variant->operand(i))) {
      return false;
    }
    if (original->operand(i)->BitCountOrDie() >
        variant->operand(i)->BitCountOrDie()) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<double> EstimateAreaForNodes(
    const AreaEstimator& area_estimator,
    std::function<absl::Status(Package* p, FunctionBuilder* fb)> build_fn) {
  Package p("area_check");
  FunctionBuilder fb("area_check", &p);
  XLS_RETURN_IF_ERROR(build_fn(&p, &fb));
  XLS_ASSIGN_OR_RETURN(Function * f, fb.Build());
  XLS_ASSIGN_OR_RETURN(
      double area,
      area_estimator.GetOperationAreaInSquareMicrons(f->return_value()));
  return area;
}

// `BitwidthExtendingEquivalenceMapping` handles nodes with identical ops where
// the original node and its operands are narrower than or equal to the variant
// node's bit widths.
class BitwidthExtendingEquivalenceMapping : public EquivalenceMapping {
 public:
  static std::optional<std::unique_ptr<EquivalenceMapping>> TryCreate(
      const Node* original, const Node* variant) {
    if (original->op() != variant->op()) {
      return std::nullopt;
    }
    if (original->operand_count() != variant->operand_count()) {
      return std::nullopt;
    }
    if (!LessThanOrEqualBitwidth(original, variant)) {
      return std::nullopt;
    }
    return std::make_unique<BitwidthExtendingEquivalenceMapping>(original,
                                                                 variant);
  }

  using EquivalenceMapping::EquivalenceMapping;

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
};

// `AddSubEquivalenceMapping` handles folding between `add` and `sub` nodes.
class AddSubEquivalenceMapping : public EquivalenceMapping {
 public:
  static std::optional<std::unique_ptr<EquivalenceMapping>> TryCreate(
      const Node* original, const Node* variant) {
    if (!original->OpIn({Op::kAdd, Op::kSub}) ||
        !variant->OpIn({Op::kAdd, Op::kSub})) {
      return std::nullopt;
    }
    if (original->operand_count() != 2 || variant->operand_count() != 2) {
      return std::nullopt;
    }
    if (!LessThanOrEqualBitwidth(original, variant)) {
      return std::nullopt;
    }
    return std::make_unique<AddSubEquivalenceMapping>(original, variant);
  }

  using EquivalenceMapping::EquivalenceMapping;

  bool RequiresOperandTransformation() const override { return true; }

  bool RequiresOutputTransformation() const override {
    return original_->BitCountOrDie() < variant_->BitCountOrDie();
  }

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
      Node* op_to_negate = operands[1];
      if (op_to_negate->op() != Op::kNeg) {
        return EstimateAreaForNodes(
            area_estimator,
            [op_to_negate](Package* p, FunctionBuilder* fb) -> absl::Status {
              XLS_ASSIGN_OR_RETURN(
                  Type * input_type,
                  p->MapTypeFromOtherPackage(op_to_negate->GetType()));
              fb->Negate(fb->Param("value_to_negate", input_type));
              return absl::OkStatus();
            });
      }
    }
    return 0.0;
  }
};

// Returns `target` XORed with a sign-extended mask created from the MSB of
// `msb_source`. Reuses existing BitSlice and SignExt nodes if available.
absl::StatusOr<Node*> XorWithSignExtendedMsb(FunctionBase* f, Node* target,
                                             Node* msb_source) {
  int64_t target_width = target->BitCountOrDie();
  int64_t source_width = msb_source->BitCountOrDie();
  XLS_ASSIGN_OR_RETURN(
      Node * msb,
      FindOrMakeBitSlice(msb_source, /*start=*/source_width - 1, /*width=*/1));

  Node* mask = nullptr;
  for (Node* user : msb->users()) {
    if (user->op() == Op::kSignExt && user->BitCountOrDie() == target_width) {
      mask = user;
      break;
    }
  }
  if (mask == nullptr) {
    XLS_ASSIGN_OR_RETURN(
        mask,
        f->MakeNode<ExtendOp>(target->loc(), msb,
                              /*new_bit_count=*/target_width, Op::kSignExt));
  }

  return f->MakeNode<NaryOp>(target->loc(), std::vector<Node*>{target, mask},
                             Op::kXor);
}

// Equivalence mapping between shift operations (shll, shrl, and shra).
//
// y = x << s is equivalent to y = reverse(reverse(x) >> s)
// y = x >> s is equivalent to y = reverse(reverse(x) << s)
// y = x >>> s is equivalent to y = ((x xor m) >> s) xor m
//    where m = sign_ext(msb(x)). If m = 0, then the XORs are no-ops. If m = 1s,
//    then xors are no-ops on shifted bits, flipping those bits twice, meanwhile
//    converting all 0s shifted into the MSBs into 1s.
// y = x >> s is equivalent to y = slice(zero_ext(x, N) >>> s, 0, W) for W < N
//    where zero-extending (padding) ensures shra shifts in 0s.
class ShiftEquivalenceMapping : public EquivalenceMapping {
 public:
  static std::optional<std::unique_ptr<EquivalenceMapping>> TryCreate(
      const Node* original, const Node* variant) {
    if (!original->OpIn({Op::kShll, Op::kShrl, Op::kShra}) ||
        !variant->OpIn({Op::kShll, Op::kShrl, Op::kShra}) ||
        original->op() == variant->op()) {
      return std::nullopt;
    }
    if (!LessThanOrEqualBitwidth(original, variant)) {
      return std::nullopt;
    }
    // Merging into shra only supported if the original node is strictly smaller
    // so we can fit the zero-extended original into the shra.
    if (variant->op() == Op::kShra &&
        original->BitCountOrDie() >= variant->BitCountOrDie()) {
      return std::nullopt;
    }
    bool same_shift_amount =
        original->operand(1)->IsDefinitelyEqualTo(variant->operand(1));
    if (!same_shift_amount) {
      return std::nullopt;
    }
    return std::make_unique<ShiftEquivalenceMapping>(original, variant);
  }

  using EquivalenceMapping::EquivalenceMapping;

  bool RequiresOperandTransformation() const override { return true; }
  bool RequiresOutputTransformation() const override { return true; }

  absl::StatusOr<std::vector<Node*>> ApplyToOperands(
      FunctionBase* f,
      absl::Span<Node* const> original_operands) const override {
    XLS_RET_CHECK_EQ(original_operands.size(), 2);
    Node* op0 = original_operands[0];
    Node* op1 = original_operands[1];
    XLS_RET_CHECK(op0->GetType()->IsBits());
    XLS_RET_CHECK(op1->GetType()->IsBits());

    // If coming from shra, xor with the sign-extended msb.
    if (original_->op() == Op::kShra) {
      XLS_ASSIGN_OR_RETURN(
          op0, XorWithSignExtendedMsb(f, /*target=*/op0, /*msb_source=*/op0));
    }

    // If going between left and right shifts, reverse bits.
    if (variant_->op() == Op::kShll || original_->op() == Op::kShll) {
      XLS_ASSIGN_OR_RETURN(op0,
                           f->MakeNode<UnOp>(op0->loc(), op0, Op::kReverse));
    }

    // Extend shifted value to target width if necessary.
    int64_t target_width = variant_->operand(0)->BitCountOrDie();
    if (op0->BitCountOrDie() < target_width) {
      XLS_ASSIGN_OR_RETURN(
          op0,
          f->MakeNode<ExtendOp>(op0->loc(), op0,
                                /*new_bit_count=*/target_width, Op::kZeroExt));
    }

    // Extend shift amount to target width if necessary.
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
    XLS_RET_CHECK(variant_output->GetType()->IsBits());
    Node* result = variant_output;
    int64_t orig_width = original_->BitCountOrDie();

    // If the original node is narrower, bit-slice to the original width.
    if (result->BitCountOrDie() > orig_width) {
      XLS_ASSIGN_OR_RETURN(
          result, f->MakeNode<BitSlice>(result->loc(), result, /*start=*/0,
                                        /*width=*/orig_width));
    }

    // If going between left and right shifts, reverse bits.
    if (variant_->op() == Op::kShll || original_->op() == Op::kShll) {
      XLS_ASSIGN_OR_RETURN(
          result, f->MakeNode<UnOp>(result->loc(), result, Op::kReverse));
    }

    // If coming from shra, xor once again with the sign-extended msb.
    if (original_->op() == Op::kShra) {
      XLS_ASSIGN_OR_RETURN(
          result, XorWithSignExtendedMsb(f, /*target=*/result,
                                         /*msb_source=*/original_->operand(0)));
    }

    return result;
  }

  absl::StatusOr<double> EstimateAreaOverhead(
      const AreaEstimator& area_estimator,
      absl::Span<Node* const> original_operands,
      const Node* original_node) const override {
    if (original_->op() == Op::kShra) {
      int64_t orig_width = original_->BitCountOrDie();
      XLS_ASSIGN_OR_RETURN(
          double xor_area,
          EstimateAreaForNodes(
              area_estimator,
              [orig_width](Package* p, FunctionBuilder* fb) -> absl::Status {
                Type* input_type = p->GetBitsType(orig_width);
                fb->Xor(fb->Param("a", input_type), fb->Param("b", input_type));
                return absl::OkStatus();
              }));
      return 2.0 * xor_area;
    }
    // kReverse, BitSlice, and ExtendOp are pure wire operations in hardware (0
    // area overhead).
    return 0.0;
  }
};

}  // namespace

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
