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

#include <algorithm>
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

// Returns true if `src` and `dst` are bit-typed and `src` is
// narrower or equal in width to `dst`.
bool LessThanOrEqualBitwidth(const Node* src, const Node* dst) {
  if (!BothBitsTyped(src, dst)) {
    return false;
  }
  if (src->BitCountOrDie() > dst->BitCountOrDie()) {
    return false;
  }
  for (int i = 0; i < src->operand_count(); ++i) {
    if (!BothBitsTyped(src->operand(i), dst->operand(i))) {
      return false;
    }
    if (src->operand(i)->BitCountOrDie() > dst->operand(i)->BitCountOrDie()) {
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
// the src node and its operands are narrower than or equal to the dst
// node's bit widths.
class BitwidthExtendingEquivalenceMapping : public EquivalenceMapping {
 public:
  static std::optional<std::unique_ptr<EquivalenceMapping>> TryCreate(
      const Node* src, const Node* dst) {
    if (src->op() != dst->op()) {
      return std::nullopt;
    }
    if (src->operand_count() != dst->operand_count()) {
      return std::nullopt;
    }
    if (!LessThanOrEqualBitwidth(src, dst)) {
      return std::nullopt;
    }
    return std::make_unique<BitwidthExtendingEquivalenceMapping>(src, dst);
  }

  using EquivalenceMapping::EquivalenceMapping;

  bool RequiresOperandTransformation() const override {
    for (int i = 0; i < src_->operand_count(); ++i) {
      if (src_->operand(i)->BitCountOrDie() <
          dst_->operand(i)->BitCountOrDie()) {
        return true;
      }
    }
    return false;
  }

  bool RequiresOutputTransformation() const override {
    return src_->BitCountOrDie() < dst_->BitCountOrDie();
  }

  absl::StatusOr<std::vector<Node*>> ApplyToOperands(
      FunctionBase* f, absl::Span<Node* const> src_operands) const override {
    XLS_RET_CHECK_EQ(src_operands.size(), dst_->operand_count());
    std::vector<Node*> result;
    result.reserve(src_operands.size());
    for (int i = 0; i < src_operands.size(); ++i) {
      Node* op = src_operands[i];
      XLS_RET_CHECK(op->GetType()->IsBits());
      XLS_RET_CHECK(dst_->operand(i)->GetType()->IsBits());
      int64_t target_width = dst_->operand(i)->BitCountOrDie();
      if (op->BitCountOrDie() < target_width) {
        Op ext_op =
            IsSigned(const_cast<Node*>(dst_)) ? Op::kSignExt : Op::kZeroExt;
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
                                      Node* dst_output) const override {
    XLS_RET_CHECK(src_->GetType()->IsBits());
    XLS_RET_CHECK(dst_->GetType()->IsBits());
    if (src_->BitCountOrDie() < dst_->BitCountOrDie()) {
      return f->MakeNode<BitSlice>(dst_output->loc(), dst_output,
                                   /*start=*/0,
                                   /*width=*/src_->BitCountOrDie());
    }
    return dst_output;
  }
};

// `AddSubEquivalenceMapping` handles folding between `add` and `sub` nodes.
class AddSubEquivalenceMapping : public EquivalenceMapping {
 public:
  static std::optional<std::unique_ptr<EquivalenceMapping>> TryCreate(
      const Node* src, const Node* dst) {
    if (!src->OpIn({Op::kAdd, Op::kSub}) || !dst->OpIn({Op::kAdd, Op::kSub})) {
      return std::nullopt;
    }
    if (src->operand_count() != 2 || dst->operand_count() != 2) {
      return std::nullopt;
    }
    if (!LessThanOrEqualBitwidth(src, dst)) {
      return std::nullopt;
    }
    return std::make_unique<AddSubEquivalenceMapping>(src, dst);
  }

  using EquivalenceMapping::EquivalenceMapping;

  bool RequiresOperandTransformation() const override { return true; }

  bool RequiresOutputTransformation() const override {
    return src_->BitCountOrDie() < dst_->BitCountOrDie();
  }

  absl::StatusOr<std::vector<Node*>> ApplyToOperands(
      FunctionBase* f, absl::Span<Node* const> src_operands) const override {
    XLS_RET_CHECK_EQ(src_operands.size(), 2);
    Node* op0 = src_operands[0];
    Node* op1 = src_operands[1];
    XLS_RET_CHECK(op0->GetType()->IsBits());
    XLS_RET_CHECK(op1->GetType()->IsBits());
    XLS_RET_CHECK(dst_->operand(0)->GetType()->IsBits());
    XLS_RET_CHECK(dst_->operand(1)->GetType()->IsBits());

    if (src_->op() != dst_->op()) {
      if (op1->op() == Op::kNeg) {
        op1 = op1->operand(0);
      } else {
        XLS_ASSIGN_OR_RETURN(op1, f->MakeNode<UnOp>(op1->loc(), op1, Op::kNeg));
      }
    }

    if (op0->BitCountOrDie() < dst_->operand(0)->BitCountOrDie()) {
      XLS_ASSIGN_OR_RETURN(
          op0, f->MakeNode<ExtendOp>(op0->loc(), op0,
                                     dst_->operand(0)->BitCountOrDie(),
                                     Op::kZeroExt));
    }
    if (op1->BitCountOrDie() < dst_->operand(1)->BitCountOrDie()) {
      XLS_ASSIGN_OR_RETURN(
          op1, f->MakeNode<ExtendOp>(op1->loc(), op1,
                                     dst_->operand(1)->BitCountOrDie(),
                                     Op::kZeroExt));
    }

    return std::vector<Node*>{op0, op1};
  }

  absl::StatusOr<Node*> ApplyToOutput(FunctionBase* f,
                                      Node* dst_output) const override {
    XLS_RET_CHECK(src_->GetType()->IsBits());
    XLS_RET_CHECK(dst_->GetType()->IsBits());
    if (src_->BitCountOrDie() < dst_->BitCountOrDie()) {
      return f->MakeNode<BitSlice>(dst_output->loc(), dst_output,
                                   /*start=*/0,
                                   /*width=*/src_->BitCountOrDie());
    }
    return dst_output;
  }

  absl::StatusOr<double> EstimateAreaOverhead(
      const AreaEstimator& area_estimator, absl::Span<Node* const> operands,
      const Node* output) const override {
    if (src_->op() != dst_->op()) {
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
      const Node* src, const Node* dst) {
    if (!src->OpIn({Op::kShll, Op::kShrl, Op::kShra}) ||
        !dst->OpIn({Op::kShll, Op::kShrl, Op::kShra}) ||
        src->op() == dst->op()) {
      return std::nullopt;
    }
    if (!BothBitsTyped(src, dst) ||
        !BothBitsTyped(src->operand(0), dst->operand(0)) ||
        !BothBitsTyped(src->operand(1), dst->operand(1))) {
      return std::nullopt;
    }
    // If dst is not shra, src must be narrower than or equal to
    // dst.
    if (dst->op() != Op::kShra && !LessThanOrEqualBitwidth(src, dst)) {
      return std::nullopt;
    }
    bool same_shift_amount =
        src->operand(1)->IsDefinitelyEqualTo(dst->operand(1));
    if (!same_shift_amount) {
      return std::nullopt;
    }
    return std::make_unique<ShiftEquivalenceMapping>(src, dst);
  }

  using EquivalenceMapping::EquivalenceMapping;

  int64_t GetTargetWidth() const {
    if (dst_->op() == Op::kShra &&
        src_->BitCountOrDie() >= dst_->BitCountOrDie()) {
      // Requires padding `src` so when we fold into an arithmetic shift
      // we can shift in 0s.
      return src_->BitCountOrDie() + 1;
    }
    return dst_->BitCountOrDie();
  }

  bool ModifiesDestinationNode() const override {
    return dst_->op() == Op::kShra && dst_->BitCountOrDie() < GetTargetWidth();
  }

  bool RequiresOperandTransformation() const override { return true; }
  bool RequiresOutputTransformation() const override { return true; }

  absl::StatusOr<std::vector<Node*>> ApplyToOperands(
      FunctionBase* f, absl::Span<Node* const> src_operands) const override {
    XLS_RET_CHECK_EQ(src_operands.size(), 2);
    Node* op0 = src_operands[0];
    Node* op1 = src_operands[1];
    XLS_RET_CHECK(op0->GetType()->IsBits());
    XLS_RET_CHECK(op1->GetType()->IsBits());

    // If coming from shra, xor with the sign-extended msb.
    if (src_->op() == Op::kShra) {
      XLS_ASSIGN_OR_RETURN(
          op0, XorWithSignExtendedMsb(f, /*target=*/op0, /*msb_source=*/op0));
    }

    // If going between left and right shifts, reverse bits.
    if (dst_->op() == Op::kShll || src_->op() == Op::kShll) {
      XLS_ASSIGN_OR_RETURN(op0,
                           f->MakeNode<UnOp>(op0->loc(), op0, Op::kReverse));
    }

    // Extend shifted value to target width if necessary.
    int64_t target_width = GetTargetWidth();
    if (op0->BitCountOrDie() < target_width) {
      XLS_ASSIGN_OR_RETURN(
          op0,
          f->MakeNode<ExtendOp>(op0->loc(), op0,
                                /*new_bit_count=*/target_width, Op::kZeroExt));
    }

    // Extend shift amount to target width if necessary.
    int64_t target_shift_width = std::max(src_->operand(1)->BitCountOrDie(),
                                          dst_->operand(1)->BitCountOrDie());
    if (op1->BitCountOrDie() < target_shift_width) {
      XLS_ASSIGN_OR_RETURN(
          op1, f->MakeNode<ExtendOp>(op1->loc(), op1,
                                     /*new_bit_count=*/target_shift_width,
                                     Op::kZeroExt));
    }

    return std::vector<Node*>{op0, op1};
  }

  absl::StatusOr<Node*> ApplyToOutput(FunctionBase* f,
                                      Node* unified_output) const override {
    XLS_RET_CHECK(unified_output->GetType()->IsBits());
    Node* result = unified_output;
    int64_t orig_width = src_->BitCountOrDie();

    // If the src node is narrower, bit-slice to the src width.
    if (result->BitCountOrDie() > orig_width) {
      XLS_ASSIGN_OR_RETURN(
          result, f->MakeNode<BitSlice>(result->loc(), result, /*start=*/0,
                                        /*width=*/orig_width));
    }

    // If going between left and right shifts, reverse bits.
    if (dst_->op() == Op::kShll || src_->op() == Op::kShll) {
      XLS_ASSIGN_OR_RETURN(
          result, f->MakeNode<UnOp>(result->loc(), result, Op::kReverse));
    }

    // If coming from shra, xor once again with the sign-extended msb.
    if (src_->op() == Op::kShra) {
      XLS_ASSIGN_OR_RETURN(
          result, XorWithSignExtendedMsb(f, /*target=*/result,
                                         /*msb_source=*/src_->operand(0)));
    }

    return result;
  }

  std::unique_ptr<EquivalenceMapping> GetDestinationMapping(
      const Node* unified_node) const override {
    if (!ModifiesDestinationNode()) {
      return std::make_unique<IdentityEquivalenceMapping>(dst_, unified_node);
    }
    return std::make_unique<BitwidthExtendingEquivalenceMapping>(dst_,
                                                                 unified_node);
  }

  absl::StatusOr<Node*> CreateUnifiedNode(
      FunctionBase* f, absl::Span<Node* const> operands) const override {
    XLS_RET_CHECK_EQ(operands.size(), 2);
    return f->MakeNode<BinOp>(dst_->loc(), operands[0], operands[1],
                              dst_->op());
  }

  absl::StatusOr<double> EstimateAreaOverhead(
      const AreaEstimator& area_estimator, absl::Span<Node* const> src_operands,
      const Node* src_node) const override {
    if (src_->op() == Op::kShra) {
      int64_t orig_width = src_->BitCountOrDie();
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
    // reverse, slice, and extend ops are pure wire operations.
    return 0.0;
  }
};

}  // namespace

std::unique_ptr<EquivalenceMapping> EquivalenceMapping::GetDestinationMapping(
    const Node* unified_node) const {
  return std::make_unique<IdentityEquivalenceMapping>(dst_, unified_node);
}

void NodeEquivalenceMapper::Register(Factory factory) {
  absl::MutexLock lock(mutex_);
  factories_.push_back(factory);
}

std::optional<std::unique_ptr<EquivalenceMapping>>
NodeEquivalenceMapper::ComputeMapping(const Node* src, const Node* dst) const {
  absl::MutexLock lock(mutex_);
  for (Factory factory : factories_) {
    if (auto mapping = factory(src, dst); mapping.has_value()) {
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
