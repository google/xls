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
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
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
bool BothBitsTyped(Node* a, Node* b) {
  return a->GetType()->IsBits() && b->GetType()->IsBits();
}

// Returns true if `src` and `dst` are bit-typed and `src` is
// narrower or equal in width to `dst`.
bool LessThanOrEqualBitwidth(Node* src, Node* dst) {
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

using NodeToMappings =
    absl::flat_hash_map<Node*, std::unique_ptr<EquivalenceMapping>>;

// `BitwidthExtendingEquivalenceMapping` handles nodes with identical ops where
// the src node and its operands are narrower than or equal to the dst
// node's bit widths.
class BitwidthExtendingEquivalenceMapping : public EquivalenceMapping {
 public:
  static absl::StatusOr<std::optional<NodeToMappings>> TryCreate(
      absl::Span<Node* const> sources, Node* dst) {
    for (Node* src : sources) {
      if (src->op() != dst->op()) {
        return std::nullopt;
      }
      if (src->operand_count() != dst->operand_count()) {
        return std::nullopt;
      }
      if (!LessThanOrEqualBitwidth(src, dst)) {
        return std::nullopt;
      }
    }
    return ComputeMappingsSourcesToDest<BitwidthExtendingEquivalenceMapping>(
        sources, dst);
  }

  using EquivalenceMapping::EquivalenceMapping;

  absl::StatusOr<std::unique_ptr<EquivalenceMapping>> Clone(
      std::optional<const absl::flat_hash_map<Node*, Node*>*>
          original_node_to_clone) const override {
    XLS_ASSIGN_OR_RETURN(Node * new_src, MapNode(src_, original_node_to_clone));
    XLS_ASSIGN_OR_RETURN(Node * new_dst, MapNode(dst_, original_node_to_clone));
    return std::make_unique<BitwidthExtendingEquivalenceMapping>(
        new_src, new_dst, tmp_package_);
  }

  absl::StatusOr<bool> RequiresOperandTransformation() const override {
    for (int i = 0; i < src_->operand_count(); ++i) {
      if (src_->operand(i)->BitCountOrDie() <
          dst_->operand(i)->BitCountOrDie()) {
        return true;
      }
    }
    return false;
  }

  absl::StatusOr<bool> RequiresOutputTransformation() const override {
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

absl::StatusOr<Node*> ExtendIf(FunctionBase* f, Node* node,
                               int64_t target_width, Op ext_op) {
  if (node->BitCountOrDie() >= target_width) {
    return node;
  }
  return f->MakeNode<ExtendOp>(node->loc(), node, target_width, ext_op);
}

// `AddSubEquivalenceMapping` handles folding between `add` and `sub` nodes.
class AddSubEquivalenceMapping : public EquivalenceMapping {
 public:
  static absl::StatusOr<std::optional<NodeToMappings>> TryCreate(
      absl::Span<Node* const> sources, Node* dst) {
    if (!dst->OpIn({Op::kAdd, Op::kSub}) || dst->operand_count() != 2) {
      return std::nullopt;
    }
    for (Node* src : sources) {
      if (!src->OpIn({Op::kAdd, Op::kSub})) {
        return std::nullopt;
      }
      if (src->operand_count() != 2) {
        return std::nullopt;
      }
      if (!LessThanOrEqualBitwidth(src, dst)) {
        return std::nullopt;
      }
    }
    return ComputeMappingsSourcesToDest<AddSubEquivalenceMapping>(sources, dst);
  }

  using EquivalenceMapping::EquivalenceMapping;

  absl::StatusOr<std::unique_ptr<EquivalenceMapping>> Clone(
      std::optional<const absl::flat_hash_map<Node*, Node*>*>
          original_node_to_clone) const override {
    return CloneEqMapping(this, original_node_to_clone);
  }

  absl::StatusOr<bool> RequiresOperandTransformation() const override {
    return true;
  }

  absl::StatusOr<bool> RequiresOutputTransformation() const override {
    return src_->BitCountOrDie() < dst_->BitCountOrDie();
  }

  absl::StatusOr<std::vector<Node*>> ApplyToOperands(
      FunctionBase* f, absl::Span<Node* const> src_operands) const override {
    XLS_RET_CHECK_EQ(src_operands.size(), 2);
    Node* op0 = src_operands[0];
    Node* op1 = src_operands[1];
    XLS_RET_CHECK(op0->GetType()->IsBits());
    XLS_RET_CHECK(op1->GetType()->IsBits());
    Node* dst_lhs = dst_->operand(0);
    Node* dst_rhs = dst_->operand(1);
    XLS_RET_CHECK(dst_lhs->GetType()->IsBits());
    XLS_RET_CHECK(dst_rhs->GetType()->IsBits());

    if (src_->op() != dst_->op()) {
      if (op1->op() == Op::kNeg) {
        op1 = op1->operand(0);
      } else {
        XLS_ASSIGN_OR_RETURN(op1, f->MakeNode<UnOp>(op1->loc(), op1, Op::kNeg));
      }
    }

    XLS_ASSIGN_OR_RETURN(
        op0, ExtendIf(f, op0, dst_lhs->BitCountOrDie(), Op::kZeroExt));
    XLS_ASSIGN_OR_RETURN(
        op1, ExtendIf(f, op1, dst_rhs->BitCountOrDie(), Op::kZeroExt));

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
      Node* output) const override {
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

// Returns true if `src` and `dst` belong to the same comparator grouping:
// equality, unsigned inequality, or signed inequality.
bool AreCompatibleComparators(Node* src, Node* dst) {
  if (!OpIsCompare(src->op()) || !OpIsCompare(dst->op())) {
    return false;
  }
  return (IsSignedCompare(src) == IsSignedCompare(dst)) &&
         (IsUnsignedCompare(src) == IsUnsignedCompare(dst));
}

// Returns the area of a single bit NOT gate, cached by AreaEstimator
absl::StatusOr<double> EstimateAreaForSingleBitInversion(
    const AreaEstimator& area_estimator) {
  static absl::Mutex mutex(absl::kConstInit);
  static absl::NoDestructor<absl::flat_hash_map<std::string, double>> cache
      ABSL_GUARDED_BY(mutex);
  std::string estimator_name = area_estimator.name();
  {
    absl::MutexLock lock(mutex);
    if (auto it = cache->find(estimator_name); it != cache->end()) {
      return it->second;
    }
  }
  XLS_ASSIGN_OR_RETURN(
      double area,
      EstimateAreaForNodes(
          area_estimator, [](Package* p, FunctionBuilder* fb) -> absl::Status {
            fb->Not(fb->Param("bit_to_invert", p->GetBitsType(1)));
            return absl::OkStatus();
          }));
  {
    absl::MutexLock lock(mutex);
    cache->emplace(estimator_name, area);
  }
  return area;
}

// `ComparatorEquivalenceMapping` handles folding within comparator operation
// groups: equality, unsigned inequalities, and signed inequalities.
class ComparatorEquivalenceMapping : public EquivalenceMapping {
 public:
  static absl::StatusOr<std::optional<NodeToMappings>> TryCreate(
      absl::Span<Node* const> sources, Node* dst) {
    if (!OpIsCompare(dst->op()) || dst->operand_count() != 2) {
      return std::nullopt;
    }
    for (Node* src : sources) {
      if (!OpIsCompare(src->op()) || src->operand_count() != 2) {
        return std::nullopt;
      }
      if (!AreCompatibleComparators(src, dst)) {
        return std::nullopt;
      }
      if (!LessThanOrEqualBitwidth(src, dst)) {
        return std::nullopt;
      }
    }
    return ComputeMappingsSourcesToDest<ComparatorEquivalenceMapping>(sources,
                                                                      dst);
  }

  using EquivalenceMapping::EquivalenceMapping;

  absl::StatusOr<std::unique_ptr<EquivalenceMapping>> Clone(
      std::optional<const absl::flat_hash_map<Node*, Node*>*>
          original_node_to_clone) const override {
    return CloneEqMapping(this, original_node_to_clone);
  }

  absl::StatusOr<bool> RequiresOperandTransformation() const override {
    XLS_ASSIGN_OR_RETURN(bool requires_swap,
                         RequiresOperandSwap(src_->op(), dst_->op()));
    if (requires_swap) {
      return true;
    }
    for (int i = 0; i < src_->operand_count(); ++i) {
      if (src_->operand(i)->BitCountOrDie() <
          dst_->operand(i)->BitCountOrDie()) {
        return true;
      }
    }
    return false;
  }

  absl::StatusOr<bool> RequiresOutputTransformation() const override {
    return RequiresOutputInversion(src_->op(), dst_->op());
  }

  absl::StatusOr<std::vector<Node*>> ApplyToOperands(
      FunctionBase* f, absl::Span<Node* const> src_operands) const override {
    XLS_RET_CHECK_EQ(src_operands.size(), 2);
    Node* op0 = src_operands[0];
    Node* op1 = src_operands[1];
    XLS_RET_CHECK(op0->GetType()->IsBits());
    XLS_RET_CHECK(op1->GetType()->IsBits());
    Node* dst_lhs = dst_->operand(0);
    Node* dst_rhs = dst_->operand(1);
    XLS_RET_CHECK(dst_lhs->GetType()->IsBits());
    XLS_RET_CHECK(dst_rhs->GetType()->IsBits());

    XLS_ASSIGN_OR_RETURN(bool requires_swap,
                         RequiresOperandSwap(src_->op(), dst_->op()));
    if (requires_swap) {
      std::swap(op0, op1);
    }

    Op ext_op = IsSigned(dst_) ? Op::kSignExt : Op::kZeroExt;
    XLS_ASSIGN_OR_RETURN(op0,
                         ExtendIf(f, op0, dst_lhs->BitCountOrDie(), ext_op));
    XLS_ASSIGN_OR_RETURN(op1,
                         ExtendIf(f, op1, dst_rhs->BitCountOrDie(), ext_op));

    return std::vector<Node*>{op0, op1};
  }

  absl::StatusOr<Node*> ApplyToOutput(FunctionBase* f,
                                      Node* dst_output) const override {
    XLS_RET_CHECK(src_->GetType()->IsBits());
    XLS_RET_CHECK(dst_->GetType()->IsBits());
    XLS_ASSIGN_OR_RETURN(bool requires_inversion,
                         RequiresOutputInversion(src_->op(), dst_->op()));
    if (requires_inversion) {
      return f->MakeNode<UnOp>(dst_output->loc(), dst_output, Op::kNot);
    }
    return dst_output;
  }

  absl::StatusOr<double> EstimateAreaOverhead(
      const AreaEstimator& area_estimator, absl::Span<Node* const> operands,
      Node* output) const override {
    XLS_ASSIGN_OR_RETURN(bool requires_inversion,
                         RequiresOutputTransformation());
    if (requires_inversion) {
      return EstimateAreaForSingleBitInversion(area_estimator);
    }
    return 0.0;
  }

  static absl::StatusOr<bool> RequiresOutputInversion(Op src_op, Op dst_op) {
    XLS_ASSIGN_OR_RETURN(Op invert_dst_op, InvertComparisonOp(dst_op));
    XLS_ASSIGN_OR_RETURN(Op rev_dst_op, ReverseComparisonOp(dst_op));
    XLS_ASSIGN_OR_RETURN(Op invert_rev_dst_op, InvertComparisonOp(rev_dst_op));
    return src_op == invert_dst_op || src_op == invert_rev_dst_op;
  }

  static absl::StatusOr<bool> RequiresOperandSwap(Op src_op, Op dst_op) {
    XLS_ASSIGN_OR_RETURN(Op rev_dst_op, ReverseComparisonOp(dst_op));
    XLS_ASSIGN_OR_RETURN(Op invert_rev_dst_op, InvertComparisonOp(rev_dst_op));
    return src_op == rev_dst_op || src_op == invert_rev_dst_op;
  }
};

struct PackageAndNode {
  std::shared_ptr<Package> package;
  Node* node;
};

template <typename BuilderFn>
absl::StatusOr<PackageAndNode> CreateModifiedDstAndPackage(BuilderFn builder) {
  auto tmp_package = std::make_shared<Package>("tmp_package");
  FunctionBuilder fb("tmp_fn", tmp_package.get());
  BValue modified_dst = builder(tmp_package.get(), fb);
  XLS_RETURN_IF_ERROR(fb.Build().status());
  return PackageAndNode{std::move(tmp_package), modified_dst.node()};
}

absl::StatusOr<PackageAndNode> CreatePaddedSubPackage(int64_t target_width) {
  return CreateModifiedDstAndPackage(
      [target_width](Package* p, FunctionBuilder& fb) -> BValue {
        Type* bits_type = p->GetBitsType(target_width);
        return fb.Subtract(fb.Param("a", bits_type), fb.Param("b", bits_type));
      });
}

absl::StatusOr<PackageAndNode> CreatePaddedShraPackage(Node* original) {
  return CreateModifiedDstAndPackage([original](Package* p,
                                                FunctionBuilder& fb) -> BValue {
    Type* padded_shift_ty =
        p->GetBitsType(original->operand(0)->BitCountOrDie() + 1);
    Type* shift_amt_ty = p->GetBitsType(original->operand(1)->BitCountOrDie());
    return fb.Shra(fb.Param("a", padded_shift_ty), fb.Param("b", shift_amt_ty));
  });
}

// `CompareToArithEquivalenceMapping` handles folding comparison operations
// into subtraction nodes then taking the msb.
class CompareToArithEquivalenceMapping : public EquivalenceMapping {
 public:
  static absl::StatusOr<std::optional<NodeToMappings>> TryCreate(
      absl::Span<Node* const> sources, Node* dst) {
    if (dst->op() != Op::kSub || dst->operand_count() != 2) {
      return std::nullopt;
    }
    Node* dst_lhs = dst->operand(0);
    Node* dst_rhs = dst->operand(1);
    if (!dst->GetType()->IsBits() || !BothBitsTyped(dst_lhs, dst_rhs)) {
      return std::nullopt;
    }
    int64_t target_width = dst->BitCountOrDie();
    bool needs_dst_widening = false;
    for (Node* src : sources) {
      if (!OpIsCompare(src->op()) || src->operand_count() != 2 ||
          src->OpIn({Op::kEq, Op::kNe})) {
        return std::nullopt;
      }
      Node* src_lhs = src->operand(0);
      Node* src_rhs = src->operand(1);
      if (!BothBitsTyped(src_lhs, src_rhs)) {
        return std::nullopt;
      }
      int64_t required_width =
          std::max(src_lhs->BitCountOrDie(), src_rhs->BitCountOrDie()) + 1;
      needs_dst_widening |= required_width > dst->BitCountOrDie();
      target_width = std::max(target_width, required_width);
    }

    if (!needs_dst_widening) {
      return ComputeMappingsSourcesToDest<CompareToArithEquivalenceMapping>(
          sources, dst);
    }

    XLS_ASSIGN_OR_RETURN((PackageAndNode p_and_sub),
                         CreatePaddedSubPackage(target_width));
    NodeToMappings mappings =
        ComputeMappingsSourcesToDest<CompareToArithEquivalenceMapping>(
            sources, p_and_sub.node, p_and_sub.package);
    mappings[dst] = std::make_unique<BitwidthExtendingEquivalenceMapping>(
        dst, p_and_sub.node, p_and_sub.package);
    return mappings;
  }

  using EquivalenceMapping::EquivalenceMapping;

  absl::StatusOr<std::unique_ptr<EquivalenceMapping>> Clone(
      std::optional<const absl::flat_hash_map<Node*, Node*>*>
          original_node_to_clone) const override {
    return CloneEqMapping(this, original_node_to_clone);
  }

  absl::StatusOr<bool> RequiresOperandTransformation() const override {
    return true;
  }
  absl::StatusOr<bool> RequiresOutputTransformation() const override {
    return true;
  }

  absl::StatusOr<std::vector<Node*>> ApplyToOperands(
      FunctionBase* f, absl::Span<Node* const> src_operands) const override {
    XLS_RET_CHECK_EQ(src_operands.size(), 2);
    Node* op0 = src_operands[0];
    Node* op1 = src_operands[1];
    XLS_RET_CHECK(op0->GetType()->IsBits());
    XLS_RET_CHECK(op1->GetType()->IsBits());
    Node* dst_lhs = dst_->operand(0);
    Node* dst_rhs = dst_->operand(1);
    XLS_RET_CHECK(dst_lhs->GetType()->IsBits());
    XLS_RET_CHECK(dst_rhs->GetType()->IsBits());

    XLS_ASSIGN_OR_RETURN(bool requires_swap, RequiresOperandSwap(src_->op()));
    if (requires_swap) {
      std::swap(op0, op1);
    }

    Op ext_op = IsSignedCompare(src_) ? Op::kSignExt : Op::kZeroExt;
    XLS_ASSIGN_OR_RETURN(op0,
                         ExtendIf(f, op0, dst_lhs->BitCountOrDie(), ext_op));
    XLS_ASSIGN_OR_RETURN(op1,
                         ExtendIf(f, op1, dst_rhs->BitCountOrDie(), ext_op));

    return std::vector<Node*>{op0, op1};
  }

  absl::StatusOr<Node*> ApplyToOutput(FunctionBase* f,
                                      Node* dst_output) const override {
    XLS_RET_CHECK(src_->GetType()->IsBits());
    XLS_RET_CHECK(dst_->GetType()->IsBits());
    XLS_ASSIGN_OR_RETURN(
        Node * msb,
        f->MakeNode<BitSlice>(dst_output->loc(), dst_output,
                              /*start=*/dst_output->BitCountOrDie() - 1,
                              /*width=*/1));
    XLS_ASSIGN_OR_RETURN(bool requires_inversion,
                         RequiresOutputInversion(src_->op()));
    if (requires_inversion) {
      return f->MakeNode<UnOp>(dst_output->loc(), msb, Op::kNot);
    }
    return msb;
  }

  absl::StatusOr<double> EstimateAreaOverhead(
      const AreaEstimator& area_estimator, absl::Span<Node* const> operands,
      Node* output) const override {
    XLS_ASSIGN_OR_RETURN(bool requires_inversion,
                         RequiresOutputInversion(src_->op()));
    if (requires_inversion) {
      return EstimateAreaForSingleBitInversion(area_estimator);
    }
    return 0.0;
  }

  static bool IsSignedCompareOp(Op src_op) {
    return src_op == Op::kSLt || src_op == Op::kSGt || src_op == Op::kSLe ||
           src_op == Op::kSGe;
  }

  static absl::StatusOr<bool> RequiresOutputInversion(Op src_op) {
    Op base_lt = IsSignedCompareOp(src_op) ? Op::kSLt : Op::kULt;
    return ComparatorEquivalenceMapping::RequiresOutputInversion(src_op,
                                                                 base_lt);
  }

  static absl::StatusOr<bool> RequiresOperandSwap(Op src_op) {
    Op base_lt = IsSignedCompareOp(src_op) ? Op::kSLt : Op::kULt;
    return ComparatorEquivalenceMapping::RequiresOperandSwap(src_op, base_lt);
  }
};

// Returns `target` XORed with a sign-extended mask created from the MSB of
// `msb_source`. Reuses existing BitSlice and SignExt nodes if available.
absl::StatusOr<Node*> XorWithSignExtendedMsb(FunctionBase* f, Node* target,
                                             Node* msb_source) {
  int64_t target_width = target->BitCountOrDie();
  int64_t source_width = msb_source->BitCountOrDie();
  XLS_ASSIGN_OR_RETURN(
      Node * msb, FindOrMakeBitSlice(msb_source, /*start=*/source_width - 1,
                                     /*width=*/1));

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
//    where m = sign_ext(msb(x)). If m = 0, then the XORs are no-ops. If m =
//    1s, then xors are no-ops on shifted bits, flipping those bits twice,
//    meanwhile converting all 0s shifted into the MSBs into 1s.
// y = x >> s is equivalent to y = slice(zero_ext(x, N) >>> s, 0, W) for W < N
//    where zero-extending (padding) ensures shra shifts in 0s.
class ShiftEquivalenceMapping : public EquivalenceMapping {
 public:
  static absl::StatusOr<std::optional<NodeToMappings>> TryCreate(
      absl::Span<Node* const> sources, Node* dst) {
    auto bit_typed_two_op_shift = [](Node* node) {
      return node->OpIn({Op::kShll, Op::kShrl, Op::kShra}) &&
             node->operand_count() == 2 && node->GetType()->IsBits() &&
             node->operand(0)->GetType()->IsBits() &&
             node->operand(1)->GetType()->IsBits();
    };
    if (!bit_typed_two_op_shift(dst)) {
      return std::nullopt;
    }
    bool needs_dst_widening = false;
    for (Node* src : sources) {
      // We gain nothing by handling edge case where src and dst have same op.
      if (!bit_typed_two_op_shift(src) || src->op() == dst->op()) {
        return std::nullopt;
      }
      // Cannot shift by a larger bit width amount.
      if (src->operand(1)->BitCountOrDie() > dst->operand(1)->BitCountOrDie()) {
        return std::nullopt;
      }
      if (dst->op() == Op::kShra) {
        // Cannot shift a value that is a larger bit width.
        if (src->BitCountOrDie() > dst->BitCountOrDie() ||
            src->operand(0)->BitCountOrDie() >
                dst->operand(0)->BitCountOrDie()) {
          return std::nullopt;
        }
        // The destination must be 0-padded if the source is not shra and the
        // bit-widths are equal, since an arithmetic shift would fill in 1s.
        if (src->op() != Op::kShra && (src->operand(0)->BitCountOrDie() ==
                                       dst->operand(0)->BitCountOrDie())) {
          needs_dst_widening = true;
        }
      } else {
        if (!LessThanOrEqualBitwidth(src, dst)) {
          return std::nullopt;
        }
      }
    }

    if (!needs_dst_widening) {
      return ComputeMappingsSourcesToDest<ShiftEquivalenceMapping>(sources,
                                                                   dst);
    }

    XLS_ASSIGN_OR_RETURN((PackageAndNode p_and_shra),
                         CreatePaddedShraPackage(dst));
    NodeToMappings mappings =
        ComputeMappingsSourcesToDest<ShiftEquivalenceMapping>(
            sources, p_and_shra.node, p_and_shra.package);
    mappings[dst] = std::make_unique<BitwidthExtendingEquivalenceMapping>(
        dst, p_and_shra.node, p_and_shra.package);
    return mappings;
  }

  using EquivalenceMapping::EquivalenceMapping;

  absl::StatusOr<std::unique_ptr<EquivalenceMapping>> Clone(
      std::optional<const absl::flat_hash_map<Node*, Node*>*>
          original_node_to_clone) const override {
    return CloneEqMapping(this, original_node_to_clone);
  }

  absl::StatusOr<bool> RequiresOperandTransformation() const override {
    return true;
  }
  absl::StatusOr<bool> RequiresOutputTransformation() const override {
    return true;
  }

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
    int64_t target_width = dst_->BitCountOrDie();
    if (op0->BitCountOrDie() < target_width) {
      XLS_ASSIGN_OR_RETURN(
          op0,
          f->MakeNode<ExtendOp>(op0->loc(), op0,
                                /*new_bit_count=*/target_width, Op::kZeroExt));
    }

    // Extend shift amount to target width if necessary.
    int64_t target_shift_width = dst_->operand(1)->BitCountOrDie();
    if (op1->BitCountOrDie() < target_shift_width) {
      XLS_ASSIGN_OR_RETURN(
          op1, f->MakeNode<ExtendOp>(op1->loc(), op1,
                                     /*new_bit_count=*/target_shift_width,
                                     Op::kZeroExt));
    }

    return std::vector<Node*>{op0, op1};
  }

  absl::StatusOr<Node*> ApplyToOutput(FunctionBase* f,
                                      Node* dst_output) const override {
    Node* result = dst_output;
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

  absl::StatusOr<double> EstimateAreaOverhead(
      const AreaEstimator& area_estimator, absl::Span<Node* const> src_operands,
      Node* src_node) const override {
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

absl::StatusOr<Node*> EquivalenceMapping::MapNode(
    Node* node, std::optional<const absl::flat_hash_map<Node*, Node*>*>
                    original_node_to_clone) const {
  // If no mapping is given, or the node is from a temporary package, return
  // the node itself.
  if (!original_node_to_clone.has_value() ||
      (tmp_package_ != nullptr && node->package() == tmp_package_.get())) {
    return node;
  }
  auto it = (*original_node_to_clone)->find(node);
  XLS_RET_CHECK(it != (*original_node_to_clone)->end())
      << "Node " << node->ToString()
      << " not found in original_node_to_clone map";
  return it->second;
}

void NodeEquivalenceMapper::Register(Factory factory) {
  absl::MutexLock lock(mutex_);
  factories_.push_back(factory);
}

absl::StatusOr<std::optional<NodeToMappings>>
NodeEquivalenceMapper::ComputeMappings(absl::Span<Node* const> sources,
                                       Node* dst) const {
  absl::MutexLock lock(mutex_);
  for (Factory factory : factories_) {
    XLS_ASSIGN_OR_RETURN((std::optional<NodeToMappings> mappings),
                         factory(sources, dst));
    if (mappings.has_value()) {
      return mappings;
    }
  }
  return std::nullopt;
}

NodeEquivalenceMapper& GetNodeEquivalenceMapper() {
  static absl::NoDestructor<NodeEquivalenceMapper> m;
  [[maybe_unused]] static const bool initialized = [&] {
    m->Register<IdentityEquivalenceMapping>();
    m->Register<BitwidthExtendingEquivalenceMapping>();
    m->Register<AddSubEquivalenceMapping>();
    m->Register<ComparatorEquivalenceMapping>();
    m->Register<CompareToArithEquivalenceMapping>();
    m->Register<ShiftEquivalenceMapping>();
    return true;
  }();
  return *m;
}

}  // namespace xls
