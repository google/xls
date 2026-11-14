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
#include <tuple>
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
#include "absl/types/optional_ref.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/area_model/area_estimator.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/topo_sort.h"
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
  double area = 0.0;
  for (Node* node : f->nodes()) {
    if (node->Is<Param>()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(double node_area,
                         area_estimator.GetOperationAreaInSquareMicrons(node));
    area += node_area;
  }
  return area;
}

using NodeToMappings =
    absl::flat_hash_map<Node*, std::unique_ptr<EquivalenceMapping>>;

absl::StatusOr<Node*> ExtendIf(FunctionBase* f, Node* node,
                               int64_t target_width, Op ext_op) {
  XLS_RET_CHECK_LE(node->BitCountOrDie(), target_width);
  if (node->BitCountOrDie() == target_width) {
    return node;
  }
  for (Node* user : node->users()) {
    if (user->function_base() == f && user->op() == ext_op &&
        user->BitCountOrDie() == target_width) {
      return user;
    }
  }
  return f->MakeNode<ExtendOp>(node->loc(), node, target_width, ext_op);
}

absl::StatusOr<Node*> FindOrMakeUnaryOp(FunctionBase* f, Node* operand, Op op) {
  for (Node* user : operand->users()) {
    if (user->function_base() == f && user->op() == op) {
      return user;
    }
  }
  return f->MakeNode<UnOp>(operand->loc(), operand, op);
}

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
      absl::optional_ref<const absl::flat_hash_map<Node*, Node*>>
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
      bool is_shift_amt_node =
          dst_->OpIn({Op::kShll, Op::kShrl, Op::kShra}) && i == 1;
      // When extending a shift amount's bit width, always zero extend.
      Op ext_op = (IsSigned(const_cast<Node*>(dst_)) && !is_shift_amt_node)
                      ? Op::kSignExt
                      : Op::kZeroExt;
      XLS_ASSIGN_OR_RETURN(Node * ext, ExtendIf(f, op, target_width, ext_op));
      result.push_back(ext);
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
      absl::optional_ref<const absl::flat_hash_map<Node*, Node*>>
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
        XLS_ASSIGN_OR_RETURN(op1, FindOrMakeUnaryOp(f, op1, Op::kNeg));
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
// equality (`eq`, `ne`) or inequality (`ult`, `ule`, `ugt`, `uge`, `slt`,
// `sle`, `sgt`, `sge`).
bool AreCompatibleComparators(Node* src, Node* dst) {
  if (!OpIsCompare(src->op()) || !OpIsCompare(dst->op())) {
    return false;
  }
  return src->OpIn({Op::kEq, Op::kNe}) == dst->OpIn({Op::kEq, Op::kNe});
}

absl::StatusOr<Node*> FindOrMakeConcat(FunctionBase* f, const SourceInfo& loc,
                                       absl::Span<Node* const> args) {
  if (!args.empty()) {
    for (Node* user : args[0]->users()) {
      if (user->Is<Concat>() && user->operands() == args) {
        return user;
      }
    }
  }
  return f->MakeNode<Concat>(loc, args);
}

// Swaps the most significant bits of `op0` and `op1`.
absl::StatusOr<std::pair<Node*, Node*>> SwapMsbs(FunctionBase* f, Node* op0,
                                                 Node* op1) {
  int64_t width = op0->BitCountOrDie();
  XLS_RET_CHECK_EQ(width, op1->BitCountOrDie());
  if (width <= 1) {
    return std::make_pair(op1, op0);
  }
  XLS_ASSIGN_OR_RETURN(
      Node * msb0, FindOrMakeBitSlice(op0, /*start=*/width - 1, /*width=*/1));
  XLS_ASSIGN_OR_RETURN(
      Node * msb1, FindOrMakeBitSlice(op1, /*start=*/width - 1, /*width=*/1));
  XLS_ASSIGN_OR_RETURN(
      Node * lsb0, FindOrMakeBitSlice(op0, /*start=*/0, /*width=*/width - 1));
  XLS_ASSIGN_OR_RETURN(
      Node * lsb1, FindOrMakeBitSlice(op1, /*start=*/0, /*width=*/width - 1));
  XLS_ASSIGN_OR_RETURN(
      Node * new_op0,
      FindOrMakeConcat(f, op0->loc(), std::vector<Node*>{msb1, lsb0}));
  XLS_ASSIGN_OR_RETURN(
      Node * new_op1,
      FindOrMakeConcat(f, op1->loc(), std::vector<Node*>{msb0, lsb1}));
  return std::make_pair(new_op0, new_op1);
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
// groups: equalities and inequalities.
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
      absl::optional_ref<const absl::flat_hash_map<Node*, Node*>>
          original_node_to_clone) const override {
    return CloneEqMapping(this, original_node_to_clone);
  }

  bool RequiresMsbSwap() const {
    if (IsSignedCompare(src_) == IsSignedCompare(dst_)) {
      return false;
    }
    // If dst is wider and we map from unsigned to signed, skip MSB swap.
    if (IsUnsignedCompare(src_) && IsSignedCompare(dst_)) {
      return src_->operand(0)->BitCountOrDie() ==
             dst_->operand(0)->BitCountOrDie();
    }
    return true;
  }

  absl::StatusOr<bool> RequiresOperandTransformation() const override {
    if (RequiresMsbSwap()) {
      return true;
    }
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

    if (RequiresMsbSwap()) {
      XLS_ASSIGN_OR_RETURN(std::tie(op0, op1), SwapMsbs(f, op0, op1));
    }

    Op ext_op = (!IsUnsignedCompare(src_) && !IsUnsignedCompare(dst_))
                    ? Op::kSignExt
                    : Op::kZeroExt;
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
    XLS_ASSIGN_OR_RETURN(bool result_inversion, RequiresOutputTransformation());
    if (result_inversion) {
      return EstimateAreaForSingleBitInversion(area_estimator);
    }
    return 0.0;
  }

  static absl::StatusOr<bool> RequiresOutputInversion(Op src_op, Op dst_op) {
    src_op = SignedCompareToUnsigned(src_op).value_or(src_op);
    dst_op = SignedCompareToUnsigned(dst_op).value_or(dst_op);
    XLS_ASSIGN_OR_RETURN(Op invert_dst_op, InvertComparisonOp(dst_op));
    XLS_ASSIGN_OR_RETURN(Op rev_dst_op, ReverseComparisonOp(dst_op));
    XLS_ASSIGN_OR_RETURN(Op invert_rev_dst_op, InvertComparisonOp(rev_dst_op));
    // Inverting the op, e.g. from ult to uge, requires not-ing the result
    return src_op == invert_dst_op || src_op == invert_rev_dst_op;
  }

  static absl::StatusOr<bool> RequiresOperandSwap(Op src_op, Op dst_op) {
    src_op = SignedCompareToUnsigned(src_op).value_or(src_op);
    dst_op = SignedCompareToUnsigned(dst_op).value_or(dst_op);
    XLS_ASSIGN_OR_RETURN(Op rev_dst_op, ReverseComparisonOp(dst_op));
    // If equality comparison, no operand swap is needed:
    if (rev_dst_op == dst_op) {
      return false;
    }
    XLS_ASSIGN_OR_RETURN(Op invert_rev_dst_op, InvertComparisonOp(rev_dst_op));
    // Reversing the op, e.g. from ult to ugt, requires swapping the operands.
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

absl::StatusOr<PackageAndNode> CreateModifiedShiftPackage(
    Node* original, int64_t target_val_width, int64_t target_shift_width) {
  return CreateModifiedDstAndPackage(
      [original, target_val_width, target_shift_width](
          Package* p, FunctionBuilder& fb) -> BValue {
        Type* val_ty = p->GetBitsType(target_val_width);
        Type* shift_amt_ty = p->GetBitsType(target_shift_width);
        return fb.AddBinOp(original->op(), fb.Param("a", val_ty),
                           fb.Param("b", shift_amt_ty));
      });
}

absl::StatusOr<PackageAndNode> CreateModifiedArithOpPackage(Op op,
                                                            int64_t lhs_width,
                                                            int64_t rhs_width,
                                                            int64_t out_width) {
  return CreateModifiedDstAndPackage(
      [op, lhs_width, rhs_width, out_width](Package* p,
                                            FunctionBuilder& fb) -> BValue {
        Type* lhs_ty = p->GetBitsType(lhs_width);
        Type* rhs_ty = p->GetBitsType(rhs_width);
        return fb.AddArithOp(op, fb.Param("a", lhs_ty), fb.Param("b", rhs_ty),
                             out_width);
      });
}

// `MulEquivalenceMapping` handles folding between multiplication operations
// (`umul` and `smul`), supporting:
// 1. Cross-sign multiplier sharing (`umul` <-> `smul`) by promoting the target
//    operation to `smul` and zero-extending any `umul` operand of width W to at
//    least W + 1 bits whenever the output width exceeds W (for output width
//    R <= W, signed and unsigned interpretations already agree modulo 2^R).
// 2. Commutative operand swapping (a * b == b * a) to align asymmetric operand
//    widths and minimize multiplier widening across N-ary folds.
class MulEquivalenceMapping : public EquivalenceMapping {
 public:
  static absl::StatusOr<std::optional<NodeToMappings>> TryCreate(
      absl::Span<Node* const> sources, Node* dst) {
    auto is_valid_mul = [](Node* n) {
      return n->OpIn({Op::kUMul, Op::kSMul}) && n->operand_count() == 2 &&
             n->GetType()->IsBits() &&
             BothBitsTyped(n->operand(0), n->operand(1)) &&
             n->BitCountOrDie() > 0 && n->operand(0)->BitCountOrDie() > 0 &&
             n->operand(1)->BitCountOrDie() > 0;
    };
    if (!is_valid_mul(dst)) {
      return std::nullopt;
    }

    const int64_t d_w0 = dst->operand(0)->BitCountOrDie();
    const int64_t d_w1 = dst->operand(1)->BitCountOrDie();
    bool has_signed = dst->op() == Op::kSMul;
    for (Node* src : sources) {
      if (!is_valid_mul(src)) {
        return std::nullopt;
      }
      const int64_t s_w0 = src->operand(0)->BitCountOrDie();
      const int64_t s_w1 = src->operand(1)->BitCountOrDie();
      // Like `LessThanOrEqualBitwidth`, only fold `src` into an equal-or-wider
      // `dst`: `src`'s output bitwidth and its sorted operand bitwidths (since
      // multiplication is commutative and operands may be swapped) must not
      // exceed `dst`'s.
      if (std::min(s_w0, s_w1) > std::min(d_w0, d_w1) ||
          std::max(s_w0, s_w1) > std::max(d_w0, d_w1) ||
          src->BitCountOrDie() > dst->BitCountOrDie()) {
        return std::nullopt;
      }
      if (src->op() == Op::kSMul) {
        has_signed = true;
      }
    }

    Op target_op = has_signed ? Op::kSMul : Op::kUMul;
    auto req_width = [target_op](Node* node, int idx) -> int64_t {
      int64_t w = node->operand(idx)->BitCountOrDie();
      if (target_op == Op::kSMul && node->op() == Op::kUMul &&
          node->BitCountOrDie() > w) {
        return w + 1;
      }
      return w;
    };

    const int64_t dst_req0 = req_width(dst, 0);
    const int64_t dst_req1 = req_width(dst, 1);
    const bool dst_0_le_1 = dst_req0 <= dst_req1;
    int64_t target_w0 = dst_req0;
    int64_t target_w1 = dst_req1;
    const int64_t target_out = dst->BitCountOrDie();
    std::vector<bool> swap_for_src;
    swap_for_src.reserve(sources.size());
    for (Node* src : sources) {
      int64_t r0 = req_width(src, 0);
      int64_t r1 = req_width(src, 1);
      bool swap_src = false;
      if (r0 != r1) {
        // Align each source's narrower and wider required operand bitwidths
        // with dst's to minimize the unified multiplier's operand bitwidths,
        // independent of source iteration order.
        swap_src = dst_0_le_1 ? (r0 > r1) : (r0 < r1);
      } else {
        int unswapped_matches = (src->operand(0) == dst->operand(0)) +
                                (src->operand(1) == dst->operand(1));
        int swapped_matches = (src->operand(0) == dst->operand(1)) +
                              (src->operand(1) == dst->operand(0));
        swap_src = swapped_matches > unswapped_matches;
      }
      swap_for_src.push_back(swap_src);
      target_w0 = std::max(target_w0, swap_src ? r1 : r0);
      target_w1 = std::max(target_w1, swap_src ? r0 : r1);
    }

    bool needs_modified_dst = target_op != dst->op() || target_w0 != d_w0 ||
                              target_w1 != d_w1 ||
                              target_out != dst->BitCountOrDie();
    Node* unified_dst = dst;
    std::shared_ptr<Package> tmp_pkg = nullptr;
    if (needs_modified_dst) {
      XLS_ASSIGN_OR_RETURN((PackageAndNode p_and_mul),
                           CreateModifiedArithOpPackage(target_op, target_w0,
                                                        target_w1, target_out));
      unified_dst = p_and_mul.node;
      tmp_pkg = std::move(p_and_mul.package);
    }

    NodeToMappings mappings;
    mappings.reserve(sources.size() + (needs_modified_dst ? 1 : 0));
    for (size_t i = 0; i < sources.size(); ++i) {
      mappings[sources[i]] = std::make_unique<MulEquivalenceMapping>(
          sources[i], unified_dst, tmp_pkg, dst, swap_for_src[i]);
    }
    if (needs_modified_dst) {
      mappings[dst] = std::make_unique<MulEquivalenceMapping>(
          dst, unified_dst, tmp_pkg, dst, /*swap_operands=*/false);
    }
    return mappings;
  }

  MulEquivalenceMapping(Node* src, Node* dst,
                        std::shared_ptr<Package> tmp_package = nullptr,
                        std::optional<Node*> orig_dst = std::nullopt,
                        bool swap_operands = false)
      : EquivalenceMapping(src, dst, std::move(tmp_package)),
        orig_dst_(orig_dst.value_or(dst)),
        swap_operands_(swap_operands) {}

  absl::StatusOr<std::unique_ptr<EquivalenceMapping>> Clone(
      absl::optional_ref<const absl::flat_hash_map<Node*, Node*>>
          original_node_to_clone) const override {
    XLS_ASSIGN_OR_RETURN(Node * new_src, MapNode(src_, original_node_to_clone));
    XLS_ASSIGN_OR_RETURN(Node * new_dst, MapNode(dst_, original_node_to_clone));
    XLS_ASSIGN_OR_RETURN(Node * new_orig_dst,
                         MapNode(orig_dst_, original_node_to_clone));
    return std::make_unique<MulEquivalenceMapping>(
        new_src, new_dst, tmp_package_, new_orig_dst, swap_operands_);
  }

  absl::StatusOr<bool> RequiresOperandTransformation() const override {
    if (swap_operands_) {
      return true;
    }
    int64_t src_w0 = src_->operand(0)->BitCountOrDie();
    int64_t src_w1 = src_->operand(1)->BitCountOrDie();
    return src_w0 < dst_->operand(0)->BitCountOrDie() ||
           src_w1 < dst_->operand(1)->BitCountOrDie();
  }

  absl::StatusOr<bool> RequiresOutputTransformation() const override {
    return src_->BitCountOrDie() < dst_->BitCountOrDie();
  }

  absl::StatusOr<std::vector<Node*>> ApplyToOperands(
      FunctionBase* f, absl::Span<Node* const> src_operands) const override {
    XLS_RET_CHECK_EQ(src_operands.size(), 2);
    Node* op0 = src_operands[0];
    Node* op1 = src_operands[1];
    XLS_RET_CHECK(BothBitsTyped(op0, op1));
    if (swap_operands_) {
      std::swap(op0, op1);
    }
    Node* dst_lhs = dst_->operand(0);
    Node* dst_rhs = dst_->operand(1);
    XLS_RET_CHECK(BothBitsTyped(dst_lhs, dst_rhs));

    Op ext_op = src_->op() == Op::kSMul ? Op::kSignExt : Op::kZeroExt;
    XLS_ASSIGN_OR_RETURN(op0,
                         ExtendIf(f, op0, dst_lhs->BitCountOrDie(), ext_op));
    XLS_ASSIGN_OR_RETURN(op1,
                         ExtendIf(f, op1, dst_rhs->BitCountOrDie(), ext_op));
    return std::vector<Node*>{op0, op1};
  }

  absl::StatusOr<Node*> ApplyToOutput(FunctionBase* f,
                                      Node* dst_output) const override {
    XLS_RET_CHECK(src_->GetType()->IsBits());
    XLS_RET_CHECK(dst_output->GetType()->IsBits());
    if (src_->BitCountOrDie() < dst_output->BitCountOrDie()) {
      return f->MakeNode<BitSlice>(dst_output->loc(), dst_output,
                                   /*start=*/0,
                                   /*width=*/src_->BitCountOrDie());
    }
    return dst_output;
  }

  absl::StatusOr<double> EstimateAreaOverhead(
      const AreaEstimator& area_estimator, absl::Span<Node* const> operands,
      Node* output) const override {
    if (dst_ != orig_dst_) {
      XLS_ASSIGN_OR_RETURN(
          double dst_area,
          area_estimator.GetOperationAreaInSquareMicrons(dst_));
      XLS_ASSIGN_OR_RETURN(
          double orig_dst_area,
          area_estimator.GetOperationAreaInSquareMicrons(orig_dst_));
      return std::max(0.0, dst_area - orig_dst_area);
    }
    return 0.0;
  }

 private:
  Node* orig_dst_;
  bool swap_operands_;
};

// `CompareToArithEquivalenceMapping` handles folding comparison operations
// into subtraction nodes then taking the msb.
//
// When `src` and `dst` have the same bitwidth, overflow is still avoided
// because either the msb differ and you can use that as the result, or the msb
// is the same in which case overflow is not possible.
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
          std::max(src_lhs->BitCountOrDie(), src_rhs->BitCountOrDie());
      needs_dst_widening |= required_width > dst->BitCountOrDie();
      target_width = std::max(target_width, required_width);
    }

    if (!needs_dst_widening) {
      return ComputeMappingsSourcesToDest<CompareToArithEquivalenceMapping>(
          sources, dst);
    }

    XLS_ASSIGN_OR_RETURN((PackageAndNode p_and_sub),
                         CreatePaddedSubPackage(target_width));
    NodeToMappings mappings;
    mappings.reserve(sources.size() + 1);
    for (Node* src : sources) {
      mappings[src] = std::make_unique<CompareToArithEquivalenceMapping>(
          src, p_and_sub.node, p_and_sub.package, dst);
    }
    mappings[dst] = std::make_unique<BitwidthExtendingEquivalenceMapping>(
        dst, p_and_sub.node, p_and_sub.package);
    return mappings;
  }

  CompareToArithEquivalenceMapping(
      Node* src, Node* dst, std::shared_ptr<Package> tmp_package = nullptr,
      std::optional<Node*> orig_dst = std::nullopt)
      : EquivalenceMapping(src, dst, std::move(tmp_package)),
        orig_dst_(orig_dst.value_or(dst)) {}

  absl::StatusOr<std::unique_ptr<EquivalenceMapping>> Clone(
      absl::optional_ref<const absl::flat_hash_map<Node*, Node*>>
          original_node_to_clone) const override {
    XLS_ASSIGN_OR_RETURN(Node * new_src, MapNode(src_, original_node_to_clone));
    XLS_ASSIGN_OR_RETURN(Node * new_dst, MapNode(dst_, original_node_to_clone));
    XLS_ASSIGN_OR_RETURN(Node * new_orig_dst,
                         MapNode(orig_dst_, original_node_to_clone));
    return std::make_unique<CompareToArithEquivalenceMapping>(
        new_src, new_dst, tmp_package_, new_orig_dst);
  }

  bool RequiresMsbSelect() const {
    return src_->operand(0)->BitCountOrDie() == dst_->BitCountOrDie();
  }

  absl::StatusOr<bool> RequiresOperandTransformation() const override {
    for (int i = 0; i < src_->operand_count(); ++i) {
      if (src_->operand(i)->BitCountOrDie() <
          dst_->operand(i)->BitCountOrDie()) {
        return true;
      }
    }
    XLS_ASSIGN_OR_RETURN(bool requires_swap, RequiresOperandSwap(src_->op()));
    return requires_swap;
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

    // Normalizing to ult/slt, e.g. for ule -> ult, we swap operands here, and
    // in ApplyToOutput, we will invert the result.
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
        Node * result,
        f->MakeNode<BitSlice>(dst_output->loc(), dst_output,
                              /*start=*/dst_output->BitCountOrDie() - 1,
                              /*width=*/1));
    if (RequiresMsbSelect()) {
      Node* op0 = src_->operand(0);
      Node* op1 = src_->operand(1);
      // Normalize to ult/slt, e.g. we swap operands for ugt -> ult, or we swap
      // operands AND invert result for sge -> slt
      XLS_ASSIGN_OR_RETURN(bool requires_swap, RequiresOperandSwap(src_->op()));
      if (requires_swap) {
        std::swap(op0, op1);
      }
      int64_t width = op0->BitCountOrDie();
      XLS_ASSIGN_OR_RETURN(
          Node * msb0,
          FindOrMakeBitSlice(op0, /*start=*/width - 1, /*width=*/1));
      XLS_ASSIGN_OR_RETURN(
          Node * msb1,
          FindOrMakeBitSlice(op1, /*start=*/width - 1, /*width=*/1));
      XLS_ASSIGN_OR_RETURN(
          Node * msb_diff,
          f->MakeNode<NaryOp>(dst_output->loc(), std::vector<Node*>{msb0, msb1},
                              Op::kXor));
      // Note we "normalize" to ult/slt. Given that, and if the msb differ,
      // msb(b) == a ult b, and msb(a) == a slt b.
      Node* msb_when_diff = IsSignedCompare(src_) ? msb0 : msb1;
      XLS_ASSIGN_OR_RETURN(
          result, f->MakeNode<Select>(dst_output->loc(), msb_diff,
                                      std::vector<Node*>{result, msb_when_diff},
                                      /*default_value=*/std::nullopt));
    }
    // Normalizing to ult/slt, e.g. for ule -> ult, we have already swapped
    // operands and now we invert the result.
    XLS_ASSIGN_OR_RETURN(bool requires_inversion,
                         RequiresOutputInversion(src_->op()));
    if (requires_inversion) {
      return f->MakeNode<UnOp>(dst_output->loc(), result, Op::kNot);
    }
    return result;
  }

  absl::StatusOr<double> EstimateAreaOverhead(
      const AreaEstimator& area_estimator, absl::Span<Node* const> operands,
      Node* output) const override {
    double area = 0.0;
    if (dst_ != orig_dst_) {
      XLS_ASSIGN_OR_RETURN(
          double dst_area,
          area_estimator.GetOperationAreaInSquareMicrons(dst_));
      XLS_ASSIGN_OR_RETURN(
          double orig_dst_area,
          area_estimator.GetOperationAreaInSquareMicrons(orig_dst_));
      area += dst_area - orig_dst_area;
    }
    if (RequiresMsbSelect()) {
      XLS_ASSIGN_OR_RETURN(
          double msb_select_area,
          EstimateAreaForNodes(
              area_estimator,
              [](Package* p, FunctionBuilder* fb) -> absl::Status {
                Type* u1 = p->GetBitsType(1);
                fb->Select(fb->Xor(fb->Param("a", u1), fb->Param("b", u1)),
                           {fb->Param("sub_msb", u1), fb->Param("c", u1)});
                return absl::OkStatus();
              }));
      area += msb_select_area;
    }
    XLS_ASSIGN_OR_RETURN(bool requires_inversion,
                         RequiresOutputInversion(src_->op()));
    if (requires_inversion) {
      XLS_ASSIGN_OR_RETURN(double inv_area,
                           EstimateAreaForSingleBitInversion(area_estimator));
      area += inv_area;
    }
    return area;
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

 private:
  // Used to estimate area overhead if the subtraction has to be widened.
  Node* orig_dst_;
};

// Returns `target` XORed with a sign-extended mask created from the MSB of
// `msb_source`. Reuses existing BitSlice, SignExt, and Xor nodes if available.
absl::StatusOr<Node*> XorWithSignExtendedMsb(FunctionBase* f, Node* target,
                                             Node* msb_source) {
  int64_t target_width = target->BitCountOrDie();
  int64_t source_width = msb_source->BitCountOrDie();
  XLS_ASSIGN_OR_RETURN(
      Node * msb, FindOrMakeBitSlice(msb_source, /*start=*/source_width - 1,
                                     /*width=*/1));
  XLS_ASSIGN_OR_RETURN(Node * mask,
                       ExtendIf(f, msb, target_width, Op::kSignExt));
  for (Node* user : target->users()) {
    if (user->function_base() == f && user->op() == Op::kXor &&
        user->operands() == absl::Span<Node* const>{target, mask}) {
      return user;
    }
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
    bool needs_msb_padding = false;
    int64_t target_shift_width = dst->operand(1)->BitCountOrDie();
    for (Node* src : sources) {
      // We gain nothing by handling edge case where src and dst have same op.
      if (!bit_typed_two_op_shift(src) || src->op() == dst->op()) {
        return std::nullopt;
      }
      // Cannot shift a value that is a larger bit width.
      if (src->BitCountOrDie() > dst->BitCountOrDie() ||
          src->operand(0)->BitCountOrDie() > dst->operand(0)->BitCountOrDie()) {
        return std::nullopt;
      }
      if (dst->op() == Op::kShra) {
        // The destination must be 0-padded if the source is not shra and the
        // bit-widths are equal, since an arithmetic shift would fill in 1s.
        if (src->op() != Op::kShra && (src->operand(0)->BitCountOrDie() ==
                                       dst->operand(0)->BitCountOrDie())) {
          needs_msb_padding = true;
        }
      }
      target_shift_width =
          std::max(target_shift_width, src->operand(1)->BitCountOrDie());
    }

    bool needs_shift_amt_wider =
        target_shift_width > dst->operand(1)->BitCountOrDie();
    bool needs_dst_wider = needs_msb_padding || needs_shift_amt_wider;

    if (!needs_dst_wider) {
      return ComputeMappingsSourcesToDest<ShiftEquivalenceMapping>(sources,
                                                                   dst);
    }

    int64_t target_val_width =
        dst->operand(0)->BitCountOrDie() + (needs_msb_padding ? 1 : 0);
    XLS_ASSIGN_OR_RETURN(
        (PackageAndNode p_and_dst),
        CreateModifiedShiftPackage(dst, target_val_width, target_shift_width));
    NodeToMappings mappings =
        ComputeMappingsSourcesToDest<ShiftEquivalenceMapping>(
            sources, p_and_dst.node, p_and_dst.package);
    mappings[dst] = std::make_unique<BitwidthExtendingEquivalenceMapping>(
        dst, p_and_dst.node, p_and_dst.package);
    return mappings;
  }

  using EquivalenceMapping::EquivalenceMapping;

  absl::StatusOr<std::unique_ptr<EquivalenceMapping>> Clone(
      absl::optional_ref<const absl::flat_hash_map<Node*, Node*>>
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
      XLS_ASSIGN_OR_RETURN(op0, FindOrMakeUnaryOp(f, op0, Op::kReverse));
    }

    // Extend shifted value to target width if necessary.
    XLS_ASSIGN_OR_RETURN(op0,
                         ExtendIf(f, op0, dst_->BitCountOrDie(), Op::kZeroExt));

    // Extend shift amount to target width if necessary.
    XLS_ASSIGN_OR_RETURN(
        op1, ExtendIf(f, op1, dst_->operand(1)->BitCountOrDie(), Op::kZeroExt));

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

// Builds the 4 multiply and adder tree decomposition for a W bit multiply using
// four W/4 bit inputs a, b, c, d, where X = (a ++ b) and Y = (c ++ d).
// Returns 2^(2Q) * ac + 2^Q * (ad + bc) + bd.
BValue BuildDecomposedMulAdderTree(FunctionBuilder& fb, BValue a, BValue b,
                                   BValue c, BValue d, int64_t width,
                                   bool is_signed) {
  int64_t half_width = width / 2;
  int64_t quarter_width = width / 4;
  auto mul = [&](BValue lhs, BValue rhs) {
    return is_signed ? fb.SMul(lhs, rhs, half_width)
                     : fb.UMul(lhs, rhs, half_width);
  };
  auto ext = [&](BValue val, int64_t new_width) {
    return is_signed ? fb.SignExtend(val, new_width)
                     : fb.ZeroExtend(val, new_width);
  };
  BValue mul_bd = mul(b, d);
  BValue mul_ad = mul(a, d);
  BValue mul_bc = mul(b, c);
  BValue mul_ac = mul(a, c);

  BValue zero_q = fb.Literal(UBits(0, quarter_width));
  BValue zero_half = fb.Literal(UBits(0, half_width));
  BValue sh_ac = fb.Concat({mul_ac, zero_half});
  BValue sh_ad = fb.Concat({ext(mul_ad, width - quarter_width), zero_q});
  BValue sh_bc = fb.Concat({ext(mul_bc, width - quarter_width), zero_q});
  BValue add_middle = fb.Add(sh_ad, sh_bc);
  BValue add_ends = fb.Add(sh_ac, ext(mul_bd, width));
  return fb.Add(add_ends, add_middle);
}

// Decomposes umul(X, Y), output width 4*Q, into four half width umul:
//   Assuming X = 2^Q * a + b and Y = 2^Q * c + d
//   Result: 2^(2Q) * ac + 2^Q * (ad + bc) + bd
absl::StatusOr<PackageAndNode> CreateDecomposedUMulPackage(int64_t width) {
  int64_t quarter_width = width / 4;
  return CreateModifiedDstAndPackage(
      [width, quarter_width](Package* p, FunctionBuilder& fb) -> BValue {
        Type* quarter_ty = p->GetBitsType(quarter_width);
        BValue a = fb.Param("a", quarter_ty);
        BValue b = fb.Param("b", quarter_ty);
        BValue c = fb.Param("c", quarter_ty);
        BValue d = fb.Param("d", quarter_ty);
        return BuildDecomposedMulAdderTree(fb, a, b, c, d, width,
                                           /*is_signed=*/false);
      });
}

// Decomposes smul(X, Y), output width 4*Q, into four half width smul:
//
// Leverage this identity, where Sx is a signed value, Ux is the same bits
// interpreted as an unsigned value:
//   X = 2^Q * Sa + Ub = 2^Q * Sa + Sb + 2^Q * msb(b)
//   which is a consequence of Ub = Sb + 2^Q * msb(b), i.e. the signed
//   interpretation subtracted 2^(Q-1) * msb(b) instead of adding it as Ub does.
//
// Expanding X * Y with four half width signed multiplies:
//   X * Y = (2^Q * Sa + Sb + 2^Q * msb(b)) * (2^Q * Sc + Sd + 2^Q * msb(d))
//   = 2^(2Q) * Sac + 2^Q * (Sad + Sbc) + Sbd + 2^Q * (
//       2^Q * Sa * msb(d) + 2^Q * Sc * msb(b)
//       + Sb * msb(d) + Sd * msb(b) + 2^Q * msb(b) * msb(d))
//   = 2^(2Q) * Sac + 2^Q * (Sad + Sbc) + Sbd + 2^Q * (
//       X * msb(d) + Y * msb(b) - 2^Q * msb(b) * msb(d))
absl::StatusOr<PackageAndNode> CreateDecomposedSMulPackage(int64_t width) {
  int64_t half_width = width / 2;
  int64_t quarter_width = width / 4;
  return CreateModifiedDstAndPackage([width, half_width, quarter_width](
                                         Package* p,
                                         FunctionBuilder& fb) -> BValue {
    Type* quarter_ty = p->GetBitsType(quarter_width);
    BValue a = fb.Param("a", quarter_ty);
    BValue b = fb.Param("b", quarter_ty);
    BValue c = fb.Param("c", quarter_ty);
    BValue d = fb.Param("d", quarter_ty);
    BValue x = fb.Concat({a, b});
    BValue y = fb.Concat({c, d});

    // Four half width multiplies and adder tree on ac, ad, bc, and bd
    // terms.
    BValue main_part = BuildDecomposedMulAdderTree(fb, a, b, c, d, width,
                                                   /*is_signed=*/true);

    // Correction terms because of using Sb and Sd instead of Ub and Ud.
    BValue msb_b = fb.BitSlice(b, quarter_width - 1, 1);
    BValue msb_d = fb.BitSlice(d, quarter_width - 1, 1);
    BValue zero_q = fb.Literal(UBits(0, quarter_width));
    BValue sh_x = fb.Concat({fb.SignExtend(x, width - quarter_width), zero_q});
    BValue sh_y = fb.Concat({fb.SignExtend(y, width - quarter_width), zero_q});
    BValue zero = fb.Literal(UBits(0, width));
    BValue corr_x_msb_d = fb.Select(msb_d, /*cases=*/{zero, sh_x});
    BValue corr_y_msb_b = fb.Select(msb_b, /*cases=*/{zero, sh_y});
    BValue corr_msb_b_and_d = fb.ZeroExtend(
        fb.Concat({fb.And(msb_b, msb_d), fb.Literal(UBits(0, half_width))}),
        width);
    BValue sum_corrs =
        fb.Subtract(fb.Add(corr_x_msb_d, corr_y_msb_b), corr_msb_b_and_d);
    return fb.Add(main_part, sum_corrs);
  });
}

// `MultiplySplitterEquivalenceMapping` decomposes a W-bit unsigned or signed
// multiply of two (W/2)-bit operands into four (W/4)x(W/4)->(W/2) unsigned or
// signed multiplies combined with shifts and additions.
class MultiplySplitterEquivalenceMapping : public EquivalenceMapping {
 public:
  static absl::StatusOr<std::shared_ptr<EquivalenceMapping>>
  TryCreateDecomposition(Node* node) {
    if (!node->OpIn({Op::kUMul, Op::kSMul}) || node->operand_count() != 2 ||
        !node->GetType()->IsBits()) {
      return nullptr;
    }
    // Require bit width to be a multiple of 4.
    int64_t width = node->BitCountOrDie();
    if (width < 4 || width % 4 != 0) {
      return nullptr;
    }
    Node* lhs = node->operand(0);
    Node* rhs = node->operand(1);
    if (!BothBitsTyped(lhs, rhs)) {
      return nullptr;
    }
    int64_t half_width = width / 2;
    if (lhs->BitCountOrDie() > half_width ||
        rhs->BitCountOrDie() > half_width) {
      return nullptr;
    }
    XLS_ASSIGN_OR_RETURN(PackageAndNode p_and_dst,
                         node->op() == Op::kSMul
                             ? CreateDecomposedSMulPackage(width)
                             : CreateDecomposedUMulPackage(width));
    return std::make_shared<MultiplySplitterEquivalenceMapping>(
        node, p_and_dst.node, std::move(p_and_dst.package));
  }

  using EquivalenceMapping::EquivalenceMapping;

  absl::StatusOr<std::unique_ptr<EquivalenceMapping>> Clone(
      absl::optional_ref<const absl::flat_hash_map<Node*, Node*>>
          original_node_to_clone) const override {
    return CloneEqMapping(this, original_node_to_clone);
  }

  absl::StatusOr<bool> RequiresOperandTransformation() const override {
    return true;
  }

  absl::StatusOr<bool> RequiresOutputTransformation() const override {
    return false;
  }

  absl::StatusOr<std::vector<Node*>> ApplyToOperands(
      FunctionBase* f, absl::Span<Node* const> src_operands) const override {
    XLS_RET_CHECK_EQ(src_operands.size(), 2);
    Node* lhs = src_operands[0];
    Node* rhs = src_operands[1];
    XLS_RET_CHECK(BothBitsTyped(lhs, rhs));
    int64_t width = src_->BitCountOrDie();
    int64_t half_width = width / 2;
    int64_t quarter_width = half_width / 2;

    Op ext_op = src_->op() == Op::kSMul ? Op::kSignExt : Op::kZeroExt;
    XLS_ASSIGN_OR_RETURN(lhs, ExtendIf(f, lhs, half_width, ext_op));
    XLS_ASSIGN_OR_RETURN(rhs, ExtendIf(f, rhs, half_width, ext_op));

    // Splits LHS=(a ++ b) and RHS=(c ++ d) into a, b, c, and d.
    XLS_ASSIGN_OR_RETURN(Node * a,
                         FindOrMakeBitSlice(lhs, /*start=*/quarter_width,
                                            /*width=*/quarter_width));
    XLS_ASSIGN_OR_RETURN(Node * b, FindOrMakeBitSlice(lhs, /*start=*/0,
                                                      /*width=*/quarter_width));
    XLS_ASSIGN_OR_RETURN(Node * c,
                         FindOrMakeBitSlice(rhs, /*start=*/quarter_width,
                                            /*width=*/quarter_width));
    XLS_ASSIGN_OR_RETURN(Node * d, FindOrMakeBitSlice(rhs, /*start=*/0,
                                                      /*width=*/quarter_width));
    return std::vector<Node*>{a, b, c, d};
  }

  absl::StatusOr<Node*> ApplyToOutput(FunctionBase* f,
                                      Node* dst_output) const override {
    return dst_output;
  }

  absl::StatusOr<double> EstimateAreaOverhead(
      const AreaEstimator& area_estimator, absl::Span<Node* const> operands,
      Node* output) const override {
    Node* orig = output != nullptr ? output : src_;
    XLS_ASSIGN_OR_RETURN(double orig_area,
                         area_estimator.GetOperationAreaInSquareMicrons(orig));
    XLS_ASSIGN_OR_RETURN(double decomp_area,
                         area_estimator.GetFunctionBaseAreaInSquareMicrons(
                             dst_->function_base()));
    return decomp_area - orig_area;
  }
};

}  // namespace

absl::StatusOr<Node*> EquivalenceMapping::MapNode(
    Node* node, absl::optional_ref<const absl::flat_hash_map<Node*, Node*>>
                    original_node_to_clone) const {
  // If no mapping is given, or the node is from a temporary package, return
  // the node itself.
  if (!original_node_to_clone.has_value() ||
      (tmp_package_ != nullptr && node->package() == tmp_package_.get())) {
    return node;
  }
  auto it = original_node_to_clone->find(node);
  XLS_RET_CHECK(it != original_node_to_clone->end())
      << "Node " << node->ToString()
      << " not found in original_node_to_clone map";
  return it->second;
}

void NodeEquivalenceMapper::Register(Factory factory) {
  absl::MutexLock lock(mutex_);
  factories_.push_back(factory);
}

void NodeEquivalenceMapper::RegisterDecomposition(
    DecompositionFactory factory) {
  absl::MutexLock lock(mutex_);
  decomposition_factories_.push_back(factory);
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

absl::StatusOr<std::vector<std::shared_ptr<EquivalenceMapping>>>
NodeEquivalenceMapper::ComputeDecompositions(Node* node) const {
  absl::MutexLock lock(mutex_);
  std::vector<std::shared_ptr<EquivalenceMapping>> decompositions;
  for (DecompositionFactory factory : decomposition_factories_) {
    XLS_ASSIGN_OR_RETURN(std::shared_ptr<EquivalenceMapping> decomp,
                         factory(node));
    if (decomp != nullptr) {
      decompositions.push_back(std::move(decomp));
    }
  }
  return decompositions;
}

NodeEquivalenceMapper& GetNodeEquivalenceMapper() {
  static absl::NoDestructor<NodeEquivalenceMapper> m;
  [[maybe_unused]] static const bool initialized = [&] {
    m->Register<IdentityEquivalenceMapping>();
    m->Register<BitwidthExtendingEquivalenceMapping>();
    m->Register<MulEquivalenceMapping>();
    m->Register<AddSubEquivalenceMapping>();
    m->Register<ComparatorEquivalenceMapping>();
    m->Register<CompareToArithEquivalenceMapping>();
    m->Register<ShiftEquivalenceMapping>();
    m->RegisterDecomposition<MultiplySplitterEquivalenceMapping>();
    return true;
  }();
  return *m;
}

absl::StatusOr<Node*> CloneWithMapping(
    Node* node, const EquivalenceMapping& mapping,
    absl::optional_ref<absl::flat_hash_map<Node*, Node*>>
        original_node_to_clone) {
  FunctionBase* f = node->function_base();
  XLS_ASSIGN_OR_RETURN(std::vector<Node*> converted_operands,
                       mapping.ApplyToOperands(f, node->operands()));
  FunctionBase* dst_fn = mapping.dst()->function_base();
  if (dst_fn == f) {
    XLS_ASSIGN_OR_RETURN(Node * unified_node, mapping.dst()->CloneInNewFunction(
                                                  converted_operands, f));
    if (original_node_to_clone.has_value()) {
      (*original_node_to_clone)[mapping.dst()] = unified_node;
    }
    return mapping.ApplyToOutput(f, unified_node);
  }

  // NOTE: The temporary function containing the destination node and its
  // dependencies has one parameter per converted operand.
  XLS_RET_CHECK_EQ(dst_fn->params().size(), converted_operands.size());
  absl::flat_hash_map<Node*, Node*> local_cloned_map;
  absl::flat_hash_map<Node*, Node*>& cloned_map =
      original_node_to_clone.has_value() ? *original_node_to_clone
                                         : local_cloned_map;
  for (int64_t i = 0; i < converted_operands.size(); ++i) {
    cloned_map[dst_fn->params()[i]] = converted_operands[i];
  }

  XLS_ASSIGN_OR_RETURN(std::vector<Node*> topo_sort, TopoSort(dst_fn));
  for (Node* part : topo_sort) {
    if (part->Is<Param>()) {
      continue;
    }
    std::vector<Node*> new_ops;
    new_ops.reserve(part->operand_count());
    for (Node* op : part->operands()) {
      auto it = cloned_map.find(op);
      XLS_RET_CHECK(it != cloned_map.end());
      new_ops.push_back(it->second);
    }
    XLS_ASSIGN_OR_RETURN(cloned_map[part],
                         part->CloneInNewFunction(new_ops, f));
  }

  auto it = cloned_map.find(mapping.dst());
  XLS_RET_CHECK(it != cloned_map.end());
  return mapping.ApplyToOutput(f, it->second);
}

}  // namespace xls
