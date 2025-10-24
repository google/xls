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

#include "xls/codegen/vast/dslx_type_fixer.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"
#include "xls/dslx/type_system_v2/type_inference_error_handler.h"

namespace xls {
namespace dslx {

class DslxTypeFixerImpl : public DslxTypeFixer {
 public:
  DslxTypeFixerImpl(Module& original_module)
      : original_module_(original_module) {}

  CloneReplacer GetErrorFixReplacer(const TypeInfo* ti) final {
    return GetErrorFixReplacerInternal(ti, /*root_node_to_preserve=*/nullptr);
  }

  CloneReplacer GetSimplifyReplacer(const TypeInfo* ti) final {
    return GetSimplifyReplacerInternal(ti, /*root_node_to_preserve=*/nullptr);
  }

  TypeInferenceErrorHandler GetErrorHandler() {
    return [this](const AstNode* node, absl::Span<const CandidateType> types) {
      return HandleError(node, types);
    };
  }

 private:
  std::vector<CandidateType> FilterToDeclarationTypes(
      absl::Span<const CandidateType> types) {
    std::vector<CandidateType> result;
    const bool any_declaration_types =
        absl::c_any_of(types, [](const CandidateType& next) {
          return next.flags.HasFlag(TypeInferenceFlag::kDeclarationType);
        });
    for (const CandidateType& next : types) {
      if (!any_declaration_types ||
          next.flags.HasFlag(TypeInferenceFlag::kDeclarationType)) {
        result.push_back(next);
      }
    }
    return result;
  }

  absl::StatusOr<const TypeAnnotation*> HandleError(
      const AstNode* node, absl::Span<const CandidateType> types) {
    VLOG(3) << "HandleError node: " << node->ToString()
            << ", kind: " << AstNodeKindToString(node->kind()) << " at "
            << (node->GetSpan().has_value()
                    ? node->GetSpan()->start().ToStringNoFile()
                    : "unknown");
    for (const CandidateType& next : types) {
      VLOG(4) << "\tCandidate type: " << next.annotation->ToString() << "\t"
              << next.flags.ToString() << " at "
              << next.annotation->span().start().ToStringNoFile();
    }

    // Verilog allows a signed index value but DSLX does not. We drop the
    // signed annotation to fix this.
    if (node->kind() == AstNodeKind::kNumber) {
      const auto* number = down_cast<const Number*>(node);
      if (number->parent() && number->type_annotation() &&
          dynamic_cast<const ArrayTypeAnnotation*>(node->parent())) {
        literals_with_dropped_annotations_.insert(number);
        return CreateU32Annotation(original_module_, number->span());
      }
    }

    // For a function invocation that is problematic, we get invoked for the
    // callee but not the invocation node. The "right answer" is the formal
    // function type.
    const TypeAnnotation* function_result = nullptr;
    std::vector<CandidateType> return_types;
    for (const CandidateType& candidate : types) {
      if (candidate.type->IsFunction()) {
        return_types.emplace_back(CandidateType{
            .annotation =
                down_cast<const FunctionTypeAnnotation*>(candidate.annotation)
                    ->return_type(),
            .type = &(
                down_cast<const FunctionType*>(candidate.type)->return_type()),
            .flags = TypeInferenceFlag::kNone});
        if (candidate.flags.HasFlag(TypeInferenceFlag::kFormalFunctionType)) {
          XLS_RET_CHECK(node->parent() != nullptr);
          XLS_RET_CHECK(node->parent()->kind() == AstNodeKind::kInvocation);
          XLS_RET_CHECK(candidate.annotation->annotation_kind() ==
                        TypeAnnotationKind::kFunction);
          const FunctionTypeAnnotation* fixed_function_type =
              down_cast<const FunctionTypeAnnotation*>(candidate.annotation);
          invocations_with_fixed_callees_.emplace(node->parent(),
                                                  fixed_function_type);
          function_result = candidate.annotation;
        }
      }
    }
    if (function_result) {
      // Do another run for the invocation node, because the result of the
      // invocation may need an inserted cast. The current type inference run
      // will just assume this cast is there based on our function type fix.
      XLS_RETURN_IF_ERROR(HandleError(node->parent(), return_types).status());
      return function_result;
    }

    // We will be invoked for arguments of problematic function calls, after we
    // are invoked for the callee. We handle these arguments here.
    if (node->parent() != nullptr &&
        node->parent()->kind() == AstNodeKind::kInvocation) {
      const auto fixed_callee =
          invocations_with_fixed_callees_.find(node->parent());
      if (fixed_callee != invocations_with_fixed_callees_.end()) {
        absl::Span<Expr* const> args =
            down_cast<Invocation*>(node->parent())->args();
        for (int i = 0; i < args.size(); i++) {
          if (args[i] == node) {
            XLS_RET_CHECK(i < fixed_callee->second->param_types().size());
            const TypeAnnotation* annotation =
                fixed_callee->second->param_types()[i];
            added_casts_.emplace(node, annotation);
            return annotation;
          }
        }
      }
    }

    // Verilog allows you to mix bits and enums, or different enums of
    // equivalent size, in an expr. In DSLX this requires casting. What follows
    // is a custom version of bits-like unification that takes this into
    // account.
    std::optional<CandidateType> result;
    int64_t result_size = 0;
    bool result_signedness = false;
    for (const CandidateType& candidate : FilterToDeclarationTypes(types)) {
      XLS_ASSIGN_OR_RETURN(TypeDim next_size_dim,
                           candidate.type->GetTotalBitCount());
      XLS_ASSIGN_OR_RETURN(int64_t next_size, next_size_dim.GetAsInt64());
      XLS_ASSIGN_OR_RETURN(bool next_signedness, IsSigned(*candidate.type));
      bool formal_member_type =
          candidate.flags.HasFlag(TypeInferenceFlag::kFormalMemberType);
      if (result && result_signedness != next_signedness) {
        if (formal_member_type) {
          result = candidate;
          result_size = next_size;
        } else {
          // This should not actually happen for correct Verilog.
          return absl::InvalidArgumentError(absl::Substitute(
              "Signed vs. unsigned mismatch: $0 vs. $1 fixing node $2 of kind "
              "$3",
              types[0].annotation->ToString(), candidate.annotation->ToString(),
              node->ToString(), AstNodeKindToString(node->kind())));
        }
      } else if (!result || next_size > result_size) {
        result = candidate;
        result_size = next_size;
        result_signedness = next_signedness;
      }
      if (formal_member_type) {
        break;
      }
    }
    if (dynamic_cast<const Expr*>(node)) {
      added_casts_.emplace(node, result->annotation);
    }

    VLOG(4) << "Yielding " << result->annotation->ToString();
    return result->annotation;
  }

  CloneReplacer GetErrorFixReplacerInternal(
      const TypeInfo* ti, const AstNode* root_node_to_preserve) {
    return [this, ti, root_node_to_preserve](
               const AstNode* node, Module* target_module,
               const absl::flat_hash_map<const AstNode*, AstNode*>&)
               -> absl::StatusOr<std::optional<AstNode*>> {
      if (node == root_node_to_preserve) {
        return std::nullopt;
      }

      // Drop literal annotations that the error handler said we don't want.
      if (literals_with_dropped_annotations_.contains(node)) {
        const auto* number = down_cast<const Number*>(node);
        return target_module->Make<Number>(number->span(), number->text(),
                                           NumberKind::kOther,
                                           /*type_annotation=*/nullptr);
      }

      // Insert casts that the error handler said we needed.
      const auto cast = added_casts_.find(node);
      if (node->kind() != AstNodeKind::kStatementBlock &&
          cast != added_casts_.end()) {
        XLS_ASSIGN_OR_RETURN(
            AstNode * clone,
            CloneAst(node, GetErrorFixReplacerInternal(ti, node)));
        return target_module->Make<Cast>(
            Span::None(),
            const_cast<Expr*>(down_cast<const dslx::Expr*>(clone)),
            const_cast<TypeAnnotation*>(cast->second));
      }

      return std::nullopt;
    };
  }

  CloneReplacer GetSimplifyReplacerInternal(
      const TypeInfo* ti, const AstNode* root_node_to_preserve) {
    return [this, ti, root_node_to_preserve](
               const AstNode* node, Module* target_module,
               const absl::flat_hash_map<const AstNode*, AstNode*>&)
               -> absl::StatusOr<std::optional<AstNode*>> {
      if (node == root_node_to_preserve) {
        return std::nullopt;
      }

      // Drop the LHS annotation on a constant def if unnecessary.
      if (node->kind() == AstNodeKind::kConstantDef) {
        const std::optional<Type*> lhs_type =
            ti->GetItem(down_cast<const ConstantDef*>(node)->name_def());
        const std::optional<Type*> rhs_type =
            ti->GetItem(down_cast<const ConstantDef*>(node)->value());
        if (TypeEq(**lhs_type, **rhs_type)) {
          XLS_ASSIGN_OR_RETURN(
              AstNode * clone,
              CloneAst(node, GetSimplifyReplacerInternal(ti, node)));
          down_cast<ConstantDef*>(clone)->set_type_annotation(nullptr);
          return clone;
        }
      }

      // Drop dead casts.
      std::optional<const Expr*> unwrapped = UnwrapDeadCast(ti, node);
      if (unwrapped.has_value()) {
        XLS_ASSIGN_OR_RETURN(
            AstNode * clone_of_unwrapped,
            CloneAst(*unwrapped, GetSimplifyReplacerInternal(ti, node)));
        Expr* result = down_cast<Expr*>(clone_of_unwrapped);
        result->set_in_parens(false);
        return result;
      }

      return std::nullopt;
    };
  }

  std::optional<const Expr*> UnwrapDeadCast(const TypeInfo* ti,
                                            const AstNode* node) {
    const std::optional<Type*> casted = ti->GetItem(node);
    std::optional<const Expr*> unwrapped;

    // Walk through the layers of a potential cast onion. Note that there are
    // usually at most 2-3 layers.
    while (node->kind() == AstNodeKind::kCast) {
      const Expr* expr = down_cast<const Cast*>(node)->expr();
      const std::optional<Type*> uncasted = ti->GetItem(expr);

      // See if we are OK to drop all layers of casts up to here.
      if (TypeEq(**casted, **uncasted)) {
        unwrapped = expr;
      }

      node = expr;
    }

    return unwrapped;
  }

  bool TypeEq(const Type& a, const Type& b) {
    if (a == b) {
      return true;
    }
    std::optional<BitsLikeProperties> a_bits = GetBitsLike(a);
    std::optional<BitsLikeProperties> b_bits = GetBitsLike(b);
    return a_bits.has_value() && b_bits.has_value() && *a_bits == *b_bits;
  }

  Module& original_module_;
  absl::flat_hash_map<const AstNode*, const TypeAnnotation*> added_casts_;
  absl::flat_hash_set<const AstNode*> literals_with_dropped_annotations_;
  absl::flat_hash_map<const AstNode*, const FunctionTypeAnnotation*>
      invocations_with_fixed_callees_;
};

}  // namespace dslx

std::unique_ptr<DslxTypeFixer> CreateDslxTypeFixer(dslx::Module& module,
                                                   const dslx::ImportData&) {
  return std::make_unique<dslx::DslxTypeFixerImpl>(module);
}

}  // namespace xls
