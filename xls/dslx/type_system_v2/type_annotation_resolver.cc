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

#include "xls/dslx/type_system_v2/type_annotation_resolver.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "xls/common/casts.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system_v2/evaluator.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/parametric_struct_instantiator.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"
#include "xls/dslx/type_system_v2/type_system_tracer.h"
#include "xls/dslx/type_system_v2/unify_type_annotations.h"

namespace xls::dslx {
namespace {

class TypeAnnotationResolverImpl : public TypeAnnotationResolver {
 public:
  TypeAnnotationResolverImpl(
      Module& module, InferenceTable& table, const FileTable& file_table,
      UnificationErrorGenerator& error_generator, Evaluator& evaluator,
      ParametricStructInstantiator& parametric_struct_instantiator,
      TypeSystemTracer& tracer)
      : module_(module),
        table_(table),
        file_table_(file_table),
        error_generator_(error_generator),
        evaluator_(evaluator),
        parametric_struct_instantiator_(parametric_struct_instantiator),
        tracer_(tracer) {}

  absl::StatusOr<std::optional<const TypeAnnotation*>>
  ResolveAndUnifyTypeAnnotationsForNode(
      std::optional<const ParametricContext*> parametric_context,
      const AstNode* node,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate) override {
    TypeSystemTrace trace = tracer_.TraceUnify(node);
    VLOG(6) << "ResolveAndUnifyTypeAnnotationsForNode " << node->ToString();
    const std::optional<const NameRef*> type_variable =
        table_.GetTypeVariable(node);
    if (type_variable.has_value()) {
      // A type variable implies unification may be needed, so don't just use
      // the type annotation of the node if it has a variable associated with
      // it.
      std::optional<Span> node_span = node->GetSpan();
      CHECK(node_span.has_value());
      if (node->parent() != nullptr &&
          ((node->parent()->kind() == AstNodeKind::kConstantDef ||
            node->parent()->kind() == AstNodeKind::kNameDef)) &&
          !VariableHasAnyExplicitTypeAnnotations(parametric_context,
                                                 *type_variable)) {
        // The motivation for disallowing this, irrespective of its
        // unifiability, is that otherwise a snippet like this would present
        // a serious ambiguity:
        //   const X = 3;
        //   const Y = X + 1;
        // If we auto-annotate the `3` as `u2` and the `1` becomes `u2` via
        // normal promotion, `X + 1` surprisingly overflows. What is really
        // desired here is probably a common type for `X` and `Y` that fits
        // both. We want the programmer to write that type on the `X` line at
        // a minimum, which will then predictably propagate to `Y` if they
        // don't say otherwise.
        return absl::InvalidArgumentError(absl::Substitute(
            "TypeInferenceError: A variable or constant cannot be defined "
            "with an implicit type. `$0` at $1 must have a type annotation "
            "on at least one side of its assignment.",
            node->parent()->ToString(), node_span->ToString(file_table_)));
      }
      return ResolveAndUnifyTypeAnnotations(parametric_context, *type_variable,
                                            *node_span, accept_predicate);
    } else {
      std::optional<const TypeAnnotation*> annotation =
          table_.GetTypeAnnotation(node);
      // If the annotation belongs to a different module, send through
      // unification to potentially create a copy in this module.
      if (annotation.has_value() && (*annotation)->owner() != &module_) {
        XLS_ASSIGN_OR_RETURN(
            const TypeAnnotation* result,
            ResolveAndUnifyTypeAnnotations(parametric_context, {*annotation},
                                           *node->GetSpan(), accept_predicate));
        return result;
      }

      return annotation;
    }
  }

  absl::StatusOr<const TypeAnnotation*> ResolveAndUnifyTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      const NameRef* type_variable, const Span& span,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate) override {
    TypeSystemTrace trace = tracer_.TraceUnify(type_variable);
    VLOG(6) << "Unifying type annotations for variable "
            << type_variable->ToString();
    XLS_ASSIGN_OR_RETURN(std::vector<const TypeAnnotation*> annotations,
                         table_.GetTypeAnnotationsForTypeVariable(
                             parametric_context, type_variable));
    if (accept_predicate.has_value()) {
      FilterAnnotations(annotations, *accept_predicate);
    }
    XLS_ASSIGN_OR_RETURN(
        const TypeAnnotation* result,
        ResolveAndUnifyTypeAnnotations(parametric_context, annotations, span,
                                       accept_predicate));
    VLOG(6) << "Unified type for variable " << type_variable->ToString() << ": "
            << result->ToString();
    return result;
  }

  absl::StatusOr<const TypeAnnotation*> ResolveAndUnifyTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      std::vector<const TypeAnnotation*> annotations, const Span& span,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate) override {
    XLS_RETURN_IF_ERROR(ResolveIndirectTypeAnnotations(
        parametric_context, annotations, accept_predicate));
    TypeSystemTrace trace = tracer_.TraceUnify(annotations);
    return UnifyTypeAnnotations(module_, table_, file_table_, error_generator_,
                                evaluator_, parametric_struct_instantiator_,
                                parametric_context, annotations, span,
                                accept_predicate);
  }

  absl::Status ResolveIndirectTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      std::vector<const TypeAnnotation*>& annotations,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate) override {
    std::vector<const TypeAnnotation*> result;
    for (const TypeAnnotation* annotation : annotations) {
      if (!accept_predicate.has_value() || (*accept_predicate)(annotation)) {
        XLS_ASSIGN_OR_RETURN(
            const TypeAnnotation* resolved_annotation,
            ResolveIndirectTypeAnnotations(parametric_context, annotation,
                                           accept_predicate));
        result.push_back(resolved_annotation);
      }
    }
    annotations = std::move(result);
    return absl::OkStatus();
  }

  absl::StatusOr<const TypeAnnotation*> ResolveIndirectTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      const TypeAnnotation* annotation,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate) override {
    TypeSystemTrace trace = tracer_.TraceResolve(annotation);
    // This is purely to avoid wasting time on annotations that clearly need no
    // resolution.
    if (GetSignednessAndBitCount(annotation).ok() || IsToken(annotation)) {
      return annotation;
    }
    VLOG(4) << "Resolving variables in: " << annotation->ToString()
            << " in context: " << ToString(parametric_context);
    constexpr int kMaxIterations = 100;
    // This loop makes it so that we don't need resolution functions to
    // recursively resolve what they come up with.
    for (int i = 0;; i++) {
      CHECK_LE(i, kMaxIterations);
      bool replaced_anything = false;
      ObservableCloneReplacer replace_indirect(
          &replaced_anything, [&](const AstNode* node) {
            return ReplaceIndirectTypeAnnotations(node, parametric_context,
                                                  annotation, accept_predicate);
          });
      ObservableCloneReplacer replace_type_aliases(
          &replaced_anything, [&](const AstNode* node) {
            return ReplaceTypeAliasWithTarget(node);
          });
      XLS_ASSIGN_OR_RETURN(
          AstNode * clone,
          table_.Clone(annotation,
                       ChainCloneReplacers(std::move(replace_indirect),
                                           std::move(replace_type_aliases))));
      if (replaced_anything) {
        annotation = down_cast<const TypeAnnotation*>(clone);
      } else {
        break;
      }
    }
    return annotation;
  }

 private:
  // Converts `member_type` into a regular `TypeAnnotation` that expresses the
  // type of the given struct member independently of the struct type. For
  // example, if `member_type` refers to `SomeStruct.foo`, and the type
  // annotation of the referenced `foo` field is `u32[5]`, then the result will
  // be the `u32[5]` annotation. The `accept_predicate` may be used to exclude
  // type annotations dependent on an implicit parametric that this utility is
  // being used to help infer.
  absl::StatusOr<const TypeAnnotation*> ResolveMemberType(
      std::optional<const ParametricContext*> parametric_context,
      const MemberTypeAnnotation* member_type,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate) {
    VLOG(6) << "Resolve member type: " << member_type->ToString();
    XLS_ASSIGN_OR_RETURN(
        const TypeAnnotation* object_type,
        ResolveIndirectTypeAnnotations(
            parametric_context, member_type->struct_type(), accept_predicate));
    absl::StatusOr<SignednessAndBitCountResult> signedness_and_bit_count =
        GetSignednessAndBitCount(object_type);
    if (signedness_and_bit_count.ok()) {
      // A member of a bits-like type would have to be one of the special
      // hard-coded members of builtins, like "MIN", "MAX", and "ZERO".
      XLS_ASSIGN_OR_RETURN(SignednessAndBitCountResult signedness_and_bit_count,
                           GetSignednessAndBitCount(object_type));
      XLS_ASSIGN_OR_RETURN(
          bool is_signed,
          evaluator_.EvaluateBoolOrExpr(parametric_context,
                                        signedness_and_bit_count.signedness));
      XLS_ASSIGN_OR_RETURN(
          uint32_t bit_count,
          evaluator_.EvaluateU32OrExpr(parametric_context,
                                       signedness_and_bit_count.bit_count));
      XLS_ASSIGN_OR_RETURN(
          InterpValueWithTypeAnnotation member,
          GetBuiltinMember(module_, is_signed, bit_count,
                           member_type->member_name(), member_type->span(),
                           object_type->ToString(), file_table_));
      return member.type_annotation;
    }
    // It's not a bits-like type, so the only other thing that would have
    // members is a struct (or impl).
    XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> struct_or_proc_ref,
                         GetStructOrProcRef(object_type, file_table_));
    if (!struct_or_proc_ref.has_value()) {
      return absl::InvalidArgumentError(absl::Substitute(
          "Invalid access of member `$0` of non-struct type: `$1`",
          member_type->member_name(), object_type->ToString()));
    }
    const StructDefBase* struct_def = struct_or_proc_ref->def;
    if (struct_def->IsParametric()) {
      XLS_ASSIGN_OR_RETURN(
          parametric_context,
          parametric_struct_instantiator_.GetOrCreateParametricStructContext(
              parametric_context, *struct_or_proc_ref, member_type));
    }
    std::optional<StructMemberNode*> member =
        struct_def->GetMemberByName(member_type->member_name());
    if (!member.has_value() && struct_def->impl().has_value()) {
      // If the member is not in the struct itself, it may be in the impl.
      std::optional<ImplMember> impl_member =
          (*struct_def->impl())->GetMember(member_type->member_name());
      if (impl_member.has_value()) {
        if (std::holds_alternative<ConstantDef*>(*impl_member)) {
          XLS_ASSIGN_OR_RETURN(
              std::optional<const TypeAnnotation*> member_type,
              ResolveAndUnifyTypeAnnotationsForNode(
                  parametric_context, std::get<ConstantDef*>(*impl_member),
                  accept_predicate));
          CHECK(member_type.has_value());
          return parametric_struct_instantiator_
              .GetParametricFreeStructMemberType(
                  parametric_context, *struct_or_proc_ref, *member_type);
        }
        if (std::holds_alternative<Function*>(*impl_member)) {
          return parametric_struct_instantiator_
              .GetParametricFreeStructMemberType(
                  parametric_context, *struct_or_proc_ref,
                  CreateFunctionTypeAnnotation(
                      module_, *std::get<Function*>(*impl_member)));
        }
        return absl::UnimplementedError(
            absl::StrCat("Impl member type is not supported: ",
                         ToAstNode(*impl_member)->ToString()));
      }
    }
    if (!member.has_value()) {
      return absl::InvalidArgumentError(absl::Substitute(
          "No member `$0` in struct `$1`.", member_type->member_name(),
          struct_def->identifier()));
    }
    return parametric_struct_instantiator_.GetParametricFreeStructMemberType(
        parametric_context, *struct_or_proc_ref, (*member)->type());
  }

  // Converts `element_type` into a regular `TypeAnnotation` that expresses the
  // element type of the given array or tuple, independently of the array or
  // tuple type. For example, if `element_type` refers to an array whose type is
  // actually `u32[5]`, then the result will be a `u32` annotation. The
  // `accept_predicate` may be used to exclude type annotations dependent on an
  // implicit parametric that this utility is being used to help infer.
  absl::StatusOr<const TypeAnnotation*> ResolveElementType(
      std::optional<const ParametricContext*> parametric_context,
      const ElementTypeAnnotation* element_type,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate) {
    XLS_ASSIGN_OR_RETURN(const TypeAnnotation* container_type,
                         ResolveIndirectTypeAnnotations(
                             parametric_context, element_type->container_type(),
                             accept_predicate));
    if (const auto* array_type =
            dynamic_cast<const ArrayTypeAnnotation*>(container_type)) {
      return array_type->element_type();
    }
    if (const auto* tuple_type =
            dynamic_cast<const TupleTypeAnnotation*>(container_type)) {
      if (!element_type->tuple_index().has_value()) {
        return TypeInferenceErrorStatusForAnnotation(
            tuple_type->span(), tuple_type,
            "Tuples should not be indexed with array-style syntax. Use "
            "`tuple.<number>` syntax instead.",
            file_table_);
      }
      XLS_ASSIGN_OR_RETURN(
          uint64_t index,
          (*element_type->tuple_index())->GetAsUint64(file_table_));
      if (index >= tuple_type->members().size()) {
        return TypeInferenceErrorStatusForAnnotation(
            tuple_type->span(), tuple_type,
            absl::StrCat("Out-of-bounds tuple index specified: ", index),
            file_table_);
      }
      return tuple_type->members()[index];
    }
    if (const auto* channel_type =
            dynamic_cast<const ChannelTypeAnnotation*>(container_type)) {
      return GetChannelArrayElementType(module_, channel_type);
    }
    if (element_type->allow_bit_vector_destructuring()) {
      absl::StatusOr<SignednessAndBitCountResult> signedness_and_bit_count =
          GetSignednessAndBitCount(container_type);
      if (signedness_and_bit_count.ok()) {
        VLOG(6) << "Destructuring bit vector type: "
                << container_type->ToString();
        XLS_ASSIGN_OR_RETURN(
            bool signedness,
            evaluator_.EvaluateBoolOrExpr(
                parametric_context, signedness_and_bit_count->signedness));
        return CreateUnOrSnElementAnnotation(module_, container_type->span(),
                                             signedness);
      }
    }
    return container_type;
  }

  absl::StatusOr<const TypeAnnotation*> ResolveReturnType(
      std::optional<const ParametricContext*> parametric_context,
      const ReturnTypeAnnotation* return_type,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate) {
    VLOG(6) << "Resolve return type: " << return_type->ToString()
            << " in context: " << ToString(parametric_context);
    XLS_ASSIGN_OR_RETURN(const TypeAnnotation* function_type,
                         ResolveIndirectTypeAnnotations(
                             parametric_context, return_type->function_type(),
                             [&](const TypeAnnotation* annotation) {
                               return annotation != return_type &&
                                      (!accept_predicate.has_value() ||
                                       (*accept_predicate)(annotation));
                             }));
    TypeAnnotation* result_type =
        dynamic_cast<const FunctionTypeAnnotation*>(function_type)
            ->return_type();
    VLOG(6) << "Resulting return type: " << result_type->ToString();
    return result_type;
  }

  absl::StatusOr<const TypeAnnotation*> ResolveParamType(
      std::optional<const ParametricContext*> parametric_context,
      const ParamTypeAnnotation* param_type,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate) {
    VLOG(6) << "Resolve param type: " << param_type->ToString();
    XLS_ASSIGN_OR_RETURN(const TypeAnnotation* function_type,
                         ResolveIndirectTypeAnnotations(
                             parametric_context, param_type->function_type(),
                             [&](const TypeAnnotation* annotation) {
                               return annotation != param_type &&
                                      (!accept_predicate.has_value() ||
                                       (*accept_predicate)(annotation));
                             }));
    const std::vector<const TypeAnnotation*>& resolved_types =
        dynamic_cast<const FunctionTypeAnnotation*>(function_type)
            ->param_types();
    CHECK(param_type->param_index() < resolved_types.size());
    VLOG(6) << "Resulting argument type: "
            << resolved_types[param_type->param_index()]->ToString();
    return resolved_types[param_type->param_index()];
  }

  std::optional<const TypeAnnotation*> ResolveSelfType(
      std::optional<const ParametricContext*> parametric_context,
      const SelfTypeAnnotation* self_type) {
    std::optional<const TypeAnnotation*> expanded =
        table_.GetTypeAnnotation(self_type);
    if (expanded.has_value()) {
      return expanded;
    }
    if (parametric_context.has_value()) {
      return (*parametric_context)->self_type();
    }
    return std::nullopt;
  }

  // Determines if the given `type_variable` has any annotations in the table
  // that were explicitly written in the DSLX source.
  bool VariableHasAnyExplicitTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      const NameRef* type_variable) {
    absl::StatusOr<std::vector<const TypeAnnotation*>> annotations =
        table_.GetTypeAnnotationsForTypeVariable(parametric_context,
                                                 type_variable);
    return annotations.ok() &&
           absl::c_any_of(*annotations,
                          [this](const TypeAnnotation* annotation) {
                            return !table_.IsAutoLiteral(annotation);
                          });
  }

  // The "replace" function for an AstCloner that replaces indirect annotations
  // with their resolved & unified versions.
  absl::StatusOr<std::optional<AstNode*>> ReplaceIndirectTypeAnnotations(
      const AstNode* node,
      std::optional<const ParametricContext*> parametric_context,
      const TypeAnnotation* annotation,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate) {
    if (const auto* variable_type_annotation =
            dynamic_cast<const TypeVariableTypeAnnotation*>(node)) {
      XLS_ASSIGN_OR_RETURN(
          const TypeAnnotation* unified,
          ResolveAndUnifyTypeAnnotations(
              parametric_context, variable_type_annotation->type_variable(),
              annotation->span(), accept_predicate));
      return const_cast<TypeAnnotation*>(unified);
    }
    if (const auto* member_type =
            dynamic_cast<const MemberTypeAnnotation*>(node)) {
      XLS_ASSIGN_OR_RETURN(
          const TypeAnnotation* result,
          ResolveMemberType(parametric_context, member_type, accept_predicate));
      VLOG(5) << "Member type expansion for: " << member_type->member_name()
              << " yielded: " << result->ToString();
      return const_cast<TypeAnnotation*>(result);
    }
    if (const auto* element_type =
            dynamic_cast<const ElementTypeAnnotation*>(node)) {
      XLS_ASSIGN_OR_RETURN(const TypeAnnotation* result,
                           ResolveElementType(parametric_context, element_type,
                                              accept_predicate));
      return const_cast<TypeAnnotation*>(result);
    }
    if (const auto* return_type =
            dynamic_cast<const ReturnTypeAnnotation*>(node)) {
      XLS_ASSIGN_OR_RETURN(
          const TypeAnnotation* result,
          ResolveReturnType(parametric_context, return_type, accept_predicate));
      return const_cast<TypeAnnotation*>(result);
    }
    if (const auto* param_type =
            dynamic_cast<const ParamTypeAnnotation*>(node)) {
      if (accept_predicate.has_value() && !(*accept_predicate)(param_type)) {
        return module_.Make<AnyTypeAnnotation>();
      }
      XLS_ASSIGN_OR_RETURN(
          const TypeAnnotation* result,
          ResolveParamType(parametric_context, param_type, accept_predicate));
      return const_cast<TypeAnnotation*>(result);
    }
    if (const auto* self_type = dynamic_cast<const SelfTypeAnnotation*>(node)) {
      std::optional<const TypeAnnotation*> expanded =
          ResolveSelfType(parametric_context, self_type);
      CHECK(expanded.has_value());
      return const_cast<TypeAnnotation*>(*expanded);
    }
    return std::nullopt;
  }

  // The "replace" function for an AstCloner that replaces type aliases with the
  // type they equate to. There are two axes of descent here: this function
  // itself iteratively unwraps chains of type aliases that equate to other type
  // aliases, while the AstCloner recursively applies this function so that if
  // the type alias usage is in a descendant of the annotation being cloned,
  // that will be discovered and handled.
  absl::StatusOr<std::optional<AstNode*>> ReplaceTypeAliasWithTarget(
      const AstNode* node) {
    bool replaced_anything = false;
    std::optional<const TypeAnnotation*> latest =
        dynamic_cast<const TypeAnnotation*>(node);
    while (latest.has_value() &&
           dynamic_cast<const TypeRefTypeAnnotation*>(*latest)) {
      const auto* type_ref =
          dynamic_cast<const TypeRefTypeAnnotation*>(*latest);
      latest = table_.GetTypeAnnotation(ToAstNode(
          TypeDefinitionGetNameDef(type_ref->type_ref()->type_definition())));
      if (latest.has_value()) {
        node = const_cast<TypeAnnotation*>(*latest);
        replaced_anything = true;
      }
    }
    return replaced_anything ? std::make_optional(const_cast<AstNode*>(node))
                             : std::nullopt;
  }

  Module& module_;
  InferenceTable& table_;
  const FileTable& file_table_;
  UnificationErrorGenerator& error_generator_;
  Evaluator& evaluator_;
  ParametricStructInstantiator& parametric_struct_instantiator_;
  TypeSystemTracer& tracer_;
};

}  // namespace

std::unique_ptr<TypeAnnotationResolver> TypeAnnotationResolver::Create(
    Module& module, InferenceTable& table, const FileTable& file_table,
    UnificationErrorGenerator& error_generator, Evaluator& evaluator,
    ParametricStructInstantiator& parametric_struct_instantiator,
    TypeSystemTracer& tracer) {
  return std::make_unique<TypeAnnotationResolverImpl>(
      module, table, file_table, error_generator, evaluator,
      parametric_struct_instantiator, tracer);
}

}  // namespace xls::dslx
