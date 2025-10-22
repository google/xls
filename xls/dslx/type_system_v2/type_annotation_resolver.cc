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
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/evaluator.h"
#include "xls/dslx/type_system_v2/import_utils.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/parametric_struct_instantiator.h"
#include "xls/dslx/type_system_v2/simplified_type_annotation_cache.h"
#include "xls/dslx/type_system_v2/type_annotation_filter.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"
#include "xls/dslx/type_system_v2/type_system_tracer.h"
#include "xls/dslx/type_system_v2/unify_type_annotations.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"

namespace xls::dslx {
namespace {

// A visitor that traverses a `TypeAnnotation` recursively until an annotation
// is found of a kind which needs resolution (e.g. member or element type). If
// no such annotation is found, then the top-level annotation can be considered
// already resolved.
class NeedsResolutionDetector : public AstNodeVisitorWithDefault {
 public:
  absl::Status HandleTypeRefTypeAnnotation(
      const TypeRefTypeAnnotation*) override {
    return MarkNeedsResolution();
  }
  absl::Status HandleTypeVariableTypeAnnotation(
      const TypeVariableTypeAnnotation*) override {
    return MarkNeedsResolution();
  }
  absl::Status HandleMemberTypeAnnotation(
      const MemberTypeAnnotation*) override {
    return MarkNeedsResolution();
  }
  absl::Status HandleElementTypeAnnotation(
      const ElementTypeAnnotation*) override {
    return MarkNeedsResolution();
  }
  absl::Status HandleReturnTypeAnnotation(
      const ReturnTypeAnnotation*) override {
    return MarkNeedsResolution();
  }
  absl::Status HandleParamTypeAnnotation(const ParamTypeAnnotation*) override {
    return MarkNeedsResolution();
  }
  absl::Status HandleSelfTypeAnnotation(const SelfTypeAnnotation*) override {
    return MarkNeedsResolution();
  }
  absl::Status HandleSliceTypeAnnotation(const SliceTypeAnnotation*) override {
    return MarkNeedsResolution();
  }

  absl::Status DefaultHandler(const AstNode* node) override {
    for (const AstNode* child : node->GetChildren(/*want_types=*/true)) {
      XLS_RETURN_IF_ERROR(child->Accept(this));
      if (needs_resolution_) {
        return absl::OkStatus();
      }
    }
    return absl::OkStatus();
  }

  bool needs_resolution() const { return needs_resolution_; }

 private:
  absl::Status MarkNeedsResolution() {
    needs_resolution_ = true;
    return absl::OkStatus();
  }

  bool needs_resolution_ = false;
};

// A resolver implementation meant for ad hoc internal use within the scope of
// one single external request. The resolver may temporarily cache information
// collected while doing the internal resolutions for that request.
class StatefulResolver : public TypeAnnotationResolver {
 public:
  StatefulResolver(
      Module& module, InferenceTable& table, const FileTable& file_table,
      UnificationErrorGenerator& error_generator, Evaluator& evaluator,
      ParametricStructInstantiator& parametric_struct_instantiator,
      TypeSystemTracer& tracer, WarningCollector& warning_collector,
      ImportData& import_data,
      SimplifiedTypeAnnotationCache& simplified_type_annotation_cache,
      ResolverErrorHandler error_handler,
      std::function<absl::Status(std::optional<const ParametricContext*>,
                                 const Invocation*)>
          invocation_converter)
      : module_(module),
        table_(table),
        file_table_(file_table),
        error_generator_(error_generator),
        evaluator_(evaluator),
        parametric_struct_instantiator_(parametric_struct_instantiator),
        tracer_(tracer),
        warning_collector_(warning_collector),
        import_data_(import_data),
        simplified_type_annotation_cache_(simplified_type_annotation_cache),
        error_handler_(error_handler),
        invocation_converter_(invocation_converter) {}

  absl::StatusOr<std::unique_ptr<TypeAnnotationResolver>> ResolverForNode(
      const AstNode* node) {
    return TypeAnnotationResolver::Create(
        *node->owner(), table_, file_table_, error_generator_, evaluator_,
        parametric_struct_instantiator_, tracer_, warning_collector_,
        import_data_, simplified_type_annotation_cache_, error_handler_,
        invocation_converter_);
  }

  absl::StatusOr<std::optional<const TypeAnnotation*>>
  ResolveAndUnifyTypeAnnotationsForNode(
      std::optional<const ParametricContext*> parametric_context,
      const AstNode* node, TypeAnnotationFilter filter) override {
    if (node->owner() != &module_) {
      XLS_ASSIGN_OR_RETURN(auto new_resolver, ResolverForNode(node));
      return new_resolver->ResolveAndUnifyTypeAnnotationsForNode(
          parametric_context, node, filter);
    }

    TypeSystemTrace trace = tracer_.TraceUnify(node);
    VLOG(6) << "ResolveAndUnifyTypeAnnotationsForNode " << node->ToString();
    const std::optional<const NameRef*> type_variable =
        table_.GetTypeVariable(node);
    if (type_variable.has_value()) {
      // A type variable implies unification may be needed, so don't just use
      // the type annotation of the node if it has a variable associated with
      // it.
      std::optional<Span> node_span = node->GetSpan();
      XLS_RET_CHECK(node_span.has_value());
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
        return TypeInferenceErrorStatus(
            *node_span, /*type=*/nullptr,
            absl::Substitute("A variable or constant cannot be defined with an "
                             "implicit type. `$0` must have a type annotation "
                             "on at least one side of its assignment.",
                             node->parent()->ToString()),
            file_table_);
      }
      if (node->kind() == AstNodeKind::kFor ||
          node->kind() == AstNodeKind::kUnrollFor) {
        const auto* loop = down_cast<const ForLoopBase*>(node);
        if (!VariableHasAnyExplicitTypeAnnotations(parametric_context,
                                                   *type_variable)) {
          // Disallow this with similar rationale to the const/NameDef case
          // above.
          return TypeInferenceErrorStatus(
              *node_span, /*type=*/nullptr,
              absl::Substitute(
                  "Loop cannot have an implicit result type derived from init "
                  "expression `$0`. Either this expression or the loop "
                  "accumulator declaration must have a type.",
                  loop->init()->ToString()),
              file_table_);
        }
      }
      absl::StatusOr<const TypeAnnotation*> result =
          ResolveAndUnifyTypeAnnotations(parametric_context, node,
                                         *type_variable, *node_span, filter,
                                         /*require_bits_like=*/false);
      if (result.ok()) {
        trace.SetResult(*result);
      }
      return result;
    } else {
      std::optional<const TypeAnnotation*> annotation =
          table_.GetTypeAnnotation(node);
      // If the annotation belongs to a different module, send through
      // unification to potentially create a copy in this module.
      if (annotation.has_value() && (*annotation)->owner() != &module_) {
        XLS_ASSIGN_OR_RETURN(const TypeAnnotation* result,
                             ResolveAndUnifyTypeAnnotations(
                                 parametric_context, node, {*annotation},
                                 *node->GetSpan(), filter,
                                 /*require_bits_like=*/false,
                                 /*used_error_handler=*/nullptr));
        trace.SetResult(result);
        return result;
      }

      if (annotation.has_value()) {
        trace.SetResult(*annotation);
      }
      return annotation;
    }
  }

  absl::StatusOr<const TypeAnnotation*> ResolveAndUnifyTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      std::optional<const AstNode*> context_node, const NameRef* type_variable,
      const Span& span, TypeAnnotationFilter filter,
      bool require_bits_like) override {
    TypeSystemTrace trace = tracer_.TraceUnify(type_variable);
    VLOG(6) << "Unifying type annotations for variable "
            << type_variable->ToString();
    // If this type variable belongs to a different table, resolve using that
    // module.
    if (type_variable->owner() != &module_) {
      XLS_ASSIGN_OR_RETURN(auto new_resolver, ResolverForNode(type_variable));
      return new_resolver->ResolveAndUnifyTypeAnnotations(
          parametric_context, context_node, type_variable, span, filter,
          require_bits_like);
    }

    const NameDef* type_variable_def =
        std::get<const NameDef*>(type_variable->name_def());
    const auto it = temp_cache_.find(type_variable_def);
    if (it != temp_cache_.end()) {
      VLOG(6) << "Using request-scoped cached type for "
              << type_variable->ToString();
      return CheckBitsLikeForDirectAnnotation(it->second, require_bits_like);
    }

    std::optional<const TypeAnnotation*> cached =
        table_.GetCachedUnifiedTypeForVariable(parametric_context,
                                               type_variable);
    if (cached.has_value() && !(*cached)->IsAnnotation<AnyTypeAnnotation>()) {
      VLOG(6) << "Using cached type for " << type_variable->ToString();
      trace.SetUsedCache(true);
      trace.SetResult(*cached);
      return CheckBitsLikeForDirectAnnotation(*cached, require_bits_like);
    }

    XLS_ASSIGN_OR_RETURN(
        std::vector<const Invocation*> prerequisite_invocations,
        table_.GetInvocationsFeedingTypeVariable(type_variable));
    for (const Invocation* invocation : prerequisite_invocations) {
      XLS_RETURN_IF_ERROR(
          invocation_converter_(parametric_context, invocation));
    }

    XLS_ASSIGN_OR_RETURN(std::vector<const TypeAnnotation*> annotations,
                         table_.GetTypeAnnotationsForTypeVariable(
                             parametric_context, type_variable));
    bool filtered_top_level = false;
    if (!filter.IsNone() && !annotations.empty()) {
      tracer_.TraceFilter(filter, annotations);
      const int original_size = annotations.size();
      FilterAnnotations(annotations, filter);
      filtered_top_level = annotations.size() != original_size;
    }
    int reject_count = 0;
    bool used_error_handler = false;
    absl::flat_hash_set<const NameRef*> variables_traversed;
    XLS_ASSIGN_OR_RETURN(
        const TypeAnnotation* result,
        ResolveAndUnifyTypeAnnotations(
            parametric_context, context_node, annotations, span,
            TypeAnnotationFilter::CaptureRejectCount(&reject_count)
                .Chain(TypeAnnotationFilter::CaptureVariables(
                    &variables_traversed))
                .Chain(filter),
            require_bits_like, &used_error_handler));
    VLOG(6) << "Unified type for variable " << type_variable->ToString() << ": "
            << result->ToString();
    trace.SetResult(result);

    // Cache the result externally only if we actually considered all the deps
    // of the variable, and came up with a non-Any result. Otherwise the result
    // is only durable/relevant enough to reuse locally within the scope of the
    // one external resolution request. We also do not cache results that were
    // generated by an external error handler, because the handler may want to
    // be invoked for every node affected by an offending type variable.
    if (!used_error_handler && !filtered_top_level && reject_count == 0 &&
        !result->IsAnnotation<AnyTypeAnnotation>()) {
      VLOG(6) << "Caching unified type: " << result->ToString()
              << " for variable: " << type_variable->ToString();
      table_.SetCachedUnifiedTypeForVariable(parametric_context, type_variable,
                                             variables_traversed, result);
      trace.SetPopulatedCache(true);
    } else {
      temp_cache_[type_variable_def] = result;
    }

    return result;
  }

  absl::StatusOr<const TypeAnnotation*> ResolveAndUnifyTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      std::optional<const AstNode*> context_node,
      std::vector<const TypeAnnotation*> annotations, const Span& span,
      TypeAnnotationFilter filter, bool require_bits_like,
      bool* used_error_handler) override {
    XLS_RETURN_IF_ERROR(ResolveIndirectTypeAnnotations(
        parametric_context, context_node, annotations, filter));

    if (require_bits_like) {
      for (const TypeAnnotation* annotation : annotations) {
        XLS_RETURN_IF_ERROR(
            CheckBitsLikeForDirectAnnotation(annotation, require_bits_like)
                .status());
      }
    }

    TypeSystemTrace trace = tracer_.TraceUnify(annotations);
    absl::StatusOr<const TypeAnnotation*> result = UnifyTypeAnnotations(
        module_, table_, file_table_, error_generator_, evaluator_,
        parametric_struct_instantiator_, parametric_context, annotations, span,
        import_data_);
    if (!result.ok() && error_handler_ && context_node.has_value()) {
      result = error_handler_(parametric_context, *context_node,
                              absl::MakeSpan(annotations));
      if (result.ok() && used_error_handler != nullptr) {
        *used_error_handler = true;
      }
    }

    if (!result.ok()) {
      return result.status();
    }

    trace.SetResult(*result);
    return result;
  }

  absl::Status ResolveIndirectTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      std::optional<const AstNode*> context_node,
      std::vector<const TypeAnnotation*>& annotations,
      TypeAnnotationFilter filter) override {
    std::vector<const TypeAnnotation*> result;
    for (const TypeAnnotation* annotation : annotations) {
      if (!filter.Filter(annotation)) {
        XLS_ASSIGN_OR_RETURN(
            const TypeAnnotation* resolved_annotation,
            ResolveIndirectTypeAnnotations(parametric_context, context_node,
                                           annotation, filter));
        result.push_back(resolved_annotation);
      }
    }
    annotations = std::move(result);
    return absl::OkStatus();
  }

  absl::StatusOr<const TypeAnnotation*> ResolveIndirectTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      std::optional<const AstNode*> context_node,
      const TypeAnnotation* annotation, TypeAnnotationFilter filter) override {
    return ResolveInternal(parametric_context, context_node, annotation, filter,
                           /*type_refs_only=*/false);
  }

  absl::StatusOr<const TypeAnnotation*> ResolveTypeRefs(
      std::optional<const ParametricContext*> parametric_context,
      std::optional<const AstNode*> context_node,
      const TypeAnnotation* annotation) override {
    return ResolveInternal(parametric_context, context_node, annotation,
                           TypeAnnotationFilter::None(),
                           /*type_refs_only=*/true);
  }

 private:
  absl::StatusOr<const TypeAnnotation*> ResolveInternal(
      std::optional<const ParametricContext*> parametric_context,
      std::optional<const AstNode*> context_node,
      const TypeAnnotation* annotation, TypeAnnotationFilter filter,
      bool type_refs_only) {
    TypeSystemTrace trace =
        tracer_.TraceResolve(annotation, parametric_context);
    // This is purely to avoid wasting time on annotations that clearly need no
    // resolution.
    if (annotation->IsAnnotation<BuiltinTypeAnnotation>() ||
        IsToken(annotation)) {
      trace.SetResult(annotation);
      return annotation;
    }
    if (GetSignednessAndBitCount(annotation).ok()) {
      XLS_ASSIGN_OR_RETURN(annotation, ReplaceForeignConstantRefs(annotation));
      trace.SetResult(annotation);
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

      // Avoid wasting time cloning the annotation for nothing.
      NeedsResolutionDetector detector;
      XLS_RETURN_IF_ERROR(annotation->Accept(&detector));
      if (!detector.needs_resolution()) {
        break;
      }

      ObservableCloneReplacer replace_indirect(
          &replaced_anything,
          [&](const AstNode* node, Module*,
              const absl::flat_hash_map<const AstNode*, AstNode*>&) {
            return ReplaceIndirectTypeAnnotations(
                node, parametric_context, context_node, annotation, filter);
          });
      ObservableCloneReplacer replace_type_aliases(
          &replaced_anything,
          [&](const AstNode* node, Module*,
              const absl::flat_hash_map<const AstNode*, AstNode*>&) {
            return ReplaceTypeAliasWithTarget(node);
          });
      XLS_ASSIGN_OR_RETURN(
          AstNode * clone,
          table_.Clone(annotation, type_refs_only
                                       ? std::move(replace_type_aliases)
                                       : ChainCloneReplacers(
                                             std::move(replace_indirect),
                                             std::move(replace_type_aliases))));
      if (replaced_anything) {
        annotation = down_cast<const TypeAnnotation*>(clone);
      } else {
        break;
      }
    }

    XLS_ASSIGN_OR_RETURN(annotation, ReplaceForeignConstantRefs(annotation));
    trace.SetResult(annotation);
    return annotation;
  }

  // Makes sure the given direct annotation is bits-like if `require_bits_like`
  // is true. This function presumes `annotation` is post-resolution. Returns
  // the passed-in annotation if it passes the check.
  absl::StatusOr<const TypeAnnotation*> CheckBitsLikeForDirectAnnotation(
      const TypeAnnotation* annotation, bool require_bits_like) {
    if (!require_bits_like ||
        GetSignednessAndBitCount(annotation, false).ok()) {
      return annotation;
    }
    return TypeInferenceErrorStatusForAnnotation(
        annotation->span(), annotation,
        absl::Substitute("Expected a bits-like type; got: `$0`",
                         annotation->ToString()),
        file_table_);
  }

  // Converts `member_type` into a regular `TypeAnnotation` that expresses the
  // type of the given struct member independently of the struct type. For
  // example, if `member_type` refers to `SomeStruct.foo`, and the type
  // annotation of the referenced `foo` field is `u32[5]`, then the result will
  // be the `u32[5]` annotation. The `filter` may be used to exclude type
  // annotations dependent on an implicit parametric that this utility is being
  // used to help infer.
  absl::StatusOr<const TypeAnnotation*> ResolveMemberType(
      std::optional<const ParametricContext*> parametric_context,
      std::optional<const AstNode*> context_node,
      const MemberTypeAnnotation* member_type, TypeAnnotationFilter filter) {
    VLOG(6) << "Resolve member type: " << member_type->ToString();
    XLS_ASSIGN_OR_RETURN(
        const TypeAnnotation* object_type,
        ResolveIndirectTypeAnnotations(parametric_context, context_node,
                                       member_type->struct_type(), filter));
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
    // members is an enum or struct (or impl).
    // The type annotation of enum member is just the enum itself.
    XLS_ASSIGN_OR_RETURN(std::optional<const EnumDef*> enum_def,
                         GetEnumDef(object_type, import_data_));
    if (enum_def.has_value()) {
      return object_type;
    }
    // It must be a struct then.
    XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> struct_or_proc_ref,
                         GetStructOrProcRef(object_type, import_data_));
    if (!struct_or_proc_ref.has_value()) {
      return TypeInferenceErrorStatus(
          member_type->span(), /*type=*/nullptr,
          absl::Substitute(
              "Invalid access of member `$0` of non-struct type: `$1`",
              member_type->member_name(), object_type->ToString()),
          file_table_);
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
                  filter));
          XLS_RET_CHECK(member_type.has_value());
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
    // If the simplified member type annotation is known, such as when we
    // process the members of a non-parametric StructDef, use it.
    // Keeping the dummy `parametric_context` argument here in case we support
    // caching parametric MemberTypeAnnotation in the future.
    if (std::optional<const TypeAnnotation*> cached =
            simplified_type_annotation_cache_.GetSimplifiedypeAnnotation(
                parametric_context, *member)) {
      return *cached;
    }
    return parametric_struct_instantiator_.GetParametricFreeStructMemberType(
        parametric_context, *struct_or_proc_ref, (*member)->type());
  }

  // Converts `element_type` into a regular `TypeAnnotation` that expresses the
  // element type of the given array or tuple, independently of the array or
  // tuple type. For example, if `element_type` refers to an array whose type is
  // actually `u32[5]`, then the result will be a `u32` annotation. The `filter`
  // may be used to exclude type annotations dependent on an implicit parametric
  // that this utility is being used to help infer.
  absl::StatusOr<const TypeAnnotation*> ResolveElementType(
      std::optional<const ParametricContext*> parametric_context,
      std::optional<const AstNode*> context_node,
      const ElementTypeAnnotation* element_type, TypeAnnotationFilter filter) {
    XLS_ASSIGN_OR_RETURN(
        const TypeAnnotation* container_type,
        ResolveIndirectTypeAnnotations(
            parametric_context, context_node, element_type->container_type(),
            filter.Chain(TypeAnnotationFilter::BlockRecursion(element_type))));
    if (container_type->IsAnnotation<ArrayTypeAnnotation>()) {
      return container_type->AsAnnotation<ArrayTypeAnnotation>()
          ->element_type();
    }
    if (container_type->IsAnnotation<TupleTypeAnnotation>()) {
      const auto* tuple_type =
          container_type->AsAnnotation<TupleTypeAnnotation>();
      if (!element_type->tuple_index().has_value()) {
        return TypeInferenceErrorStatusForAnnotation(
            tuple_type->span(), tuple_type,
            "Tuples should not be indexed with array-style syntax. Use "
            "`tuple.<number>` syntax instead.",
            file_table_);
      }
      XLS_ASSIGN_OR_RETURN(uint32_t index, evaluator_.EvaluateU32OrExpr(
                                               parametric_context,
                                               *element_type->tuple_index()));
      if (index >= tuple_type->members().size()) {
        return TypeInferenceErrorStatusForAnnotation(
            tuple_type->span(), tuple_type,
            absl::StrCat("Out-of-bounds tuple index specified: ", index),
            file_table_);
      }
      return tuple_type->members()[index];
    }
    if (container_type->IsAnnotation<ChannelTypeAnnotation>()) {
      return GetChannelArrayElementType(
          module_, container_type->AsAnnotation<ChannelTypeAnnotation>());
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
      std::optional<const AstNode*> context_node,
      const ReturnTypeAnnotation* return_type, TypeAnnotationFilter filter) {
    VLOG(6) << "Resolve return type: " << return_type->ToString()
            << " in context: " << ToString(parametric_context);
    XLS_ASSIGN_OR_RETURN(
        const TypeAnnotation* function_type,
        ResolveIndirectTypeAnnotations(
            parametric_context, context_node, return_type->function_type(),
            filter.Chain(TypeAnnotationFilter::BlockRecursion(return_type))));
    TypeAnnotation* result_type =
        function_type->AsAnnotation<FunctionTypeAnnotation>()->return_type();
    VLOG(6) << "Resulting return type: " << result_type->ToString();
    return result_type;
  }

  absl::StatusOr<const TypeAnnotation*> ResolveParamType(
      std::optional<const ParametricContext*> parametric_context,
      std::optional<const AstNode*> context_node,
      const ParamTypeAnnotation* param_type, TypeAnnotationFilter filter) {
    VLOG(6) << "Resolve param type: " << param_type->ToString();
    XLS_ASSIGN_OR_RETURN(
        const TypeAnnotation* function_type,
        ResolveIndirectTypeAnnotations(
            parametric_context, context_node, param_type->function_type(),
            filter.Chain(TypeAnnotationFilter::BlockRecursion(param_type))));
    const std::vector<const TypeAnnotation*>& resolved_types =
        function_type->AsAnnotation<FunctionTypeAnnotation>()->param_types();
    XLS_RET_CHECK(param_type->param_index() < resolved_types.size());
    VLOG(6) << "Resulting argument type: "
            << resolved_types[param_type->param_index()]->ToString();
    return resolved_types[param_type->param_index()];
  }

  absl::StatusOr<const TypeAnnotation*> ResolveSelfType(
      std::optional<const ParametricContext*> parametric_context,
      const SelfTypeAnnotation* self_type) {
    if (parametric_context.has_value() &&
        (*parametric_context)->self_type().has_value()) {
      return *(*parametric_context)->self_type();
    }
    return self_type->struct_ref();
  }

  absl::StatusOr<const TypeAnnotation*> ResolveSliceType(
      std::optional<const ParametricContext*> parametric_context,
      std::optional<const AstNode*> context_node,
      const SliceTypeAnnotation* slice_type, TypeAnnotationFilter filter) {
    XLS_ASSIGN_OR_RETURN(
        const TypeAnnotation* source_type,
        ResolveIndirectTypeAnnotations(parametric_context, context_node,
                                       slice_type->source_type(), filter));
    int64_t source_size = 0;
    TypeAnnotation* element_type = nullptr;
    if (source_type->IsAnnotation<ArrayTypeAnnotation>()) {
      const auto* array_type = source_type->AsAnnotation<ArrayTypeAnnotation>();
      element_type = array_type->element_type();
      XLS_ASSIGN_OR_RETURN(
          source_size,
          evaluator_.EvaluateU32OrExpr(parametric_context, array_type->dim()));
    } else {
      XLS_ASSIGN_OR_RETURN(SignednessAndBitCountResult signedness_and_bit_count,
                           GetSignednessAndBitCount(source_type));
      XLS_ASSIGN_OR_RETURN(
          bool is_signed,
          evaluator_.EvaluateBoolOrExpr(parametric_context,
                                        signedness_and_bit_count.signedness));
      if (is_signed) {
        return TypeInferenceErrorStatusForAnnotation(
            slice_type->span(), source_type, "Bit slice LHS must be unsigned.",
            file_table_);
      }
      element_type =
          CreateUnOrSnElementAnnotation(module_, slice_type->span(), false);
      XLS_ASSIGN_OR_RETURN(
          source_size,
          evaluator_.EvaluateU32OrExpr(parametric_context,
                                       signedness_and_bit_count.bit_count));
    }

    if (std::holds_alternative<Slice*>(slice_type->slice())) {
      const auto* slice = std::get<Slice*>(slice_type->slice());
      std::optional<int64_t> start;
      std::optional<int64_t> limit;
      if (slice->start() != nullptr) {
        XLS_ASSIGN_OR_RETURN(start, evaluator_.EvaluateS32OrExpr(
                                        parametric_context, slice->start()));
      }
      if (slice->limit() != nullptr) {
        XLS_ASSIGN_OR_RETURN(limit, evaluator_.EvaluateS32OrExpr(
                                        parametric_context, slice->limit()));
      }

      XLS_ASSIGN_OR_RETURN(StartAndWidth start_and_width,
                           ResolveBitSliceIndices(source_size, start, limit));
      XLS_RETURN_IF_ERROR(table_.SetSliceStartAndWidthExprs(
          slice, StartAndWidthExprs{.start = start_and_width.start,
                                    .width = start_and_width.width}));
      return module_.Make<ArrayTypeAnnotation>(
          slice_type->span(), element_type,
          module_.Make<Number>(slice_type->span(),
                               absl::StrCat(start_and_width.width),
                               NumberKind::kOther,
                               /*type_annotation=*/nullptr));
    }

    const auto* width_slice = std::get<WidthSlice*>(slice_type->slice());
    StartAndWidthExprs start_and_width;
    absl::StatusOr<int64_t> constexpr_start =
        evaluator_.EvaluateU32OrExpr(parametric_context, width_slice->start());
    if (constexpr_start.ok()) {
      start_and_width.start = *constexpr_start;
    } else {
      start_and_width.start = width_slice->start();
    }

    XLS_ASSIGN_OR_RETURN(
        const TypeAnnotation* width_type,
        ResolveIndirectTypeAnnotations(parametric_context, context_node,
                                       width_slice->width(), filter));
    absl::StatusOr<SignednessAndBitCountResult> width_signedness_and_bit_count =
        GetSignednessAndBitCount(width_type);
    if (!width_signedness_and_bit_count.ok()) {
      return TypeInferenceErrorStatusForAnnotation(
          slice_type->span(), width_type,
          "A bits type is required for a width-based slice.", file_table_);
    }
    XLS_ASSIGN_OR_RETURN(
        int64_t width,
        evaluator_.EvaluateU32OrExpr(
            parametric_context, width_signedness_and_bit_count->bit_count));
    start_and_width.width = width;

    if (width > source_size) {
      // The width is actually a type annotation in the source code, so using a
      // width larger than the source is considered an error by both v1 and v2.
      return TypeInferenceErrorStatus(
          slice_type->span(), nullptr,
          absl::StrCat("Slice range out of bounds for array of size ",
                       source_size),
          file_table_);
    }

    if (constexpr_start.ok()) {
      // If start index is a signed value or a negative number literal it would
      // have a signed type annotation which contradicts with the type
      // annotation of a widthslice `uN[width]` and it would have been caught
      // earlier at the unification of the index itself, so start index is
      // expected to be unsigned, and the only reason that constexpr_start may
      // be negative is that the value being evaluated is a uint64_t with MSB
      // set, which overflows when casted to int64_t. It is obvious that a start
      // index of 2^63 or greater is always out of range.
      if (*constexpr_start < 0 || *constexpr_start > source_size ||
          width > source_size - *constexpr_start) {
        // In v2, if the start happens to be constexpr and makes the width too
        // far, there is an added warning that is not in v1.
        warning_collector_.Add(
            slice_type->span(), WarningKind::kWidthSliceOutOfRange,
            absl::StrCat("Slice range out of bounds for array of size ",
                         source_size));
      }
    }

    XLS_RETURN_IF_ERROR(
        table_.SetSliceStartAndWidthExprs(width_slice, start_and_width));
    return width_type;
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
           absl::c_any_of(
               *annotations, [this](const TypeAnnotation* annotation) {
                 TypeInferenceFlag flag = table_.GetAnnotationFlag(annotation);
                 return !flag.HasNonExplicitTypeSemantics();
               });
  }

  // Helper for `ReplaceIndirectTypeAnnotations`. A visitor instance is intended
  // to be used once, and the result of handling that one annotation can be
  // obtained by calling `result()`.
  class ReplaceIndirectTypeAnnotationsVisitor
      : public AstNodeVisitorWithDefault {
   public:
    ReplaceIndirectTypeAnnotationsVisitor(
        Module& module, const InferenceTable& table, StatefulResolver& resolver,
        std::optional<const ParametricContext*> parametric_context,
        std::optional<const AstNode*> context_node, TypeAnnotationFilter filter)
        : module_(module),
          table_(table),
          resolver_(resolver),
          parametric_context_(parametric_context),
          context_node_(context_node),
          filter_(filter) {}

    absl::Status HandleTypeVariableTypeAnnotation(
        const TypeVariableTypeAnnotation* variable_type_annotation) override {
      const bool require_bits_like =
          table_.GetAnnotationFlag(variable_type_annotation)
              .HasFlag(TypeInferenceFlag::kBitsLikeType);
      XLS_ASSIGN_OR_RETURN(
          result_,
          resolver_.ResolveAndUnifyTypeAnnotations(
              parametric_context_, context_node_,
              variable_type_annotation->type_variable(),
              variable_type_annotation->span(), filter_, require_bits_like));
      return absl::OkStatus();
    }

    absl::Status HandleMemberTypeAnnotation(
        const MemberTypeAnnotation* member_type) override {
      XLS_ASSIGN_OR_RETURN(result_, resolver_.ResolveMemberType(
                                        parametric_context_, context_node_,
                                        member_type, filter_));
      VLOG(5) << "Member type expansion for: " << member_type->member_name()
              << " yielded: " << result_->ToString();
      return absl::OkStatus();
    }

    absl::Status HandleElementTypeAnnotation(
        const ElementTypeAnnotation* element_type) override {
      XLS_ASSIGN_OR_RETURN(result_, resolver_.ResolveElementType(
                                        parametric_context_, context_node_,
                                        element_type, filter_));
      return absl::OkStatus();
    }

    absl::Status HandleReturnTypeAnnotation(
        const ReturnTypeAnnotation* return_type) override {
      XLS_ASSIGN_OR_RETURN(result_, resolver_.ResolveReturnType(
                                        parametric_context_, context_node_,
                                        return_type, filter_));
      return absl::OkStatus();
    }

    absl::Status HandleParamTypeAnnotation(
        const ParamTypeAnnotation* param_type) override {
      if (filter_.Filter(param_type)) {
        result_ = module_.Make<AnyTypeAnnotation>();
        return absl::OkStatus();
      }
      XLS_ASSIGN_OR_RETURN(result_, resolver_.ResolveParamType(
                                        parametric_context_, context_node_,
                                        param_type, filter_));
      return absl::OkStatus();
    }

    absl::Status HandleSelfTypeAnnotation(
        const SelfTypeAnnotation* self_type) override {
      XLS_ASSIGN_OR_RETURN(
          result_, resolver_.ResolveSelfType(parametric_context_, self_type));
      XLS_RET_CHECK(result_ != nullptr);
      return absl::OkStatus();
    }

    absl::Status HandleSliceTypeAnnotation(
        const SliceTypeAnnotation* slice_type) override {
      XLS_ASSIGN_OR_RETURN(result_, resolver_.ResolveSliceType(
                                        parametric_context_, context_node_,
                                        slice_type, filter_));
      return absl::OkStatus();
    }

    // The result is either `nullopt`, if a direct annotation was visited; or
    // the direct replacement, if an indirect annotation was visited.
    std::optional<AstNode*> result() const {
      return result_ == nullptr
                 ? std::nullopt
                 : std::make_optional(const_cast<TypeAnnotation*>(result_));
    }

   private:
    Module& module_;
    const InferenceTable& table_;
    StatefulResolver& resolver_;
    std::optional<const ParametricContext*> parametric_context_;
    std::optional<const AstNode*> context_node_;
    TypeAnnotationFilter filter_;
    const TypeAnnotation* result_ = nullptr;
  };

  // The "replace" function for an AstCloner that replaces indirect annotations
  // with their resolved & unified versions.
  absl::StatusOr<std::optional<AstNode*>> ReplaceIndirectTypeAnnotations(
      const AstNode* node,
      std::optional<const ParametricContext*> parametric_context,
      std::optional<const AstNode*> context_node,
      const TypeAnnotation* annotation, TypeAnnotationFilter filter) {
    ReplaceIndirectTypeAnnotationsVisitor visitor(
        module_, table_, *this, parametric_context, context_node, filter);
    XLS_RETURN_IF_ERROR(node->Accept(&visitor));
    return visitor.result();
  }

  // The "replace" function for an AstCloner that replaces type aliases with the
  // type they equate to. There are two axes of descent here: this function
  // itself iteratively unwraps chains of type aliases that equate to other type
  // aliases, while the AstCloner recursively applies this function so that if
  // the type alias usage is in a descendant of the annotation being cloned,
  // that will be discovered and handled.
  absl::StatusOr<std::optional<AstNode*>> ReplaceTypeAliasWithTarget(
      const AstNode* node) {
    if (node->kind() != AstNodeKind::kTypeAnnotation) {
      return std::nullopt;
    }

    bool replaced_anything = false;
    std::optional<const TypeAnnotation*> latest;
    if (node->kind() == AstNodeKind::kTypeAnnotation) {
      latest = down_cast<const TypeAnnotation*>(node);
    }
    while (latest.has_value() &&
           (*latest)->IsAnnotation<TypeRefTypeAnnotation>()) {
      const auto* type_ref_annotation =
          (*latest)->AsAnnotation<TypeRefTypeAnnotation>();
      if (std::holds_alternative<ColonRef*>(
              type_ref_annotation->type_ref()->type_definition())) {
        // `ColonRef` needs specific handling, because
        // `TypeDefinitionGetNameDef` will not do what we want for that.
        const ColonRef* colon_ref = std::get<ColonRef*>(
            type_ref_annotation->type_ref()->type_definition());
        std::optional<const AstNode*> target =
            table_.GetColonRefTarget(colon_ref);
        if (target.has_value() &&
            (*target)->kind() == AstNodeKind::kTypeAnnotation) {
          latest = down_cast<const TypeAnnotation*>(*target);
        } else {
          latest = table_.GetTypeAnnotation(colon_ref);
          // We expect the ColonRef in this context to be a type alias, so it
          // must have a TypeRefTypeAnnotation.
          if (latest.has_value() &&
              (*latest)->annotation_kind() != TypeAnnotationKind::kTypeRef) {
            return NotATypeErrorStatus(*(colon_ref)->GetSpan(), colon_ref,
                                       file_table_);
          }
        }
      } else {
        const TypeDefinition& type_def =
            type_ref_annotation->type_ref()->type_definition();
        // If the simplified type of a type alias is known, use it.
        if (std::holds_alternative<TypeAlias*>(type_def)) {
          if (std::optional<const TypeAnnotation*> cached =
                  simplified_type_annotation_cache_.GetSimplifiedypeAnnotation(
                      std::nullopt, std::get<TypeAlias*>(type_def))) {
            return const_cast<TypeAnnotation*>(*cached);
          }
        }
        latest = table_.GetTypeAnnotation(
            ToAstNode(TypeDefinitionGetNameDef(type_def)));
      }

      // If the TRTA unwrapped in this iteration specified parametrics, add them
      // to the unwrapped type.
      if (latest.has_value() &&
          (*latest)->IsAnnotation<TypeRefTypeAnnotation>() &&
          !type_ref_annotation->parametrics().empty()) {
        latest = (*latest)->owner()->Make<TypeRefTypeAnnotation>(
            (*latest)->span(),
            (*latest)->AsAnnotation<TypeRefTypeAnnotation>()->type_ref(),
            type_ref_annotation->parametrics());
      }

      if (latest.has_value()) {
        node = const_cast<TypeAnnotation*>(*latest);
        replaced_anything = true;
      }
    }
    return replaced_anything ? std::make_optional(const_cast<AstNode*>(node))
                             : std::nullopt;
  }

  // Replaces any references to foreign constants in `annotation`, i.e.
  // constants that are not owned by annotation->owner(). In the returned
  // annotation, any such references have been replaced with fabricated literal
  // values owned by annotation->owner().
  absl::StatusOr<const TypeAnnotation*> ReplaceForeignConstantRefs(
      const TypeAnnotation* annotation) {
    // Find all the foreign name refs in `annotation`.
    FreeVariables vars =
        GetFreeVariablesByLambda(annotation, [&](const NameRef& ref) -> bool {
          if (!std::holds_alternative<const NameDef*>(ref.name_def())) {
            return false;
          }
          const auto* def = std::get<const NameDef*>(ref.name_def());
          if (def->owner() == &module_) {
            return false;
          }
          if (def->parent() != nullptr &&
              def->parent()->kind() == AstNodeKind::kConstantDef) {
            return true;
          }
          return false;
        });

    // Nothing to do if there are no foreign refs.
    if (vars.name_refs().empty()) {
      return annotation;
    }

    // Clone `annotation` and replace the foreign refs with literals.
    XLS_ASSIGN_OR_RETURN(
        AstNode * result,
        table_.Clone(
            annotation,
            [&](const AstNode* node, Module* new_module,
                const absl::flat_hash_map<const AstNode*, AstNode*>&)
                -> absl::StatusOr<std::optional<AstNode*>> {
              if (node->kind() == AstNodeKind::kTypeRef) {
                return const_cast<AstNode*>(node);
              }
              if (node->kind() != AstNodeKind::kNameRef ||
                  !vars.name_refs().contains(down_cast<const NameRef*>(node))) {
                return std::nullopt;
              }

              const NameRef* ref = down_cast<const NameRef*>(node);
              const NameDef* def = std::get<const NameDef*>(ref->name_def());
              XLS_ASSIGN_OR_RETURN(TypeInfo * ti,
                                   import_data_.GetRootTypeInfoForNode(def));
              XLS_ASSIGN_OR_RETURN(InterpValue value, ti->GetConstExpr(def));
              if (value.IsTuple()) {
                return std::nullopt;
              }

              auto* result = new_module->Make<Number>(
                  Span::None(), value.ToString(/*humanize=*/true),
                  NumberKind::kOther, nullptr, /*in_parens=*/false,
                  /*leave_span_intact=*/true);
              return result;
            }));

    return down_cast<const TypeAnnotation*>(result);
  }

  Module& module_;
  InferenceTable& table_;
  const FileTable& file_table_;
  UnificationErrorGenerator& error_generator_;
  Evaluator& evaluator_;
  ParametricStructInstantiator& parametric_struct_instantiator_;
  TypeSystemTracer& tracer_;
  WarningCollector& warning_collector_;
  ImportData& import_data_;
  SimplifiedTypeAnnotationCache& simplified_type_annotation_cache_;
  ResolverErrorHandler error_handler_;
  std::function<absl::Status(std::optional<const ParametricContext*>,
                             const Invocation*)>
      invocation_converter_;
  absl::flat_hash_map<const NameDef*, const TypeAnnotation*> temp_cache_;
};

// A resolver implementation that creates and uses an ad hoc stateful resolver
// for each external request that is made, so that all internal resolution
// within the processing of that external request can benefit from shared
// temporary state.
class StatelessResolver : public TypeAnnotationResolver {
 public:
  StatelessResolver(
      Module& module, InferenceTable& table, const FileTable& file_table,
      UnificationErrorGenerator& error_generator, Evaluator& evaluator,
      ParametricStructInstantiator& parametric_struct_instantiator,
      TypeSystemTracer& tracer, WarningCollector& warning_collector,
      ImportData& import_data,
      SimplifiedTypeAnnotationCache& simplified_type_annotation_cache,
      ResolverErrorHandler error_handler,
      std::function<absl::Status(std::optional<const ParametricContext*>,
                                 const Invocation*)>
          invocation_converter)
      : module_(module),
        table_(table),
        file_table_(file_table),
        error_generator_(error_generator),
        evaluator_(evaluator),
        parametric_struct_instantiator_(parametric_struct_instantiator),
        tracer_(tracer),
        warning_collector_(warning_collector),
        import_data_(import_data),
        simplified_type_annotation_cache_(simplified_type_annotation_cache),
        error_handler_(error_handler),
        invocation_converter_(std::move(invocation_converter)) {}

  absl::StatusOr<std::optional<const TypeAnnotation*>>
  ResolveAndUnifyTypeAnnotationsForNode(
      std::optional<const ParametricContext*> parametric_context,
      const AstNode* node, TypeAnnotationFilter filter) override {
    return CreateStatefulResolver()->ResolveAndUnifyTypeAnnotationsForNode(
        parametric_context, node, filter);
  }

  absl::StatusOr<const TypeAnnotation*> ResolveAndUnifyTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      std::optional<const AstNode*> context_node, const NameRef* type_variable,
      const Span& span, TypeAnnotationFilter filter,
      bool require_bits_like) final {
    return CreateStatefulResolver()->ResolveAndUnifyTypeAnnotations(
        parametric_context, context_node, type_variable, span, filter,
        require_bits_like);
  }

  absl::StatusOr<const TypeAnnotation*> ResolveAndUnifyTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      std::optional<const AstNode*> context_node,
      std::vector<const TypeAnnotation*> annotations, const Span& span,
      TypeAnnotationFilter filter, bool require_bits_like,
      bool* used_error_handler) final {
    return CreateStatefulResolver()->ResolveAndUnifyTypeAnnotations(
        parametric_context, context_node, annotations, span, filter,
        require_bits_like, used_error_handler);
  }

  absl::StatusOr<const TypeAnnotation*> ResolveIndirectTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      std::optional<const AstNode*> context_node,
      const TypeAnnotation* annotation, TypeAnnotationFilter filter) override {
    return CreateStatefulResolver()->ResolveIndirectTypeAnnotations(
        parametric_context, context_node, annotation, filter);
  }

  absl::Status ResolveIndirectTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      std::optional<const AstNode*> context_node,
      std::vector<const TypeAnnotation*>& annotations,
      TypeAnnotationFilter filter) override {
    return CreateStatefulResolver()->ResolveIndirectTypeAnnotations(
        parametric_context, context_node, annotations, filter);
  }

  absl::StatusOr<const TypeAnnotation*> ResolveTypeRefs(
      std::optional<const ParametricContext*> parametric_context,
      std::optional<const AstNode*> context_node,
      const TypeAnnotation* annotation) override {
    return CreateStatefulResolver()->ResolveTypeRefs(parametric_context,
                                                     context_node, annotation);
  }

 private:
  std::unique_ptr<TypeAnnotationResolver> CreateStatefulResolver() {
    return std::make_unique<StatefulResolver>(
        module_, table_, file_table_, error_generator_, evaluator_,
        parametric_struct_instantiator_, tracer_, warning_collector_,
        import_data_, simplified_type_annotation_cache_, error_handler_,
        invocation_converter_);
  }

  Module& module_;
  InferenceTable& table_;
  const FileTable& file_table_;
  UnificationErrorGenerator& error_generator_;
  Evaluator& evaluator_;
  ParametricStructInstantiator& parametric_struct_instantiator_;
  TypeSystemTracer& tracer_;
  WarningCollector& warning_collector_;
  ImportData& import_data_;
  SimplifiedTypeAnnotationCache& simplified_type_annotation_cache_;
  ResolverErrorHandler error_handler_;
  std::function<absl::Status(std::optional<const ParametricContext*>,
                             const Invocation*)>
      invocation_converter_;
};

}  // namespace

std::unique_ptr<TypeAnnotationResolver> TypeAnnotationResolver::Create(
    Module& module, InferenceTable& table, const FileTable& file_table,
    UnificationErrorGenerator& error_generator, Evaluator& evaluator,
    ParametricStructInstantiator& parametric_struct_instantiator,
    TypeSystemTracer& tracer, WarningCollector& warning_collector,
    ImportData& import_data,
    SimplifiedTypeAnnotationCache& simplified_type_annotation_cache,
    ResolverErrorHandler error_handler,
    std::function<absl::Status(std::optional<const ParametricContext*>,
                               const Invocation*)>
        invocation_converter) {
  return std::make_unique<StatelessResolver>(
      module, table, file_table, error_generator, evaluator,
      parametric_struct_instantiator, tracer, warning_collector, import_data,
      simplified_type_annotation_cache, error_handler,
      std::move(invocation_converter));
}

}  // namespace xls::dslx
