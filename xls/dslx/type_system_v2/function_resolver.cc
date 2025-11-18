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

// The result of resolving the target of a function call. If the `target_object`
// is specified, then it is an instance method being invoked on `target_object`.
// Otherwise, it is a static function which may or may not be a member.

#include "xls/dslx/type_system_v2/function_resolver.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/builtin_stubs_utils.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/import_utils.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/inference_table_converter.h"
#include "xls/dslx/type_system_v2/parametric_struct_instantiator.h"
#include "xls/dslx/type_system_v2/populate_table_visitor.h"
#include "xls/dslx/type_system_v2/trait_deriver.h"
#include "xls/dslx/type_system_v2/type_annotation_resolver.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"
#include "xls/dslx/type_system_v2/type_system_tracer.h"

namespace xls::dslx {
namespace {

class FunctionResolverImpl : public FunctionResolver {
 public:
  FunctionResolverImpl(
      Module& module, ImportData& import_data, InferenceTable& table,
      InferenceTableConverter& converter,
      TypeAnnotationResolver& type_annotation_resolver,
      ParametricStructInstantiator& parametric_struct_instantiator,
      std::optional<TraitDeriver*> trait_deriver, TypeSystemTracer& tracer)
      : module_(module),
        file_table_(*module.file_table()),
        import_data_(import_data),
        table_(table),
        converter_(converter),
        type_annotation_resolver_(type_annotation_resolver),
        parametric_struct_instantiator_(parametric_struct_instantiator),
        trait_deriver_(trait_deriver),
        tracer_(tracer) {}

  absl::StatusOr<const FunctionAndTargetObject> ResolveFunction(
      const Expr* callee, std::optional<const Function*> caller_function,
      std::optional<const ParametricContext*> caller_context) override {
    const AstNode* function_node = nullptr;
    std::optional<Expr*> target_object;
    std::optional<const TypeAnnotation*> target_object_type;
    std::optional<const ParametricContext*> target_struct_context =
        caller_context.has_value() && (*caller_context)->is_struct()
            ? caller_context
            : std::nullopt;

    if (callee->kind() == AstNodeKind::kColonRef) {
      const auto* colon_ref = down_cast<const ColonRef*>(callee);
      std::optional<const AstNode*> target =
          table_.GetColonRefTarget(colon_ref);
      if (target.has_value()) {
        function_node = *target;
      }
    } else if (callee->kind() == AstNodeKind::kNameRef) {
      // Either a local function or a built-in function call.
      const auto* name_ref = down_cast<const NameRef*>(callee);
      if (std::holds_alternative<const NameDef*>(name_ref->name_def())) {
        const NameDef* def = std::get<const NameDef*>(name_ref->name_def());
        function_node = def->definer();
      } else if (std::holds_alternative<BuiltinNameDef*>(
                     name_ref->name_def())) {
        const Module* module = &module_;
        if (module->name() != kBuiltinStubsModuleName) {
          XLS_ASSIGN_OR_RETURN(module, import_data_.GetBuiltinStubsModule());
        }
        // Look it up in our module
        BuiltinNameDef* def = std::get<BuiltinNameDef*>(name_ref->name_def());
        auto fn_name = def->identifier();
        std::optional<Function*> builtin_fn = module->GetFunction(fn_name);
        if (builtin_fn.has_value()) {
          function_node = *builtin_fn;
        } else {
          return TypeInferenceErrorStatus(
              name_ref->span(), nullptr,
              absl::Substitute("Cannot find built-in method `$0`", fn_name),
              file_table_);
        }
      }
    } else if (callee->kind() == AstNodeKind::kAttr) {
      const auto* attr = down_cast<const Attr*>(callee);

      // Disallow the form `module.fn()`. If they really want to do that, it
      // should be `module::fn()`.
      if (attr->lhs()->kind() == AstNodeKind::kNameRef &&
          IsImportedModuleReference(down_cast<NameRef*>(attr->lhs()))) {
        return TypeInferenceErrorStatus(
            attr->span(), nullptr,
            "An invocation callee must be a function, with a possible scope.",
            file_table_);
      }

      XLS_RETURN_IF_ERROR(converter_.ConvertSubtree(
          attr->lhs(), caller_function, caller_context));
      target_object = attr->lhs();
      XLS_ASSIGN_OR_RETURN(
          target_object_type,
          type_annotation_resolver_.ResolveAndUnifyTypeAnnotationsForNode(
              caller_context, *target_object));
      XLS_ASSIGN_OR_RETURN(
          std::optional<StructOrProcRef> struct_or_proc_ref,
          GetStructOrProcRef(*target_object_type, import_data_));
      if (!struct_or_proc_ref.has_value()) {
        return TypeInferenceErrorStatus(
            attr->span(), nullptr,
            absl::Substitute(
                "Cannot invoke method `$0` on non-struct type `$1`",
                attr->attr(), (*target_object_type)->ToString()),
            file_table_);
      }
      if (struct_or_proc_ref->def->IsParametric()) {
        XLS_ASSIGN_OR_RETURN(
            target_struct_context,
            parametric_struct_instantiator_.GetOrCreateParametricStructContext(
                caller_context, *struct_or_proc_ref, callee));
      }

      // For a struct, the function may be in a derived trait.
      std::string function_name(attr->attr());
      if (struct_or_proc_ref->def->kind() == AstNodeKind::kStructDef) {
        // Derive the traits for the struct in the effective parametric struct
        // context. This is a no-op if already done.
        XLS_ASSIGN_OR_RETURN(
            TypeInfo * ti,
            converter_.GetTypeInfo((*target_object)->owner(), caller_context));
        std::optional<Type*> struct_type = ti->GetItem(*target_object);
        XLS_RET_CHECK(struct_type.has_value());
        XLS_RET_CHECK((*struct_type)->IsStruct());
        StructDef* struct_def = const_cast<StructDef*>(
            down_cast<const StructDef*>(struct_or_proc_ref->def));
        XLS_RETURN_IF_ERROR(DeriveTraits(target_struct_context, *struct_def,
                                         (*struct_type)->AsStruct()));

        // Now check if the function we are looking for is derived from a trait.
        const auto providing_trait_it = providing_trait_.find(
            StructFunctionKey{.struct_def = struct_def,
                              .function_name = std::string(function_name)});
        if (providing_trait_it != providing_trait_.end() &&
            providing_trait_it->second.has_value()) {
          VLOG(6) << "Function " << function_name << " is provided by trait: "
                  << (*providing_trait_it->second)->identifier();
          function_node = derived_functions_.at(TraitFunctionKey{
              .trait = *providing_trait_it->second,
              .struct_def = struct_def,
              .parametric_struct_context = target_struct_context,
              .function_name = function_name});
        }
      }

      // Non-trait impl function case.
      if (function_node == nullptr) {
        std::optional<Impl*> impl = struct_or_proc_ref->def->impl();
        if (!impl.has_value()) {
          return TypeInferenceErrorStatus(
              callee->span(), /*type=*/nullptr,
              absl::Substitute("No function `$0` on object of type: `$1`.",
                               function_name,
                               (*target_object_type)->ToString()),
              file_table_);
        }
        std::optional<Function*> instance_method =
            (*impl)->GetFunction(attr->attr());
        if (instance_method.has_value()) {
          function_node = *instance_method;
        } else {
          return TypeInferenceErrorStatusForAnnotation(
              callee->span(), *target_object_type,
              absl::Substitute(
                  "Name '$0' is not defined by the impl for struct '$1'.",
                  attr->attr(), struct_or_proc_ref->def->identifier()),
              file_table_);
        }
      }
    }

    if (function_node != nullptr) {
      const auto* fn = dynamic_cast<const Function*>(function_node);
      if (fn == nullptr) {
        return TypeInferenceErrorStatus(
            callee->span(), nullptr,
            absl::Substitute("Invocation callee `$0` is not a function",
                             callee->ToString()),
            file_table_);
      }
      if (!fn->IsCompilerDerived() &&
          (fn->parent() == nullptr || !fn->impl().has_value())) {
        // A call like `std::clog2(X)` that is a default expr for a struct
        // parametric does not count as being in that struct's context.
        target_struct_context = std::nullopt;
      }
      return FunctionAndTargetObject{fn, target_object, target_struct_context,
                                     target_object_type};
    }
    return TypeInferenceErrorStatus(
        callee->span(), nullptr,
        "An invocation callee must be a function, with a possible scope "
        "indicated using `::` or `.`",
        file_table_);
  }

 private:
  absl::StatusOr<Trait*> ResolveTrait(const Span& span, std::string_view name) {
    std::optional<Trait*> local_trait = module_.GetMember<Trait>(name);
    if (local_trait.has_value()) {
      return *local_trait;
    }
    XLS_ASSIGN_OR_RETURN(Module * builtin_stubs,
                         import_data_.GetBuiltinStubsModule());
    std::optional<Trait*> builtin_trait = builtin_stubs->GetMember<Trait>(name);
    if (builtin_trait.has_value()) {
      return *builtin_trait;
    }
    return TypeInferenceErrorStatus(
        span, /*type=*/nullptr, absl::Substitute("Unknown trait: `$0`", name),
        file_table_);
  }

  absl::Status DeriveTrait(
      std::optional<const ParametricContext*> parametric_context,
      const Span& attribute_span, StructDef& struct_def, Impl& impl,
      TypeAnnotation* actual_self_type, const StructType& struct_type,
      const Trait& trait) {
    VLOG(5) << "Deriving trait `" << trait.identifier() << "` for struct `"
            << struct_def.identifier() << "`";
    TypeSystemTrace trace =
        tracer_.TraceDeriveTrait(&trait, &struct_def, struct_type);
    for (Function* function : trait.members()) {
      VLOG(5) << "Deriving trait member `" << function->identifier() << "`";

      StructFunctionKey struct_function_key{
          .struct_def = &struct_def, .function_name = function->identifier()};
      const auto providing_trait_it =
          providing_trait_.find(struct_function_key);
      bool existed_before = providing_trait_it != providing_trait_.end();
      if (existed_before && (!providing_trait_it->second.has_value() ||
                             (*providing_trait_it->second) != &trait)) {
        return TypeInferenceErrorStatus(
            attribute_span, /*type=*/nullptr,
            absl::Substitute(
                "Attempting to derive conflicting function `$0` from trait "
                "`$1`. Struct $2 already has a function by this name from $3.",
                function->identifier(), trait.identifier(),
                struct_def.identifier(),
                providing_trait_it->second.has_value()
                    ? absl::StrCat("trait: ",
                                   (*providing_trait_it->second)->identifier())
                    : "its explicit impl"),
            file_table_);
      }

      providing_trait_.emplace_hint(providing_trait_it, struct_function_key,
                                    &trait);

      // Clone the function stub and repoint any Self type references in the
      // copy to the struct at hand. We will then turn the copied stub into a
      // full-fledged function.
      XLS_ASSIGN_OR_RETURN(
          (absl::flat_hash_map<const AstNode*, AstNode*> clones),
          CloneAstAndGetAllPairs(
              function, &module_,
              ChainCloneReplacers(
                  &PreserveTypeDefinitionsReplacer,
                  [&](const AstNode* node, Module*,
                      const absl::flat_hash_map<const AstNode*, AstNode*>&)
                      -> absl::StatusOr<std::optional<AstNode*>> {
                    if (const auto* self_type =
                            dynamic_cast<const SelfTypeAnnotation*>(node)) {
                      return module_.Make<SelfTypeAnnotation>(
                          self_type->span(), self_type->explicit_type(),
                          actual_self_type);
                    }
                    return std::nullopt;
                  })));

      Function* derived = down_cast<Function*>(clones.at(function));

      if (!existed_before) {
        // Put a stub for the derived function in the impl so that things like
        // `TypeAnnotationResolver` can deal with a `MemberTypeAnnotation`
        // referring to it.
        XLS_ASSIGN_OR_RETURN(
            AstNode * derived_stub,
            CloneAst(derived, &PreserveTypeDefinitionsReplacer));
        impl.AddMember(down_cast<Function*>(derived_stub));
      }

      derived->set_compiler_derived(true);
      derived->set_impl(&impl);
      XLS_ASSIGN_OR_RETURN(StatementBlock * body,
                           (*trait_deriver_)
                               ->DeriveFunctionBody(module_, trait, struct_def,
                                                    struct_type, *derived));
      derived->set_body(body);
      derived->SetParentage();

      std::unique_ptr<PopulateTableVisitor> visitor =
          CreatePopulateTableVisitor(&module_, &table_, &import_data_,
                                     /*typecheck_imported_module=*/nullptr);
      XLS_RETURN_IF_ERROR(visitor->PopulateFromFunction(derived));

      VLOG(5) << "Derived function: " << derived->ToString();
      derived_functions_[TraitFunctionKey{
          .trait = &trait,
          .struct_def = &struct_def,
          .parametric_struct_context = parametric_context,
          .function_name = function->identifier()}] = derived;
    }
    return absl::OkStatus();
  }

  absl::Status DeriveTraits(
      std::optional<const ParametricContext*> parametric_context,
      StructDef& struct_def, const StructType& struct_type) {
    if (parametric_contexts_with_completed_derivation_[&struct_def].contains(
            parametric_context)) {
      return absl::OkStatus();
    }

    std::optional<const Attribute*> attribute =
        GetAttribute(&struct_def, AttributeKind::kDerive);
    if (!attribute.has_value()) {
      return absl::OkStatus();
    }

    if (!trait_deriver_.has_value()) {
      return absl::UnimplementedError(
          "Trait deriver must be supplied in order to derive traits.");
    }

    // We merge the trait derivations with the user-written impl if it exists.
    // If it doesn't, we fabricate an impl. In standard Rust usage, there can
    // be separate per-trait impls, but to keep things simple we don't
    // currently support that.
    std::optional<Impl*> impl = struct_def.impl();
    if (impl.has_value()) {
      // Record which functions are provided by the user-written impl.
      for (const ImplMember& member : (*struct_def.impl())->members()) {
        if (const auto* fn = std::get_if<Function*>(&member);
            fn && !(*fn)->IsStub()) {
          providing_trait_[StructFunctionKey{
              .struct_def = &struct_def,
              .function_name = (*fn)->identifier()}] = std::nullopt;
        }
      }
    } else {
      // Create a synthetic impl if there isn't one.
      TypeAnnotation* struct_ref = module_.Make<TypeRefTypeAnnotation>(
          Span::None(), module_.Make<TypeRef>(Span::None(), &struct_def),
          std::vector<ExprOrType>(), std::nullopt);

      impl =
          module_.Make<Impl>(Span::None(), struct_ref,
                             std::vector<ImplMember>{}, struct_def.is_public());
      struct_def.set_impl(*impl);
    }

    std::vector<ExprOrType> parametrics;
    if (parametric_context.has_value()) {
      XLS_ASSIGN_OR_RETURN(
          (absl::flat_hash_map<const NameDef*, ExprOrType> parametric_map),
          table_.GetParametricValueExprs(*parametric_context));
      parametrics.reserve(parametric_map.size());
      for (const ParametricBinding* binding :
           struct_def.parametric_bindings()) {
        const auto it = parametric_map.find(binding->name_def());
        if (it == parametric_map.end()) {
          break;
        }
        parametrics.push_back(it->second);
      }
    }

    TypeAnnotation* actual_self_type = module_.Make<TypeRefTypeAnnotation>(
        Span::None(), module_.Make<TypeRef>(Span::None(), &struct_def),
        parametrics, std::nullopt);

    // Derive the traits for the struct.
    Span attribute_span = *(*attribute)->GetSpan();
    for (Attribute::Argument arg : (*attribute)->args()) {
      // This should already have been checked by the parser.
      XLS_RET_CHECK(std::holds_alternative<std::string>(arg));
      XLS_ASSIGN_OR_RETURN(
          const Trait* trait,
          ResolveTrait(attribute_span, std::get<std::string>(arg)));
      XLS_RETURN_IF_ERROR(DeriveTrait(parametric_context, attribute_span,
                                      struct_def, **impl, actual_self_type,
                                      struct_type, *trait));
    }

    return absl::OkStatus();
  }

  struct StructFunctionKey {
    const StructDef* struct_def;
    std::string function_name;

    template <typename H>
    friend H AbslHashValue(H h, const StructFunctionKey& key) {
      return H::combine(std::move(h), key.struct_def, key.function_name);
    }

    bool operator==(const StructFunctionKey& other) const = default;
    bool operator!=(const StructFunctionKey& other) const = default;
  };

  struct TraitFunctionKey {
    const Trait* trait;
    const StructDef* struct_def;
    std::optional<const ParametricContext*> parametric_struct_context;
    std::string function_name;

    template <typename H>
    friend H AbslHashValue(H h, const TraitFunctionKey& key) {
      return H::combine(std::move(h), key.trait, key.struct_def,
                        key.parametric_struct_context, key.function_name);
    }

    bool operator==(const TraitFunctionKey& other) const = default;
    bool operator!=(const TraitFunctionKey& other) const = default;
  };

  Module& module_;
  const FileTable& file_table_;
  ImportData& import_data_;
  InferenceTable& table_;
  InferenceTableConverter& converter_;
  TypeAnnotationResolver& type_annotation_resolver_;
  ParametricStructInstantiator& parametric_struct_instantiator_;
  std::optional<TraitDeriver*> trait_deriver_;

  absl::flat_hash_map<
      const StructDef*,
      absl::flat_hash_set<std::optional<const ParametricContext*>>>
      parametric_contexts_with_completed_derivation_;

  // The derived trait function object for a given trait function and target
  // struct/parametric context.
  absl::flat_hash_map<TraitFunctionKey, Function*> derived_functions_;

  // Which trait provides a given function for a given struct (value is
  // nullopt if it's in the user-written impl).
  absl::flat_hash_map<StructFunctionKey, std::optional<const Trait*>>
      providing_trait_;

  TypeSystemTracer& tracer_;
};

}  // namespace

std::unique_ptr<FunctionResolver> CreateFunctionResolver(
    Module& module, ImportData& import_data, InferenceTable& table,
    InferenceTableConverter& converter,
    TypeAnnotationResolver& type_annotation_resolver,
    ParametricStructInstantiator& parametric_struct_instantiator,
    std::optional<TraitDeriver*> trait_deriver, TypeSystemTracer& tracer) {
  return std::make_unique<FunctionResolverImpl>(
      module, import_data, table, converter, type_annotation_resolver,
      parametric_struct_instantiator, trait_deriver, tracer);
}

}  // namespace xls::dslx
