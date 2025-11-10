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

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/builtin_stubs_utils.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system_v2/import_utils.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/inference_table_converter.h"
#include "xls/dslx/type_system_v2/parametric_struct_instantiator.h"
#include "xls/dslx/type_system_v2/type_annotation_resolver.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"

namespace xls::dslx {
namespace {

class FunctionResolverImpl : public FunctionResolver {
 public:
  FunctionResolverImpl(
      Module& module, ImportData& import_data, InferenceTable& table,
      InferenceTableConverter& converter,
      TypeAnnotationResolver& type_annotation_resolver,
      ParametricStructInstantiator& parametric_struct_instantiator)
      : module_(module),
        file_table_(*module.file_table()),
        import_data_(import_data),
        table_(table),
        converter_(converter),
        type_annotation_resolver_(type_annotation_resolver),
        parametric_struct_instantiator_(parametric_struct_instantiator) {}

  absl::StatusOr<const FunctionAndTargetObject> ResolveFunction(
      const Expr* callee, std::optional<const Function*> caller_function,
      std::optional<const ParametricContext*> caller_context) override {
    const AstNode* function_node = nullptr;
    std::optional<Expr*> target_object;
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
          std::optional<const TypeAnnotation*> target_object_type,
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

      std::optional<Impl*> impl = struct_or_proc_ref->def->impl();
      XLS_RET_CHECK(impl.has_value());
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

    if (function_node != nullptr) {
      const auto* fn = dynamic_cast<const Function*>(function_node);
      if (fn == nullptr) {
        return TypeInferenceErrorStatus(
            callee->span(), nullptr,
            absl::Substitute("Invocation callee `$0` is not a function",
                             callee->ToString()),
            file_table_);
      }
      return FunctionAndTargetObject{fn, target_object, target_struct_context};
    }
    return TypeInferenceErrorStatus(
        callee->span(), nullptr,
        "An invocation callee must be a function, with a possible scope "
        "indicated using `::` or `.`",
        file_table_);
  }

 private:
  Module& module_;
  const FileTable& file_table_;
  ImportData& import_data_;
  InferenceTable& table_;
  InferenceTableConverter& converter_;
  TypeAnnotationResolver& type_annotation_resolver_;
  ParametricStructInstantiator& parametric_struct_instantiator_;
};

}  // namespace

std::unique_ptr<FunctionResolver> CreateFunctionResolver(
    Module& module, ImportData& import_data, InferenceTable& table,
    InferenceTableConverter& converter,
    TypeAnnotationResolver& type_annotation_resolver,
    ParametricStructInstantiator& parametric_struct_instantiator) {
  return std::make_unique<FunctionResolverImpl>(
      module, import_data, table, converter, type_annotation_resolver,
      parametric_struct_instantiator);
}

}  // namespace xls::dslx
