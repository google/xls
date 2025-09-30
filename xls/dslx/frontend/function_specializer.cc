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

#include "xls/dslx/frontend/function_specializer.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/module.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"

namespace xls::dslx {

namespace {

absl::StatusOr<Number*> CreateLiteralFromValue(Module* module, const Span& span,
                                               const InterpValue& value) {
  if (value.IsBool()) {
    return module->Make<Number>(span, value.IsTrue() ? "true" : "false",
                                NumberKind::kBool, /*type=*/nullptr);
  }

  if (!value.IsBits()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Cannot specialize param binding with unsupported value: %s",
        value.ToHumanString()));
  }

  XLS_ASSIGN_OR_RETURN(const Bits& bits, value.GetBits());
  std::string text = BitsToString(bits, FormatPreference::kHex);
  return module->Make<Number>(span, text, NumberKind::kOther,
                              /*type=*/nullptr);
}

}  // namespace

absl::StatusOr<Function*> InsertFunctionSpecialization(
    Function* source_function, const ParametricEnv& param_env,
    std::string_view specialized_name) {
  CHECK_NE(source_function, nullptr);
  Module* module = source_function->owner();
  CHECK_NE(module, nullptr);

  if (!source_function->IsParametric()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Source function %s is not parametric", source_function->identifier()));
  }

  auto binding_values =
      std::make_shared<absl::flat_hash_map<const NameDef*, InterpValue>>();
  for (ParametricBinding* binding : source_function->parametric_bindings()) {
    std::optional<InterpValue> value = param_env.GetValue(binding->name_def());
    if (!value.has_value()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Parametric binding %s missing from environment when specializing %s",
          binding->identifier(), source_function->identifier()));
    }
    binding_values->emplace(binding->name_def(), *value);
  }

  auto make_replacer = [binding_values](
                            const absl::flat_hash_map<const NameDef*, NameDef*>*
                                param_name_replacements) -> CloneReplacer {
    return [binding_values, param_name_replacements](
               const AstNode* original, Module* target_module,
               const absl::flat_hash_map<const AstNode*, AstNode*>& old_to_new)
               -> absl::StatusOr<std::optional<AstNode*>> {
      if (original->kind() != AstNodeKind::kNameRef) {
        return std::nullopt;
      }
      const NameRef* name_ref = down_cast<const NameRef*>(original);
      if (std::holds_alternative<const NameDef*>(name_ref->name_def())) {
        const NameDef* def = std::get<const NameDef*>(name_ref->name_def());
        if (param_name_replacements != nullptr) {
          auto param_it = param_name_replacements->find(def);
          if (param_it != param_name_replacements->end()) {
            NameDef* replacement = param_it->second;
            return std::optional<AstNode*>(
                target_module->Make<NameRef>(name_ref->span(),
                                             name_ref->identifier(),
                                             replacement, name_ref->in_parens()));
          }
        }
        auto binding_it = binding_values->find(def);
        if (binding_it != binding_values->end()) {
          XLS_ASSIGN_OR_RETURN(Number * literal,
                               CreateLiteralFromValue(target_module,
                                                      name_ref->span(),
                                                      binding_it->second));
          return std::optional<AstNode*>(literal);
        }
      }
      return std::nullopt;
    };
  };

  std::vector<Param*> new_params;
  new_params.reserve(source_function->params().size());
  absl::flat_hash_map<const NameDef*, NameDef*> param_name_replacements;
  for (Param* param : source_function->params()) {
    XLS_ASSIGN_OR_RETURN(Param * cloned_param,
                         CloneNode<Param>(param,
                                          make_replacer(/*param_name_replacements=*/nullptr)));
    param_name_replacements.emplace(param->name_def(),
                                    cloned_param->name_def());
    new_params.push_back(cloned_param);
  }

  TypeAnnotation* new_return_type = nullptr;
  if (source_function->return_type() != nullptr) {
    XLS_ASSIGN_OR_RETURN(new_return_type,
                         CloneNode<TypeAnnotation>(source_function->return_type(),
                                                   make_replacer(&param_name_replacements)));
  }

  XLS_ASSIGN_OR_RETURN(
      StatementBlock * new_body,
      CloneNode<StatementBlock>(source_function->body(),
                                make_replacer(&param_name_replacements)));

  NameDef* new_name_def = module->Make<NameDef>(
      source_function->name_def()->span(), std::string(specialized_name),
      /*definer=*/nullptr);

  Function* new_function = module->Make<Function>(
      source_function->span(), new_name_def,
      /*parametric_bindings=*/std::vector<ParametricBinding*>{}, new_params,
      new_return_type, new_body, source_function->tag(),
      source_function->is_public(), source_function->is_test_utility());
  new_name_def->set_definer(new_function);

  if (source_function->extern_verilog_module().has_value()) {
    new_function->set_extern_verilog_module(
        *source_function->extern_verilog_module());
  }
  new_function->set_disable_format(source_function->disable_format());
  XLS_RETURN_IF_ERROR(module->InsertTopAfter(source_function, new_function,
                                             /*make_collision_error=*/nullptr));

  return new_function;
}

}  // namespace xls::dslx
