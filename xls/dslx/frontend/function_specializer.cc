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
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/module.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"

namespace xls::dslx {

namespace {

// SyntheticSpanAllocator
// Assigns deterministic synthetic Spans to fabricated specialized-function ASTs
// so the function span encloses all child spans and diagnostics can reference positions.
class SyntheticSpanAllocator {
 public:
  SyntheticSpanAllocator(Module* module, const Function* source_function,
                         std::string_view specialized_name)
      : module_(module),
        file_table_(module != nullptr ? module->file_table() : nullptr) {
    if (file_table_ != nullptr) {
      synthetic_file_ = file_table_->GetOrCreate(absl::StrFormat(
          "<specialization:%s:%s@%p>", module_->name(), specialized_name,
          static_cast<const void*>(source_function)));
    }
  }

  bool enabled() const { return file_table_ != nullptr; }

  absl::StatusOr<Span> Assign(AstNode* node) {
    if (node == nullptr || file_table_ == nullptr) {
      return Span::Fake();
    }
    return AssignInternal(node);
  }

  Span Enclose(absl::Span<const Span> spans) const {
    if (!enabled() || spans.empty()) {
      return Span::Fake();
    }
    Span result = spans.front();
    for (size_t i = 1; i < spans.size(); ++i) {
      result = MergeSpans(result, spans[i]);
    }
    return result;
  }

 private:
  absl::StatusOr<Span> AssignInternal(AstNode* node) {
    std::optional<Span> aggregate;
    for (AstNode* child : node->GetChildren(/*want_types=*/true)) {
      if (child == nullptr) {
        continue;
      }
      XLS_ASSIGN_OR_RETURN(Span child_span, AssignInternal(child));
      if (!aggregate.has_value()) {
        aggregate = child_span;
      } else {
        aggregate = MergeSpans(*aggregate, child_span);
      }
    }

    Span result = aggregate.has_value() ? *aggregate : AllocateLeafSpan();
    XLS_RETURN_IF_ERROR(ApplySpan(node, result));
    return result;
  }

  Span MergeSpans(const Span& a, const Span& b) const {
    Pos start = a.start() < b.start() ? a.start() : b.start();
    Pos limit = a.limit() < b.limit() ? b.limit() : a.limit();
    return Span(start, limit);
  }

  Span AllocateLeafSpan() {
    Pos start(synthetic_file_, next_line_, 0);
    Pos limit(synthetic_file_, next_line_, 1);
    ++next_line_;
    return Span(start, limit);
  }

  absl::Status ApplySpan(AstNode* node, const Span& span) {
    if (auto* expr = dynamic_cast<Expr*>(node)) {
      const_cast<Span&>(expr->span()) = span;
      return absl::OkStatus();
    }
    if (auto* _ = dynamic_cast<Statement*>(node)) {
      // Statement span derives from wrapped node; nothing to assign.
      return absl::OkStatus();
    }
    if (auto* type_alias = dynamic_cast<TypeAlias*>(node)) {
      const_cast<Span&>(type_alias->span()) = span;
      return absl::OkStatus();
    }
    if (auto* const_assert = dynamic_cast<ConstAssert*>(node)) {
      const_cast<Span&>(const_assert->span()) = span;
      return absl::OkStatus();
    }
    if (auto* type = dynamic_cast<TypeAnnotation*>(node)) {
      const_cast<Span&>(type->span()) = span;
      return absl::OkStatus();
    }
    if (auto* type_ref = dynamic_cast<TypeRef*>(node)) {
      const_cast<Span&>(type_ref->span()) = span;
      return absl::OkStatus();
    }
    if (auto* slice = dynamic_cast<Slice*>(node)) {
      const_cast<Span&>(slice->span()) = span;
      return absl::OkStatus();
    }
    if (auto* width_slice = dynamic_cast<WidthSlice*>(node)) {
      const_cast<Span&>(width_slice->span()) = span;
      return absl::OkStatus();
    }
    if (auto* name_def = dynamic_cast<NameDef*>(node)) {
      const_cast<Span&>(name_def->span()) = span;
      return absl::OkStatus();
    }
    if (auto* name_def_tree = dynamic_cast<NameDefTree*>(node)) {
      const_cast<Span&>(name_def_tree->span()) = span;
      return absl::OkStatus();
    }
    if (auto* wildcard = dynamic_cast<WildcardPattern*>(node)) {
      const_cast<Span&>(wildcard->span()) = span;
      return absl::OkStatus();
    }
    if (auto* rest_of_tuple = dynamic_cast<RestOfTuple*>(node)) {
      const_cast<Span&>(rest_of_tuple->span()) = span;
      return absl::OkStatus();
    }
    if (auto* param = dynamic_cast<Param*>(node)) {
      const_cast<Span&>(param->span()) = span;
      return absl::OkStatus();
    }
    if (auto* let_node = dynamic_cast<Let*>(node)) {
      const_cast<Span&>(let_node->span()) = span;
      return absl::OkStatus();
    }
    if (auto* match_arm = dynamic_cast<MatchArm*>(node)) {
      const_cast<Span&>(match_arm->span()) = span;
      return absl::OkStatus();
    }
    if (auto* param_binding = dynamic_cast<ParametricBinding*>(node)) {
      const_cast<Span&>(param_binding->name_def()->span()) = span;
      return absl::OkStatus();
    }
    // Other nodes are not expected to be encountered in ApplySpan as they
    // cannot be part of a function body.
    return absl::InvalidArgumentError(absl::StrFormat(
        "AST node %s encountered in ApplySpan", node->ToString()));
  }

  Module* module_;
  FileTable* file_table_;
  Fileno synthetic_file_ = Fileno(0);
  int64_t next_line_ = 0;
};

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
  if (text.empty()) {
    text = "0x0";
  }
  return module->Make<Number>(span, text, NumberKind::kOther,
                              /*type=*/nullptr);
}

}  // namespace

absl::StatusOr<Function*> InsertFunctionSpecialization(
    Function* source_function, const ParametricEnv& param_env,
    std::string_view specialized_name) {
  XLS_RET_CHECK_NE(source_function, nullptr)
      << "InsertFunctionSpecialization requires a non-null source function";
  Module* module = source_function->owner();
  XLS_RET_CHECK_NE(module, nullptr) << absl::StrFormat(
      "Source function %s has no owning module", source_function->identifier());

  if (!source_function->IsParametric()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Source function %s is not parametric", source_function->identifier()));
  }

  auto binding_values =
      std::make_shared<absl::flat_hash_map<const NameDef*, InterpValue>>();
  auto binding_types =
      std::make_shared<absl::flat_hash_map<const NameDef*, TypeAnnotation*>>();
  for (ParametricBinding* binding : source_function->parametric_bindings()) {
    std::optional<InterpValue> value = param_env.GetValue(binding->name_def());
    if (!value.has_value()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Parametric binding %s missing from environment when specializing %s",
          binding->identifier(), source_function->identifier()));
    }
    binding_values->emplace(binding->name_def(), *value);
    if (binding->type_annotation() != nullptr) {
      binding_types->emplace(binding->name_def(), binding->type_annotation());
    }
  }

  auto make_replacer = [binding_values, binding_types](
                           const absl::flat_hash_map<const NameDef*, NameDef*>*
                               param_name_replacements) -> CloneReplacer {
    return [binding_values, binding_types, param_name_replacements](
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
            return std::optional<AstNode*>(target_module->Make<NameRef>(
                name_ref->span(), name_ref->identifier(), replacement,
                name_ref->in_parens()));
          }
        }
        auto binding_it = binding_values->find(def);
        if (binding_it != binding_values->end()) {
          XLS_ASSIGN_OR_RETURN(
              Number * literal,
              CreateLiteralFromValue(target_module, name_ref->span(),
                                     binding_it->second));
          auto type_it = binding_types->find(def);
          if (type_it != binding_types->end() && type_it->second != nullptr) {
            XLS_ASSIGN_OR_RETURN(TypeAnnotation * cloned_type,
                                 CloneNode<TypeAnnotation>(type_it->second));
            literal->SetTypeAnnotation(cloned_type);
          }
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
    XLS_ASSIGN_OR_RETURN(
        Param * cloned_param,
        CloneNode<Param>(param,
                         make_replacer(/*param_name_replacements=*/nullptr)));
    param_name_replacements.emplace(param->name_def(),
                                    cloned_param->name_def());
    new_params.push_back(cloned_param);
  }

  TypeAnnotation* new_return_type = nullptr;
  if (source_function->return_type() != nullptr) {
    XLS_ASSIGN_OR_RETURN(
        new_return_type,
        CloneNode<TypeAnnotation>(source_function->return_type(),
                                  make_replacer(&param_name_replacements)));
  }

  XLS_ASSIGN_OR_RETURN(
      StatementBlock * new_body,
      CloneNode<StatementBlock>(source_function->body(),
                                make_replacer(&param_name_replacements)));

  NameDef* new_name_def =
      module->Make<NameDef>(Span::Fake(), std::string(specialized_name),
                            /*definer=*/nullptr);

  SyntheticSpanAllocator span_allocator(module, source_function,
                                        specialized_name);
  std::vector<Span> function_spans;
  {
    XLS_ASSIGN_OR_RETURN(Span s, span_allocator.Assign(new_name_def));
    function_spans.push_back(s);
  }
  for (Param* param : new_params) {
    XLS_ASSIGN_OR_RETURN(Span s, span_allocator.Assign(param));
    function_spans.push_back(s);
  }
  if (new_return_type != nullptr) {
    XLS_ASSIGN_OR_RETURN(Span s, span_allocator.Assign(new_return_type));
    function_spans.push_back(s);
  }
  {
    XLS_ASSIGN_OR_RETURN(Span s, span_allocator.Assign(new_body));
    function_spans.push_back(s);
  }

  Span function_span = source_function->span();
  if (span_allocator.enabled()) {
    function_span = span_allocator.Enclose(function_spans);
  }

  Function* new_function = module->Make<Function>(
      function_span, new_name_def,
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
