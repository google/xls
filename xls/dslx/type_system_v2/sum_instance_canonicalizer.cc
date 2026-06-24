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

#include "xls/dslx/type_system_v2/sum_instance_canonicalizer.h"

#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/bindings.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system_v2/import_utils.h"

namespace xls::dslx {
namespace {

absl::StatusOr<ColonRef*> GetConstructorRef(
    const TypeAnnotation* type_annotation) {
  auto* type_ref_type_annotation =
      dynamic_cast<const TypeRefTypeAnnotation*>(type_annotation);
  XLS_RET_CHECK_NE(type_ref_type_annotation, nullptr);
  TypeDefinition type_definition =
      type_ref_type_annotation->type_ref()->type_definition();
  XLS_RET_CHECK(std::holds_alternative<ColonRef*>(type_definition));
  return std::get<ColonRef*>(type_definition);
}

// Follow syntactic edges once to distinguish values from references and
// patterns. In particular, TypeRef::GetChildren does not follow its borrowed
// definition, which may be a ColonRef that must retain its structural type.
absl::StatusOr<absl::flat_hash_set<const AstNode*>> CollectConstructors(
    const Module& module, const ImportData& import_data) {
  absl::flat_hash_set<const AstNode*> constructors;
  absl::flat_hash_set<const ColonRef*> expressions;
  absl::flat_hash_set<const AstNode*> visited;
  std::vector<const AstNode*> pending = {&module};
  while (!pending.empty()) {
    const AstNode* node = pending.back();
    pending.pop_back();
    if (!visited.insert(node).second) {
      continue;
    }

    if (const auto* invocation = dynamic_cast<const Invocation*>(node)) {
      if (const auto* constructor =
              dynamic_cast<const ColonRef*>(invocation->callee())) {
        if (!invocation->explicit_parametrics().empty()) {
          XLS_ASSIGN_OR_RETURN(std::optional<SumRef> sum_ref,
                               GetSumRefForSubject(constructor, import_data));
          if (sum_ref.has_value()) {
            return ParseErrorStatus(
                invocation->span(),
                "Explicit parametrics belong on the sum type, not the "
                "constructor; use `Name<T>::Variant(...)`.",
                import_data.file_table());
          }
        }
        XLS_ASSIGN_OR_RETURN(std::optional<SumConstructorRef> resolved,
                             ResolveSumConstructor(constructor, import_data));
        if (resolved.has_value()) {
          if (!resolved->variant->is_tuple()) {
            return TypeInferenceErrorStatus(
                invocation->span(), nullptr,
                absl::Substitute("Constructor `$0` is not callable here.",
                                 constructor->ToString()),
                import_data.file_table());
          } else {
            constructors.insert(node);
          }
        }
      }
    } else if (const auto* instance =
                   dynamic_cast<const StructInstanceBase*>(node)) {
      XLS_ASSIGN_OR_RETURN(
          std::optional<SumConstructorRef> resolved,
          ResolveSumConstructor(instance->struct_ref(), import_data));
      if (resolved.has_value()) {
        const auto* annotation =
            instance->struct_ref()->AsAnnotation<TypeRefTypeAnnotation>();
        if (!annotation->parametrics().empty()) {
          return ParseErrorStatus(
              instance->span(),
              "Explicit parametrics belong on the sum type, not the "
              "constructor; use `Name<T>::Variant { ... }`.",
              import_data.file_table());
        } else if (!resolved->variant->is_struct()) {
          return TypeInferenceErrorStatusForAnnotation(
              instance->span(), instance->struct_ref(),
              absl::Substitute(
                  "Attempted to instantiate non-struct type `$0` as a struct.",
                  instance->struct_ref()->ToString()),
              import_data.file_table());
        } else if (dynamic_cast<const SplatStructInstance*>(instance) !=
                   nullptr) {
          return TypeInferenceErrorStatusForAnnotation(
              instance->span(), instance->struct_ref(),
              "Struct-style sum constructors do not support splat syntax.",
              import_data.file_table());
        } else {
          constructors.insert(node);
        }
      }
    } else if (const auto* constructor = dynamic_cast<const ColonRef*>(node);
               constructor != nullptr && expressions.contains(constructor)) {
      XLS_ASSIGN_OR_RETURN(std::optional<SumConstructorRef> resolved,
                           ResolveSumConstructor(constructor, import_data));
      if (resolved.has_value() && resolved->variant->is_unit()) {
        constructors.insert(node);
      }
    }

    std::vector<AstNode*> children;
    const AstNode* pattern = nullptr;
    const AstNode* non_value_ref = nullptr;
    if (const auto* arm = dynamic_cast<const MatchArm*>(node)) {
      children.push_back(arm->expr());
    } else {
      children = node->GetChildren(/*want_types=*/true);
      if (const auto* let = dynamic_cast<const Let*>(node)) {
        pattern = ToAstNode(let->pattern());
      } else if (const auto* loop = dynamic_cast<const ForLoopBase*>(node)) {
        pattern = ToAstNode(loop->pattern());
      }
      if (const auto* instantiation =
              dynamic_cast<const Instantiation*>(node)) {
        non_value_ref = instantiation->callee();
      } else if (const auto* colon_ref = dynamic_cast<const ColonRef*>(node)) {
        non_value_ref = ToAstNode(colon_ref->subject());
      } else if (const auto* sum = dynamic_cast<const SumInstance*>(node)) {
        non_value_ref = sum->constructor_ref();
      } else if (const auto* proc_alias =
                     dynamic_cast<const ProcAlias*>(node)) {
        non_value_ref = ToAstNode(proc_alias->target());
      }
      // Struct references are not children, but their parametrics can contain
      // expressions. Their TypeRef still keeps its borrowed definition out of
      // this traversal.
      if (const auto* instance =
              dynamic_cast<const StructInstanceBase*>(node)) {
        children.push_back(instance->struct_ref());
      }
    }
    for (const AstNode* child : children) {
      if (child != nullptr && child != pattern) {
        pending.push_back(child);
        if (child != non_value_ref && child->kind() == AstNodeKind::kColonRef) {
          expressions.insert(absl::down_cast<const ColonRef*>(child));
        }
      }
    }
  }
  return constructors;
}

}  // namespace

absl::StatusOr<std::optional<std::unique_ptr<Module>>> CanonicalizeSumInstances(
    const Module& module, const ImportData& import_data) {
  XLS_ASSIGN_OR_RETURN(absl::flat_hash_set<const AstNode*> constructors,
                       CollectConstructors(module, import_data));
  if (constructors.empty()) {
    return std::nullopt;
  } else {
    ClonePostReplacer replacer =
        [&constructors](const AstNode* node,
                        AstNode* cloned_node) -> absl::StatusOr<AstNode*> {
      Module* target_module = cloned_node->owner();
      if (!constructors.contains(node)) {
        return cloned_node;
      } else if (node->kind() == AstNodeKind::kInvocation) {
        auto* cloned = absl::down_cast<Invocation*>(cloned_node);
        return target_module->Make<SumInstance>(
            cloned->span(), absl::down_cast<ColonRef*>(cloned->callee()),
            SumInstance::PayloadShape::kTuple,
            std::vector<Expr*>(cloned->args().begin(), cloned->args().end()),
            std::vector<SumInstance::StructPayloadFieldArg>{},
            cloned->in_parens());
      } else if (node->kind() == AstNodeKind::kStructInstance) {
        auto* cloned = absl::down_cast<StructInstance*>(cloned_node);
        XLS_ASSIGN_OR_RETURN(ColonRef * cloned_constructor,
                             GetConstructorRef(cloned->struct_ref()));
        return target_module->Make<SumInstance>(
            cloned->span(), cloned_constructor,
            SumInstance::PayloadShape::kStruct, std::vector<Expr*>{},
            cloned->members(), cloned->in_parens());
      } else {
        auto* cloned = absl::down_cast<ColonRef*>(cloned_node);
        const bool in_parens = cloned->in_parens();
        cloned->set_in_parens(false);
        return target_module->Make<SumInstance>(
            cloned->span(), cloned, SumInstance::PayloadShape::kUnit,
            std::vector<Expr*>{},
            std::vector<SumInstance::StructPayloadFieldArg>{}, in_parens);
      }
    };

    CloneReplacer preserve_external_nodes =
        [&module](const AstNode* node, Module*,
                  const absl::flat_hash_map<const AstNode*, AstNode*>&)
        -> std::optional<AstNode*> {
      // Borrowed declarations, including builtin State, keep their identity.
      if (node->owner() != &module) {
        return const_cast<AstNode*>(node);
      } else {
        return std::nullopt;
      }
    };
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Module> cloned,
                         CloneModule(module, std::move(preserve_external_nodes),
                                     std::move(replacer)));
    XLS_RETURN_IF_ERROR(
        VerifyClone(&module, cloned.get(), *module.file_table()));
    return std::make_optional(std::move(cloned));
  }
}

}  // namespace xls::dslx
