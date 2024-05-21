// Copyright 2021 The XLS Authors
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

#include "xls/dslx/frontend/ast_utils.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/variant.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/builtins_metadata.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/token_utils.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {
namespace {

void FlattenToSetInternal(const AstNode* node,
                          absl::flat_hash_set<const AstNode*>* the_set) {
  the_set->insert(node);
  for (const AstNode* child : node->GetChildren(/*want_types=*/true)) {
    FlattenToSetInternal(child, the_set);
  }
}

}  // namespace

bool IsParametricFunction(const AstNode* n) {
  // Convenience so we can check things like "definer" when the definer may be
  // unspecified in the AST. This generally only happens with programmatically
  // built ASTs.
  if (n == nullptr) {
    return false;
  }

  const auto* f = dynamic_cast<const Function*>(n);
  return f != nullptr && f->IsParametric();
}

bool ParentIsInvocationWithCallee(const NameRef* n) {
  CHECK(n != nullptr);
  const AstNode* parent = n->parent();
  CHECK(parent != nullptr);
  const auto* invocation = dynamic_cast<const Invocation*>(parent);
  return invocation != nullptr && invocation->callee() == n;
}

bool IsBuiltinFn(Expr* callee, std::optional<std::string_view> target) {
  NameRef* name_ref = dynamic_cast<NameRef*>(callee);
  if (name_ref == nullptr) {
    return false;
  }

  if (!std::holds_alternative<BuiltinNameDef*>(name_ref->name_def())) {
    return false;
  }

  if (target.has_value()) {
    auto* bnd = std::get<BuiltinNameDef*>(name_ref->name_def());
    return bnd->identifier() == target.value();
  }

  return true;
}

absl::StatusOr<std::string> GetBuiltinName(Expr* callee) {
  if (!IsBuiltinFn(callee)) {
    return absl::InvalidArgumentError("Callee is not a builtin function.");
  }

  NameRef* name_ref = dynamic_cast<NameRef*>(callee);
  return name_ref->identifier();
}

static absl::StatusOr<StructDef*> ResolveLocalStructDef(
    TypeAnnotation* type_annotation, const TypeDefinition& td) {
  auto error = [&](const AstNode* latest) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Could not resolve local struct definition from %s -- "
                        "%s was not a struct definition",
                        ToAstNode(td)->ToString(), latest->ToString()));
  };

  TypeRefTypeAnnotation* type_ref_type_annotation =
      dynamic_cast<TypeRefTypeAnnotation*>(type_annotation);
  if (type_ref_type_annotation == nullptr) {
    return error(type_annotation);
  }

  return ResolveLocalStructDef(
      type_ref_type_annotation->type_ref()->type_definition());
}

absl::StatusOr<StructDef*> ResolveLocalStructDef(TypeDefinition td) {
  auto error = [&](const AstNode* latest) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Could not resolve local struct definition from %s -- "
                        "%s was not a struct definition",
                        ToAstNode(td)->ToString(), latest->ToString()));
  };

  return absl::visit(
      Visitor{
          [&](TypeAlias* n) -> absl::StatusOr<StructDef*> {
            return ResolveLocalStructDef(&n->type_annotation(), td);
          },
          [&](StructDef* n) -> absl::StatusOr<StructDef*> { return n; },
          [&](EnumDef* n) -> absl::StatusOr<StructDef*> { return error(n); },
          [&](ColonRef* n) -> absl::StatusOr<StructDef*> { return error(n); },
      },
      td);
}

absl::Status VerifyParentage(const Module* module) {
  for (const ModuleMember member : module->top()) {
    if (std::holds_alternative<Function*>(member)) {
      return VerifyParentage(std::get<Function*>(member));
    }
    if (std::holds_alternative<Proc*>(member)) {
      return VerifyParentage(std::get<Proc*>(member));
    }
    if (std::holds_alternative<TestFunction*>(member)) {
      return VerifyParentage(std::get<TestFunction*>(member));
    }
    if (std::holds_alternative<TestProc*>(member)) {
      return VerifyParentage(std::get<TestProc*>(member));
    }
    if (std::holds_alternative<QuickCheck*>(member)) {
      return VerifyParentage(std::get<QuickCheck*>(member));
    }
    if (std::holds_alternative<TypeAlias*>(member)) {
      return VerifyParentage(std::get<TypeAlias*>(member));
    }
    if (std::holds_alternative<StructDef*>(member)) {
      return VerifyParentage(std::get<StructDef*>(member));
    }
    if (std::holds_alternative<ConstantDef*>(member)) {
      return VerifyParentage(std::get<ConstantDef*>(member));
    }
    if (std::holds_alternative<EnumDef*>(member)) {
      return VerifyParentage(std::get<EnumDef*>(member));
    }
    if (std::holds_alternative<Import*>(member)) {
      return VerifyParentage(std::get<Import*>(member));
    }
  }

  return absl::OkStatus();
}

absl::Status VerifyParentage(const AstNode* root) {
  CHECK(root != nullptr);

  if (const Module* module = dynamic_cast<const Module*>(root);
      module != nullptr) {
    return VerifyParentage(module);
  }

  for (const auto* child : root->GetChildren(/*want_types=*/true)) {
    CHECK(child != nullptr);
    XLS_RETURN_IF_ERROR(VerifyParentage(child));

    if (child->parent() == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Child \"%s\" (%s) of node \"%s\" (%s) had "
                          "no parent.",
                          child->ToString(), child->GetNodeTypeName(),
                          root->ToString(), root->GetNodeTypeName()));
    }

    if (child->parent() != root) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Child \"%s\" (%s, %s) of node \"%s\" (%s, %s) had "
          "node \"%s\" (%s, %s) as its parent.",
          child->ToString(), child->GetNodeTypeName(),
          child->GetSpan()->ToString(), root->ToString(),
          root->GetNodeTypeName(), root->GetSpan()->ToString(),
          child->parent()->ToString(), child->parent()->GetNodeTypeName(),
          child->parent()->GetSpan()->ToString()));
    }
  }

  return absl::OkStatus();
}

absl::flat_hash_set<const AstNode*> FlattenToSet(const AstNode* node) {
  absl::flat_hash_set<const AstNode*> the_set;
  FlattenToSetInternal(node, &the_set);
  return the_set;
}

absl::StatusOr<InterpValue> GetBuiltinNameDefColonAttr(
    const BuiltinNameDef* builtin_name_def, std::string_view attr) {
  const auto& sized_type_keywords = GetSizedTypeKeywordsMetadata();
  auto it = sized_type_keywords.find(builtin_name_def->identifier());
  // We should have checked this was a valid type keyword in typechecking.
  XLS_RET_CHECK(it != sized_type_keywords.end());
  auto [is_signed, width] = it->second;
  if (attr == "ZERO") {
    return InterpValue::MakeZeroValue(is_signed, width);
  }
  if (attr == "MAX") {
    return InterpValue::MakeMaxValue(is_signed, width);
  }
  // We only support the above attributes on builtin types at the moment -- this
  // is checked during typechecking.
  return absl::InvalidArgumentError(
      absl::StrFormat("Invalid attribute of builtin name %s: %s",
                      builtin_name_def->identifier(), attr));
}

absl::StatusOr<InterpValue> GetArrayTypeColonAttr(
    const ArrayTypeAnnotation* array_type, uint64_t constexpr_dim,
    std::string_view attr) {
  auto* builtin_type =
      dynamic_cast<BuiltinTypeAnnotation*>(array_type->element_type());
  if (builtin_type == nullptr) {
    return absl::InvalidArgumentError(
        "Can only take '::' attributes of uN/sN/bits array types.");
  }
  bool is_signed;
  switch (builtin_type->builtin_type()) {
    case BuiltinType::kUN:
    case BuiltinType::kBits:
      is_signed = false;
      break;
    case BuiltinType::kSN:
      is_signed = true;
      break;
    default:
      return absl::InvalidArgumentError(
          "Can only take '::' attributes of uN/sN/bits array types.");
  }

  if (attr == "ZERO") {
    return InterpValue::MakeZeroValue(is_signed, constexpr_dim);
  }
  if (attr == "MAX") {
    return InterpValue::MakeMaxValue(is_signed, constexpr_dim);
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Invalid attribute of builtin array type: %s", attr));
}

int64_t DetermineIndentLevel(const AstNode& n) {
  switch (n.kind()) {
    case AstNodeKind::kModule:
      return 0;
    case AstNodeKind::kBlock: {
      CHECK(n.parent() != nullptr);
      return DetermineIndentLevel(*n.parent()) + 1;
    }
    case AstNodeKind::kFunction: {
      const Function* function = down_cast<const Function*>(&n);
      switch (function->tag()) {
        case FunctionTag::kProcInit:
        case FunctionTag::kProcNext:
        case FunctionTag::kProcConfig:
          return 1;
        case FunctionTag::kNormal:
          return 0;
      }
    }
    default: {
      AstNode* parent = n.parent();
      CHECK(parent != nullptr);
      return DetermineIndentLevel(*parent);
    }
  }
}

std::optional<BitVectorMetadata> ExtractBitVectorMetadata(
    const TypeAnnotation* type_annotation) {
  bool is_enum = false;
  bool is_alias = false;
  const TypeAnnotation* type = type_annotation;
  while (dynamic_cast<const TypeRefTypeAnnotation*>(type) != nullptr) {
    auto type_ref = dynamic_cast<const TypeRefTypeAnnotation*>(type);
    if (std::holds_alternative<TypeAlias*>(
            type_ref->type_ref()->type_definition())) {
      is_alias = true;
      type = &std::get<TypeAlias*>(type_ref->type_ref()->type_definition())
                  ->type_annotation();
    } else if (std::holds_alternative<EnumDef*>(
                   type_ref->type_ref()->type_definition())) {
      is_enum = true;
      type = std::get<EnumDef*>(type_ref->type_ref()->type_definition())
                 ->type_annotation();
    } else {
      break;
    }
  }

  BitVectorKind kind;
  if (is_enum && is_alias) {
    kind = BitVectorKind::kEnumTypeAlias;
  } else if (is_enum && !is_alias) {
    kind = BitVectorKind::kEnumType;
  } else if (!is_enum && is_alias) {
    kind = BitVectorKind::kBitTypeAlias;
  } else {
    kind = BitVectorKind::kBitType;
  }

  if (const BuiltinTypeAnnotation* builtin =
          dynamic_cast<const BuiltinTypeAnnotation*>(type);
      builtin != nullptr) {
    switch (builtin->builtin_type()) {
      case BuiltinType::kToken:
      case BuiltinType::kChannelIn:
      case BuiltinType::kChannelOut:
        return std::nullopt;
      default:
        break;
    }
    return BitVectorMetadata{.bit_count = builtin->GetBitCount(),
                             .is_signed = builtin->GetSignedness(),
                             .kind = kind};
  }
  if (const ArrayTypeAnnotation* array_type =
          dynamic_cast<const ArrayTypeAnnotation*>(type);
      array_type != nullptr) {
    // bits[..], uN[..], and sN[..] are bit-vector types but a represented with
    // ArrayTypeAnnotations.
    const BuiltinTypeAnnotation* builtin_element_type =
        dynamic_cast<const BuiltinTypeAnnotation*>(array_type->element_type());
    if (builtin_element_type == nullptr) {
      return std::nullopt;
    }
    if (builtin_element_type->builtin_type() == BuiltinType::kBits ||
        builtin_element_type->builtin_type() == BuiltinType::kUN ||
        builtin_element_type->builtin_type() == BuiltinType::kSN) {
      return BitVectorMetadata{
          .bit_count = array_type->dim(),
          .is_signed = builtin_element_type->builtin_type() == BuiltinType::kSN,
          .kind = kind};
    }
  }
  return std::nullopt;
}

absl::StatusOr<std::vector<AstNode*>> CollectUnder(AstNode* root,
                                                   bool want_types) {
  std::vector<AstNode*> nodes;

  class CollectVisitor : public AstNodeVisitor {
   public:
    explicit CollectVisitor(std::vector<AstNode*>& nodes) : nodes_(nodes) {}

#define DECLARE_HANDLER(__type)                           \
  absl::Status Handle##__type(const __type* n) override { \
    nodes_.push_back(const_cast<__type*>(n));             \
    return absl::OkStatus();                              \
  }
    XLS_DSLX_AST_NODE_EACH(DECLARE_HANDLER)
#undef DECLARE_HANDLER

   private:
    std::vector<AstNode*>& nodes_;
  } collect_visitor(nodes);

  XLS_RETURN_IF_ERROR(WalkPostOrder(root, &collect_visitor, want_types));
  return nodes;
}

absl::StatusOr<std::vector<const AstNode*>> CollectUnder(const AstNode* root,
                                                         bool want_types) {
  // Implementation note: delegate to non-const version and turn result values
  // back to const.
  XLS_ASSIGN_OR_RETURN(std::vector<AstNode*> got,
                       CollectUnder(const_cast<AstNode*>(root), want_types));

  std::vector<const AstNode*> result;
  result.reserve(got.size());
  for (AstNode* n : got) {
    result.push_back(n);
  }
  return result;
}

absl::StatusOr<std::vector<const NameDef*>> CollectReferencedUnder(
    const AstNode* root, bool want_types) {
  XLS_ASSIGN_OR_RETURN(std::vector<const AstNode*> nodes,
                       CollectUnder(root, want_types));
  std::vector<const NameDef*> name_defs;
  for (const AstNode* n : nodes) {
    if (const NameRef* nr = dynamic_cast<const NameRef*>(n)) {
      if (std::holds_alternative<const NameDef*>(nr->name_def())) {
        name_defs.push_back(std::get<const NameDef*>(nr->name_def()));
      }
    }
  }
  return name_defs;
}

bool IsBuiltinParametricNameRef(const NameRef* name_ref) {
  // Implementation note: we also check IsNameParametricBuiltin() as future
  // proofing -- we may add a built-in name that is not a type or parametric
  // function.
  return std::holds_alternative<BuiltinNameDef*>(name_ref->name_def()) &&
         IsNameParametricBuiltin(name_ref->identifier());
}

const Number* IsBareNumber(const AstNode* node, bool* is_boolean) {
  if (const Number* number = dynamic_cast<const Number*>(node)) {
    if (is_boolean != nullptr) {
      *is_boolean = number->number_kind() == NumberKind::kBool;
    }
    if (number->type_annotation() == nullptr) {
      return number;
    }
    return nullptr;
  }

  return nullptr;
}

bool ContainedWithinFunction(const Invocation& invocation,
                             const Function& caller) {
  VLOG(10) << absl::StreamFormat(
      "Checking whether invocation `%s` @ %v is contained within caller `%s` @ "
      "%v",
      invocation.ToString(), invocation.span(), caller.identifier(),
      caller.span());
  const AstNode* parent = invocation.parent();
  CHECK(parent != nullptr) << absl::StreamFormat(
      "invocation node had no parent set: `%s` @ %v", invocation.ToString(),
      invocation.span());
  VLOG(10) << absl::StreamFormat("node `%s` has parent: `%s`",
                                 invocation.ToString(), parent->ToString());

  while (parent->kind() != AstNodeKind::kFunction) {
    const AstNode* new_parent = parent->parent();
    CHECK(new_parent != nullptr);
    VLOG(10) << absl::StreamFormat("transitive; node `%s` has parent: `%s`",
                                   parent->ToString(), new_parent->ToString());
    parent = new_parent;
  }

  bool contained = &caller == parent;
  VLOG(10) << absl::StreamFormat(
      "caller: %p vs found parent: %p; invocation contained? %s", &caller,
      parent, contained ? "true" : "false");

  // Here we check that, if the parent links indicate the node is contained, it
  // is also lexically/positionally contained.
  CHECK_EQ(contained, caller.span().Contains(invocation.span()));

  return contained;
}

}  // namespace xls::dslx
