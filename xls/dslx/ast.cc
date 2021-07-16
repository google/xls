// Copyright 2020 The XLS Authors
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

#include "xls/dslx/ast.h"

#include "absl/status/statusor.h"
#include "absl/strings/strip.h"
#include "xls/common/indent.h"
#include "xls/common/status/ret_check.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/number_parser.h"

namespace xls::dslx {

absl::Status WalkPostOrder(AstNode* root, AstNodeVisitor* visitor,
                           bool want_types) {
  for (AstNode* child : root->GetChildren(want_types)) {
    XLS_RETURN_IF_ERROR(WalkPostOrder(child, visitor, want_types));
  }
  return root->Accept(visitor);
}

absl::StatusOr<ColonRef::Subject> ToColonRefSubject(Expr* e) {
  if (auto* n = dynamic_cast<NameRef*>(e)) {
    return ColonRef::Subject(n);
  }
  if (auto* n = dynamic_cast<ColonRef*>(e)) {
    return ColonRef::Subject(n);
  }
  return absl::InvalidArgumentError(
      "Expression AST node is not a ColonRef subject.");
}

absl::StatusOr<TypeDefinition> ToTypeDefinition(AstNode* node) {
  if (auto* n = dynamic_cast<TypeDef*>(node)) {
    return TypeDefinition(n);
  }
  if (auto* n = dynamic_cast<StructDef*>(node)) {
    return TypeDefinition(n);
  }
  if (auto* n = dynamic_cast<EnumDef*>(node)) {
    return TypeDefinition(n);
  }
  if (auto* n = dynamic_cast<ColonRef*>(node)) {
    return TypeDefinition(n);
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("AST node is not a type definition: (%s) %s",
                      node->GetNodeTypeName(), node->ToString()));
}

FreeVariables FreeVariables::DropBuiltinDefs() const {
  FreeVariables result;
  for (const auto& [identifier, name_refs] : values_) {
    for (NameRef* ref : name_refs) {
      auto def = ref->name_def();
      if (absl::holds_alternative<BuiltinNameDef*>(def)) {
        continue;
      }
      result.Add(identifier, ref);
    }
  }
  return result;
}

std::vector<std::pair<std::string, AnyNameDef>>
FreeVariables::GetNameDefTuples() const {
  std::vector<std::pair<std::string, AnyNameDef>> result;
  for (const auto& item : values_) {
    NameRef* ref = item.second[0];
    result.push_back({item.first, ref->name_def()});
  }
  std::sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.first < rhs.first;
  });
  return result;
}

std::vector<ConstRef*> FreeVariables::GetConstRefs() {
  std::vector<ConstRef*> const_refs;
  for (const auto& [name, refs] : values_) {
    for (NameRef* name_ref : refs) {
      if (ConstRef* const_ref = dynamic_cast<ConstRef*>(name_ref)) {
        const_refs.push_back(const_ref);
      }
    }
  }
  return const_refs;
}

std::vector<AnyNameDef> FreeVariables::GetNameDefs() const {
  std::vector<AnyNameDef> result;
  for (auto& pair : GetNameDefTuples()) {
    result.push_back(pair.second);
  }
  return result;
}

void FreeVariables::Add(std::string identifier, NameRef* name_ref) {
  auto it = values_.insert({identifier, {name_ref}});
  if (!it.second) {
    it.first->second.push_back(name_ref);
  }
}

absl::flat_hash_set<std::string> FreeVariables::Keys() const {
  absl::flat_hash_set<std::string> result;
  for (const auto& item : values_) {
    result.insert(item.first);
  }
  return result;
}

std::string BuiltinTypeToString(BuiltinType t) {
  switch (t) {
#define CASE(__enum, B, __str, ...) \
  case BuiltinType::__enum:         \
    return __str;
    XLS_DSLX_BUILTIN_TYPE_EACH(CASE)
#undef CASE
  }
  return absl::StrFormat("<invalid BuiltinType(%d)>", static_cast<int>(t));
}

absl::StatusOr<BuiltinType> GetBuiltinType(bool is_signed, int64_t width) {
#define TEST(__enum, __name, __str, __signedness, __width) \
  if (__signedness == is_signed && __width == width) {     \
    return BuiltinType::__enum;                            \
  }
  XLS_DSLX_BUILTIN_TYPE_EACH(TEST)
#undef TEST
  return absl::NotFoundError(
      absl::StrFormat("Cannot find built in type with signedness: %d width: %d",
                      is_signed, width));
}

absl::StatusOr<BuiltinType> BuiltinTypeFromString(absl::string_view s) {
#define CASE(__enum, __unused, __str, ...) \
  if (s == __str) {                        \
    return BuiltinType::__enum;            \
  }
  XLS_DSLX_BUILTIN_TYPE_EACH(CASE)
#undef CASE
  return absl::InvalidArgumentError(
      absl::StrFormat("String is not a BuiltinType: \"%s\"", s));
}

class DfsIteratorNoTypes {
 public:
  DfsIteratorNoTypes(AstNode* start) : to_visit_({start}) {}

  bool HasNext() const { return !to_visit_.empty(); }

  AstNode* Next() {
    AstNode* result = to_visit_.front();
    to_visit_.pop_front();
    std::vector<AstNode*> children = result->GetChildren(/*want_types=*/false);
    std::reverse(children.begin(), children.end());
    for (AstNode* c : children) {
      to_visit_.push_front(c);
    }
    return result;
  }

 private:
  std::deque<AstNode*> to_visit_;
};

FreeVariables AstNode::GetFreeVariables(const Pos* start_pos) {
  DfsIteratorNoTypes it(this);
  FreeVariables freevars;
  while (it.HasNext()) {
    AstNode* n = it.Next();
    if (auto* name_ref = dynamic_cast<NameRef*>(n)) {
      // If a start position was given we test whether the name definition
      // occurs before that start positions. (If none was given we accept all
      // name refs.)
      if (start_pos == nullptr) {
        freevars.Add(name_ref->identifier(), name_ref);
      } else {
        absl::optional<Pos> name_def_start = name_ref->GetNameDefStart();
        if (!name_def_start.has_value() || *name_def_start < *start_pos) {
          freevars.Add(name_ref->identifier(), name_ref);
        }
      }
    }
  }
  return freevars;
}

const absl::btree_set<BinopKind>& GetBinopSameTypeKinds() {
  static auto* singleton = new absl::btree_set<BinopKind>{
      BinopKind::kAdd, BinopKind::kSub, BinopKind::kMul, BinopKind::kAnd,
      BinopKind::kOr,  BinopKind::kXor, BinopKind::kDiv,
  };
  return *singleton;
}

const absl::btree_set<BinopKind>& GetBinopComparisonKinds() {
  static auto* singleton = new absl::btree_set<BinopKind>{
      BinopKind::kGe, BinopKind::kGt, BinopKind::kLe,
      BinopKind::kLt, BinopKind::kEq, BinopKind::kNe,
  };
  return *singleton;
}

const absl::btree_set<BinopKind>& GetBinopShifts() {
  static auto* singleton = new absl::btree_set<BinopKind>{
      BinopKind::kShl,
      BinopKind::kShr,
  };
  return *singleton;
}

std::string BinopKindFormat(BinopKind kind) {
  switch (kind) {
    // clang-format off
    // Shifts.
    case BinopKind::kShl:       return "<<";
    case BinopKind::kShr:       return ">>";
    // Comparisons.
    case BinopKind::kGe:         return ">=";
    case BinopKind::kGt:         return ">";
    case BinopKind::kLe:         return "<=";
    case BinopKind::kLt:         return "<";
    case BinopKind::kEq:         return "==";
    case BinopKind::kNe:         return "!=";

    case BinopKind::kAdd:        return "+";
    case BinopKind::kSub:        return "-";
    case BinopKind::kMul:        return "*";
    case BinopKind::kAnd:        return "&";
    case BinopKind::kOr:         return "|";
    case BinopKind::kXor:        return "^";
    case BinopKind::kDiv:        return "/";
    case BinopKind::kLogicalAnd: return "&&";
    case BinopKind::kLogicalOr:  return "||";
    case BinopKind::kConcat:     return "++";
      // clang-format on
  }
  return absl::StrFormat("<invalid BinopKind(%d)>", static_cast<int>(kind));
}

std::string BinopKindToString(BinopKind kind) {
  switch (kind) {
#define CASIFY(__enum, __str, ...) \
  case BinopKind::__enum:          \
    return __str;
    XLS_DSLX_BINOP_KIND_EACH(CASIFY)
#undef CASIFY
  }
  return absl::StrFormat("<invalid BinopKind(%d)>", static_cast<int>(kind));
}

// -- class NameDef

NameDef::NameDef(Module* owner, Span span, std::string identifier,
                 AstNode* definer)
    : AstNode(owner),
      span_(span),
      identifier_(std::move(identifier)),
      definer_(definer) {}

// -- class Ternary

Ternary::Ternary(Module* owner, Span span, Expr* test, Expr* consequent,
                 Expr* alternate)
    : Expr(owner, std::move(span)),
      test_(test),
      consequent_(consequent),
      alternate_(alternate) {}

// -- class ParametricBinding

ParametricBinding::ParametricBinding(Module* owner, NameDef* name_def,
                                     TypeAnnotation* type_annotation,
                                     Expr* expr)
    : AstNode(owner),
      name_def_(name_def),
      type_annotation_(type_annotation),
      expr_(expr) {
  XLS_CHECK_EQ(name_def_->owner(), owner);
  XLS_CHECK_EQ(type_annotation_->owner(), owner);
}

std::string ParametricBinding::ToString() const {
  std::string suffix;
  if (expr_ != nullptr) {
    suffix = absl::StrFormat(" = %s", expr_->ToString());
  }
  return absl::StrFormat("%s: %s%s", name_def_->ToString(),
                         type_annotation_->ToString(), suffix);
}

std::string ParametricBinding::ToReprString() const {
  return absl::StrFormat("ParametricBinding(name_def=%s, type=%s, expr=%s)",
                         name_def_->ToReprString(),
                         type_annotation_->ToString(),
                         expr_ == nullptr ? "null" : expr_->ToString());
}

std::vector<AstNode*> ParametricBinding::GetChildren(bool want_types) const {
  std::vector<AstNode*> results = {name_def_};
  if (want_types) {
    results.push_back(type_annotation_);
  }
  if (expr_ != nullptr) {
    results.push_back(expr_);
  }
  return results;
}

std::string MatchArm::ToString() const {
  std::string patterns_or = absl::StrJoin(
      patterns_, " | ", [](std::string* out, NameDefTree* name_def_tree) {
        absl::StrAppend(out, name_def_tree->ToString());
      });
  return absl::StrFormat("%s => %s", patterns_or, expr_->ToString());
}

std::vector<AstNode*> StructInstance::GetChildren(bool want_types) const {
  std::vector<AstNode*> results;
  if (want_types) {
    results.push_back(GetStructNode());
  }
  for (auto& item : members_) {
    results.push_back(item.second);
  }
  return results;
}

std::string StructInstance::ToString() const {
  std::string type_name;
  if (absl::holds_alternative<StructDef*>(struct_ref_)) {
    type_name = absl::get<StructDef*>(struct_ref_)->identifier();
  } else {
    type_name = ToAstNode(struct_ref_)->ToString();
  }

  std::string members_str = absl::StrJoin(
      members_, ", ",
      [](std::string* out, const std::pair<std::string, Expr*>& member) {
        absl::StrAppendFormat(out, "%s: %s", member.first,
                              member.second->ToString());
      });
  return absl::StrFormat("%s { %s }", type_name, members_str);
}

std::string For::ToString() const {
  std::string type_str;
  if (type_annotation_ != nullptr) {
    type_str = absl::StrCat(": ", type_annotation_->ToString());
  }
  return absl::StrFormat(R"(for %s%s in %s {
%s
}(%s))",
                         names_->ToString(), type_str, iterable_->ToString(),
                         Indent(body_->ToString()), init_->ToString());
}

ConstantDef::ConstantDef(Module* owner, Span span, NameDef* name_def,
                         Expr* value, bool is_public, bool is_local)
    : AstNode(owner),
      span_(std::move(span)),
      name_def_(name_def),
      value_(value),
      is_public_(is_public),
      is_local_(is_local) {}

std::string ConstantDef::ToString() const {
  std::string privacy;
  if (is_public_) {
    privacy = "pub ";
  }
  return absl::StrFormat("%sconst %s = %s;", privacy, name_def_->ToString(),
                         value_->ToString());
}

std::string ConstantDef::ToReprString() const {
  return absl::StrFormat("ConstantDef(%s)", name_def_->ToReprString());
}

Array::Array(Module* owner, Span span, std::vector<Expr*> members,
             bool has_ellipsis)
    : Expr(owner, std::move(span)),
      members_(std::move(members)),
      has_ellipsis_(has_ellipsis) {}

ConstantArray::ConstantArray(Module* owner, Span span,
                             std::vector<Expr*> members, bool has_ellipsis)
    : Array(owner, std::move(span), std::move(members), has_ellipsis) {
  for (Expr* expr : members) {
    XLS_CHECK(IsConstant(expr))
        << "non-constant in constant array: " << expr->ToString();
  }
}

TypeRef::TypeRef(Module* owner, Span span, std::string text,
                 TypeDefinition type_definition)
    : AstNode(owner),
      span_(std::move(span)),
      text_(std::move(text)),
      type_definition_(type_definition) {}

Import::Import(Module* owner, Span span, std::vector<std::string> subject,
               NameDef* name_def, absl::optional<std::string> alias)
    : AstNode(owner),
      span_(std::move(span)),
      subject_(std::move(subject)),
      name_def_(name_def),
      alias_(std::move(alias)) {
  XLS_CHECK(!subject_.empty());
  XLS_CHECK(name_def != nullptr);
}

std::string Import::ToString() const {
  if (alias_.has_value()) {
    return absl::StrFormat("import %s as %s", absl::StrJoin(subject_, "."),
                           *alias_);
  }
  return absl::StrFormat("import %s", absl::StrJoin(subject_, "."));
}

// -- class ColonRef

ColonRef::ColonRef(Module* owner, Span span, Subject subject, std::string attr)
    : Expr(owner, std::move(span)), subject_(subject), attr_(std::move(attr)) {}

absl::optional<Import*> ColonRef::ResolveImportSubject() const {
  if (!absl::holds_alternative<NameRef*>(subject_)) {
    return absl::nullopt;
  }
  auto* name_ref = absl::get<NameRef*>(subject_);
  AnyNameDef any_name_def = name_ref->name_def();
  if (!absl::holds_alternative<NameDef*>(any_name_def)) {
    return absl::nullopt;
  }
  auto* name_def = absl::get<NameDef*>(any_name_def);
  AstNode* definer = name_def->definer();
  Import* import = dynamic_cast<Import*>(definer);
  if (import == nullptr) {
    return absl::nullopt;
  }
  return import;
}

// -- class Param

Param::Param(Module* owner, NameDef* name_def, TypeAnnotation* type_annotation)
    : AstNode(owner),
      name_def_(name_def),
      type_annotation_(type_annotation),
      span_(name_def_->span().start(), type_annotation_->span().limit()) {}

// -- class Module

absl::optional<Function*> Module::GetFunction(absl::string_view target_name) {
  for (ModuleMember& member : top_) {
    if (absl::holds_alternative<Function*>(member)) {
      Function* f = absl::get<Function*>(member);
      if (f->identifier() == target_name) {
        return f;
      }
    }
  }
  return absl::nullopt;
}

absl::StatusOr<Function*> Module::GetFunctionOrError(
    absl::string_view target_name) {
  if (absl::optional<Function*> f = GetFunction(target_name)) {
    return f.value();
  }
  return absl::NotFoundError(absl::StrFormat(
      "No function in module %s with name \"%s\"", name_, target_name));
}

absl::StatusOr<TestFunction*> Module::GetTest(absl::string_view target_name) {
  for (ModuleMember& member : top_) {
    if (absl::holds_alternative<TestFunction*>(member)) {
      TestFunction* t = absl::get<TestFunction*>(member);
      if (t->identifier() == target_name) {
        return t;
      }
    }
  }
  return absl::NotFoundError(absl::StrFormat(
      "No test in module %s with name \"%s\"", name_, target_name));
}

std::vector<std::string> Module::GetTestNames() const {
  std::vector<std::string> result;
  for (auto& member : top_) {
    if (absl::holds_alternative<TestFunction*>(member)) {
      TestFunction* t = absl::get<TestFunction*>(member);
      result.push_back(t->identifier());
    }
  }
  return result;
}

std::vector<std::string> Module::GetFunctionNames() const {
  std::vector<std::string> result;
  for (auto& member : top_) {
    if (absl::holds_alternative<Function*>(member)) {
      Function* f = absl::get<Function*>(member);
      result.push_back(f->identifier());
    }
  }
  return result;
}

absl::optional<ModuleMember*> Module::FindMemberWithName(
    absl::string_view target) {
  for (ModuleMember& member : top_) {
    if (absl::holds_alternative<Function*>(member)) {
      if (absl::get<Function*>(member)->identifier() == target) {
        return &member;
      }
    } else if (absl::holds_alternative<TestFunction*>(member)) {
      if (absl::get<TestFunction*>(member)->identifier() == target) {
        return &member;
      }
    } else if (absl::holds_alternative<QuickCheck*>(member)) {
      if (absl::get<QuickCheck*>(member)->identifier() == target) {
        return &member;
      }
    } else if (absl::holds_alternative<TypeDef*>(member)) {
      if (absl::get<TypeDef*>(member)->identifier() == target) {
        return &member;
      }
    } else if (absl::holds_alternative<StructDef*>(member)) {
      if (absl::get<StructDef*>(member)->identifier() == target) {
        return &member;
      }
    } else if (absl::holds_alternative<ConstantDef*>(member)) {
      if (absl::get<ConstantDef*>(member)->identifier() == target) {
        return &member;
      }
    } else if (absl::holds_alternative<EnumDef*>(member)) {
      if (absl::get<EnumDef*>(member)->identifier() == target) {
        return &member;
      }
    } else if (absl::holds_alternative<Import*>(member)) {
      if (absl::get<Import*>(member)->identifier() == target) {
        return &member;
      }
    } else {
      XLS_LOG(FATAL) << "Unhandled module member variant: "
                     << ToAstNode(member)->GetNodeTypeName();
    }
  }
  return absl::nullopt;
}

absl::StatusOr<ConstantDef*> Module::GetConstantDef(absl::string_view target) {
  absl::optional<ModuleMember*> member = FindMemberWithName(target);
  if (!member.has_value()) {
    return absl::NotFoundError(
        absl::StrFormat("Could not find member named '%s' in module.", target));
  }
  if (!absl::holds_alternative<ConstantDef*>(*member.value())) {
    return absl::NotFoundError(absl::StrFormat(
        "Member named '%s' in module was not a constant.", target));
  }
  return absl::get<ConstantDef*>(*member.value());
}

absl::flat_hash_map<std::string, TypeDefinition>
Module::GetTypeDefinitionByName() const {
  absl::flat_hash_map<std::string, TypeDefinition> result;
  for (auto& member : top_) {
    if (absl::holds_alternative<TypeDef*>(member)) {
      TypeDef* td = absl::get<TypeDef*>(member);
      result[td->identifier()] = td;
    } else if (absl::holds_alternative<EnumDef*>(member)) {
      EnumDef* enum_ = absl::get<EnumDef*>(member);
      result[enum_->identifier()] = enum_;
    } else if (absl::holds_alternative<StructDef*>(member)) {
      StructDef* struct_ = absl::get<StructDef*>(member);
      result[struct_->identifier()] = struct_;
    }
  }
  return result;
}

std::vector<TypeDefinition> Module::GetTypeDefinitions() const {
  std::vector<TypeDefinition> results;
  for (auto& member : top_) {
    if (absl::holds_alternative<TypeDef*>(member)) {
      TypeDef* td = absl::get<TypeDef*>(member);
      results.push_back(td);
    } else if (absl::holds_alternative<EnumDef*>(member)) {
      EnumDef* enum_def = absl::get<EnumDef*>(member);
      results.push_back(enum_def);
    } else if (absl::holds_alternative<StructDef*>(member)) {
      StructDef* struct_def = absl::get<StructDef*>(member);
      results.push_back(struct_def);
    }
  }
  return results;
}

std::vector<AstNode*> Module::GetChildren(bool want_types) const {
  std::vector<AstNode*> results;
  for (ModuleMember member : top_) {
    results.push_back(ToAstNode(member));
  }
  return results;
}

absl::StatusOr<TypeDefinition> Module::GetTypeDefinition(
    absl::string_view name) const {
  absl::flat_hash_map<std::string, TypeDefinition> map =
      GetTypeDefinitionByName();
  auto it = map.find(name);
  if (it == map.end()) {
    return absl::NotFoundError(
        absl::StrCat("Could not find type definition for name: ", name));
  }
  return it->second;
}

absl::StatusOr<ModuleMember> AsModuleMember(AstNode* node) {
  // clang-format off
  if (auto* n = dynamic_cast<Function*    >(node)) { return ModuleMember(n); }
  if (auto* n = dynamic_cast<TestFunction*>(node)) { return ModuleMember(n); }
  if (auto* n = dynamic_cast<QuickCheck*  >(node)) { return ModuleMember(n); }
  if (auto* n = dynamic_cast<TypeDef*     >(node)) { return ModuleMember(n); }
  if (auto* n = dynamic_cast<StructDef*   >(node)) { return ModuleMember(n); }
  if (auto* n = dynamic_cast<ConstantDef* >(node)) { return ModuleMember(n); }
  if (auto* n = dynamic_cast<EnumDef*     >(node)) { return ModuleMember(n); }
  if (auto* n = dynamic_cast<Import*      >(node)) { return ModuleMember(n); }
  // clang-format on
  return absl::InvalidArgumentError("AST node is not a module-level member: " +
                                    node->ToString());
}

absl::StatusOr<IndexRhs> AstNodeToIndexRhs(AstNode* node) {
  // clang-format off
  if (auto* n = dynamic_cast<Slice*     >(node)) { return IndexRhs(n); }
  if (auto* n = dynamic_cast<WidthSlice*>(node)) { return IndexRhs(n); }
  if (auto* n = dynamic_cast<Expr*      >(node)) { return IndexRhs(n); }
  // clang-format on
  return absl::InvalidArgumentError("AST node is not a valid 'index': " +
                                    node->ToString());
}

TypeRefTypeAnnotation::TypeRefTypeAnnotation(Module* owner, Span span,
                                             TypeRef* type_ref,
                                             std::vector<Expr*> parametrics)
    : TypeAnnotation(owner, std::move(span)),
      type_ref_(type_ref),
      parametrics_(std::move(parametrics)) {}

ArrayTypeAnnotation::ArrayTypeAnnotation(Module* owner, Span span,
                                         TypeAnnotation* element_type,
                                         Expr* dim)
    : TypeAnnotation(owner, std::move(span)),
      element_type_(element_type),
      dim_(dim) {}

std::vector<AstNode*> ArrayTypeAnnotation::GetChildren(bool want_types) const {
  return {element_type_, dim_};
}

std::string ArrayTypeAnnotation::ToString() const {
  return absl::StrFormat("%s[%s]", element_type_->ToString(), dim_->ToString());
}

bool IsConstant(AstNode* node) {
  if (IsOneOf<ConstantArray, Number, ConstRef, ColonRef>(node)) {
    return true;
  }
  if (Cast* n = dynamic_cast<Cast*>(node)) {
    return IsConstant(n->expr());
  }
  if (StructInstance* n = dynamic_cast<StructInstance*>(node)) {
    for (const auto& [name, expr] : n->GetUnorderedMembers()) {
      if (!IsConstant(expr)) {
        return false;
      }
    }
    return true;
  }
  if (XlsTuple* n = dynamic_cast<XlsTuple*>(node)) {
    return std::all_of(n->members().begin(), n->members().end(), IsConstant);
  }
  if (Expr* e = dynamic_cast<Expr*>(node)) {
    auto children = e->GetChildren(/*want_types=*/false);
    return std::all_of(children.begin(), children.end(), IsConstant);
  }
  return false;
}

std::vector<AstNode*> SplatStructInstance::GetChildren(bool want_types) const {
  std::vector<AstNode*> results;
  if (want_types) {
    results.push_back(ToAstNode(struct_ref_));
  }
  for (auto& item : members_) {
    results.push_back(item.second);
  }
  results.push_back(splatted_);
  return results;
}

std::string SplatStructInstance::ToString() const {
  std::string members_str = absl::StrJoin(
      members_, ", ",
      [](std::string* out, const std::pair<std::string, Expr*>& member) {
        absl::StrAppendFormat(out, "%s: %s", member.first,
                              member.second->ToString());
      });
  return absl::StrFormat("%s { %s, ..%s }", ToAstNode(struct_ref_)->ToString(),
                         members_str, splatted_->ToString());
}

std::vector<AstNode*> MatchArm::GetChildren(bool want_types) const {
  std::vector<AstNode*> results;
  for (NameDefTree* ndt : patterns_) {
    results.push_back(ndt);
  }
  results.push_back(expr_);
  return results;
}

// -- class Match

std::vector<AstNode*> Match::GetChildren(bool want_types) const {
  std::vector<AstNode*> results = {matched_};
  for (MatchArm* arm : arms_) {
    results.push_back(arm);
  }
  return results;
}

std::string Match::ToString() const {
  std::string result = absl::StrFormat("match %s {\n", matched_->ToString());
  for (MatchArm* arm : arms_) {
    absl::StrAppend(&result, "  ", arm->ToString(), ",\n");
  }
  absl::StrAppend(&result, "}");
  return result;
}

// -- class Index

std::string Index::ToString() const {
  return absl::StrFormat("(%s)[%s]", lhs_->ToString(),
                         ToAstNode(rhs_)->ToString());
}

// -- class WidthSlice

std::string WidthSlice::ToString() const {
  return absl::StrFormat("%s+:%s", start_->ToString(), width_->ToString());
}

// -- class Slice

std::vector<AstNode*> Slice::GetChildren(bool want_types) const {
  std::vector<AstNode*> results;
  if (start_ != nullptr) {
    results.push_back(start_);
  }
  if (limit_ != nullptr) {
    results.push_back(limit_);
  }
  return results;
}

std::string Slice::ToString() const {
  if (start_ != nullptr && limit_ != nullptr) {
    return absl::StrFormat("%s:%s", start_->ToString(), limit_->ToString());
  }
  if (start_ != nullptr) {
    return absl::StrFormat("%s:", start_->ToString());
  }
  if (limit_ != nullptr) {
    return absl::StrFormat(":%s", limit_->ToString());
  }
  return ":";
}

// -- class EnumDef

EnumDef::EnumDef(Module* owner, Span span, NameDef* name_def,
                 TypeAnnotation* type_annotation,
                 std::vector<EnumMember> values, bool is_public)
    : AstNode(owner),
      span_(std::move(span)),
      name_def_(name_def),
      type_annotation_(type_annotation),
      values_(std::move(values)),
      is_public_(is_public) {}

bool EnumDef::HasValue(absl::string_view name) const {
  for (const auto& item : values_) {
    if (item.name_def->identifier() == name) {
      return true;
    }
  }
  return false;
}

absl::StatusOr<Expr*> EnumDef::GetValue(absl::string_view name) const {
  for (const EnumMember& item : values_) {
    if (item.name_def->identifier() == name) {
      return item.value;
    }
  }
  return absl::NotFoundError(absl::StrFormat(
      "Enum %s has no value with name \"%s\"", identifier(), name));
}

std::string EnumDef::ToString() const {
  std::string result =
      absl::StrFormat("%senum %s : %s {\n", is_public_ ? "pub " : "",
                      identifier(), type_annotation_->ToString());

  auto value_to_string = [](Expr* value) -> std::string {
    if (Number* number = dynamic_cast<Number*>(value)) {
      return number->ToStringNoType();
    }
    return value->ToString();
  };

  for (const auto& item : values_) {
    absl::StrAppendFormat(&result, "  %s = %s,\n", item.name_def->identifier(),
                          value_to_string(item.value));
  }
  absl::StrAppend(&result, "}");
  return result;
}

// -- class Invocation

Invocation::Invocation(Module* owner, Span span, Expr* callee,
                       std::vector<Expr*> args, std::vector<Expr*> parametrics)
    : Expr(owner, std::move(span)),
      callee_(callee),
      args_(std::move(args)),
      parametrics_(std::move(parametrics)) {}

std::vector<AstNode*> Invocation::GetChildren(bool want_types) const {
  std::vector<AstNode*> results = {callee_};
  for (Expr* arg : args_) {
    results.push_back(arg);
  }
  return results;
}

std::string Invocation::FormatArgs() const {
  return absl::StrJoin(args_, ", ", [](std::string* out, Expr* e) {
    absl::StrAppend(out, e->ToString());
  });
}

std::string Invocation::FormatParametrics() const {
  if (parametrics_.empty()) {
    return "";
  }

  return absl::StrCat("<",
                      absl::StrJoin(parametrics_, ", ",
                                    [](std::string* out, Expr* e) {
                                      absl::StrAppend(out, e->ToString());
                                    }),
                      ">");
}

// -- class StructDef

StructDef::StructDef(Module* owner, Span span, NameDef* name_def,
                     std::vector<ParametricBinding*> parametric_bindings,
                     std::vector<std::pair<NameDef*, TypeAnnotation*>> members,
                     bool is_public)
    : AstNode(owner),
      span_(std::move(span)),
      name_def_(name_def),
      parametric_bindings_(std::move(parametric_bindings)),
      members_(std::move(members)),
      public_(is_public) {}

std::vector<AstNode*> StructDef::GetChildren(bool want_types) const {
  std::vector<AstNode*> results = {name_def_};
  for (auto* pb : parametric_bindings_) {
    results.push_back(pb);
  }
  for (const auto& pair : members_) {
    results.push_back(pair.first);
    results.push_back(pair.second);
  }
  return results;
}

std::string StructDef::ToString() const {
  std::string parametric_str;
  if (!parametric_bindings_.empty()) {
    std::string guts =
        absl::StrJoin(parametric_bindings_, ", ",
                      [](std::string* out, ParametricBinding* binding) {
                        absl::StrAppend(out, binding->ToString());
                      });
    parametric_str = absl::StrFormat("<%s>", guts);
  }
  std::string result = absl::StrFormat(
      "%sstruct %s%s {\n", public_ ? "pub " : "", identifier(), parametric_str);
  for (const auto& item : members_) {
    absl::StrAppendFormat(&result, "  %s: %s,\n", item.first->ToString(),
                          item.second->ToString());
  }
  absl::StrAppend(&result, "}");
  return result;
}

std::vector<std::string> StructDef::GetMemberNames() const {
  std::vector<std::string> names;
  for (auto& item : members_) {
    names.push_back(item.first->identifier());
  }
  return names;
}

absl::optional<int64_t> StructDef::GetMemberIndex(
    absl::string_view name) const {
  for (int64_t i = 0; i < members_.size(); ++i) {
    if (members_[i].first->identifier() == name) {
      return i;
    }
  }
  return absl::nullopt;
}

// -- class StructInstance

StructInstance::StructInstance(
    Module* owner, Span span, StructRef struct_ref,
    std::vector<std::pair<std::string, Expr*>> members)
    : Expr(owner, std::move(span)),
      struct_ref_(struct_ref),
      members_(std::move(members)) {}

std::vector<std::pair<std::string, Expr*>> StructInstance::GetOrderedMembers(
    StructDef* struct_def) const {
  std::vector<std::pair<std::string, Expr*>> result;
  for (const std::string& name : struct_def->GetMemberNames()) {
    result.push_back({name, GetExpr(name).value()});
  }
  return result;
}

absl::StatusOr<Expr*> StructInstance::GetExpr(absl::string_view name) const {
  for (const auto& item : members_) {
    if (item.first == name) {
      return item.second;
    }
  }
  return absl::NotFoundError(
      absl::StrFormat("Name is not present in struct instance: \"%s\"", name));
}

// -- class SplatStructInstance

SplatStructInstance::SplatStructInstance(
    Module* owner, Span span, StructRef struct_ref,
    std::vector<std::pair<std::string, Expr*>> members, Expr* splatted)
    : Expr(owner, std::move(span)),
      struct_ref_(std::move(struct_ref)),
      members_(std::move(members)),
      splatted_(splatted) {}

std::vector<AstNode*> TypeRefTypeAnnotation::GetChildren(
    bool want_types) const {
  std::vector<AstNode*> results = {type_ref_};
  for (Expr* e : parametrics_) {
    results.push_back(e);
  }
  return results;
}

std::string TypeRefTypeAnnotation::ToString() const {
  std::string parametric_str = "";
  if (!parametrics_.empty()) {
    std::vector<std::string> pieces;
    for (Expr* e : parametrics_) {
      pieces.push_back(e->ToString());
    }
    parametric_str = absl::StrCat("<", absl::StrJoin(pieces, ", "), ">");
  }
  return absl::StrCat(type_ref_->ToString(), parametric_str);
}

absl::StatusOr<BinopKind> BinopKindFromString(absl::string_view s) {
#define HANDLE(__enum, __unused, __operator) \
  if (s == __operator) {                     \
    return BinopKind::__enum;                \
  }
  XLS_DSLX_BINOP_KIND_EACH(HANDLE)
#undef HANDLE
  return absl::InvalidArgumentError(
      absl::StrFormat("Invalid BinopKind string: \"%s\"", s));
}

Binop::Binop(Module* owner, Span span, BinopKind kind, Expr* lhs, Expr* rhs)
    : Expr(owner, span), kind_(kind), lhs_(lhs), rhs_(rhs) {}

absl::StatusOr<UnopKind> UnopKindFromString(absl::string_view s) {
  if (s == "!") {
    return UnopKind::kInvert;
  }
  if (s == "-") {
    return UnopKind::kNegate;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Invalid UnopKind string: \"%s\"", s));
}

std::string UnopKindToString(UnopKind k) {
  switch (k) {
    case UnopKind::kInvert:
      return "!";
    case UnopKind::kNegate:
      return "-";
  }
  return absl::StrFormat("<invalid UnopKind(%d)>", static_cast<int>(k));
}

// -- class For

For::For(Module* owner, Span span, NameDefTree* names,
         TypeAnnotation* type_annotation, Expr* iterable, Expr* body,
         Expr* init)
    : Expr(owner, std::move(span)),
      names_(names),
      type_annotation_(type_annotation),
      iterable_(iterable),
      body_(body),
      init_(init) {}

std::vector<AstNode*> For::GetChildren(bool want_types) const {
  std::vector<AstNode*> results = {names_};
  if (want_types && type_annotation_ != nullptr) {
    results.push_back(type_annotation_);
  }
  results.push_back(iterable_);
  results.push_back(body_);
  results.push_back(init_);
  return results;
}

Function::Function(Module* owner, Span span, NameDef* name_def,
                   std::vector<ParametricBinding*> parametric_bindings,
                   std::vector<Param*> params, TypeAnnotation* return_type,
                   Expr* body, bool is_public)
    : AstNode(owner),
      span_(span),
      name_def_(XLS_DIE_IF_NULL(name_def)),
      params_(std::move(params)),
      parametric_bindings_(std::move(parametric_bindings)),
      return_type_(return_type),
      body_(XLS_DIE_IF_NULL(body)),
      is_public_(is_public) {}

std::vector<AstNode*> Function::GetChildren(bool want_types) const {
  std::vector<AstNode*> results;
  results.push_back(name_def_);
  for (ParametricBinding* binding : parametric_bindings_) {
    results.push_back(binding);
  }
  if (return_type_ != nullptr && want_types) {
    results.push_back(return_type_);
  }
  results.push_back(body_);
  return results;
}

std::vector<std::string> Function::GetFreeParametricKeys() const {
  std::vector<std::string> results;
  for (ParametricBinding* b : parametric_bindings_) {
    if (b->expr() == nullptr) {
      results.push_back(b->name_def()->identifier());
    }
  }
  return results;
}

std::string Function::Format(bool include_body) const {
  std::string parametric_str;
  if (!parametric_bindings_.empty()) {
    parametric_str = absl::StrFormat(
        "<%s>",
        absl::StrJoin(
            parametric_bindings_, ", ",
            [](std::string* out, ParametricBinding* parametric_binding) {
              absl::StrAppend(out, parametric_binding->ToString());
            }));
  }
  std::string params_str =
      absl::StrJoin(params_, ", ", [](std::string* out, Param* param) {
        absl::StrAppend(out, param->ToString());
      });
  std::string return_type_str = " ";
  if (return_type_ != nullptr) {
    return_type_str = " -> " + return_type_->ToString() + " ";
  }
  std::string pub_str = is_public_ ? "pub " : "";
  std::string name_str = name_def_->ToString();
  std::string body_str = Indent(body_->ToString());
  return absl::StrFormat("%sfn %s%s(%s)%s{\n%s\n}", pub_str, name_str,
                         parametric_str, params_str, return_type_str, body_str);
}

// -- class MatchArm

MatchArm::MatchArm(Module* owner, Span span, std::vector<NameDefTree*> patterns,
                   Expr* expr)
    : AstNode(owner),
      span_(std::move(span)),
      patterns_(std::move(patterns)),
      expr_(expr) {
  XLS_CHECK(!patterns_.empty());
}

Span MatchArm::GetPatternSpan() const {
  return Span(patterns_[0]->span().start(), patterns_.back()->span().limit());
}

Match::Match(Module* owner, Span span, Expr* matched,
             std::vector<MatchArm*> arms)
    : Expr(owner, std::move(span)), matched_(matched), arms_(std::move(arms)) {}

// -- class Proc

Proc::Proc(Module* owner, Span span, NameDef* name_def,
           std::vector<Param*> proc_params, std::vector<Param*> iter_params,
           Expr* iter_body, bool is_public)
    : AstNode(owner),
      span_(std::move(span)),
      name_def_(name_def),
      proc_params_(std::move(proc_params)),
      iter_params_(std::move(iter_params)),
      iter_body_(iter_body),
      is_public_(is_public) {}

std::vector<AstNode*> Proc::GetChildren(bool want_types) const {
  std::vector<AstNode*> results = {name_def_};
  for (Param* p : proc_params_) {
    results.push_back(p);
  }
  for (Param* p : iter_params_) {
    results.push_back(p);
  }
  results.push_back(iter_body_);
  return results;
}

std::string Proc::ToString() const {
  std::string pub_str = is_public_ ? "pub " : "";
  auto param_append = [](std::string* out, const Param* param) {
    absl::StrAppend(out, param->ToString());
  };
  std::string proc_params_str = absl::StrJoin(proc_params_, ", ", param_append);
  std::string iter_params_str = absl::StrJoin(iter_params_, ", ", param_append);
  return absl::StrFormat("%sproc %s(%s) {\n  next(%s) {\n%s\n  }\n}", pub_str,
                         name_def_->identifier(), proc_params_str,
                         iter_params_str,
                         Indent(iter_body_->ToString(), /*spaces=*/4));
}

BuiltinTypeAnnotation::BuiltinTypeAnnotation(Module* owner, Span span,
                                             BuiltinType builtin_type)
    : TypeAnnotation(owner, std::move(span)), builtin_type_(builtin_type) {}

int64_t BuiltinTypeAnnotation::GetBitCount() const {
  switch (builtin_type_) {
#define CASE(__enum, _unused1, _unused2, _unused3, __bit_count) \
  case BuiltinType::__enum:                                     \
    return __bit_count;
    XLS_DSLX_BUILTIN_TYPE_EACH(CASE)
#undef CASE
  }
  XLS_LOG(FATAL) << "Invalid builtin type: " << static_cast<int>(builtin_type_);
}

bool BuiltinTypeAnnotation::GetSignedness() const {
  switch (builtin_type_) {
#define CASE(__enum, _unused1, _unused2, __signedness, ...) \
  case BuiltinType::__enum:                                 \
    return __signedness;
    XLS_DSLX_BUILTIN_TYPE_EACH(CASE)
#undef CASE
  }
  XLS_LOG(FATAL) << "Invalid builtin type: " << static_cast<int>(builtin_type_);
}

TupleTypeAnnotation::TupleTypeAnnotation(Module* owner, Span span,
                                         std::vector<TypeAnnotation*> members)
    : TypeAnnotation(owner, std::move(span)), members_(std::move(members)) {}

std::string TupleTypeAnnotation::ToString() const {
  std::string guts =
      absl::StrJoin(members_, ", ", [](std::string* out, TypeAnnotation* t) {
        absl::StrAppend(out, t->ToString());
      });
  return absl::StrFormat("(%s%s)", guts, members_.size() == 1 ? "," : "");
}

// -- class QuickCheck

QuickCheck::QuickCheck(Module* owner, Span span, Function* f,
                       absl::optional<int64_t> test_count)
    : AstNode(owner), span_(span), f_(f), test_count_(std::move(test_count)) {}

std::string QuickCheck::ToString() const {
  std::string test_count_str;
  if (test_count_.has_value()) {
    test_count_str = absl::StrFormat("(test_count=%d)", *test_count_);
  }
  return absl::StrFormat("#![quickcheck%s]\n%s", test_count_str,
                         f_->ToString());
}

std::string XlsTuple::ToString() const {
  std::string result = "(";
  for (int64_t i = 0; i < members_.size(); ++i) {
    absl::StrAppend(&result, members_[i]->ToString());
    if (i != members_.size() - 1) {
      absl::StrAppend(&result, ", ");
    }
  }
  if (members_.size() == 1) {
    absl::StrAppend(&result, ",");  // Singleton tuple.
  }
  absl::StrAppend(&result, ")");
  return result;
}

std::string StructRefToText(const StructRef& struct_ref) {
  if (absl::holds_alternative<StructDef*>(struct_ref)) {
    return absl::get<StructDef*>(struct_ref)->identifier();
  }
  if (absl::holds_alternative<ColonRef*>(struct_ref)) {
    return absl::get<ColonRef*>(struct_ref)->ToString();
  }
  XLS_LOG(FATAL)
      << "Unhandled alternative for converting struct reference to string.";
}

// -- class NameDefTree

std::vector<AstNode*> NameDefTree::GetChildren(bool want_types) const {
  if (absl::holds_alternative<Leaf>(tree_)) {
    return {ToAstNode(absl::get<Leaf>(tree_))};
  }
  const Nodes& nodes = absl::get<Nodes>(tree_);
  return ToAstNodes<NameDefTree>(nodes);
}

std::string NameDefTree::ToString() const {
  if (is_leaf()) {
    return ToAstNode(leaf())->ToString();
  } else {
    std::string guts =
        absl::StrJoin(nodes(), ", ", [](std::string* out, NameDefTree* node) {
          absl::StrAppend(out, node->ToString());
        });
    return absl::StrFormat("(%s)", guts);
  }
}

std::vector<NameDefTree::Leaf> NameDefTree::Flatten() const {
  if (is_leaf()) {
    return {leaf()};
  }
  std::vector<Leaf> results;
  for (const NameDefTree* node : absl::get<Nodes>(tree_)) {
    auto node_leaves = node->Flatten();
    results.insert(results.end(), node_leaves.begin(), node_leaves.end());
  }
  return results;
}

std::vector<NameDef*> NameDefTree::GetNameDefs() const {
  std::vector<NameDef*> results;
  for (Leaf leaf : Flatten()) {
    if (absl::holds_alternative<NameDef*>(leaf)) {
      results.push_back(absl::get<NameDef*>(leaf));
    }
  }
  return results;
}

std::vector<absl::variant<NameDefTree::Leaf, NameDefTree*>>
NameDefTree::Flatten1() {
  if (is_leaf()) {
    return {leaf()};
  }
  std::vector<absl::variant<Leaf, NameDefTree*>> result;
  for (NameDefTree* ndt : nodes()) {
    if (ndt->is_leaf()) {
      result.push_back(ndt->leaf());
    } else {
      result.push_back(ndt);
    }
  }
  return result;
}

// -- class Let

Let::Let(Module* owner, Span span, NameDefTree* name_def_tree,
         TypeAnnotation* type_annotation, Expr* rhs, Expr* body,
         ConstantDef* const_def)
    : Expr(owner, std::move(span)),
      name_def_tree_(name_def_tree),
      type_annotation_(type_annotation),
      rhs_(rhs),
      body_(body),
      constant_def_(const_def) {}

std::vector<AstNode*> Let::GetChildren(bool want_types) const {
  std::vector<AstNode*> results = {name_def_tree_};
  if (type_annotation_ != nullptr && want_types) {
    results.push_back(type_annotation_);
  }
  results.push_back(rhs_);
  results.push_back(body_);
  if (constant_def_ != nullptr) {
    results.push_back(constant_def_);
  }
  return results;
}

std::string Let::ToString() const {
  return absl::StrFormat(
      "%s %s%s = %s;\n%s", constant_def_ == nullptr ? "let" : "const",
      name_def_tree_->ToString(),
      type_annotation_ == nullptr ? "" : ": " + type_annotation_->ToString(),
      rhs_->ToString(), body_->ToString());
}

// -- class Number

Number::Number(Module* owner, Span span, std::string text, NumberKind kind,
               TypeAnnotation* type_annotation)
    : Expr(owner, std::move(span)),
      text_(std::move(text)),
      kind_(kind),
      type_annotation_(type_annotation) {}

std::vector<AstNode*> Number::GetChildren(bool want_types) const {
  if (type_annotation_ == nullptr) {
    return {};
  }
  return {type_annotation_};
}

std::string Number::ToString() const {
  if (type_annotation_ != nullptr) {
    return absl::StrFormat("%s:%s", type_annotation_->ToString(), text_);
  }
  return text_;
}

std::string Number::ToStringNoType() const { return text_; }

absl::StatusOr<Bits> Number::GetBits(int64_t bit_count) const {
  switch (kind_) {
    case NumberKind::kBool: {
      Bits result(bit_count);
      return result.UpdateWithSet(0, text_ == "true");
    }
    case NumberKind::kCharacter: {
      XLS_RET_CHECK_EQ(text_.size(), 1);
      Bits result = Bits::FromBytes(/*bytes=*/{static_cast<uint8_t>(text_[0])},
                                    /*bit_count=*/CHAR_BIT);
      return bits_ops::ZeroExtend(result, bit_count);
    }
    case NumberKind::kOther: {
      XLS_ASSIGN_OR_RETURN(auto sm, GetSignAndMagnitude(text_));
      auto [sign, bits] = sm;
      if (bit_count < bits.bit_count()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Internal error: %s Cannot fit number value %s in %d bits; %d "
            "required: `%s`",
            span().ToString(), text_, bit_count, bits.bit_count(), ToString()));
      }
      bits = bits_ops::ZeroExtend(bits, bit_count);
      if (sign) {
        bits = bits_ops::Negate(bits);
      }
      return bits;
    }
  }
  return absl::InternalError(
      absl::StrFormat("Invalid NumberKind: %d", static_cast<int64_t>(kind_)));
}

TypeDef::TypeDef(Module* owner, Span span, NameDef* name_def,
                 TypeAnnotation* type, bool is_public)
    : AstNode(owner),
      span_(std::move(span)),
      name_def_(name_def),
      type_annotation_(type),
      is_public_(is_public) {}

std::string Array::ToString() const {
  std::string type_prefix;
  if (type_annotation_ != nullptr) {
    type_prefix = absl::StrCat(type_annotation_->ToString(), ":");
  }
  return absl::StrFormat("%s[%s%s]", type_prefix,
                         absl::StrJoin(members_, ", ",
                                       [](std::string* out, Expr* expr) {
                                         absl::StrAppend(out, expr->ToString());
                                       }),
                         has_ellipsis_ ? ", ..." : "");
}

std::vector<AstNode*> Array::GetChildren(bool want_types) const {
  std::vector<AstNode*> results;
  if (want_types && type_annotation_ != nullptr) {
    results.push_back(type_annotation_);
  }
  for (Expr* member : members_) {
    XLS_CHECK(member != nullptr);
    results.push_back(member);
  }
  return results;
}

}  // namespace xls::dslx
