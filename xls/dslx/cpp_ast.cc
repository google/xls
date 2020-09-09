// Copyright 2020 Google LLC
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

#include "xls/dslx/cpp_ast.h"

#include "xls/common/indent.h"

namespace xls::dslx {

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

FreeVariables AstNode::GetFreeVariables(Pos start_pos) {
  DfsIteratorNoTypes it(this);
  FreeVariables freevars;
  while (it.HasNext()) {
    AstNode* n = it.Next();
    if (auto* name_ref = dynamic_cast<NameRef*>(n)) {
      absl::optional<Pos> name_def_start = name_ref->GetNameDefStart();
      if (!name_def_start.has_value() || *name_def_start < start_pos) {
        freevars.Add(name_ref->identifier(), name_ref);
      }
    }
  }
  return freevars;
}

std::string EnumRef::GetEnumIdentifier() const {
  if (absl::holds_alternative<TypeDef*>(enum_def_)) {
    return absl::get<TypeDef*>(enum_def_)->identifier();
  }
  return absl::get<Enum*>(enum_def_)->identifier();
}

std::string BinopKindFormat(BinopKind kind) {
  switch (kind) {
    // clang-format off
    case BinopKind::kShll:       return "<<";
    case BinopKind::kShrl:       return ">>";
    case BinopKind::kShra:       return ">>>";
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
  std::string members_str = absl::StrJoin(
      members_, ", ",
      [](std::string* out, const std::pair<std::string, Expr*>& member) {
        absl::StrAppendFormat(out, "%s: %s", member.first,
                              member.second->ToString());
      });
  return absl::StrFormat("%s { %s }", GetStructNode()->ToString(), members_str);
}

std::string For::ToString() const {
  return absl::StrFormat(R"(for %s: %s in %s {
%s
}(%s))",
                         names_->ToString(), type_->ToString(),
                         iterable_->ToString(), Indent(body_->ToString()),
                         init_->ToString());
}

std::string ConstantDef::ToString() const {
  return absl::StrFormat("const %s = %s;", name_def_->ToString(),
                         value_->ToString());
}

ConstantArray::ConstantArray(Span span, std::vector<Expr*> members,
                             bool has_ellipsis)
    : Array(std::move(span), std::move(members), has_ellipsis) {
  for (Expr* expr : members) {
    XLS_CHECK(IsConstant(expr))
        << "non-constant in constant array: " << expr->ToString();
  }
}

std::vector<AstNode*> Module::GetChildren(bool want_types) const {
  std::vector<AstNode*> results;
  for (ModuleMember member : top_) {
    results.push_back(ToAstNode(member));
  }
  return results;
}

xabsl::StatusOr<ModuleMember> AsModuleMember(AstNode* node) {
  // clang-format off
  if (auto* n = dynamic_cast<Function*   >(node)) { return ModuleMember(n); }
  if (auto* n = dynamic_cast<Test*       >(node)) { return ModuleMember(n); }
  if (auto* n = dynamic_cast<QuickCheck* >(node)) { return ModuleMember(n); }
  if (auto* n = dynamic_cast<TypeDef*    >(node)) { return ModuleMember(n); }
  if (auto* n = dynamic_cast<Struct*     >(node)) { return ModuleMember(n); }
  if (auto* n = dynamic_cast<ConstantDef*>(node)) { return ModuleMember(n); }
  if (auto* n = dynamic_cast<Enum*       >(node)) { return ModuleMember(n); }
  if (auto* n = dynamic_cast<Import*     >(node)) { return ModuleMember(n); }
  // clang-format on
  return absl::InvalidArgumentError("AST node is not a module-level member: " +
                                    node->ToString());
}

xabsl::StatusOr<IndexRhs> AstNodeToIndexRhs(AstNode* node) {
  // clang-format off
  if (auto* n = dynamic_cast<Slice*     >(node)) { return IndexRhs(n); }
  if (auto* n = dynamic_cast<WidthSlice*>(node)) { return IndexRhs(n); }
  if (auto* n = dynamic_cast<Expr*      >(node)) { return IndexRhs(n); }
  // clang-format on
  return absl::InvalidArgumentError("AST node is not a valid 'index': " +
                                    node->ToString());
}

std::vector<AstNode*> ArrayTypeAnnotation::GetChildren(bool want_types) const {
  return {element_type_, dim_};
}

std::string ArrayTypeAnnotation::ToString() const {
  return absl::StrFormat("%s[%s]", element_type_->ToString(), dim_->ToString());
}

bool IsConstant(AstNode* node) {
  if (IsOneOf<ConstantArray, EnumRef, Number, ConstRef>(node)) {
    return true;
  }
  if (Cast* n = dynamic_cast<Cast*>(node)) {
    return IsConstant(n->expr());
  }
  if (XlsTuple* n = dynamic_cast<XlsTuple*>(node)) {
    return std::all_of(n->members().begin(), n->members().end(), IsConstant);
  }
  return false;
}

std::vector<AstNode*> SplatStructInstance::GetChildren(bool want_types) const {
  std::vector<AstNode*> results;
  if (want_types) {
    results.push_back(ToAstNode(struct_def_));
  }
  for (auto& item : members_) {
    results.push_back(item.second);
  }
  results.push_back(splatted_);
  return results;
}

std::vector<AstNode*> MatchArm::GetChildren(bool want_types) const {
  std::vector<AstNode*> results;
  for (NameDefTree* ndt : patterns_) {
    results.push_back(ndt);
  }
  results.push_back(expr_);
  return results;
}

std::string Match::ToString() const {
  std::string result = absl::StrFormat("match %s {\n", matched_->ToString());
  for (MatchArm* arm : arms_) {
    absl::StrAppend(&result, arm->ToString(), "\n");
  }
  absl::StrAppend(&result, "}");
  return result;
}

std::string Struct::ToString() const {
  std::string parametric_str;
  if (!parametric_bindings_.empty()) {
    std::string guts =
        absl::StrJoin(parametric_bindings_, ", ",
                      [](std::string* out, ParametricBinding* binding) {
                        absl::StrAppend(out, binding->ToString());
                      });
    parametric_str = absl::StrFormat("[%s] ", guts);
  }
  std::string result =
      absl::StrFormat("struct %s%s {\n", parametric_str, identifier());
  for (const auto& item : members_) {
    absl::StrAppendFormat(&result, "  %s: %s,\n", item.first->ToString(),
                          item.second->ToString());
  }
  absl::StrAppend(&result, "}");
  return result;
}

std::vector<AstNode*> TypeRefTypeAnnotation::GetChildren(
    bool want_types) const {
  std::vector<AstNode*> results = {type_ref_};
  for (Expr* e : parametrics_) {
    results.push_back(e);
  }
  return results;
}

std::string TypeRefTypeAnnotation::ToString() const {
  return type_ref_->ToString();
}

BinopKind BinopKindFromString(absl::string_view s) {
#define HANDLE(__enum, __unused, __operator) \
  if (s == __operator) {                     \
    return BinopKind::__enum;                \
  }
  BINOP_KIND_EACH(HANDLE)
#undef HANDLE
  XLS_LOG(FATAL) << "Invalid BinopKind string: \"" << s << "\"";
}

UnopKind UnopKindFromString(absl::string_view s) {
  if (s == "!") {
    return UnopKind::kInvert;
  }
  if (s == "-") {
    return UnopKind::kNegate;
  }
  XLS_LOG(FATAL) << "Invalid UnopKind string: \"" << s << "\"";
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

std::vector<AstNode*> For::GetChildren(bool want_types) const {
  std::vector<AstNode*> results = {names_};
  if (want_types) {
    results.push_back(type_);
  }
  results.push_back(iterable_);
  results.push_back(body_);
  results.push_back(init_);
  return results;
}

std::string Function::Format(bool include_body) const {
  std::string parametric_str;
  if (!parametric_bindings_.empty()) {
    parametric_str = absl::StrFormat(
        " [%s] ",
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
  std::string return_type_str;
  if (return_type_ != nullptr) {
    return_type_str = " -> " + return_type_->ToString() + " ";
  }
  return absl::StrFormat("%sfn %s%s(%s)%s{\n%s\n}", is_public_ ? "pub " : "",
                         parametric_str, name_def_->ToString(), params_str,
                         return_type_str, Indent(body_->ToString()));
}

std::string Proc::ToString() const { XLS_LOG(FATAL) << "boom"; }

int64 BuiltinTypeAnnotation::GetBitCount() const {
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

std::string XlsTuple::ToString() const {
  std::string result = "(";
  for (int64 i = 0; i < members_.size(); ++i) {
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

std::string StructDefinitionToText(absl::variant<Struct*, ModRef*> struct_) {
  if (absl::holds_alternative<Struct*>(struct_)) {
    return absl::get<Struct*>(struct_)->identifier();
  }
  if (absl::holds_alternative<ModRef*>(struct_)) {
    return absl::get<ModRef*>(struct_)->ToString();
  }
  XLS_LOG(FATAL)
      << "Unhandled alternative for converting struct definition to string.";
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

std::string Let::ToString() const {
  return absl::StrFormat("%s %s%s = %s;\n%s",
                         constant_def_ == nullptr ? "let" : "const",
                         name_def_tree_->ToString(),
                         type_ == nullptr ? "" : ": " + type_->ToString(),
                         rhs_->ToString(), body_->ToString());
}

}  // namespace xls::dslx
