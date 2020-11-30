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

#include "xls/dslx/cpp_ast.h"

#include "absl/status/statusor.h"
#include "absl/strings/strip.h"
#include "xls/common/indent.h"
#include "xls/common/status/ret_check.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/number_parser.h"

namespace xls::dslx {

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
  return absl::InvalidArgumentError("AST node is not a type definition.");
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
  std::string privacy;
  if (is_public_) {
    privacy = "pub ";
  }
  return absl::StrFormat("%sconst %s = %s;", privacy, name_def_->ToString(),
                         value_->ToString());
}

ConstantArray::ConstantArray(Module* owner, Span span,
                             std::vector<Expr*> members, bool has_ellipsis)
    : Array(owner, std::move(span), std::move(members), has_ellipsis) {
  for (Expr* expr : members) {
    XLS_CHECK(IsConstant(expr))
        << "non-constant in constant array: " << expr->ToString();
  }
}

absl::optional<ModuleMember*> Module::FindMemberWithName(
    absl::string_view target) {
  for (ModuleMember& member : top_) {
    if (absl::holds_alternative<Function*>(member)) {
      if (absl::get<Function*>(member)->identifier() == target) {
        return &member;
      }
    } else if (absl::holds_alternative<Test*>(member)) {
      if (absl::get<Test*>(member)->identifier() == target) {
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
  if (auto* n = dynamic_cast<Function*   >(node)) { return ModuleMember(n); }
  if (auto* n = dynamic_cast<Test*       >(node)) { return ModuleMember(n); }
  if (auto* n = dynamic_cast<QuickCheck* >(node)) { return ModuleMember(n); }
  if (auto* n = dynamic_cast<TypeDef*    >(node)) { return ModuleMember(n); }
  if (auto* n = dynamic_cast<StructDef*  >(node)) { return ModuleMember(n); }
  if (auto* n = dynamic_cast<ConstantDef*>(node)) { return ModuleMember(n); }
  if (auto* n = dynamic_cast<EnumDef*    >(node)) { return ModuleMember(n); }
  if (auto* n = dynamic_cast<Import*     >(node)) { return ModuleMember(n); }
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
  if (XlsTuple* n = dynamic_cast<XlsTuple*>(node)) {
    return std::all_of(n->members().begin(), n->members().end(), IsConstant);
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

std::string Match::ToString() const {
  std::string result = absl::StrFormat("match %s {\n", matched_->ToString());
  for (MatchArm* arm : arms_) {
    absl::StrAppend(&result, arm->ToString(), "\n");
  }
  absl::StrAppend(&result, "}");
  return result;
}

EnumDef::EnumDef(Module* owner, Span span, NameDef* name_def,
                 TypeAnnotation* type, std::vector<EnumMember> values,
                 bool is_public)
    : AstNode(owner),
      span_(std::move(span)),
      name_def_(name_def),
      type_(type),
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

absl::StatusOr<absl::variant<Number*, NameRef*>> EnumDef::GetValue(
    absl::string_view name) const {
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
      absl::StrFormat("enum %s : %s {\n", identifier(), type_->ToString());
  for (const auto& item : values_) {
    absl::StrAppendFormat(&result, "  %s = %s,\n", item.name_def->identifier(),
                          ToAstNode(item.value)->ToString());
  }
  absl::StrAppend(&result, "}");
  return result;
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

std::string StructDef::ToString() const {
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
  std::string pub_str = is_public_ ? "pub " : "";
  std::string name_str = name_def_->ToString();
  std::string body_str = Indent(body_->ToString());
  return absl::StrFormat("%sfn %s%s(%s)%s{\n%s\n}", pub_str, parametric_str,
                         name_str, params_str, return_type_str, body_str);
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

absl::StatusOr<Bits> Number::GetBits(int64 bit_count) const {
  switch (kind_) {
    case NumberKind::kBool: {
      Bits result(bit_count);
      return result.UpdateWithSet(0, text_ == "true");
    }
    case NumberKind::kCharacter: {
      XLS_RET_CHECK_EQ(text_.size(), 1);
      Bits result = Bits::FromBytes(/*bytes=*/{static_cast<uint8>(text_[0])},
                                    /*bit_count=*/CHAR_BIT);
      return bits_ops::ZeroExtend(result, bit_count);
    }
    case NumberKind::kOther: {
      XLS_ASSIGN_OR_RETURN(auto sm, GetSignAndMagnitude(text_));
      auto [sign, bits] = sm;
      if (bit_count < bits.bit_count()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Cannot fit number value %s in %d bits; %d required", text_,
            bit_count, bits.bit_count()));
      }
      bits = bits_ops::ZeroExtend(bits, bit_count);
      if (sign) {
        bits = bits_ops::Negate(bits);
      }
      return bits;
    }
  }
  return absl::InternalError(
      absl::StrFormat("Invalid NumberKind: %d", static_cast<int64>(kind_)));
}

}  // namespace xls::dslx
