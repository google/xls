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

#include "xls/dslx/frontend/ast.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/casts.h"
#include "xls/common/indent.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/frontend/ast_builtin_types.inc"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/token_utils.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/format_strings.h"
#include "xls/ir/number_parser.h"

namespace xls::dslx {
namespace {

constexpr std::string_view kRustOneIndent = "    ";

class DfsIteratorNoTypes {
 public:
  explicit DfsIteratorNoTypes(const AstNode* start) : to_visit_({start}) {}

  bool HasNext() const { return !to_visit_.empty(); }

  const AstNode* Next() {
    const AstNode* result = to_visit_.front();
    to_visit_.pop_front();
    std::vector<AstNode*> children = result->GetChildren(/*want_types=*/false);
    std::reverse(children.begin(), children.end());
    for (AstNode* c : children) {
      to_visit_.push_front(c);
    }
    return result;
  }

 private:
  std::deque<const AstNode*> to_visit_;
};

static AnyNameDef GetSubjectNameDef(const ColonRef::Subject& subject) {
  return absl::visit(
      Visitor{
          [](NameRef* n) { return n->name_def(); },
          [](ColonRef* n) { return GetSubjectNameDef(n->subject()); },
      },
      subject);
}

void Parenthesize(std::string* s) { *s = absl::StrCat("(", *s, ")"); }

std::string MakeExternTypeAttr(const std::optional<std::string>& name) {
  if (name) {
    return absl::StrFormat("#[sv_type(\"%s\")]\n", *name);
  }
  return "";
}
}  // namespace

std::string_view FunctionTagToString(FunctionTag tag) {
  switch (tag) {
    case FunctionTag::kNormal:
      return "normal";
    case FunctionTag::kProcConfig:
      return "proc config";
    case FunctionTag::kProcNext:
      return "proc next";
    case FunctionTag::kProcInit:
      return "proc init";
  }
  LOG(FATAL) << "Out-of-range function tag: " << static_cast<int>(tag);
}

std::string_view PrecedenceToString(Precedence p) {
  switch (p) {
    case Precedence::kStrongest:
      return "strongest";
    case Precedence::kPaths:
      return "paths";
    case Precedence::kMethodCall:
      return "method-call";

    case Precedence::kFieldExpression:
      return "field-expression";
    case Precedence::kFunctionCallOrArrayIndex:
      return "function-call-or-array-index";
    case Precedence::kQuestionMark:
      return "question-mark";
    case Precedence::kUnaryOp:
      return "unary";
    case Precedence::kAs:
      return "as";
    case Precedence::kStrongArithmetic:
      return "strong-arithmetic";
    case Precedence::kWeakArithmetic:
      return "weak-arithmetic";
    case Precedence::kShift:
      return "shift";

    case Precedence::kConcat:
      return "concat";

    case Precedence::kBitwiseAnd:
      return "bitwise-and";
    case Precedence::kBitwiseXor:
      return "bitwise-xor";
    case Precedence::kBitwiseOr:
      return "bitwise-or";
    case Precedence::kComparison:
      return "comparison";
    case Precedence::kLogicalAnd:
      return "logical-and";
    case Precedence::kLogicalOr:
      return "logical-or";
    case Precedence::kRange:
      return "range";
    case Precedence::kEquals:
      return "equals";
    case Precedence::kReturn:
      return "return";

    case Precedence::kWeakest:
      return "weakest";
  }

  LOG(FATAL) << "Invalid precedence value: " << static_cast<int>(p);
}

constexpr int64_t kTargetLineChars = 80;

ExprOrType ToExprOrType(AstNode* n) {
  if (Expr* e = dynamic_cast<Expr*>(n)) {
    return e;
  }
  auto* type = down_cast<TypeAnnotation*>(n);
  CHECK_NE(type, nullptr);
  return type;
}

std::string_view AstNodeKindToString(AstNodeKind kind) {
  switch (kind) {
    case AstNodeKind::kConstAssert:
      return "const assert";
    case AstNodeKind::kStatement:
      return "statement";
    case AstNodeKind::kTypeAnnotation:
      return "type annotation";
    case AstNodeKind::kModule:
      return "module";
    case AstNodeKind::kNameDef:
      return "name definition";
    case AstNodeKind::kBuiltinNameDef:
      return "builtin name definition";
    case AstNodeKind::kConditional:
      return "conditional";
    case AstNodeKind::kTypeAlias:
      return "type alias";
    case AstNodeKind::kNumber:
      return "number";
    case AstNodeKind::kTypeRef:
      return "type reference";
    case AstNodeKind::kImport:
      return "import";
    case AstNodeKind::kImpl:
      return "impl";
    case AstNodeKind::kUnop:
      return "unary op";
    case AstNodeKind::kBinop:
      return "binary op";
    case AstNodeKind::kColonRef:
      return "colon reference";
    case AstNodeKind::kParam:
      return "parameter";
    case AstNodeKind::kFunction:
      return "function";
    case AstNodeKind::kProc:
      return "proc";
    case AstNodeKind::kProcMember:
      return "proc member";
    case AstNodeKind::kNameRef:
      return "name reference";
    case AstNodeKind::kConstRef:
      return "const reference";
    case AstNodeKind::kArray:
      return "array";
    case AstNodeKind::kString:
      return "string";
    case AstNodeKind::kStructInstance:
      return "struct instance";
    case AstNodeKind::kSplatStructInstance:
      return "splat struct instance";
    case AstNodeKind::kNameDefTree:
      return "name definition tree";
    case AstNodeKind::kIndex:
      return "index";
    case AstNodeKind::kRange:
      return "range";
    case AstNodeKind::kRecv:
      return "receive";
    case AstNodeKind::kRecvNonBlocking:
      return "receive-non-blocking";
    case AstNodeKind::kRecvIf:
      return "receive-if";
    case AstNodeKind::kRecvIfNonBlocking:
      return "receive-if-non-blocking";
    case AstNodeKind::kSend:
      return "send";
    case AstNodeKind::kSendIf:
      return "send-if";
    case AstNodeKind::kJoin:
      return "join";
    case AstNodeKind::kTestFunction:
      return "test function";
    case AstNodeKind::kTestProc:
      return "test proc";
    case AstNodeKind::kWidthSlice:
      return "width slice";
    case AstNodeKind::kWildcardPattern:
      return "wildcard pattern";
    case AstNodeKind::kMatchArm:
      return "match arm";
    case AstNodeKind::kMatch:
      return "match";
    case AstNodeKind::kAttr:
      return "attribute";
    case AstNodeKind::kInstantiation:
      return "instantiation";
    case AstNodeKind::kInvocation:
      return "invocation";
    case AstNodeKind::kSpawn:
      return "spawn";
    case AstNodeKind::kFormatMacro:
      return "format macro";
    case AstNodeKind::kZeroMacro:
      return "zero macro";
    case AstNodeKind::kAllOnesMacro:
      return "all-ones macro";
    case AstNodeKind::kSlice:
      return "slice";
    case AstNodeKind::kEnumDef:
      return "enum definition";
    case AstNodeKind::kStructDef:
      return "struct definition";
    case AstNodeKind::kQuickCheck:
      return "quick-check";
    case AstNodeKind::kXlsTuple:
      return "tuple";
    case AstNodeKind::kRestOfTuple:
      return "rest of tuple";
    case AstNodeKind::kFor:
      return "for";
    case AstNodeKind::kStatementBlock:
      return "statement-block";
    case AstNodeKind::kCast:
      return "cast";
    case AstNodeKind::kConstantDef:
      return "constant definition";
    case AstNodeKind::kLet:
      return "let";
    case AstNodeKind::kChannelDecl:
      return "channel declaration";
    case AstNodeKind::kParametricBinding:
      return "parametric binding";
    case AstNodeKind::kTupleIndex:
      return "tuple index";
    case AstNodeKind::kUnrollFor:
      return "unroll-for";
  }
  LOG(FATAL) << "Out-of-range AstNodeKind: " << static_cast<int>(kind);
}

AnyNameDef TypeDefinitionGetNameDef(const TypeDefinition& td) {
  return absl::visit(
      Visitor{
          [](TypeAlias* n) -> AnyNameDef { return &n->name_def(); },
          [](StructDef* n) -> AnyNameDef { return n->name_def(); },
          [](EnumDef* n) -> AnyNameDef { return n->name_def(); },
          [](ColonRef* n) -> AnyNameDef {
            return GetSubjectNameDef(n->subject());
          },
      },
      td);
}

absl::StatusOr<TypeDefinition> ToTypeDefinition(AstNode* node) {
  if (auto* n = dynamic_cast<TypeAlias*>(node)) {
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

const Span& FreeVariables::GetFirstNameRefSpan(
    std::string_view identifier) const {
  std::vector<const NameRef*> name_refs = values_.at(identifier);
  CHECK(!name_refs.empty());
  return name_refs.at(0)->span();
}

FreeVariables FreeVariables::DropBuiltinDefs() const {
  FreeVariables result;
  for (const auto& [identifier, name_refs] : values_) {
    for (const NameRef* ref : name_refs) {
      auto def = ref->name_def();
      if (std::holds_alternative<BuiltinNameDef*>(def)) {
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
    const NameRef* ref = item.second[0];
    result.push_back({item.first, ref->name_def()});
  }
  std::sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.first < rhs.first;
  });
  return result;
}

std::vector<const ConstRef*> FreeVariables::GetConstRefs() {
  std::vector<const ConstRef*> const_refs;
  for (const auto& [name, refs] : values_) {
    for (const NameRef* name_ref : refs) {
      if (auto* const_ref = dynamic_cast<const ConstRef*>(name_ref)) {
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

void FreeVariables::Add(std::string identifier, const NameRef* name_ref) {
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

FreeVariables GetFreeVariablesByLambda(
    const AstNode* node,
    const std::function<bool(const NameRef&)>& consider_free) {
  DfsIteratorNoTypes it(node);
  FreeVariables freevars;
  while (it.HasNext()) {
    const AstNode* n = it.Next();
    if (const auto* name_ref = dynamic_cast<const NameRef*>(n)) {
      if (consider_free == nullptr || consider_free(*name_ref)) {
        freevars.Add(name_ref->identifier(), name_ref);
      }
    }
  }
  return freevars;
}

FreeVariables GetFreeVariablesByPos(const AstNode* node, const Pos* start_pos) {
  std::function<bool(const NameRef&)> consider_free = nullptr;
  if (start_pos != nullptr) {
    consider_free = [start_pos](const NameRef& name_ref) {
      std::optional<Pos> name_def_start = name_ref.GetNameDefStart();
      return !name_def_start.has_value() || name_def_start.value() < *start_pos;
    };
  }
  return GetFreeVariablesByLambda(node, consider_free);
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
#define TEST(__enum, __name, __str, __signedness, __width)           \
  {                                                                  \
    std::optional<bool> signedness = __signedness;                   \
    if (signedness.has_value() && signedness.value() == is_signed && \
        __width == width) {                                          \
      return BuiltinType::__enum;                                    \
    }                                                                \
  }
  XLS_DSLX_BUILTIN_TYPE_EACH(TEST)
#undef TEST
  return absl::NotFoundError(
      absl::StrFormat("Cannot find built in type with signedness: %d width: %d",
                      is_signed, width));
}

absl::StatusOr<bool> GetBuiltinTypeSignedness(BuiltinType type) {
  switch (type) {
#define CASE(__enum, _unused1, __str, __signedness, _unused3)           \
  case BuiltinType::__enum: {                                           \
    std::optional<bool> signedness = __signedness;                      \
    if (!signedness.has_value()) {                                      \
      return absl::InvalidArgumentError("Type " #__str                  \
                                        " has no defined signedness."); \
    }                                                                   \
    return signedness.value();                                          \
  }
    XLS_DSLX_BUILTIN_TYPE_EACH(CASE)
#undef CASE
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unknown builtin type: ", static_cast<int64_t>(type)));
}

absl::StatusOr<int64_t> GetBuiltinTypeBitCount(BuiltinType type) {
  switch (type) {
#define CASE(__enum, _unused1, _unused2, _unused3, __width) \
  case BuiltinType::__enum:                                 \
    return __width;
    XLS_DSLX_BUILTIN_TYPE_EACH(CASE)
#undef CASE
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unknown builtin type: ", static_cast<int64_t>(type)));
}

absl::StatusOr<BuiltinType> BuiltinTypeFromString(std::string_view s) {
#define CASE(__enum, __unused, __str, ...) \
  if (s == __str) {                        \
    return BuiltinType::__enum;            \
  }
  XLS_DSLX_BUILTIN_TYPE_EACH(CASE)
#undef CASE
  return absl::InvalidArgumentError(
      absl::StrFormat("String is not a BuiltinType: \"%s\"", s));
}

const absl::btree_set<BinopKind>& GetBinopSameTypeKinds() {
  static const absl::NoDestructor<absl::btree_set<BinopKind>> singleton({
      BinopKind::kAdd,
      BinopKind::kSub,
      BinopKind::kMul,
      BinopKind::kAnd,
      BinopKind::kOr,
      BinopKind::kXor,
      BinopKind::kDiv,
  });
  return *singleton;
}

const absl::btree_set<BinopKind>& GetBinopComparisonKinds() {
  static const absl::NoDestructor<absl::btree_set<BinopKind>> singleton({
      BinopKind::kGe,
      BinopKind::kGt,
      BinopKind::kLe,
      BinopKind::kLt,
      BinopKind::kEq,
      BinopKind::kNe,
  });
  return *singleton;
}

const absl::btree_set<BinopKind>& GetBinopLogicalKinds() {
  static const absl::NoDestructor<absl::btree_set<BinopKind>> singleton({
      BinopKind::kLogicalOr,
      BinopKind::kLogicalAnd,
  });
  return *singleton;
}

const absl::btree_set<BinopKind>& GetBinopShifts() {
  static const absl::NoDestructor<absl::btree_set<BinopKind>> singleton({
      BinopKind::kShl,
      BinopKind::kShr,
  });
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
    case BinopKind::kMod:        return "%";
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
      span_(std::move(span)),
      identifier_(std::move(identifier)),
      definer_(definer) {}

NameDef::~NameDef() = default;

// -- class Conditional

Conditional::Conditional(Module* owner, Span span, Expr* test,
                         StatementBlock* consequent,
                         std::variant<StatementBlock*, Conditional*> alternate)
    : Expr(owner, std::move(span)),
      test_(test),
      consequent_(consequent),
      alternate_(alternate) {}

Conditional::~Conditional() = default;

std::string Conditional::ToStringInternal() const {
  std::string inline_str = absl::StrFormat(
      R"(if %s %s else %s)", test_->ToInlineString(),
      consequent_->ToInlineString(), ToAstNode(alternate_)->ToInlineString());
  if (inline_str.size() <= kTargetLineChars) {
    return inline_str;
  }
  return absl::StrFormat(R"(if %s %s else %s)", test_->ToString(),
                         consequent_->ToString(),
                         ToAstNode(alternate_)->ToString());
}

bool Conditional::HasMultiStatementBlocks() const {
  if (consequent_->size() > 1) {
    return true;
  }
  return absl::visit(
      Visitor{
          [](const StatementBlock* block) { return block->size() > 1; },
          [](const Conditional* elseif) {
            return elseif->HasMultiStatementBlocks();
          },
      },
      alternate_);
}

// -- class Attr

Attr::~Attr() = default;

// -- class ParametricBinding

ParametricBinding::ParametricBinding(Module* owner, NameDef* name_def,
                                     TypeAnnotation* type_annotation,
                                     Expr* expr)
    : AstNode(owner),
      name_def_(name_def),
      type_annotation_(type_annotation),
      expr_(expr) {
  CHECK_EQ(name_def_->owner(), owner);
  CHECK_EQ(type_annotation_->owner(), owner);
}

ParametricBinding::~ParametricBinding() = default;

std::string ParametricBinding::ToString() const {
  std::string suffix;
  if (expr_ != nullptr) {
    suffix = absl::StrFormat(" = {%s}", expr_->ToString());
  }
  return absl::StrFormat("%s: %s%s", name_def_->ToString(),
                         type_annotation_->ToString(), suffix);
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

For::~For() = default;

UnrollFor::~UnrollFor() = default;

ConstantDef::ConstantDef(Module* owner, Span span, NameDef* name_def,
                         TypeAnnotation* type_annotation, Expr* value,
                         bool is_public)
    : AstNode(owner),
      span_(std::move(span)),
      name_def_(name_def),
      type_annotation_(type_annotation),
      value_(value),
      is_public_(is_public) {}

ConstantDef::~ConstantDef() = default;

std::string ConstantDef::ToString() const {
  std::string privacy;
  if (is_public_) {
    privacy = "pub ";
  }
  std::string type_annotation_str;
  if (type_annotation_ != nullptr) {
    type_annotation_str = absl::StrCat(": ", type_annotation_->ToString());
  }
  return absl::StrFormat("%sconst %s%s = %s;", privacy, name_def_->ToString(),
                         type_annotation_str, value_->ToString());
}

Array::Array(Module* owner, Span span, std::vector<Expr*> members,
             bool has_ellipsis)
    : Expr(owner, std::move(span)),
      members_(std::move(members)),
      has_ellipsis_(has_ellipsis) {}

ConstantArray::ConstantArray(Module* owner, Span span,
                             std::vector<Expr*> members, bool has_ellipsis)
    : Array(owner, std::move(span), std::move(members), has_ellipsis) {
  for (Expr* expr : this->members()) {
    CHECK(IsConstant(expr))
        << "non-constant in constant array: " << expr->ToString();
  }
}

ConstantArray::~ConstantArray() = default;

// -- class TypeRef

TypeRef::TypeRef(Module* owner, Span span, TypeDefinition type_definition)
    : AstNode(owner),
      span_(std::move(span)),
      type_definition_(type_definition) {}

std::string TypeRef::ToString() const {
  return absl::visit(Visitor{[&](TypeAlias* n) { return n->identifier(); },
                             [&](StructDef* n) { return n->identifier(); },
                             [&](EnumDef* n) { return n->identifier(); },
                             [&](ColonRef* n) { return n->ToString(); }},
                     type_definition_);
}

TypeRef::~TypeRef() = default;

// -- class Import

Import::Import(Module* owner, Span span, std::vector<std::string> subject,
               NameDef& name_def, std::optional<std::string> alias)
    : AstNode(owner),
      span_(std::move(span)),
      subject_(std::move(subject)),
      name_def_(name_def),
      alias_(std::move(alias)) {
  CHECK(!subject_.empty());
}

Import::~Import() = default;

std::string Import::ToString() const {
  if (alias_.has_value()) {
    return absl::StrFormat("import %s as %s;", absl::StrJoin(subject_, "."),
                           *alias_);
  }
  return absl::StrFormat("import %s;", absl::StrJoin(subject_, "."));
}

// -- class ColonRef

ColonRef::ColonRef(Module* owner, Span span, Subject subject, std::string attr)
    : Expr(owner, std::move(span)), subject_(subject), attr_(std::move(attr)) {}

ColonRef::~ColonRef() = default;

std::optional<Import*> ColonRef::ResolveImportSubject() const {
  if (!std::holds_alternative<NameRef*>(subject_)) {
    return std::nullopt;
  }
  auto* name_ref = std::get<NameRef*>(subject_);
  AnyNameDef any_name_def = name_ref->name_def();
  if (!std::holds_alternative<const NameDef*>(any_name_def)) {
    return std::nullopt;
  }
  const auto* name_def = std::get<const NameDef*>(any_name_def);
  AstNode* definer = name_def->definer();
  Import* import = dynamic_cast<Import*>(definer);
  if (import == nullptr) {
    return std::nullopt;
  }
  return import;
}

// -- class Param

Param::Param(Module* owner, NameDef* name_def, TypeAnnotation* type_annotation)
    : AstNode(owner),
      name_def_(name_def),
      type_annotation_(type_annotation),
      span_(name_def_->span().start(), type_annotation_->span().limit()) {}

Param::~Param() = default;

// -- class ChannelDecl

ChannelDecl::~ChannelDecl() = default;

std::string ChannelDecl::ToStringInternal() const {
  std::vector<std::string> dims;
  if (dims_.has_value()) {
    for (const Expr* dim : dims_.value()) {
      dims.push_back(absl::StrCat("[", dim->ToString(), "]"));
    }
  }

  std::string fifo_depth_str;
  if (fifo_depth_.has_value()) {
    fifo_depth_str = absl::StrCat(", ", fifo_depth_.value()->ToString());
  }
  return absl::StrFormat("chan<%s%s>%s(%s)", type_->ToString(), fifo_depth_str,
                         absl::StrJoin(dims, ""),
                         channel_name_expr_.ToString());
}

TypeAnnotation::~TypeAnnotation() = default;

// -- class TypeRefTypeAnnotation

TypeRefTypeAnnotation::TypeRefTypeAnnotation(
    Module* owner, Span span, TypeRef* type_ref,
    std::vector<ExprOrType> parametrics)
    : TypeAnnotation(owner, std::move(span)),
      type_ref_(type_ref),
      parametrics_(std::move(parametrics)) {}

TypeRefTypeAnnotation::~TypeRefTypeAnnotation() = default;

std::vector<AstNode*> TypeRefTypeAnnotation::GetChildren(
    bool want_types) const {
  std::vector<AstNode*> results = {type_ref_};
  for (const ExprOrType& e : parametrics_) {
    if (std::holds_alternative<TypeAnnotation*>(e)) {
      if (want_types) {
        results.push_back(std::get<TypeAnnotation*>(e));
      }
    } else {
      results.push_back(std::get<Expr*>(e));
    }
  }
  return results;
}

std::string TypeRefTypeAnnotation::ToString() const {
  std::string parametric_str;
  if (!parametrics_.empty()) {
    std::vector<std::string> pieces;
    pieces.reserve(parametrics_.size());
    for (const ExprOrType& e : parametrics_) {
      pieces.push_back(ToAstNode(e)->ToString());
    }
    parametric_str = absl::StrCat("<", absl::StrJoin(pieces, ", "), ">");
  }
  return absl::StrCat(type_ref_->ToString(), parametric_str);
}

// -- class ArrayTypeAnnotation

ArrayTypeAnnotation::ArrayTypeAnnotation(Module* owner, Span span,
                                         TypeAnnotation* element_type,
                                         Expr* dim)
    : TypeAnnotation(owner, std::move(span)),
      element_type_(element_type),
      dim_(dim) {}

ArrayTypeAnnotation::~ArrayTypeAnnotation() = default;

std::vector<AstNode*> ArrayTypeAnnotation::GetChildren(bool want_types) const {
  return {element_type_, dim_};
}

std::string ArrayTypeAnnotation::ToString() const {
  return absl::StrFormat("%s[%s]", element_type_->ToString(), dim_->ToString());
}

// -- class BuiltinNameDef

BuiltinNameDef::~BuiltinNameDef() = default;

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

std::vector<AstNode*> MatchArm::GetChildren(bool want_types) const {
  std::vector<AstNode*> results;
  results.reserve(patterns_.size());
  for (NameDefTree* ndt : patterns_) {
    results.push_back(ndt);
  }
  results.push_back(expr_);
  return results;
}

// -- class Match

Match::~Match() = default;

std::vector<AstNode*> Match::GetChildren(bool want_types) const {
  std::vector<AstNode*> results = {matched_};
  for (MatchArm* arm : arms_) {
    results.push_back(arm);
  }
  return results;
}

std::string Match::ToStringInternal() const {
  std::string result = absl::StrFormat("match %s {\n", matched_->ToString());
  for (MatchArm* arm : arms_) {
    absl::StrAppend(&result, Indent(absl::StrCat(arm->ToString(), ",\n"),
                                    kRustSpacesPerIndent));
  }
  absl::StrAppend(&result, "}");
  return result;
}

// -- class Index

Index::~Index() = default;

std::string Index::ToStringInternal() const {
  std::string lhs = lhs_->ToString();
  if (WeakerThan(lhs_->GetPrecedence(), GetPrecedenceWithoutParens())) {
    Parenthesize(&lhs);
  }
  return absl::StrFormat("%s[%s]", lhs, ToAstNode(rhs_)->ToString());
}

// -- class WidthSlice

WidthSlice::~WidthSlice() = default;

std::string WidthSlice::ToString() const {
  return absl::StrFormat("%s+:%s", start_->ToString(), width_->ToString());
}

// -- class Slice

Slice::~Slice() = default;

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

EnumDef::~EnumDef() = default;

NameDef* EnumDef::GetNameDef(std::string_view target) {
  for (const EnumMember& member : values_) {
    if (member.name_def->identifier() == target) {
      return member.name_def;
    }
  }
  LOG(FATAL) << "EnumDef::GetNameDef; no member: " << target;
}

bool EnumDef::HasValue(std::string_view name) const {
  for (const auto& item : values_) {
    if (item.name_def->identifier() == name) {
      return true;
    }
  }
  return false;
}

absl::StatusOr<Expr*> EnumDef::GetValue(std::string_view name) const {
  for (const EnumMember& item : values_) {
    if (item.name_def->identifier() == name) {
      return item.value;
    }
  }
  return absl::NotFoundError(absl::StrFormat(
      "Enum %s has no value with name \"%s\"", identifier(), name));
}

std::string EnumDef::ToString() const {
  std::string type_str;
  std::string extern_attr = MakeExternTypeAttr(extern_type_name_);
  if (type_annotation_ != nullptr) {
    type_str = absl::StrCat(" : " + type_annotation_->ToString());
  }
  std::string result =
      absl::StrFormat("%s%senum %s%s {\n", extern_attr,
                      is_public_ ? "pub " : "", identifier(), type_str);

  auto value_to_string = [](Expr* value) -> std::string {
    if (Number* number = dynamic_cast<Number*>(value)) {
      return number->ToStringNoType();
    }
    return value->ToString();
  };

  for (const auto& item : values_) {
    absl::StrAppendFormat(&result, "%s%s = %s,\n", kRustOneIndent,
                          item.name_def->identifier(),
                          value_to_string(item.value));
  }
  absl::StrAppend(&result, "}");
  return result;
}

// -- class Instantiation

Instantiation::Instantiation(Module* owner, Span span, Expr* callee,
                             std::vector<ExprOrType> explicit_parametrics)
    : Expr(owner, std::move(span)),
      callee_(callee),
      explicit_parametrics_(std::move(explicit_parametrics)) {}

Instantiation::~Instantiation() = default;

std::string Instantiation::FormatParametrics() const {
  if (explicit_parametrics_.empty()) {
    return "";
  }

  return absl::StrCat("<",
                      absl::StrJoin(explicit_parametrics_, ", ",
                                    [](std::string* out, ExprOrType e) {
                                      absl::StrAppend(out,
                                                      ToAstNode(e)->ToString());
                                    }),
                      ">");
}

// -- class Invocation

Invocation::Invocation(Module* owner, Span span, Expr* callee,
                       std::vector<Expr*> args,
                       std::vector<ExprOrType> explicit_parametrics)
    : Instantiation(owner, std::move(span), callee,
                    std::move(explicit_parametrics)),
      args_(std::move(args)) {}

Invocation::~Invocation() = default;

std::vector<AstNode*> Invocation::GetChildren(bool want_types) const {
  std::vector<AstNode*> results = {callee()};
  for (const ExprOrType& eot : explicit_parametrics()) {
    results.push_back(ToAstNode(eot));
  }
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

// -- class Spawn

Spawn::Spawn(Module* owner, Span span, Expr* callee, Invocation* config,
             Invocation* next, std::vector<ExprOrType> explicit_parametrics)
    : Instantiation(owner, std::move(span), callee,
                    std::move(explicit_parametrics)),
      config_(config),
      next_(next) {}

Spawn::~Spawn() = default;

std::vector<AstNode*> Spawn::GetChildren(bool want_types) const {
  return {config_, next_};
}

std::string Spawn::ToStringInternal() const {
  std::string param_str;
  if (!explicit_parametrics().empty()) {
    param_str = FormatParametrics();
  }

  std::string config_args = absl::StrJoin(
      config_->args(), ", ",
      [](std::string* out, Expr* e) { absl::StrAppend(out, e->ToString()); });

  return absl::StrFormat("spawn %s%s(%s)", callee()->ToString(), param_str,
                         config_args);
}

// -- class ConstAssert

ConstAssert::ConstAssert(Module* owner, Span span, Expr* arg)
    : AstNode(owner), span_(std::move(span)), arg_(arg) {}

ConstAssert::~ConstAssert() = default;

std::vector<AstNode*> ConstAssert::GetChildren(bool want_types) const {
  return std::vector<AstNode*>{arg()};
}

std::string ConstAssert::ToString() const {
  return absl::StrFormat("const_assert!(%s);", arg()->ToString());
}

// -- class ZeroMacro

ZeroMacro::ZeroMacro(Module* owner, Span span, ExprOrType type)
    : Expr(owner, std::move(span)), type_(type) {}

ZeroMacro::~ZeroMacro() = default;

std::vector<AstNode*> ZeroMacro::GetChildren(bool want_types) const {
  if (want_types) {
    return {ToAstNode(type_)};
  }
  return {};
}

std::string ZeroMacro::ToStringInternal() const {
  return absl::StrFormat("zero!<%s>()", ToAstNode(type_)->ToString());
}

// -- class AllOnesMacro

AllOnesMacro::AllOnesMacro(Module* owner, Span span, ExprOrType type)
    : Expr(owner, std::move(span)), type_(type) {}

AllOnesMacro::~AllOnesMacro() = default;

std::vector<AstNode*> AllOnesMacro::GetChildren(bool want_types) const {
  if (want_types) {
    return {ToAstNode(type_)};
  }
  return {};
}

std::string AllOnesMacro::ToStringInternal() const {
  return absl::StrFormat("all_ones!<%s>()", ToAstNode(type_)->ToString());
}

// -- class FormatMacro

FormatMacro::FormatMacro(Module* owner, Span span, std::string macro,
                         std::vector<FormatStep> format,
                         std::vector<Expr*> args)
    : Expr(owner, std::move(span)),
      macro_(std::move(macro)),
      format_(std::move(format)),
      args_(std::move(args)) {}

FormatMacro::~FormatMacro() = default;

std::vector<AstNode*> FormatMacro::GetChildren(bool want_types) const {
  std::vector<AstNode*> results;
  results.reserve(args_.size());
  for (Expr* arg : args_) {
    results.push_back(arg);
  }
  return results;
}

std::string FormatMacro::ToStringInternal() const {
  std::string format_string = "\"";
  for (const auto& step : format_) {
    if (std::holds_alternative<std::string>(step)) {
      absl::StrAppend(&format_string, std::get<std::string>(step));
    } else {
      absl::StrAppend(&format_string,
                      std::string(FormatPreferenceToXlsSpecifier(
                          std::get<FormatPreference>(step))));
    }
  }
  absl::StrAppend(&format_string, "\"");
  return absl::StrFormat("%s(%s, %s)", macro_, format_string, FormatArgs());
}

std::string FormatMacro::FormatArgs() const {
  return absl::StrJoin(args_, ", ", [](std::string* out, Expr* e) {
    absl::StrAppend(out, e->ToString());
  });
}

// -- class StructDef

StructDef::StructDef(Module* owner, Span span, NameDef* name_def,
                     std::vector<ParametricBinding*> parametric_bindings,
                     std::vector<StructMember> members, bool is_public)
    : AstNode(owner),
      span_(std::move(span)),
      name_def_(name_def),
      parametric_bindings_(std::move(parametric_bindings)),
      members_(std::move(members)),
      public_(is_public) {}

StructDef::~StructDef() = default;

std::vector<AstNode*> StructDef::GetChildren(bool want_types) const {
  std::vector<AstNode*> results = {name_def_};
  for (auto* pb : parametric_bindings_) {
    results.push_back(pb);
  }
  if (want_types) {
    for (const auto& member : members_) {
      results.push_back(member.type);
    }
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
  std::string extern_type_def = MakeExternTypeAttr(extern_type_name_);
  std::string result =
      absl::StrFormat("%s%sstruct %s%s {\n", extern_type_def,
                      public_ ? "pub " : "", identifier(), parametric_str);
  for (const auto& item : members_) {
    absl::StrAppendFormat(&result, "%s%s: %s,\n", kRustOneIndent, item.name,
                          item.type->ToString());
  }
  absl::StrAppend(&result, "}");
  return result;
}

std::vector<std::string> StructDef::GetMemberNames() const {
  std::vector<std::string> names;
  names.reserve(members_.size());
  for (auto& item : members_) {
    names.push_back(item.name);
  }
  return names;
}

// -- class Impl

Impl::Impl(Module* owner, Span span, TypeAnnotation* struct_ref, bool is_public)
    : AstNode(owner),
      span_(span),
      struct_ref_(struct_ref),
      public_(is_public) {}

Impl::~Impl() = default;

std::string Impl::ToString() const {
  std::string type_name = ToAstNode(struct_ref_)->ToString();
  return absl::StrFormat("%simpl %s {}", public_ ? "pub " : "", type_name);
}

// -- class StructInstance

StructInstance::StructInstance(
    Module* owner, Span span, TypeAnnotation* struct_ref,
    std::vector<std::pair<std::string, Expr*>> members)
    : Expr(owner, std::move(span)),
      struct_ref_(struct_ref),
      members_(std::move(members)) {}

StructInstance::~StructInstance() = default;

std::vector<std::pair<std::string, Expr*>> StructInstance::GetOrderedMembers(
    const StructDef* struct_def) const {
  std::vector<std::pair<std::string, Expr*>> result;
  for (const std::string& name : struct_def->GetMemberNames()) {
    result.push_back({name, GetExpr(name).value()});
  }
  return result;
}

absl::StatusOr<Expr*> StructInstance::GetExpr(std::string_view name) const {
  for (const auto& item : members_) {
    if (item.first == name) {
      return item.second;
    }
  }
  return absl::NotFoundError(
      absl::StrFormat("Name is not present in struct instance: \"%s\"", name));
}

std::vector<AstNode*> StructInstance::GetChildren(bool want_types) const {
  std::vector<AstNode*> results;
  results.reserve(members_.size());
  for (auto& item : members_) {
    results.push_back(item.second);
  }
  return results;
}

std::string StructInstance::ToStringInternal() const {
  std::string type_name = ToAstNode(struct_ref_)->ToString();

  std::string members_str = absl::StrJoin(
      members_, ", ",
      [](std::string* out, const std::pair<std::string, Expr*>& member) {
        absl::StrAppendFormat(out, "%s: %s", member.first,
                              member.second->ToString());
      });
  return absl::StrFormat("%s { %s }", type_name, members_str);
}

// -- class SplatStructInstance

SplatStructInstance::SplatStructInstance(
    Module* owner, Span span, TypeAnnotation* struct_ref,
    std::vector<std::pair<std::string, Expr*>> members, Expr* splatted)
    : Expr(owner, std::move(span)),
      struct_ref_(struct_ref),
      members_(std::move(members)),
      splatted_(splatted) {}

SplatStructInstance::~SplatStructInstance() = default;

std::vector<AstNode*> SplatStructInstance::GetChildren(bool want_types) const {
  std::vector<AstNode*> results;
  results.reserve(members_.size() + 1);
  for (auto& item : members_) {
    results.push_back(item.second);
  }
  results.push_back(splatted_);
  return results;
}

std::string SplatStructInstance::ToStringInternal() const {
  std::string type_name = ToAstNode(struct_ref_)->ToString();

  std::string members_str = absl::StrJoin(
      members_, ", ",
      [](std::string* out, const std::pair<std::string, Expr*>& member) {
        absl::StrAppendFormat(out, "%s: %s", member.first,
                              member.second->ToString());
      });
  return absl::StrFormat("%s { %s, ..%s }", type_name, members_str,
                         splatted_->ToString());
}

// -- class Unop

Unop::~Unop() = default;

std::string Unop::ToStringInternal() const {
  std::string operand = operand_->ToString();
  if (WeakerThan(operand_->GetPrecedence(), GetPrecedenceWithoutParens())) {
    Parenthesize(&operand);
  }
  return absl::StrFormat("%s%s", UnopKindToString(unop_kind_), operand);
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

// -- class Binop

Binop::~Binop() = default;

Precedence Binop::GetPrecedenceWithoutParens() const {
  switch (binop_kind_) {
    case BinopKind::kShl:
      return Precedence::kShift;
    case BinopKind::kShr:
      return Precedence::kShift;
    case BinopKind::kLogicalAnd:
      return Precedence::kLogicalAnd;
    case BinopKind::kLogicalOr:
      return Precedence::kLogicalOr;
    // bitwise
    case BinopKind::kXor:
      return Precedence::kBitwiseXor;
    case BinopKind::kOr:
      return Precedence::kBitwiseOr;
    case BinopKind::kAnd:
      return Precedence::kBitwiseAnd;
    // comparisons
    case BinopKind::kEq:
    case BinopKind::kNe:
    case BinopKind::kGe:
    case BinopKind::kGt:
    case BinopKind::kLt:
    case BinopKind::kLe:
      return Precedence::kComparison;
    // weak arithmetic
    case BinopKind::kAdd:
    case BinopKind::kSub:
      return Precedence::kWeakArithmetic;
    // strong arithmetic
    case BinopKind::kMul:
    case BinopKind::kDiv:
    case BinopKind::kMod:
      return Precedence::kStrongArithmetic;
    case BinopKind::kConcat:
      return Precedence::kConcat;
  }
  LOG(FATAL) << "Invalid binop kind: " << static_cast<int>(binop_kind_);
}

absl::StatusOr<BinopKind> BinopKindFromString(std::string_view s) {
#define HANDLE(__enum, __unused, __operator) \
  if (s == __operator) {                     \
    return BinopKind::__enum;                \
  }
  XLS_DSLX_BINOP_KIND_EACH(HANDLE)
#undef HANDLE
  return absl::InvalidArgumentError(
      absl::StrFormat("Invalid BinopKind string: \"%s\"", s));
}

Binop::Binop(Module* owner, Span span, BinopKind binop_kind, Expr* lhs,
             Expr* rhs)
    : Expr(owner, std::move(span)),
      binop_kind_(binop_kind),
      lhs_(lhs),
      rhs_(rhs) {}

std::string Binop::ToStringInternal() const {
  Precedence op_precedence = GetPrecedenceWithoutParens();
  std::string lhs = lhs_->ToString();
  {
    Precedence lhs_precedence = lhs_->GetPrecedence();
    VLOG(10) << "lhs_expr: `" << lhs << "` precedence: " << lhs_precedence
             << " op_precedence: " << op_precedence;
    if (WeakerThan(lhs_precedence, op_precedence)) {
      Parenthesize(&lhs);
    } else if (binop_kind_ == BinopKind::kLt &&
               lhs_->kind() == AstNodeKind::kCast && !lhs_->in_parens()) {
      // If there is an open angle bracket, and the LHS is suffixed with a type,
      // we parenthesize it to avoid ambiguity; e.g.
      //
      //    foo as bar < baz
      //           ^~~~~~~~^
      //
      // We don't know whether `bar<baz` is the start of a parametric type
      // instantiation, so we force conservative parenthesization:
      //
      //    (foo as bar) < baz
      Parenthesize(&lhs);
    }
  }

  std::string rhs = rhs_->ToString();
  {
    if (WeakerThan(rhs_->GetPrecedence(), op_precedence)) {
      Parenthesize(&rhs);
    }
  }
  return absl::StrFormat("%s %s %s", lhs, BinopKindFormat(binop_kind_), rhs);
}

// -- class StatementBlock

StatementBlock::StatementBlock(Module* owner, Span span,
                               std::vector<Statement*> statements,
                               bool trailing_semi)
    : Expr(owner, std::move(span)),
      statements_(std::move(statements)),
      trailing_semi_(trailing_semi) {
  if (statements_.empty()) {
    CHECK(trailing_semi) << "empty block but trailing_semi is false";
  }
}

StatementBlock::~StatementBlock() = default;

std::string StatementBlock::ToInlineString() const {
  // A formatting special case: if there are no statements (and implicitly a
  // trailing semi since an empty block gives unit type) we just give back
  // braces without any semicolon inside.
  if (statements_.empty()) {
    CHECK(trailing_semi_);
    return "{}";
  }

  std::string s = absl::StrCat(
      "{ ",
      absl::StrJoin(statements_, "; ", [](std::string* out, Statement* stmt) {
        absl::StrAppend(out, stmt->ToString());
      }));
  if (trailing_semi_) {
    absl::StrAppend(&s, ";");
  }
  absl::StrAppend(&s, " }");
  return s;
}

std::string StatementBlock::ToStringInternal() const {
  // A formatting special case: if there are no statements (and implicitly a
  // trailing semi since an empty block gives unit type) we just give back
  // braces without any semicolon inside.
  if (statements_.empty()) {
    CHECK(trailing_semi_);
    return "{}";
  }

  std::vector<std::string> stmts;
  for (size_t i = 0; i < statements_.size(); ++i) {
    Statement* stmt = statements_[i];
    if (std::holds_alternative<Expr*>(stmt->wrapped())) {
      if (i + 1 == statements_.size() && !trailing_semi_) {
        stmts.push_back(stmt->ToString());
      } else {
        stmts.push_back(stmt->ToString() + ";");
      }
    } else {
      stmts.push_back(stmt->ToString());
    }
  }
  return absl::StrFormat(
      "{\n%s\n}", Indent(absl::StrJoin(stmts, "\n"), kRustSpacesPerIndent));
}

// -- class ForLoopBase

ForLoopBase::ForLoopBase(Module* owner, Span span, NameDefTree* names,
                         TypeAnnotation* type_annotation, Expr* iterable,
                         StatementBlock* body, Expr* init)
    : Expr(owner, span),
      names_(names),
      type_annotation_(type_annotation),
      iterable_(iterable),
      body_(body),
      init_(init) {}

std::vector<AstNode*> ForLoopBase::GetChildren(bool want_types) const {
  std::vector<AstNode*> results = {names_};
  if (want_types && type_annotation_ != nullptr) {
    results.push_back(type_annotation_);
  }
  results.push_back(iterable_);
  results.push_back(body_);
  results.push_back(init_);
  return results;
}

std::string ForLoopBase::ToStringInternal() const {
  std::string type_str;
  if (type_annotation_ != nullptr) {
    type_str = absl::StrCat(": ", type_annotation_->ToString());
  }
  return absl::StrFormat("%s %s%s in %s %s(%s)", keyword(), names_->ToString(),
                         type_str, iterable_->ToString(), body_->ToString(),
                         init_->ToString());
}

// -- class Function

Function::Function(Module* owner, Span span, NameDef* name_def,
                   std::vector<ParametricBinding*> parametric_bindings,
                   std::vector<Param*> params, TypeAnnotation* return_type,
                   StatementBlock* body, FunctionTag tag, bool is_public)
    : AstNode(owner),
      span_(std::move(span)),
      name_def_(name_def),
      parametric_bindings_(std::move(parametric_bindings)),
      params_(std::move(params)),
      return_type_(return_type),
      body_(body),
      tag_(tag),
      is_public_(is_public) {
  for (const ParametricBinding* pb : parametric_bindings_) {
    CHECK(parametric_keys_.insert(pb->identifier()).second);
  }
}

Function::~Function() = default;

std::vector<AstNode*> Function::GetChildren(bool want_types) const {
  std::vector<AstNode*> results;
  results.push_back(name_def());
  if (tag_ == FunctionTag::kNormal) {
    // The parametric bindings of a proc are shared between the proc itself and
    // the two functions it contains. Thus, they should have a single owner, the
    // proc, and the other two functions "borrow" them.
    for (ParametricBinding* binding : parametric_bindings()) {
      results.push_back(binding);
    }
  }
  for (Param* p : params_) {
    results.push_back(p);
  }
  if (return_type_ != nullptr && want_types) {
    results.push_back(return_type_);
  }
  results.push_back(body());
  return results;
}

std::string Function::ToString() const {
  std::string parametric_str;
  if (!parametric_bindings().empty()) {
    parametric_str = absl::StrFormat(
        "<%s>",
        absl::StrJoin(
            parametric_bindings(), ", ",
            [](std::string* out, ParametricBinding* parametric_binding) {
              absl::StrAppend(out, parametric_binding->ToString());
            }));
  }
  std::string params_str =
      absl::StrJoin(params(), ", ", [](std::string* out, Param* param) {
        absl::StrAppend(out, param->ToString());
      });
  std::string return_type_str = " ";
  if (return_type_ != nullptr) {
    return_type_str = " -> " + return_type_->ToString() + " ";
  }
  std::string pub_str = is_public() ? "pub " : "";
  std::string annotation_str;
  if (extern_verilog_module_.has_value()) {
    annotation_str = absl::StrFormat("#[extern_verilog(\"%s\")]\n",
                                     extern_verilog_module_->code_template());
  }
  return absl::StrFormat("%s%sfn %s%s(%s)%s%s", annotation_str, pub_str,
                         name_def_->ToString(), parametric_str, params_str,
                         return_type_str, body_->ToString());
}

std::string Function::ToUndecoratedString(std::string_view identifier) const {
  std::string params_str =
      absl::StrJoin(params(), ", ", [](std::string* out, Param* param) {
        absl::StrAppend(out, param->ToString());
      });
  return absl::StrFormat("%s(%s) %s", identifier, params_str,
                         body_->ToString());
}

absl::btree_set<std::string> Function::GetFreeParametricKeySet() const {
  std::vector<std::string> keys = GetFreeParametricKeys();
  return absl::btree_set<std::string>(keys.begin(), keys.end());
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

// -- class TestFunction

TestFunction::~TestFunction() = default;

// -- class MatchArm

MatchArm::MatchArm(Module* owner, Span span, std::vector<NameDefTree*> patterns,
                   Expr* expr)
    : AstNode(owner),
      span_(std::move(span)),
      patterns_(std::move(patterns)),
      expr_(expr) {
  CHECK(!patterns_.empty());
}

MatchArm::~MatchArm() = default;

Span MatchArm::GetPatternSpan() const {
  return Span(patterns_[0]->span().start(), patterns_.back()->span().limit());
}

Match::Match(Module* owner, Span span, Expr* matched,
             std::vector<MatchArm*> arms)
    : Expr(owner, std::move(span)), matched_(matched), arms_(std::move(arms)) {}

// -- class NameRef

NameRef::~NameRef() = default;

// -- class ConstRef

ConstRef::~ConstRef() = default;

// -- class Range

Range::Range(Module* owner, Span span, Expr* start, Expr* end)
    : Expr(owner, std::move(span)), start_(start), end_(end) {}

Range::~Range() = default;

std::string Range::ToStringInternal() const {
  return absl::StrFormat("%s..%s", start_->ToString(), end_->ToString());
}

// -- class Cast

Cast::~Cast() = default;

std::string Cast::ToStringInternal() const {
  std::string lhs = expr_->ToString();
  Precedence arg_precedence = expr_->GetPrecedence();
  if (WeakerThan(arg_precedence, Precedence::kAs)) {
    VLOG(10) << absl::StreamFormat("expr `%s` precedence: %s weaker than 'as'",
                                   lhs, PrecedenceToString(arg_precedence));
    Parenthesize(&lhs);
  }
  return absl::StrFormat("%s as %s", lhs, type_annotation_->ToString());
}

// -- class BuiltinTypeAnnotation

BuiltinTypeAnnotation::BuiltinTypeAnnotation(Module* owner, Span span,
                                             BuiltinType builtin_type,
                                             BuiltinNameDef* builtin_name_def)
    : TypeAnnotation(owner, std::move(span)),
      builtin_type_(builtin_type),
      builtin_name_def_(builtin_name_def) {}

BuiltinTypeAnnotation::~BuiltinTypeAnnotation() = default;

std::vector<AstNode*> BuiltinTypeAnnotation::GetChildren(
    bool want_types) const {
  return std::vector<AstNode*>{};
}

int64_t BuiltinTypeAnnotation::GetBitCount() const {
  return GetBuiltinTypeBitCount(builtin_type_).value();
}

bool BuiltinTypeAnnotation::GetSignedness() const {
  return GetBuiltinTypeSignedness(builtin_type_).value();
}

// -- class ChannelTypeAnnotation

ChannelTypeAnnotation::ChannelTypeAnnotation(
    Module* owner, Span span, ChannelDirection direction,
    TypeAnnotation* payload, std::optional<std::vector<Expr*>> dims)
    : TypeAnnotation(owner, std::move(span)),
      direction_(direction),
      payload_(payload),
      dims_(std::move(dims)) {}

ChannelTypeAnnotation::~ChannelTypeAnnotation() = default;

std::string ChannelTypeAnnotation::ToString() const {
  std::vector<std::string> dims;
  if (dims_.has_value()) {
    for (const Expr* dim : dims_.value()) {
      dims.push_back(absl::StrCat("[", dim->ToString(), "]"));
    }
  }
  return absl::StrFormat("chan<%s>%s %s", payload_->ToString(),
                         absl::StrJoin(dims, ""),
                         direction_ == ChannelDirection::kIn ? "in" : "out");
}

// -- class TupleTypeAnnotation

TupleTypeAnnotation::TupleTypeAnnotation(Module* owner, Span span,
                                         std::vector<TypeAnnotation*> members)
    : TypeAnnotation(owner, std::move(span)), members_(std::move(members)) {}

TupleTypeAnnotation::~TupleTypeAnnotation() = default;

std::string TupleTypeAnnotation::ToString() const {
  std::string guts =
      absl::StrJoin(members_, ", ", [](std::string* out, TypeAnnotation* t) {
        absl::StrAppend(out, t->ToString());
      });
  return absl::StrFormat("(%s%s)", guts, members_.size() == 1 ? "," : "");
}

// -- class Statement

/* static */ absl::StatusOr<std::variant<Expr*, TypeAlias*, Let*, ConstAssert*>>
Statement::NodeToWrapped(AstNode* n) {
  if (auto* e = dynamic_cast<Expr*>(n)) {
    return e;
  }
  if (auto* t = dynamic_cast<TypeAlias*>(n)) {
    return t;
  }
  if (auto* l = dynamic_cast<Let*>(n)) {
    return l;
  }
  if (auto* d = dynamic_cast<ConstAssert*>(n)) {
    return d;
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "AST node could not be wrapped in a statement: ", n->GetNodeTypeName()));
}

Statement::Statement(Module* owner, Statement::Wrapped wrapped)
    : AstNode(owner), wrapped_(wrapped) {
  CHECK_NE(ToAstNode(wrapped_), this);
}

std::optional<Span> Statement::GetSpan() const {
  AstNode* wrapped = ToAstNode(wrapped_);
  CHECK_NE(wrapped, nullptr);
  CHECK_NE(wrapped, this);
  return wrapped->GetSpan();
}

// -- class WildcardPattern

WildcardPattern::~WildcardPattern() = default;

// -- class RestOfTuple

RestOfTuple::~RestOfTuple() = default;

// -- class QuickCheck

QuickCheck::QuickCheck(Module* owner, Span span, Function* f,
                       std::optional<int64_t> test_count)
    : AstNode(owner), span_(std::move(span)), f_(f), test_count_(test_count) {}

QuickCheck::~QuickCheck() = default;

std::string QuickCheck::ToString() const {
  std::string test_count_str;
  if (test_count_.has_value()) {
    test_count_str = absl::StrFormat("(test_count=%d)", *test_count_);
  }
  return absl::StrFormat("#[quickcheck%s]\n%s", test_count_str, f_->ToString());
}

// -- class TupleIndex

TupleIndex::~TupleIndex() = default;

TupleIndex::TupleIndex(Module* owner, Span span, Expr* lhs, Number* index)
    : Expr(owner, std::move(span)), lhs_(lhs), index_(index) {}

absl::Status TupleIndex::Accept(AstNodeVisitor* v) const {
  return v->HandleTupleIndex(this);
}

absl::Status TupleIndex::AcceptExpr(ExprVisitor* v) const {
  return v->HandleTupleIndex(this);
}

std::string TupleIndex::ToStringInternal() const {
  return absl::StrCat(lhs_->ToString(), ".", index_->ToString());
}

std::vector<AstNode*> TupleIndex::GetChildren(bool want_types) const {
  return {lhs_, index_};
}

// -- class XlsTuple

XlsTuple::~XlsTuple() = default;

std::string XlsTuple::ToStringInternal() const {
  std::string result = "(";
  for (int64_t i = 0; i < members_.size(); ++i) {
    absl::StrAppend(&result, members_[i]->ToString());
    if (i != members_.size() - 1) {
      absl::StrAppend(&result, ", ");
    }
  }
  if (members_.size() == 1 || has_trailing_comma()) {
    // Singleton tuple requires a trailing comma to avoid being parsed as a
    // parenthesized expression.
    absl::StrAppend(&result, ",");
  }
  absl::StrAppend(&result, ")");
  return result;
}

// -- class NameDefTree

NameDefTree::~NameDefTree() = default;

std::vector<AstNode*> NameDefTree::GetChildren(bool want_types) const {
  if (std::holds_alternative<Leaf>(tree_)) {
    return {ToAstNode(std::get<Leaf>(tree_))};
  }
  const Nodes& nodes = std::get<Nodes>(tree_);
  return ToAstNodes<NameDefTree>(nodes);
}

std::string NameDefTree::ToString() const {
  if (is_leaf()) {
    return ToAstNode(leaf())->ToString();
  }

  std::string guts =
      absl::StrJoin(nodes(), ", ", [](std::string* out, NameDefTree* node) {
        absl::StrAppend(out, node->ToString());
      });
  return absl::StrFormat("(%s)", guts);
}

std::vector<NameDefTree::Leaf> NameDefTree::Flatten() const {
  if (is_leaf()) {
    return {leaf()};
  }
  std::vector<Leaf> results;
  for (const NameDefTree* node : std::get<Nodes>(tree_)) {
    auto node_leaves = node->Flatten();
    results.insert(results.end(), node_leaves.begin(), node_leaves.end());
  }
  return results;
}

std::vector<NameDef*> NameDefTree::GetNameDefs() const {
  std::vector<NameDef*> results;
  for (Leaf leaf : Flatten()) {
    if (std::holds_alternative<NameDef*>(leaf)) {
      results.push_back(std::get<NameDef*>(leaf));
    }
  }
  return results;
}

std::vector<std::variant<NameDefTree::Leaf, NameDefTree*>>
NameDefTree::Flatten1() const {
  if (is_leaf()) {
    return {leaf()};
  }
  std::vector<std::variant<Leaf, NameDefTree*>> result;
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
         TypeAnnotation* type_annotation, Expr* rhs, bool is_const)
    : AstNode(owner),
      span_(std::move(span)),
      name_def_tree_(name_def_tree),
      type_annotation_(type_annotation),
      rhs_(rhs),
      is_const_(is_const) {}

Let::~Let() = default;

std::vector<AstNode*> Let::GetChildren(bool want_types) const {
  std::vector<AstNode*> results = {name_def_tree_};
  if (type_annotation_ != nullptr && want_types) {
    results.push_back(type_annotation_);
  }
  results.push_back(rhs_);
  return results;
}

std::string Let::ToString() const {
  return absl::StrFormat(
      "%s %s%s = %s;", is_const_ ? "const" : "let", name_def_tree_->ToString(),
      type_annotation_ == nullptr ? "" : ": " + type_annotation_->ToString(),
      rhs_->ToString());
}

// -- class Expr

Expr::~Expr() = default;

std::string Expr::ToString() const {
  std::string s = ToStringInternal();
  if (in_parens()) {
    Parenthesize(&s);
  }
  return s;
}

// -- class String

String::~String() = default;

std::string String::ToStringInternal() const {
  return absl::StrCat("\"", Escape(text_), "\"");
}

// -- class Number

static Span MakeNumberSpan(Span expr_span,
                           const TypeAnnotation* type_annotation) {
  if (type_annotation == nullptr) {
    return expr_span;
  }
  return Span(type_annotation->span().start(), expr_span.limit());
}

Number::Number(Module* owner, Span span, std::string text,
               NumberKind number_kind, TypeAnnotation* type_annotation)
    : Expr(owner, MakeNumberSpan(std::move(span), type_annotation)),
      text_(std::move(text)),
      number_kind_(number_kind),
      type_annotation_(type_annotation) {}

Number::~Number() = default;

void Number::SetTypeAnnotation(TypeAnnotation* type_annotation) {
  type_annotation_ = type_annotation;
  UpdateSpan(MakeNumberSpan(span(), type_annotation));
}

std::vector<AstNode*> Number::GetChildren(bool want_types) const {
  if (type_annotation_ == nullptr) {
    return {};
  }
  return {type_annotation_};
}

std::string Number::ToStringInternal() const {
  std::string formatted_text = text_;
  if (number_kind_ == NumberKind::kCharacter) {
    if (text_[0] == '\'' || text_[0] == '\\') {
      formatted_text = absl::StrCat(R"(\)", formatted_text);
    }
    formatted_text = absl::StrCat("'", formatted_text, "'");
  }
  if (type_annotation_ != nullptr) {
    return absl::StrFormat("%s:%s", type_annotation_->ToString(),
                           formatted_text);
  }
  return formatted_text;
}

std::string Number::ToStringNoType() const { return text_; }

absl::StatusOr<bool> Number::FitsInType(int64_t bit_count) const {
  XLS_RET_CHECK_GE(bit_count, 0);
  switch (number_kind_) {
    case NumberKind::kBool:
      return bit_count >= 1;
    case NumberKind::kCharacter:
      return bit_count >= CHAR_BIT;
    case NumberKind::kOther: {
      XLS_ASSIGN_OR_RETURN(auto sm, GetSignAndMagnitude(text_));
      auto [sign, bits] = sm;
      return bit_count >= bits.bit_count();
    }
  }
  return absl::InternalError(
      absl::StrFormat("Unreachable; invalid number kind: %d", number_kind_));
}

absl::StatusOr<Bits> Number::GetBits(int64_t bit_count,
                                     const FileTable& file_table) const {
  XLS_RET_CHECK_GE(bit_count, 0);
  switch (number_kind_) {
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
      XLS_RET_CHECK_GE(bits.bit_count(), 0);
      XLS_RET_CHECK(bit_count >= bits.bit_count()) << absl::StreamFormat(
          "Internal error: %s Cannot fit number value %s in %d bits; %d "
          "required: `%s`",
          span().ToString(file_table), text_, bit_count, bits.bit_count(),
          ToString());
      bits = bits_ops::ZeroExtend(bits, bit_count);
      if (sign) {
        bits = bits_ops::Negate(bits);
      }
      return bits;
    }
  }
  return absl::InternalError(absl::StrFormat(
      "Invalid NumberKind: %d", static_cast<int64_t>(number_kind_)));
}

TypeAlias::TypeAlias(Module* owner, Span span, NameDef& name_def,
                     TypeAnnotation& type, bool is_public)
    : AstNode(owner),
      span_(span),
      name_def_(name_def),
      type_annotation_(type),
      is_public_(is_public) {}

TypeAlias::~TypeAlias() = default;

std::string TypeAlias::ToString() const {
  return absl::StrFormat(
      "%s%stype %s = %s;", MakeExternTypeAttr(extern_type_name_),
      is_public_ ? "pub " : "", identifier(), type_annotation_.ToString());
}
// -- class Array

Array::~Array() = default;

std::string Array::ToStringInternal() const {
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
    CHECK(member != nullptr);
    results.push_back(member);
  }
  return results;
}

std::vector<AstNode*> Statement::GetChildren(bool want_types) const {
  return {ToAstNode(wrapped_)};
}

Span ExprOrTypeSpan(const ExprOrType& expr_or_type) {
  return absl::visit(Visitor{
                         [](Expr* expr) { return expr->span(); },
                         [](TypeAnnotation* type) { return type->span(); },
                     },
                     expr_or_type);
}

}  // namespace xls::dslx
