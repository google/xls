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

#ifndef XLS_DSLX_AST_H_
#define XLS_DSLX_AST_H_

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/logging/logging.h"
#include "xls/dslx/cpp_pos.h"

namespace xls::dslx {

// Helper function for downcast-based membership testing.
//
// Not fast, but this is not performance critical code at the moment.
template <typename ObjT>
bool IsOneOf(ObjT* obj) {
  return false;
}
template <typename FirstT, typename... Rest, typename ObjT>
bool IsOneOf(ObjT* obj) {
  if (dynamic_cast<FirstT*>(obj) != nullptr) {
    return true;
  }
  return IsOneOf<Rest...>(obj);
}

// Forward decls.
class BuiltinNameDef;
class Enum;
class Expr;
class ModRef;
class NameDef;
class NameDefTree;
class NameRef;
class Struct;
class TypeDef;
class TypeRef;

// Name definitions can be either built in (BuiltinNameDef, in which case they
// have no effective position) or defined in the user AST (NameDef).
using AnyNameDef = absl::variant<NameDef*, BuiltinNameDef*>;

// Holds a mapping {identifier: NameRefs} -- this is used for accumulating free
// variable references (the NameRefs) in the source program; see
// AstNode::GetFreeVariables().
//
// Note: this is generally used as an immutable collection -- it gets built, and
// once built it can be queried but not mutated.
class FreeVariables {
 public:
  // Returns a (stably ordered) sequence of (identifier, name_def).
  std::vector<std::pair<std::string, AnyNameDef>> GetNameDefTuples() const;

  // As above, but without the identifier.
  std::vector<AnyNameDef> GetNameDefs() const;

  // Drops any BuiltinNameDef from this FreeVariables set (note this is *not* a
  // mutating operation).
  FreeVariables DropBuiltinDefs() const;

  // Adds a free variable reference to this set -- "identifier" is the free
  // variable identifier and "name_ref" is the AST node reference that is the
  // free variable reference.
  void Add(std::string identifier, NameRef* name_ref);

  // Returns the identifiers in this free variable set.
  absl::flat_hash_set<std::string> Keys() const;

  // Underlying data for this free variables set.
  const absl::flat_hash_map<std::string, std::vector<NameRef*>>& values()
      const {
    return values_;
  }

 private:
  absl::flat_hash_map<std::string, std::vector<NameRef*>> values_;
};

// Abstract base class for AST nodes.
class AstNode {
 public:
  virtual ~AstNode() = default;

  virtual std::string ToString() const = 0;

  // Retrieves all the child nodes for this AST node.
  //
  // If want_types is false, then type annotations should be excluded from the
  // returned child nodes. This exclusion of types is useful e.g. when
  // attempting to find free variables that are referred to during program
  // execution, since all type information must be resolved to constants at type
  // inference time.
  virtual std::vector<AstNode*> GetChildren(bool want_types) const = 0;

  // Retrieves all the free variables (references to names that are defined
  // prior to start_pos) that are transitively in this AST subtree.
  FreeVariables GetFreeVariables(Pos start_pos);
};

// Helpers for converting variants of "AstNode subtype" pointers and their
// variants to the base `AstNode*`.
template <typename... Types>
inline AstNode* ToAstNode(const absl::variant<Types...>& v) {
  return absl::ConvertVariantTo<AstNode*>(v);
}
inline AstNode* ToAstNode(AstNode* n) { return n; }

// As above, but for Expr base.
template <typename... Types>
inline Expr* ToExprNode(const absl::variant<Types...>& v) {
  return absl::ConvertVariantTo<Expr*>(v);
}

// Converts sequence of AstNode subtype pointers to vector of the base AstNode*.
template <typename NodeT>
inline std::vector<AstNode*> ToAstNodes(absl::Span<NodeT* const> source) {
  std::vector<AstNode*> result;
  for (NodeT* item : source) {
    result.push_back(item);
  }
  return result;
}

// Abstract base class for type annotations.
class TypeAnnotation : public AstNode {
 public:
  explicit TypeAnnotation(Span span) : span_(std::move(span)) {}

  const Span& span() const { return span_; }

 private:
  Span span_;
};

#include "xls/dslx/cpp_ast_builtin_types.inc"

// Enumeration of types that are built-in keywords; e.g. `u32`, `bool`, etc.
enum class BuiltinType {
#define FIRST_COMMA(A, ...) A,
  XLS_DSLX_BUILTIN_TYPE_EACH(FIRST_COMMA)
#undef FIRST_COMMA
};

std::string BuiltinTypeToString(BuiltinType t);
absl::StatusOr<BuiltinType> BuiltinTypeFromString(absl::string_view s);

// Represents a built-in type annotation; e.g. `u32`, `bits`, etc.
class BuiltinTypeAnnotation : public TypeAnnotation {
 public:
  BuiltinTypeAnnotation(Span span, BuiltinType builtin_type)
      : TypeAnnotation(std::move(span)), builtin_type_(builtin_type) {}

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {};
  }

  std::string ToString() const override {
    return BuiltinTypeToString(builtin_type_);
  }

  int64 GetBitCount() const;
  bool GetSignedness() const;

 private:
  BuiltinType builtin_type_;
};

// Represents a tuple type annotation; e.g. `(u32, s42)`.
class TupleTypeAnnotation : public TypeAnnotation {
 public:
  TupleTypeAnnotation(Span span, std::vector<TypeAnnotation*> members)
      : TypeAnnotation(std::move(span)), members_(std::move(members)) {}

  std::string ToString() const override {
    std::string guts =
        absl::StrJoin(members_, ", ", [](std::string* out, TypeAnnotation* t) {
          absl::StrAppend(out, t->ToString());
        });
    return absl::StrFormat("(%s%s)", guts, members_.size() == 1 ? "," : "");
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return ToAstNodes<TypeAnnotation>(members_);
  }

  const std::vector<TypeAnnotation*>& members() const { return members_; }

 private:
  std::vector<TypeAnnotation*> members_;
};

// Represents a type reference annotation.
class TypeRefTypeAnnotation : public TypeAnnotation {
 public:
  TypeRefTypeAnnotation(Span span, TypeRef* type_ref,
                        std::vector<Expr*> parametrics)
      : TypeAnnotation(std::move(span)),
        type_ref_(type_ref),
        parametrics_(std::move(parametrics)) {}

  TypeRef* type_ref() const { return type_ref_; }

  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override;

  const std::vector<Expr*>& parametrics() const { return parametrics_; }

 private:
  TypeRef* type_ref_;
  std::vector<Expr*> parametrics_;
};

// Represents an array type annotation; e.g. `u32[5]`.
class ArrayTypeAnnotation : public TypeAnnotation {
 public:
  ArrayTypeAnnotation(Span span, TypeAnnotation* element_type, Expr* dim)
      : TypeAnnotation(std::move(span)),
        element_type_(element_type),
        dim_(dim) {}

  TypeAnnotation* element_type() const { return element_type_; }
  Expr* dim() const { return dim_; }

  std::vector<AstNode*> GetChildren(bool want_types) const override;

  std::string ToString() const override;

 private:
  TypeAnnotation* element_type_;
  Expr* dim_;
};

// Represents the definition point of a built-in name.
//
// This node is for representation consistency; all references to names must
// have a corresponding definition where the name was bound. For primitive
// builtins there is no textual point, so we create positionless (in the text)
// definition points for them.
class BuiltinNameDef : public AstNode {
 public:
  explicit BuiltinNameDef(std::string identifier)
      : identifier_(std::move(identifier)) {}

  std::string ToString() const override { return identifier_; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {};
  }

  const std::string& identifier() const { return identifier_; }

 private:
  std::string identifier_;
};

// Represents a wildcard pattern in a 'match' construct.
class WildcardPattern : public AstNode {
 public:
  WildcardPattern(Span span) : span_(std::move(span)) {}

  std::string ToString() const override { return "_"; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {};
  }

  const Span& span() const { return span_; }

 private:
  Span span_;
};

// Represents the definition of a name (identifier).
class NameDef : public AstNode {
 public:
  NameDef(Span span, std::string identifier)
      : span_(span), identifier_(std::move(identifier)) {}

  const Span& span() const { return span_; }
  const std::string& identifier() const { return identifier_; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {};
  }
  std::string ToString() const override { return identifier_; }

 private:
  Span span_;
  std::string identifier_;
};

// Abstract base class for AST node that can appear in expression positions
// (i.e. can produce runtime values).
class Expr : public AstNode {
 public:
  explicit Expr(Span span) : span_(span) {}
  virtual ~Expr() = default;

  const Span& span() const { return span_; }
  void set_span(const Span& span) { span_ = span; }

 private:
  Span span_;
};

// Represents a reference to a name (identifier).
class NameRef : public Expr {
 public:
  NameRef(Span span, std::string identifier, AnyNameDef name_def)
      : Expr(std::move(span)),
        name_def_(name_def),
        identifier_(std::move(identifier)) {}

  const std::string& identifier() const { return identifier_; }

  std::string ToString() const override { return identifier_; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {ToAstNode(name_def_)};
  }

  absl::optional<Pos> GetNameDefStart() const {
    if (absl::holds_alternative<NameDef*>(name_def_)) {
      return absl::get<NameDef*>(name_def_)->span().start();
    }
    return absl::nullopt;
  }

  absl::variant<NameDef*, BuiltinNameDef*> name_def() const {
    return name_def_;
  }

 private:
  AnyNameDef name_def_;
  std::string identifier_;
};

// Used to represent a named reference to a Constant name definition.
class ConstRef : public NameRef {
 public:
  using NameRef::NameRef;
};

// Represents an enum-value reference (via `::`, i.e. `Foo::BAR`).
//
// TODO(leary): 2020-08-27 More appropriate name would be EnumMemberRef or
// something.
class EnumRef : public Expr {
 public:
  EnumRef(Span span, absl::variant<TypeDef*, Enum*> enum_def, std::string attr)
      : Expr(std::move(span)), enum_def_(enum_def), attr_(std::move(attr)) {}

  std::string ToString() const override {
    return absl::StrFormat("%s::%s", GetEnumIdentifier(), attr_);
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {ToAstNode(enum_def_)};
  }

  absl::variant<TypeDef*, Enum*> enum_def() const { return enum_def_; }
  const std::string& attr() const { return attr_; }

 private:
  std::string GetEnumIdentifier() const;

  absl::variant<TypeDef*, Enum*> enum_def_;
  std::string attr_;
};

enum class NumberKind {
  kBool,
  kCharacter,
  kOther,
};

// Represents a literal number value.
class Number : public Expr {
 public:
  explicit Number(Span span, std::string text, NumberKind kind,
                  TypeAnnotation* type)
      : Expr(std::move(span)),
        text_(std::move(text)),
        kind_(kind),
        type_(type) {}

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    if (type_ == nullptr) {
      return {};
    }
    return {type_};
  }

  std::string ToString() const override {
    if (type_ != nullptr) {
      return absl::StrFormat("%s:%s", type_->ToString(), text_);
    }
    return text_;
  }

  TypeAnnotation* type() const { return type_; }
  void set_type(TypeAnnotation* type) { type_ = type; }

  const std::string& text() const { return text_; }

  NumberKind kind() const { return kind_; }

 private:
  std::string text_;
  NumberKind kind_;
  TypeAnnotation* type_;  // May be null.
};

// Represents a user-defined-type definition; e.g.
//    type Foo = (u32, u32);
//    type Bar = (u32, Foo);
//
// TODO(leary): 2020-09-15 Rename to TypeAlias, less of a loaded term.
class TypeDef : public AstNode {
 public:
  TypeDef(Span span, NameDef* name_def, TypeAnnotation* type, bool is_public)
      : span_(std::move(span)),
        name_def_(name_def),
        type_(type),
        is_public_(is_public) {}

  const std::string& identifier() const { return name_def_->identifier(); }

  std::string ToString() const override {
    return absl::StrFormat("type %s = %s;", identifier(), type_->ToString());
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {name_def_, type_};
  }

  NameDef* name_def() const { return name_def_; }
  TypeAnnotation* type() const { return type_; }
  bool is_public() const { return is_public_; }
  const Span& span() const { return span_; }

 private:
  Span span_;
  NameDef* name_def_;
  TypeAnnotation* type_;
  bool is_public_;
};

// Represents an array expression.
class Array : public Expr {
 public:
  Array(Span span, std::vector<Expr*> members, bool has_ellipsis)
      : Expr(std::move(span)),
        members_(std::move(members)),
        has_ellipsis_(has_ellipsis) {}

  std::string ToString() const override {
    return absl::StrFormat(
        "[%s]", absl::StrJoin(members_, ", ", [](std::string* out, Expr* expr) {
          absl::StrAppend(out, expr->ToString());
        }));
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    std::vector<AstNode*> results;
    if (type_ != nullptr) {
      results.push_back(type_);
    }
    for (Expr* member : members_) {
      XLS_CHECK(member != nullptr);
      results.push_back(member);
    }
    return results;
  }

  const std::vector<Expr*>& members() const { return members_; }
  TypeAnnotation* type() const { return type_; }
  void set_type(TypeAnnotation* type) { type_ = type; }

  bool has_ellipsis() const { return has_ellipsis_; }

 private:
  TypeAnnotation* type_ = nullptr;
  std::vector<Expr*> members_;
  bool has_ellipsis_;
};

// A constant array expression is an array expression where all of the
// expressions contained within it are constant.
class ConstantArray : public Array {
 public:
  // Adds checking for constant-expression-ness of the members beyond
  // Array::Array.
  ConstantArray(Span span, std::vector<Expr*> members, bool has_ellipsis);
};

// Several different AST nodes define types that can be referred to by a
// TypeRef.
using TypeDefinition = absl::variant<TypeDef*, Struct*, Enum*, ModRef*>;

// Represents a name that refers to a defined type.
//
// TODO(leary): 2020-09-04 This should not be an expr, change the base class to
// AstNode.
class TypeRef : public Expr {
 public:
  TypeRef(Span span, std::string text, TypeDefinition type_definition)
      : Expr(std::move(span)),
        text_(std::move(text)),
        type_definition_(type_definition) {}

  std::string ToString() const override { return text_; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {ToAstNode(type_definition_)};
  }

  const std::string& text() const { return text_; }
  const TypeDefinition& type_definition() const { return type_definition_; }

 private:
  std::string text_;
  TypeDefinition type_definition_;
};

// Represents an import statement; e.g.
//  import std as my_std
class Import : public AstNode {
 public:
  Import(Span span, std::vector<std::string> name, NameDef* name_def,
         absl::optional<std::string> alias)
      : span_(std::move(span)),
        name_(std::move(name)),
        name_def_(name_def),
        alias_(std::move(alias)) {
    XLS_CHECK(!name_.empty());
  }

  const std::string& identifier() const { return name_def_->identifier(); }

  std::string ToString() const override {
    if (alias_) {
      return absl::StrFormat("import %s as %s", absl::StrJoin(name_, "."),
                             *alias_);
    }
    return absl::StrFormat("import %s", absl::StrJoin(name_, "."));
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {name_def_};
  }

  const std::vector<std::string>& name() const { return name_; }
  NameDef* name_def() const { return name_def_; }
  const Span& span() const { return span_; }

 private:
  // Span of the import in the text.
  Span span_;
  // Name of the module being imported ("original" name before aliasing); e.g.
  // "std". Only present if the import is aliased.
  std::vector<std::string> name_;
  // The name definition we bind the import to.
  NameDef* name_def_;
  // The identifier text we bind the import to.
  absl::optional<std::string> alias_;
};

// Represents a module-value reference (via `::` i.e. `std::FOO`).
class ModRef : public Expr {
 public:
  ModRef(Span span, Import* mod, std::string attr)
      : Expr(std::move(span)), mod_(mod), attr_(std::move(attr)) {}

  std::string ToString() const override {
    return absl::StrFormat("%s::%s", mod_->identifier(), attr_);
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {mod_};
  }

  Import* import() const { return mod_; }
  const std::string& attr() const { return attr_; }

 private:
  Import* mod_;
  std::string attr_;
};

// Represents a function parameter.
class Param : public AstNode {
 public:
  Param(NameDef* name_def, TypeAnnotation* type)
      : name_def_(name_def),
        type_(type),
        span_(name_def_->span().start(), type_->span().limit()) {}

  std::string ToString() const override {
    return absl::StrFormat("%s: %s", name_def_->ToString(), type_->ToString());
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {name_def_, type_};
  }

  const Span& span() const { return span_; }
  NameDef* name_def() const { return name_def_; }
  TypeAnnotation* type() const { return type_; }

 private:
  NameDef* name_def_;
  TypeAnnotation* type_;
  Span span_;
};

enum class UnopKind {
  kInvert,  // one's complement inversion (bit flip)
  kNegate,  // two's complement aritmetic negation (~x+1)
};

absl::StatusOr<UnopKind> UnopKindFromString(absl::string_view s);
std::string UnopKindToString(UnopKind k);

// Represents a unary operation expression; e.g. `!x`.
class Unop : public Expr {
 public:
  Unop(Span span, UnopKind kind, Expr* operand)
      : Expr(std::move(span)), kind_(kind), operand_(operand) {}

  std::string ToString() const override {
    return absl::StrFormat("%s(%s)", UnopKindToString(kind_),
                           operand_->ToString());
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {operand_};
  }

  UnopKind kind() const { return kind_; }
  Expr* operand() const { return operand_; }

 private:
  UnopKind kind_;
  Expr* operand_;
};

#define XLS_DSLX_BINOP_KIND_EACH(X)       \
  /* enum member, python attr, tok str */ \
  X(kShll, "SHLL", "<<")                  \
  X(kShrl, "SHRL", ">>")                  \
  X(kShra, "SHRA", ">>>")                 \
  X(kGe, "GE", ">=")                      \
  X(kGt, "GT", ">")                       \
  X(kLe, "LE", "<=")                      \
  X(kLt, "LT", "<")                       \
  X(kEq, "EQ", "==")                      \
  X(kNe, "NE", "!=")                      \
  X(kAdd, "ADD", "+")                     \
  X(kSub, "SUB", "-")                     \
  X(kMul, "MUL", "*")                     \
  X(kAnd, "AND", "&")                     \
  X(kOr, "OR", "|")                       \
  X(kXor, "XOR", "^")                     \
  X(kDiv, "DIV", "/")                     \
  X(kLogicalAnd, "LOGICAL_AND", "&&")     \
  X(kLogicalOr, "LOGICAL_OR", "||")       \
  X(kConcat, "CONCAT", "++")

enum class BinopKind {
#define FIRST_COMMA(A, ...) A,
  XLS_DSLX_BINOP_KIND_EACH(FIRST_COMMA)
#undef FIRST_COMMA
};

absl::StatusOr<BinopKind> BinopKindFromString(absl::string_view s);

std::string BinopKindFormat(BinopKind kind);

// Represents a binary operation expression; e.g. `x + y`.
class Binop : public Expr {
 public:
  Binop(Span span, BinopKind kind, Expr* lhs, Expr* rhs)
      : Expr(span), kind_(kind), lhs_(lhs), rhs_(rhs) {}

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {lhs_, rhs_};
  }

  std::string ToString() const override {
    return absl::StrFormat("(%s) %s (%s)", lhs_->ToString(),
                           BinopKindFormat(kind_), rhs_->ToString());
  }

  BinopKind kind() const { return kind_; }
  Expr* lhs() const { return lhs_; }
  Expr* rhs() const { return rhs_; }

 private:
  BinopKind kind_;
  Expr* lhs_;
  Expr* rhs_;
};

// Represents the ternary expression; e.g. in Pythonic style:
//
//  consequent if test else alternate
class Ternary : public Expr {
 public:
  Ternary(Span span, Expr* test, Expr* consequent, Expr* alternate)
      : Expr(std::move(span)),
        test_(test),
        consequent_(consequent),
        alternate_(alternate) {}

  std::string ToString() const override {
    return absl::StrFormat("(%s) if (%s) else (%s)", consequent_->ToString(),
                           test_->ToString(), alternate_->ToString());
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {test_, consequent_, alternate_};
  }

  Expr* test() const { return test_; }
  Expr* consequent() const { return consequent_; }
  Expr* alternate() const { return alternate_; }

 private:
  Expr* test_;
  Expr* consequent_;
  Expr* alternate_;
};

// Represents a member in a parametric binding list.
//
// That is, in:
//
//  fn [X: u32, Y: u32 = X+X] f(x: bits[X]) -> bits[Y] {
//    x ++ x
//  }
//
// There are two parametric bindings:
//
// * X is a u32.
// * Y is a value derived from the parametric binding of X.
class ParametricBinding : public AstNode {
 public:
  ParametricBinding(NameDef* name_def, TypeAnnotation* type, Expr* expr)
      : name_def_(name_def), type_(type), expr_(expr) {}

  // TODO(leary): 2020-08-21 Fix this, the span is more than just the name def's
  // span, it must include the type/expr.
  const Span& span() const { return name_def_->span(); }

  std::string ToString() const override {
    std::string suffix;
    if (expr_ != nullptr) {
      suffix = absl::StrFormat("= %s", expr_->ToString());
    }
    return absl::StrFormat("%s: %s%s", name_def_->ToString(), type_->ToString(),
                           suffix);
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    std::vector<AstNode*> results = {name_def_};
    if (want_types) {
      results.push_back(type_);
    }
    if (expr_ != nullptr) {
      results.push_back(expr_);
    }
    return results;
  }

  NameDef* name_def() const { return name_def_; }
  TypeAnnotation* type() const { return type_; }
  Expr* expr() const { return expr_; }

 private:
  NameDef* name_def_;
  TypeAnnotation* type_;
  Expr* expr_;  // May be null.
};

// Represents a function definition.
class Function : public AstNode {
 public:
  Function(Span span, NameDef* name_def,
           std::vector<ParametricBinding*> parametric_bindings,
           std::vector<Param*> params, TypeAnnotation* return_type, Expr* body,
           bool is_public)
      : span_(span),
        name_def_(XLS_DIE_IF_NULL(name_def)),
        params_(std::move(params)),
        parametric_bindings_(std::move(parametric_bindings)),
        return_type_(return_type),
        body_(XLS_DIE_IF_NULL(body)),
        is_public_(is_public) {}

  std::vector<AstNode*> GetChildren(bool want_types) const override {
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

  std::string Format(bool include_body = true) const;

  std::string ToString() const override {
    return Format(/*include_body=*/false);
  }

  NameDef* name_def() const { return name_def_; }
  Expr* body() const { return body_; }
  const Span& span() const { return span_; }

  const std::string& identifier() const { return name_def_->identifier(); }
  const std::vector<Param*>& params() const { return params_; }
  const std::vector<ParametricBinding*>& parametric_bindings() const {
    return parametric_bindings_;
  }
  bool is_parametric() const { return !parametric_bindings_.empty(); }
  bool is_public() const { return is_public_; }

  TypeAnnotation* return_type() const { return return_type_; }

  std::vector<std::string> GetFreeParametricKeys() const {
    std::vector<std::string> results;
    for (ParametricBinding* b : parametric_bindings_) {
      if (b->expr() == nullptr) {
        results.push_back(b->name_def()->identifier());
      }
    }
    return results;
  }

 private:
  Span span_;
  NameDef* name_def_;
  std::vector<Param*> params_;
  std::vector<ParametricBinding*> parametric_bindings_;
  TypeAnnotation* return_type_;  // May be null.
  Expr* body_;
  bool is_public_;
};

// Represents a single arm in a match expression.
//
// Attributes:
//   patterns: The pattern to match against to yield the value of 'expr'.
//   expr: The expression to yield on a match.
//   span: The span of the match arm (both matcher and expr).
class MatchArm : public AstNode {
 public:
  MatchArm(Span span, std::vector<NameDefTree*> patterns, Expr* expr)
      : span_(std::move(span)), patterns_(std::move(patterns)), expr_(expr) {}

  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override;

  const std::vector<NameDefTree*>& patterns() const { return patterns_; }
  Expr* expr() const { return expr_; }
  const Span& span() const { return span_; }

 private:
  Span span_;
  std::vector<NameDefTree*> patterns_;
  Expr* expr_;  // Expression that is executed if one of the patterns matches.
};

// Represents a match (pattern match) expression.
class Match : public Expr {
 public:
  Match(Span span, Expr* matched, std::vector<MatchArm*> arms)
      : Expr(std::move(span)), matched_(matched), arms_(std::move(arms)) {}

  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    std::vector<AstNode*> results = {matched_};
    for (MatchArm* arm : arms_) {
      results.push_back(arm);
    }
    return results;
  }

  const std::vector<MatchArm*>& arms() const { return arms_; }
  Expr* matched() const { return matched_; }

 private:
  Expr* matched_;
  std::vector<MatchArm*> arms_;
};

// Represents an attribute access expression; e.g. `a.x`.
class Attr : public Expr {
 public:
  Attr(Span span, Expr* lhs, NameDef* attr)
      : Expr(std::move(span)), lhs_(lhs), attr_(attr) {}

  std::string ToString() const override {
    return absl::StrFormat("%s.%s", lhs_->ToString(), attr_->ToString());
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {lhs_, attr_};
  }

  Expr* lhs() const { return lhs_; }
  NameDef* attr() const { return attr_; }

 private:
  Expr* lhs_;
  NameDef* attr_;
};

// Represents an invocation expression; e.g. `f(a, b, c)`
class Invocation : public Expr {
 public:
  Invocation(Span span, Expr* callee, std::vector<Expr*> args)
      : Expr(std::move(span)), callee_(callee), args_(std::move(args)) {}

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    std::vector<AstNode*> results = {callee_};
    for (Expr* arg : args_) {
      results.push_back(arg);
    }
    return results;
  }

  std::string FormatArgs() const {
    return absl::StrJoin(args_, ", ", [](std::string* out, Expr* e) {
      absl::StrAppend(out, e->ToString());
    });
  }

  std::string ToString() const override {
    return absl::StrFormat("%s(%s)", callee_->ToString(), FormatArgs());
  };

  const std::vector<Expr*> args() const { return args_; }
  Expr* callee() const { return callee_; }
  const std::vector<std::pair<std::string, int64>> symbolic_bindings() const {
    return symbolic_bindings_;
  }

 private:
  Expr* callee_;
  std::vector<Expr*> args_;
  std::vector<std::pair<std::string, int64>> symbolic_bindings_;
};

// Represents a slice in the AST.
//
// For example, we can have `x[-4:-2]`, where x is of bit width N.
class Slice : public AstNode {
 public:
  Slice(Span span, Number* start, Number* limit)
      : span_(std::move(span)), start_(start), limit_(limit) {}

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    std::vector<AstNode*> results;
    if (start_ != nullptr) {
      results.push_back(start_);
    }
    if (limit_ != nullptr) {
      results.push_back(limit_);
    }
    return results;
  }

  std::string ToString() const override {
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

  Number* start() const { return start_; }
  Number* limit() const { return limit_; }

 private:
  Span span_;
  Number* start_;  // May be nullptr.
  Number* limit_;  // May be nullptr.
};

// Helper struct for members items defined inside of enums.
struct EnumMember {
  NameDef* name_def;
  // TODO(leary): 2020-09-14 This should be ConstRef*.
  absl::variant<Number*, NameRef*> value;
};

// Represents a user-defined enum definition; e.g.
//
//  type MyTypeDef = u2;
//  enum Foo : MyTypeDef {
//    A = 0,
//    B = 1,
//    C = 2,
//  }
class Enum : public AstNode {
 public:
  Enum(Span span, NameDef* name_def, TypeAnnotation* type,
       std::vector<EnumMember> values, bool is_public)
      : span_(std::move(span)),
        name_def_(name_def),
        type_(type),
        values_(std::move(values)),
        is_public_(is_public) {}

  bool HasValue(absl::string_view name) const {
    for (const auto& item : values_) {
      if (item.name_def->identifier() == name) {
        return true;
      }
    }
    return false;
  }

  absl::StatusOr<absl::variant<Number*, NameRef*>> GetValue(
      absl::string_view name) const {
    for (const EnumMember& item : values_) {
      if (item.name_def->identifier() == name) {
        return item.value;
      }
    }
    return absl::NotFoundError(absl::StrFormat(
        "Enum %s has no value with name \"%s\"", identifier(), name));
  }

  std::string ToString() const override {
    std::string result =
        absl::StrFormat("enum %s : %s {\n", identifier(), type_->ToString());
    for (const auto& item : values_) {
      absl::StrAppendFormat(&result, "  %s = %s,\n",
                            item.name_def->identifier(),
                            ToAstNode(item.value)->ToString());
    }
    absl::StrAppend(&result, "}");
    return result;
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    std::vector<AstNode*> results = {name_def_, type_};
    for (const EnumMember& item : values_) {
      results.push_back(item.name_def);
      results.push_back(ToAstNode(item.value));
    }
    return results;
  }

  const std::string& identifier() const { return name_def_->identifier(); }

  const Span& span() const { return span_; }
  NameDef* name_def() const { return name_def_; }
  const std::vector<EnumMember>& values() const { return values_; }
  TypeAnnotation* type() const { return type_; }
  bool is_public() const { return is_public_; }

  void set_signedness(bool is_signed) { is_signed_ = is_signed; }
  absl::optional<bool> signedness() const { return is_signed_; }

 private:
  Span span_;
  NameDef* name_def_;
  TypeAnnotation* type_;
  std::vector<EnumMember> values_;
  absl::optional<bool> is_signed_;
  bool is_public_;
};

// Represents a struct definition.
class Struct : public AstNode {
 public:
  Struct(Span span, NameDef* name_def,
         std::vector<ParametricBinding*> parametric_bindings,
         std::vector<std::pair<NameDef*, TypeAnnotation*>> members,
         bool is_public)
      : span_(std::move(span)),
        name_def_(name_def),
        parametric_bindings_(std::move(parametric_bindings)),
        members_(std::move(members)),
        public_(is_public) {}

  bool is_parametric() const { return !parametric_bindings_.empty(); }

  const std::string& identifier() const { return name_def_->identifier(); }

  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
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

  NameDef* name_def() const { return name_def_; }
  const std::vector<ParametricBinding*> parametric_bindings() const {
    return parametric_bindings_;
  }
  const std::vector<std::pair<NameDef*, TypeAnnotation*>> members() const {
    return members_;
  }
  bool is_public() const { return public_; }
  const Span& span() const { return span_; }

  std::vector<std::string> GetMemberNames() const {
    std::vector<std::string> names;
    for (auto& item : members_) {
      names.push_back(item.first->identifier());
    }
    return names;
  }

 private:
  Span span_;
  NameDef* name_def_;
  std::vector<ParametricBinding*> parametric_bindings_;
  std::vector<std::pair<NameDef*, TypeAnnotation*>> members_;
  bool public_;
};

using StructDef = absl::variant<Struct*, ModRef*>;

std::string StructDefinitionToText(StructDef struct_);

// Represents instantiation of a struct via member expressions.
//
// TODO(leary): 2020-09-08 Break out a StructMember type in lieu of the pair.
class StructInstance : public Expr {
 public:
  StructInstance(Span span, StructDef struct_def,
                 std::vector<std::pair<std::string, Expr*>> members)
      : Expr(std::move(span)),
        struct_def_(struct_def),
        members_(std::move(members)) {}

  std::vector<AstNode*> GetChildren(bool want_types) const override;

  std::string ToString() const override;

  absl::Span<const std::pair<std::string, Expr*>> GetUnorderedMembers() const {
    return members_;
  }

  // Returns the members for the struct instance, ordered by the (resolved)
  // struct definition "struct_def".
  std::vector<std::pair<std::string, Expr*>> GetOrderedMembers(
      Struct* struct_def) const {
    std::vector<std::pair<std::string, Expr*>> result;
    for (std::string name : struct_def->GetMemberNames()) {
      result.push_back({name, GetExpr(name).value()});
    }
    return result;
  }

  absl::StatusOr<Expr*> GetExpr(absl::string_view name) const {
    for (const auto& item : members_) {
      if (item.first == name) {
        return item.second;
      }
    }
    return absl::NotFoundError(absl::StrFormat(
        "Name is not present in struct instance: \"%s\"", name));
  }

  StructDef struct_def() const { return struct_def_; }

 private:
  AstNode* GetStructNode() const {
    if (absl::holds_alternative<ModRef*>(struct_def_)) {
      return absl::get<ModRef*>(struct_def_);
    }
    return absl::get<Struct*>(struct_def_);
  }

  StructDef struct_def_;
  std::vector<std::pair<std::string, Expr*>> members_;
};

// Represents a struct instantiation as a "delta" from a 'splatted' original;
// e.g.
//    Point { y: new_y, ..orig_p }
class SplatStructInstance : public Expr {
 public:
  SplatStructInstance(Span span, StructDef struct_def,
                      std::vector<std::pair<std::string, Expr*>> members,
                      Expr* splatted)
      : Expr(std::move(span)),
        struct_def_(std::move(struct_def)),
        members_(std::move(members)),
        splatted_(splatted) {}

  std::vector<AstNode*> GetChildren(bool want_types) const override;

  std::string ToString() const override {
    std::string members_str = absl::StrJoin(
        members_, ", ",
        [](std::string* out, const std::pair<std::string, Expr*>& member) {
          absl::StrAppendFormat(out, "%s: %s", member.first,
                                member.second->ToString());
        });
    return absl::StrFormat("%s { %s, ..%s }",
                           ToAstNode(struct_def_)->ToString(), members_str,
                           splatted_->ToString());
  }

  Expr* splatted() const { return splatted_; }
  StructDef struct_def() const { return struct_def_; }
  const std::vector<std::pair<std::string, Expr*>>& members() const {
    return members_;
  }

 private:
  // The struct being instantiated.
  StructDef struct_def_;

  // Sequenc eof members being changed from the splatted original; e.g. in the
  // above example this is [('y', new_y)].
  std::vector<std::pair<std::string, Expr*>> members_;

  // Expression that's used as the original struct instance (that we're
  // instantiating a delta from); e.g. orig_p in the example above.
  Expr* splatted_;
};

// Represents a slice in the AST; e.g. `-4+:u2`
class WidthSlice : public AstNode {
 public:
  WidthSlice(Span span, Expr* start, TypeAnnotation* width)
      : span_(std::move(span)), start_(start), width_(width) {}

  std::string ToString() const override {
    return absl::StrFormat("%s+:%s", start_->ToString(), width_->ToString());
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {start_, width_};
  }

  Expr* start() const { return start_; }
  TypeAnnotation* width() const { return width_; }

 private:
  Span span_;
  Expr* start_;
  TypeAnnotation* width_;
};

// Helper type that holds the allowed variations for an index AST node's right
// hand side.
using IndexRhs = std::variant<Expr*, Slice*, WidthSlice*>;

absl::StatusOr<IndexRhs> AstNodeToIndexRhs(AstNode* node);

// Represents an index expression; e.g. `a[i]`
class Index : public Expr {
 public:
  Index(Span span, Expr* lhs, IndexRhs rhs)
      : Expr(std::move(span)), lhs_(lhs), rhs_(rhs) {}

  std::string ToString() const override {
    return absl::StrFormat("(%s)[%s]", lhs_->ToString(),
                           ToAstNode(rhs_)->ToString());
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {lhs_, ToAstNode(rhs_)};
  }

  Expr* lhs() const { return lhs_; }
  IndexRhs rhs() const { return rhs_; }

 private:
  // Expression that yields the value being indexed into; e.g. `a` in `a[10]`.
  Expr* lhs_;
  // Index expression; e.g. `10` in `a[10]`.
  IndexRhs rhs_;
};

// Represents a parsed 'process' specification in the DSL.
class Proc : public AstNode {
 public:
  Proc(Span span, NameDef* name_def, std::vector<Param*> proc_params,
       std::vector<Param*> iter_params, Expr* iter_body, bool is_public)
      : span_(std::move(span)),
        name_def_(name_def),
        proc_params_(std::move(proc_params)),
        iter_params_(std::move(iter_params)),
        iter_body_(iter_body),
        is_public_(is_public) {}

  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
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

  NameDef* name_def() const { return name_def_; }
  bool is_public() const { return is_public_; }

 private:
  Span span_;
  NameDef* name_def_;
  std::vector<Param*> proc_params_;
  std::vector<Param*> iter_params_;
  Expr* iter_body_;
  bool is_public_;
};

// Represents a 'test' definition in the DSL.
//
// TODO(leary): 2020-08-21 Delete this in favor of test directives on functions.
class Test : public AstNode {
 public:
  Test(NameDef* name_def, Expr* body) : name_def_(name_def), body_(body) {}

  std::string ToString() const override {
    return absl::StrFormat("test %s { ... }", name_def_->ToString());
  }

  NameDef* name_def() const { return name_def_; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {name_def_, body_};
  }

  const std::string& identifier() const { return name_def_->identifier(); }
  Expr* body() const { return body_; }

 private:
  NameDef* name_def_;
  Expr* body_;
};

// Represents a new-style unit test construct.
//
// These are specified as follows:
//
// ```dslx
// #![test]
// fn test_foo() { ... }
// ```
//
// We keep Test for backwards compatibility with old-style test constructs.
class TestFunction : public Test {
 public:
  explicit TestFunction(Function* fn)
      : Test(fn->name_def(), fn->body()) {}
};

// Represents a function to be quick-check'd.
class QuickCheck : public AstNode {
 public:
  static constexpr int64 kDefaultTestCount = 1000;

  QuickCheck(Span span, Function* f,
             absl::optional<int64> test_count = absl::nullopt)
      : span_(span),
        f_(f),
        test_count_(test_count ? *test_count : kDefaultTestCount) {}

  std::string ToString() const override {
    return absl::StrFormat("#![quickcheck]\n%s", f_->ToString());
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {f_};
  }

  const std::string& identifier() const { return f_->identifier(); }

  Function* f() const { return f_; }
  int64 test_count() const { return test_count_; }

 private:
  Span span_;
  Function* f_;
  int64 test_count_;
};

// Represents an XLS tuple expression.
class XlsTuple : public Expr {
 public:
  XlsTuple(Span span, std::vector<Expr*> members)
      : Expr(std::move(span)), members_(members) {}

  absl::Span<Expr* const> members() const { return members_; }

  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return ToAstNodes<Expr>(members_);
  }

 private:
  std::vector<Expr*> members_;
};

// Represents a for-loop expression.
class For : public Expr {
 public:
  For(Span span, NameDefTree* names, TypeAnnotation* type, Expr* iterable,
      Expr* body, Expr* init)
      : Expr(std::move(span)),
        names_(names),
        type_(type),
        iterable_(iterable),
        body_(body),
        init_(init) {}

  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override;

  NameDefTree* names() const { return names_; }
  Expr* iterable() const { return iterable_; }
  Expr* body() const { return body_; }
  Expr* init() const { return init_; }
  TypeAnnotation* type() const { return type_; }

 private:
  NameDefTree* names_;    // NameDefTree bound in the body of the loop.
  TypeAnnotation* type_;  // Annotation corresponding to "names".
  Expr* iterable_;        // Expression for "thing to iterate over".
  Expr* body_;  // Expression for the loop body, should evaluate to type type_.
  Expr* init_;  // Initial expression for the loop (start values expr).
};

// Represents a while-loop expression.
class While : public Expr {
 public:
  While(Span span) : Expr(std::move(span)) {}

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {test_, body_, init_};
  }

  std::string ToString() const override {
    return absl::StrFormat("while %s { %s }(%s)", test_->ToString(),
                           body_->ToString(), init_->ToString());
  }

  Expr* test() const { return test_; }
  void set_test(Expr* test) { test_ = test; }
  Expr* body() const { return body_; }
  void set_body(Expr* body) { body_ = body; }
  Expr* init() const { return init_; }
  void set_init(Expr* init) { init_ = init; }

 private:
  Expr* test_;  // Expression that determines whether the body should execute.
  Expr* body_;  // Body to execute each time the test is true.
  Expr* init_;  // Initial value to use for loop carry data.
};

// Represents a cast expression; converting a new value to a target type.
//
// For example:
//
//  foo() as u32
//
// Casts the result of the foo() invocation to a u32 value.
class Cast : public Expr {
 public:
  Cast(Span span, Expr* expr, TypeAnnotation* type)
      : Expr(std::move(span)), expr_(expr), type_(type) {}

  std::string ToString() const override {
    return absl::StrFormat("((%s) as %s)", expr_->ToString(),
                           type_->ToString());
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    if (want_types) {
      return {expr_, type_};
    }
    return {expr_};
  }

  Expr* expr() const { return expr_; }
  TypeAnnotation* type() const { return type_; }

 private:
  Expr* expr_;
  TypeAnnotation* type_;
};

// Represents `next` keyword, refers to the implicit loop-carry call in `Proc`.
class Next : public Expr {
 public:
  using Expr::Expr;

  std::string ToString() const override { return "next"; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {};
  }
};

// Represents `carry` keyword, refers to the implicit loop-carry data in
// `While`.
class Carry : public Expr {
 public:
  Carry(Span span, While* loop) : Expr(std::move(span)), loop_(loop) {}

  std::string ToString() const override { return "carry"; }

  // Note: the while loop is a back reference, it is not a child.
  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {};
  }

  While* loop() const { return loop_; }

 private:
  While* loop_;
};

// Represents a constant definition.
class ConstantDef : public AstNode {
 public:
  ConstantDef(Span span, NameDef* name_def, Expr* value)
      : span_(std::move(span)), name_def_(name_def), value_(value) {}

  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {name_def_, value_};
  }

  const std::string& identifier() const { return name_def_->identifier(); }
  NameDef* name_def() const { return name_def_; }
  Expr* value() const { return value_; }
  const Span& span() const { return span_; }

 private:
  Span span_;
  NameDef* name_def_;
  Expr* value_;
};

// Tree of name definition nodes; e.g.
//
// in LHS of let bindings.
//
// For example:
//
//   let (a, (b, (c)), d) = ...
//
// Makes a:
//
//   NameDefTree((NameDef('a'),
//                NameDefTree((
//                  NameDef('b'),
//                  NameDefTree((
//                    NameDef('c'))))),
//                NameDef('d')))
//
// A "NameDef" is an AST node that signifies an identifier is being bound, so
// this is simply a tree of those (with the tree being constructed via tuples;
// leaves are NameDefs, interior nodes are tuples).
//
// Attributes:
//   span: The span of the names at this level of the tree.
//   tree: The subtree this represents (either a tuple of subtrees or a leaf).
class NameDefTree : public AstNode {
 public:
  using Nodes = std::vector<NameDefTree*>;
  using Leaf = absl::variant<NameDef*, NameRef*, EnumRef*, ModRef*,
                             WildcardPattern*, Number*>;

  NameDefTree(Span span, absl::variant<Nodes, Leaf> tree)
      : span_(std::move(span)), tree_(tree) {}

  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    if (absl::holds_alternative<Leaf>(tree_)) {
      return {ToAstNode(absl::get<Leaf>(tree_))};
    }
    const Nodes& nodes = absl::get<Nodes>(tree_);
    return ToAstNodes<NameDefTree>(nodes);
  }

  bool is_leaf() const { return absl::holds_alternative<Leaf>(tree_); }
  Leaf leaf() const { return absl::get<Leaf>(tree_); }

  const Nodes& nodes() const { return absl::get<Nodes>(tree_); }

  // Flattens this NameDefTree a single level, unwrapping any leaf NDTs; e.g.
  //
  //    LEAF:a => [LEAF:a]
  //    LEAF:a, NODES:[NDT:b, NDT:c], LEAF:d => [LEAF:a, NDT:b, NDT:c, LEAF:d]
  //    NODES:[NDT:a, NDT:LEAF:c], NODES[NDT:c] => [NDT:a, NDT:b, LEAF:c, NDT:d]
  //
  // This is useful for flattening a tuple a single level; e.g. where a
  // NameDefTree is going to be used as variadic args in for-loop to function
  // conversion.
  std::vector<absl::variant<Leaf, NameDefTree*>> Flatten1() {
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

  // Flattens the (recursive) NameDefTree into a list of leaves.
  std::vector<Leaf> Flatten() const {
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

  // A pattern is irrefutable if it always causes a successful match.
  //
  // Returns whether this NameDefTree is known-irrefutable.
  bool IsIrrefutable() const {
    auto leaves = Flatten();
    return std::all_of(leaves.begin(), leaves.end(), [](Leaf leaf) {
      return absl::holds_alternative<NameDef*>(leaf) ||
             absl::holds_alternative<WildcardPattern*>(leaf);
    });
  }

  const absl::variant<Nodes, Leaf>& tree() const { return tree_; }
  const Span& span() const { return span_; }

 private:
  Span span_;
  absl::variant<Nodes, Leaf> tree_;
};

// Represents a let-binding expression.
class Let : public Expr {
 public:
  Let(Span span, NameDefTree* name_def_tree, TypeAnnotation* type, Expr* rhs,
      Expr* body, ConstantDef* const_def)
      : Expr(std::move(span)),
        name_def_tree_(name_def_tree),
        type_(type),
        rhs_(rhs),
        body_(body),
        constant_def_(const_def) {}

  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    std::vector<AstNode*> results = {name_def_tree_};
    if (type_ != nullptr && want_types) {
      results.push_back(type_);
    }
    results.push_back(rhs_);
    results.push_back(body_);
    if (constant_def_ != nullptr) {
      results.push_back(constant_def_);
    }
    return results;
  }

  NameDefTree* name_def_tree() const { return name_def_tree_; }
  TypeAnnotation* type() const { return type_; }
  Expr* rhs() const { return rhs_; }
  Expr* body() const { return body_; }
  ConstantDef* constant_def() const { return constant_def_; }

 private:
  // Names that are bound by this let expression; e.g. in
  //  let (a, b, (c)) = (1, 2, (3,));
  //  ...
  //
  // the name_def_tree is `(a, b, (c))`
  NameDefTree* name_def_tree_;

  // The optional annotated type on the let expression, may be null.
  TypeAnnotation* type_;

  // Right hand side of the let; e.g. in `let a = b; c` this is `b`.
  Expr* rhs_;

  // The body of the let: it has the expression to be evaluated with the let
  // bindings; e.g. in `let a = b; c` this is `c`.
  Expr* body_;

  // Whether or not this is a constant binding; constant bindings cannot be
  // shadowed. May be null.
  ConstantDef* constant_def_;
};

using ModuleMember = absl::variant<Function*, Test*, QuickCheck*, TypeDef*,
                                   Struct*, ConstantDef*, Enum*, Import*>;

absl::StatusOr<ModuleMember> AsModuleMember(AstNode* node);

// Represents a syntactic module in the AST.
//
// Modules contain top-level definitions such as functions and tests.
//
// Attributes:
//   name: Name of this module.
//   top: Top-level module constructs; e.g. functions, tests. Given as a
//   sequence
//     instead of a mapping in case there are unnamed constructs at the module
//     level (e.g. metadata, docstrings).
class Module : public AstNode {
 public:
  explicit Module(std::string name) : name_(std::move(name)) {}

  std::vector<AstNode*> GetChildren(bool want_types) const override;

  std::string ToString() const override {
    return absl::StrJoin(top_, "\n",
                         [](std::string* out, const ModuleMember& member) {
                           absl::StrAppend(out, ToAstNode(member)->ToString());
                         });
  }

  template <typename T, typename... Args>
  T* Make(Args&&... args) {
    std::unique_ptr<T> node = absl::make_unique<T>(std::forward<Args>(args)...);
    T* ptr = node.get();
    nodes_.push_back(std::move(node));
    return ptr;
  }

  void AddTop(ModuleMember member) { top_.push_back(member); }

  absl::StatusOr<Function*> GetFunction(absl::string_view target_name) {
    for (ModuleMember& member : top_) {
      if (absl::holds_alternative<Function*>(member)) {
        Function* f = absl::get<Function*>(member);
        if (f->identifier() == target_name) {
          return f;
        }
      }
    }
    return absl::NotFoundError(absl::StrFormat(
        "No function in module %s with name \"%s\"", name_, target_name));
  }

  absl::StatusOr<Test*> GetTest(absl::string_view target_name) {
    for (ModuleMember& member : top_) {
      if (absl::holds_alternative<Test*>(member)) {
        return absl::get<Test*>(member);
      }
    }
    return absl::NotFoundError(absl::StrFormat(
        "No test in module %s with name \"%s\"", name_, target_name));
  }

  absl::Span<ModuleMember const> top() const { return top_; }
  std::vector<ModuleMember>* mutable_top() { return &top_; }

  // Obtains all the type definition nodes in the module:
  //    TypeDef, Struct, Enum
  absl::flat_hash_map<std::string, TypeDefinition> GetTypeDefinitionByName()
      const {
    absl::flat_hash_map<std::string, TypeDefinition> result;
    for (auto& member : top_) {
      if (absl::holds_alternative<TypeDef*>(member)) {
        TypeDef* td = absl::get<TypeDef*>(member);
        result[td->identifier()] = td;
      } else if (absl::holds_alternative<Enum*>(member)) {
        Enum* enum_ = absl::get<Enum*>(member);
        result[enum_->identifier()] = enum_;
      } else if (absl::holds_alternative<Struct*>(member)) {
        Struct* struct_ = absl::get<Struct*>(member);
        result[struct_->identifier()] = struct_;
      }
    }
    return result;
  }

  absl::flat_hash_map<std::string, ConstantDef*> GetConstantByName() const {
    return GetTopWithTByName<ConstantDef>();
  }

  absl::flat_hash_map<std::string, Function*> GetFunctionByName() const {
    return GetTopWithTByName<Function>();
  }

  std::vector<QuickCheck*> GetQuickChecks() const {
    return GetTopWithT<QuickCheck>();
  }
  std::vector<Struct*> GetStructs() const { return GetTopWithT<Struct>(); }
  std::vector<Function*> GetFunctions() const {
    return GetTopWithT<Function>();
  }
  std::vector<Test*> GetTests() const { return GetTopWithT<Test>(); }
  std::vector<ConstantDef*> GetConstantDefs() const {
    return GetTopWithT<ConstantDef>();
  }

  std::vector<std::string> GetTestNames() const {
    std::vector<std::string> result;
    for (auto& member : top_) {
      if (absl::holds_alternative<Test*>(member)) {
        Test* t = absl::get<Test*>(member);
        result.push_back(t->identifier());
      }
    }
    return result;
  }

  const std::string& name() const { return name_; }

 private:
  // Returns all of the elements of top_ that have the given variant type T.
  template <typename T>
  std::vector<T*> GetTopWithT() const {
    std::vector<T*> result;
    for (auto& member : top_) {
      if (absl::holds_alternative<T*>(member)) {
        result.push_back(absl::get<T*>(member));
      }
    }
    return result;
  }

  // Return sall the elements of top_ that have the given variant type T, using
  // T's identifier as a key. (T must have a string identifier.)
  template <typename T>
  absl::flat_hash_map<std::string, T*> GetTopWithTByName() const {
    absl::flat_hash_map<std::string, T*> result;
    for (auto& member : top_) {
      if (absl::holds_alternative<T*>(member)) {
        auto* c = absl::get<T*>(member);
        result.insert({c->identifier(), c});
      }
    }
    return result;
  }

  std::string name_;               // Name of this module.
  std::vector<ModuleMember> top_;  // Top-level members of this module.
  std::vector<std::unique_ptr<AstNode>> nodes_;  // Lifetime-owned AST nodes.
};

// Helper for determining whether an AST node is constant (e.g. can be
// considered a constant value in a ConstantArray).
bool IsConstant(AstNode* n);

}  // namespace xls::dslx

#endif  // XLS_DSLX_AST_H_
