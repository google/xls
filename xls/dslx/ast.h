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

#ifndef XLS_DSLX_AST_H_
#define XLS_DSLX_AST_H_

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/casts.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/pos.h"
#include "xls/ir/bits.h"

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

// Higher-order macro for all the Expr node leaf types (non-abstract).
#define XLS_DSLX_EXPR_NODE_EACH(X) \
  X(Array)                         \
  X(Attr)                          \
  X(Binop)                         \
  X(Carry)                         \
  X(Cast)                          \
  X(ColonRef)                      \
  X(ConstantArray)                 \
  X(ConstRef)                      \
  X(For)                           \
  X(Index)                         \
  X(Invocation)                    \
  X(Let)                           \
  X(Match)                         \
  X(NameRef)                       \
  X(Next)                          \
  X(Number)                        \
  X(SplatStructInstance)           \
  X(String)                        \
  X(StructInstance)                \
  X(Ternary)                       \
  X(Unop)                          \
  X(While)                         \
  X(XlsTuple)

// Higher-order macro for all the AST node leaf types (non-abstract).
//
// (Note that this includes all the Expr node leaf kinds listed in
// XLS_DSLX_EXPR_NODE_EACH).
#define XLS_DSLX_AST_NODE_EACH(X) \
  X(BuiltinNameDef)               \
  X(ConstantDef)                  \
  X(EnumDef)                      \
  X(Function)                     \
  X(Import)                       \
  X(MatchArm)                     \
  X(Module)                       \
  X(NameDef)                      \
  X(NameDefTree)                  \
  X(Param)                        \
  X(ParametricBinding)            \
  X(Proc)                         \
  X(QuickCheck)                   \
  X(Slice)                        \
  X(StructDef)                    \
  X(TestFunction)                 \
  X(TypeDef)                      \
  X(TypeRef)                      \
  X(WidthSlice)                   \
  X(WildcardPattern)              \
  /* type annotations */          \
  X(ArrayTypeAnnotation)          \
  X(BuiltinTypeAnnotation)        \
  X(TupleTypeAnnotation)          \
  X(TypeRefTypeAnnotation)        \
  XLS_DSLX_EXPR_NODE_EACH(X)

// Forward decl of non-leaf type.
class Expr;

// Forward decls of all leaf types.
#define FORWARD_DECL(__type) class __type;
XLS_DSLX_AST_NODE_EACH(FORWARD_DECL)
#undef FORWARD_DECL

// Helper type (abstract base) for double dispatch on AST nodes.
class AstNodeVisitor {
 public:
  virtual ~AstNodeVisitor() = default;

#define DECLARE_HANDLER(__type) \
  virtual absl::Status Handle##__type(__type* n) = 0;
  XLS_DSLX_AST_NODE_EACH(DECLARE_HANDLER)
#undef DECLARE_HANDLER
};

// Subtype of abstract AstNodeVisitor that returns ok status (does nothing) for
// every node type.
class AstNodeVisitorWithDefault : public AstNodeVisitor {
 public:
  ~AstNodeVisitorWithDefault() override = default;

#define DECLARE_HANDLER(__type) \
  absl::Status Handle##__type(__type* n) override { return absl::OkStatus(); }
  XLS_DSLX_AST_NODE_EACH(DECLARE_HANDLER)
#undef DECLARE_HANDLER
};

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

  // Returns all of the free variable NameRefs that are references to constants
  // (as the ConstRef subtype of NameRef).
  std::vector<ConstRef*> GetConstRefs();

  // Returns the number of unique free variables (note: not the number of
  // references, but the number of free variables).
  int64_t GetFreeVariableCount() const { return values_.size(); }

 private:
  absl::flat_hash_map<std::string, std::vector<NameRef*>> values_;
};

// Abstract base class for AST nodes.
class AstNode {
 public:
  explicit AstNode(Module* owner) : owner_(owner) {}
  virtual ~AstNode() = default;

  // Retrieves the name of the leafmost-derived class, suitable for debugging;
  // e.g. "NameDef", "BuiltinTypeAnnotation", etc.
  virtual absl::string_view GetNodeTypeName() const = 0;
  virtual std::string ToString() const = 0;

  virtual absl::optional<Span> GetSpan() const = 0;

  // Retrieves all the child nodes for this AST node.
  //
  // If want_types is false, then type annotations should be excluded from the
  // returned child nodes. This exclusion of types is useful e.g. when
  // attempting to find free variables that are referred to during program
  // execution, since all type information must be resolved to constants at type
  // inference time.
  virtual std::vector<AstNode*> GetChildren(bool want_types) const = 0;

  // Used for double-dispatch (making the actual type of an apparent AstNode
  // available to calling code).
  virtual absl::Status Accept(AstNodeVisitor* v) = 0;

  // Retrieves all the free variables (references to names that are defined
  // prior to start_pos) that are transitively in this AST subtree.
  //
  // For example, if given the AST node for this function:
  //
  //    const FOO = u32:42;
  //    fn main(x: u32) { FOO+x }
  //
  // And using the starting point of the function as the start_pos, the FOO will
  // be flagged as a free variable and returned.
  FreeVariables GetFreeVariables(const Pos* start_pos = nullptr);

  Module* owner() const { return owner_; }

 private:
  Module* owner_;
};

// Visits transitively from the root down using post-order visitation (visit
// children, then node). want_types is as in AstNode::GetChildren().
absl::Status WalkPostOrder(AstNode* root, AstNodeVisitor* visitor,
                           bool want_types);

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
  TypeAnnotation(Module* owner, Span span)
      : AstNode(owner), span_(std::move(span)) {}

  const Span& span() const { return span_; }
  absl::optional<Span> GetSpan() const override { return span_; }

 private:
  Span span_;
};

#include "xls/dslx/ast_builtin_types.inc"

// Enumeration of types that are built-in keywords; e.g. `u32`, `bool`, etc.
enum class BuiltinType {
#define FIRST_COMMA(A, ...) A,
  XLS_DSLX_BUILTIN_TYPE_EACH(FIRST_COMMA)
#undef FIRST_COMMA
};

// All builtin types up to this limit have a concrete width and sign -- above
// this point are things like "bits", "uN", "sN" which need a corresponding
// array dimension to have a known bit count.
constexpr int64_t kConcreteBuiltinTypeLimit =
    static_cast<int64_t>(BuiltinType::kS64) + 1;

std::string BuiltinTypeToString(BuiltinType t);
absl::StatusOr<BuiltinType> BuiltinTypeFromString(absl::string_view s);

absl::StatusOr<BuiltinType> GetBuiltinType(bool is_signed, int64_t width);

// Represents a built-in type annotation; e.g. `u32`, `bits`, etc.
class BuiltinTypeAnnotation : public TypeAnnotation {
 public:
  BuiltinTypeAnnotation(Module* owner, Span span, BuiltinType builtin_type);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleBuiltinTypeAnnotation(this);
  }

  absl::string_view GetNodeTypeName() const override {
    return "BuiltinTypeAnnotation";
  }
  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {};
  }

  std::string ToString() const override {
    return BuiltinTypeToString(builtin_type_);
  }

  int64_t GetBitCount() const;

  // Returns true if signed, false if unsigned.
  bool GetSignedness() const;

  BuiltinType builtin_type() const { return builtin_type_; }

 private:
  BuiltinType builtin_type_;
};

// Represents a tuple type annotation; e.g. `(u32, s42)`.
class TupleTypeAnnotation : public TypeAnnotation {
 public:
  TupleTypeAnnotation(Module* owner, Span span,
                      std::vector<TypeAnnotation*> members);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleTupleTypeAnnotation(this);
  }

  absl::string_view GetNodeTypeName() const override {
    return "TupleTypeAnnotation";
  }
  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return ToAstNodes<TypeAnnotation>(members_);
  }

  const std::vector<TypeAnnotation*>& members() const { return members_; }
  int64_t size() const { return members_.size(); }
  bool empty() const { return members_.empty(); }

 private:
  std::vector<TypeAnnotation*> members_;
};

// Represents a type reference annotation; e.g.
//
//  type Foo = u32;
//  fn f(x: Foo) -> Foo { ... }
//
// `Foo` is a type reference (TypeRef) in the function signature, and since
// parameters and return types are both type annotations (as are let bindings,
// cast type targets, etc.) we wrap that up in the TypeAnnotation AST construct
// using this type.
class TypeRefTypeAnnotation : public TypeAnnotation {
 public:
  TypeRefTypeAnnotation(Module* owner, Span span, TypeRef* type_ref,
                        std::vector<Expr*> parametrics);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleTypeRefTypeAnnotation(this);
  }

  TypeRef* type_ref() const { return type_ref_; }

  std::string ToString() const override;

  absl::string_view GetNodeTypeName() const override {
    return "TypeRefTypeAnnotation";
  }
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  const std::vector<Expr*>& parametrics() const { return parametrics_; }
  bool HasParametrics() const { return !parametrics_.empty(); }

 private:
  TypeRef* type_ref_;
  std::vector<Expr*> parametrics_;
};

// Represents an array type annotation; e.g. `u32[5]`.
class ArrayTypeAnnotation : public TypeAnnotation {
 public:
  ArrayTypeAnnotation(Module* owner, Span span, TypeAnnotation* element_type,
                      Expr* dim);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleArrayTypeAnnotation(this);
  }

  absl::string_view GetNodeTypeName() const override {
    return "ArrayTypeAnnotation";
  }
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
  BuiltinNameDef(Module* owner, std::string identifier)
      : AstNode(owner), identifier_(std::move(identifier)) {}

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleBuiltinNameDef(this);
  }
  absl::optional<Span> GetSpan() const override { return absl::nullopt; }

  absl::string_view GetNodeTypeName() const override {
    return "BuiltinNameDef";
  }
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
  WildcardPattern(Module* owner, Span span)
      : AstNode(owner), span_(std::move(span)) {}

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleWildcardPattern(this);
  }

  absl::string_view GetNodeTypeName() const override {
    return "WildcardPattern";
  }
  std::string ToString() const override { return "_"; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {};
  }

  const Span& span() const { return span_; }
  absl::optional<Span> GetSpan() const override { return span_; }

 private:
  Span span_;
};

// Represents the definition of a name (identifier).
class NameDef : public AstNode {
 public:
  NameDef(Module* owner, Span span, std::string identifier, AstNode* definer);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleNameDef(this);
  }

  absl::string_view GetNodeTypeName() const override { return "NameDef"; }
  const Span& span() const { return span_; }
  absl::optional<Span> GetSpan() const override { return span_; }
  const std::string& identifier() const { return identifier_; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {};
  }
  std::string ToString() const override { return identifier_; }
  std::string ToReprString() const {
    return absl::StrFormat("NameDef(identifier=\"%s\")", identifier_);
  }

  void set_definer(AstNode* definer) { definer_ = definer; }
  AstNode* definer() const { return definer_; }

 private:
  Span span_;
  std::string identifier_;
  // Node that caused this name to be defined.
  // May be null, generally because we make NameDefs then wrap those in the
  // defining nodes, and have to "circle back" and note what that resulting
  // "definer" node was.
  AstNode* definer_;
};

// Abstract base class for visitation of expression nodes in the AST.
class ExprVisitor {
 public:
  virtual ~ExprVisitor() = default;

  virtual void HandleArray(Array* expr) = 0;
  virtual void HandleAttr(Attr* expr) = 0;
  virtual void HandleBinop(Binop* expr) = 0;
  virtual void HandleCarry(Carry* expr) = 0;
  virtual void HandleCast(Cast* expr) = 0;
  virtual void HandleConstRef(ConstRef* expr) = 0;
  virtual void HandleColonRef(ColonRef* expr) = 0;
  virtual void HandleFor(For* expr) = 0;
  virtual void HandleIndex(Index* expr) = 0;
  virtual void HandleInvocation(Invocation* expr) = 0;
  virtual void HandleLet(Let* expr) = 0;
  virtual void HandleMatch(Match* expr) = 0;
  virtual void HandleNameRef(NameRef* expr) = 0;
  virtual void HandleNext(Next* expr) = 0;
  virtual void HandleNumber(Number* expr) = 0;
  virtual void HandleString(String* expr) = 0;
  virtual void HandleStructInstance(StructInstance* expr) = 0;
  virtual void HandleSplatStructInstance(SplatStructInstance* expr) = 0;
  virtual void HandleTernary(Ternary* expr) = 0;
  virtual void HandleUnop(Unop* expr) = 0;
  virtual void HandleWhile(While* expr) = 0;
  virtual void HandleXlsTuple(XlsTuple* expr) = 0;
};

// Abstract base class for AST node that can appear in expression positions
// (i.e. can produce runtime values).
class Expr : public AstNode {
 public:
  Expr(Module* owner, Span span) : AstNode(owner), span_(span) {}
  virtual ~Expr() = default;

  const Span& span() const { return span_; }
  void set_span(const Span& span) { span_ = span; }
  absl::optional<Span> GetSpan() const override { return span_; }

  virtual void AcceptExpr(ExprVisitor* v) = 0;

 private:
  Span span_;
};

// Represents a reference to a name (identifier).
class NameRef : public Expr {
 public:
  NameRef(Module* owner, Span span, std::string identifier, AnyNameDef name_def)
      : Expr(owner, std::move(span)),
        name_def_(name_def),
        identifier_(std::move(identifier)) {}

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleNameRef(this);
  }
  void AcceptExpr(ExprVisitor* v) override { v->HandleNameRef(this); }

  const std::string& identifier() const { return identifier_; }

  absl::string_view GetNodeTypeName() const override { return "NameRef"; }
  std::string ToString() const override { return identifier_; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {ToAstNode(name_def_)};
  }

  template <typename T>
  bool DefinerIs() {
    if (absl::holds_alternative<BuiltinNameDef*>(name_def_)) {
      return false;
    }
    auto* name_def = absl::get<NameDef*>(name_def_);
    return dynamic_cast<T*>(name_def->definer()) != nullptr;
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

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleConstRef(this);
  }
  void AcceptExpr(ExprVisitor* v) override { v->HandleConstRef(this); }

  absl::string_view GetNodeTypeName() const override { return "ConstRef"; }

  // When holding a ConstRef we know that the corresponding NameDef cannot be
  // builtin (since consts are user constructs).
  NameDef* name_def() { return absl::get<NameDef*>(NameRef::name_def()); }

  // Returns the constant definition that this ConstRef is referring to.
  ConstantDef* GetConstantDef() {
    AstNode* definer = name_def()->definer();
    XLS_CHECK(definer != nullptr);
    return down_cast<ConstantDef*>(definer);
  }
};

enum class NumberKind {
  kBool,
  kCharacter,
  kOther,
};

// Represents a literal number value.
class Number : public Expr {
 public:
  Number(Module* owner, Span span, std::string text, NumberKind kind,
         TypeAnnotation* type);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleNumber(this);
  }
  void AcceptExpr(ExprVisitor* v) override { v->HandleNumber(this); }

  absl::string_view GetNodeTypeName() const override { return "Number"; }
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  std::string ToString() const override;

  // Returns this number without a leading type AST node prefix (even if it is
  // present).
  std::string ToStringNoType() const;

  TypeAnnotation* type_annotation() const { return type_annotation_; }

  // TODO(leary): 2021-05-18 We should remove this setter -- it is currently
  // used because of the `$type_annotation:$number` syntax, where the type is
  // parsed, the number is parsed independent of type context, but then the
  // type_annotation is imbued *into* the number. Cleaner would be to make a
  // TypedNumber construct that decorated a bare number with its literal
  // type_annotation context.
  void set_type_annotation(TypeAnnotation* type_annotation) {
    type_annotation_ = type_annotation;
  }

  const std::string& text() const { return text_; }

  // Turns the text for this number into a Bits object with the given bit_count.
  absl::StatusOr<Bits> GetBits(int64_t bit_count) const;

  // Note: fails if the value doesn't fit in 64 bits.
  absl::StatusOr<uint64_t> GetAsUint64() const {
    XLS_ASSIGN_OR_RETURN(Bits bits, GetBits(64));
    return bits.ToUint64();
  }

  NumberKind kind() const { return kind_; }

 private:
  std::string text_;  // Will never be empty.
  NumberKind kind_;
  TypeAnnotation* type_annotation_;  // May be null.
};

// A literal string of u8s. Does not internally include opening and closing
// quotation marks.
class String : public Expr {
 public:
  String(Module* owner, Span span, absl::string_view text)
      : Expr(owner, span), text_(text) {}

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleString(this);
  }
  void AcceptExpr(ExprVisitor* v) override { v->HandleString(this); }
  absl::string_view GetNodeTypeName() const override { return "String"; }
  std::string ToString() const override {
    // We need to re-insert the quote-escaping slash.
    return absl::StrFormat("\"%s\"",
                           absl::StrReplaceAll(text_, {{"\"", "\\\""}}));
  }
  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {};
  }

  const std::string& text() const { return text_; }

 private:
  std::string text_;
};

// Represents a user-defined-type definition; e.g.
//    type Foo = (u32, u32);
//    type Bar = (u32, Foo);
//
// TODO(leary): 2020-09-15 Rename to TypeAlias, less of a loaded term.
class TypeDef : public AstNode {
 public:
  TypeDef(Module* owner, Span span, NameDef* name_def, TypeAnnotation* type,
          bool is_public);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleTypeDef(this);
  }
  absl::string_view GetNodeTypeName() const override { return "TypeDef"; }
  const std::string& identifier() const { return name_def_->identifier(); }

  std::string ToString() const override {
    return absl::StrFormat("%stype %s = %s;", is_public_ ? "pub " : "",
                           identifier(), type_annotation_->ToString());
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {name_def_, type_annotation_};
  }

  NameDef* name_def() const { return name_def_; }
  TypeAnnotation* type_annotation() const { return type_annotation_; }
  bool is_public() const { return is_public_; }
  const Span& span() const { return span_; }
  absl::optional<Span> GetSpan() const override { return span_; }

 private:
  Span span_;
  NameDef* name_def_;
  TypeAnnotation* type_annotation_;
  bool is_public_;
};

// Represents an array expression; e.g. `[a, b, c]`.
class Array : public Expr {
 public:
  Array(Module* owner, Span span, std::vector<Expr*> members,
        bool has_ellipsis);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleArray(this);
  }
  void AcceptExpr(ExprVisitor* v) override { v->HandleArray(this); }

  absl::string_view GetNodeTypeName() const override { return "Array"; }
  std::string ToString() const override;
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  const std::vector<Expr*>& members() const { return members_; }
  TypeAnnotation* type_annotation() const { return type_annotation_; }

  // TODO(leary): 2021-05-18 See TODO comment on Number::set_type_annotation for
  // the reason this exists (prefix types for literal values), but it should be
  // removed in favor of a decorator construct instead of using mutability.
  void set_type_annotation(TypeAnnotation* type_annotation) {
    type_annotation_ = type_annotation;
  }

  bool has_ellipsis() const { return has_ellipsis_; }

 private:
  TypeAnnotation* type_annotation_ = nullptr;
  std::vector<Expr*> members_;
  bool has_ellipsis_;
};

// A constant array expression is an array expression where all of the
// expressions contained within it are constant.
class ConstantArray : public Array {
 public:
  // Adds checking for constant-expression-ness of the members beyond
  // Array::Array.
  ConstantArray(Module* owner, Span span, std::vector<Expr*> members,
                bool has_ellipsis);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleConstantArray(this);
  }
};

// Several different AST nodes define types that can be referred to by a
// TypeRef.
using TypeDefinition = absl::variant<TypeDef*, StructDef*, EnumDef*, ColonRef*>;

absl::StatusOr<TypeDefinition> ToTypeDefinition(AstNode* node);

// Represents an AST construct that refers to a defined type.
//
// Attrs:
//  type_definition: The resolved type if it can be resolved locally, or a
//    ColonRef if the type lives in an external module.
class TypeRef : public AstNode {
 public:
  TypeRef(Module* owner, Span span, std::string text,
          TypeDefinition type_definition);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleTypeRef(this);
  }

  absl::string_view GetNodeTypeName() const override { return "TypeRef"; }
  std::string ToString() const override { return text_; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {ToAstNode(type_definition_)};
  }

  const std::string& text() const { return text_; }
  const TypeDefinition& type_definition() const { return type_definition_; }
  const Span& span() const { return span_; }
  absl::optional<Span> GetSpan() const override { return span_; }

 private:
  Span span_;
  std::string text_;
  TypeDefinition type_definition_;
};

// Represents an import statement; e.g.
//  import std as my_std
class Import : public AstNode {
 public:
  Import(Module* owner, Span span, std::vector<std::string> subject,
         NameDef* name_def, absl::optional<std::string> alias);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleImport(this);
  }
  absl::string_view GetNodeTypeName() const override { return "Import"; }
  const std::string& identifier() const { return name_def_->identifier(); }

  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {name_def_};
  }

  const std::vector<std::string>& subject() const { return subject_; }
  NameDef* name_def() const { return name_def_; }
  const Span& span() const { return span_; }
  absl::optional<Span> GetSpan() const override { return span_; }

 private:
  // Span of the import in the text.
  Span span_;
  // Name of the module being imported ("original" name before aliasing); e.g.
  // "std". Only present if the import is aliased.
  std::vector<std::string> subject_;
  // The name definition we bind the import to.
  NameDef* name_def_;
  // The identifier text we bind the import to.
  absl::optional<std::string> alias_;
};

// Represents a module-value or enum-value style reference when the LHS
// expression is unknown; e.g. when accessing a member in a module:
//
//    some_mod::SomeEnum::VALUE
//
// Then the ColonRef `some_mod::SomeEnum` is the LHS.
class ColonRef : public Expr {
 public:
  using Subject = absl::variant<NameRef*, ColonRef*>;

  ColonRef(Module* owner, Span span, Subject subject, std::string attr);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleColonRef(this);
  }
  void AcceptExpr(ExprVisitor* v) override { v->HandleColonRef(this); }

  absl::string_view GetNodeTypeName() const override { return "ColonRef"; }
  std::string ToString() const override {
    return absl::StrFormat("%s::%s", ToAstNode(subject_)->ToString(), attr_);
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {ToAstNode(subject_)};
  }

  Subject subject() const { return subject_; }
  const std::string& attr() const { return attr_; }

  // Resolves the subject of this ColonRef to an import node, or returns a
  // nullopt if the subject is not an imported module.
  //
  // Note: if the value is not nullopt, it will be a valid pointer (not
  // nullptr).
  absl::optional<Import*> ResolveImportSubject() const;

 private:
  Subject subject_;
  std::string attr_;
};

absl::StatusOr<ColonRef::Subject> ToColonRefSubject(Expr* e);

// Represents a function parameter.
class Param : public AstNode {
 public:
  Param(Module* owner, NameDef* name_def, TypeAnnotation* type);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleParam(this);
  }

  absl::string_view GetNodeTypeName() const override { return "Param"; }
  std::string ToString() const override {
    return absl::StrFormat("%s: %s", name_def_->ToString(),
                           type_annotation_->ToString());
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {name_def_, type_annotation_};
  }

  const Span& span() const { return span_; }
  NameDef* name_def() const { return name_def_; }
  TypeAnnotation* type_annotation() const { return type_annotation_; }
  const std::string& identifier() const { return name_def_->identifier(); }
  absl::optional<Span> GetSpan() const override { return span_; }

 private:
  NameDef* name_def_;
  TypeAnnotation* type_annotation_;
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
  Unop(Module* owner, Span span, UnopKind kind, Expr* operand)
      : Expr(owner, std::move(span)), kind_(kind), operand_(operand) {}

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleUnop(this);
  }
  void AcceptExpr(ExprVisitor* v) override { v->HandleUnop(this); }

  absl::string_view GetNodeTypeName() const override { return "Unop"; }
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
  X(kShl, "SHLL", "<<")                   \
  X(kShr, "SHRL", ">>")                   \
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

// Returns the "operator token" corresponding to the given binop kind; e.g. "+"
// for kAdd.
std::string BinopKindFormat(BinopKind kind);

// Returns a string representation of the given binary operation kind; e.g.
// "LOGICAL_AND".
std::string BinopKindToString(BinopKind kind);

inline std::ostream& operator<<(std::ostream& os, BinopKind kind) {
  os << BinopKindToString(kind);
  return os;
}

// The binary operators that have signature `(T, T) -> T` (i.e. they take and
// produce the "same type").
const absl::btree_set<BinopKind>& GetBinopSameTypeKinds();

// Binary operators that have signature `(T, T) -> bool`.
const absl::btree_set<BinopKind>& GetBinopComparisonKinds();

// Binary operators that are shift operations.
const absl::btree_set<BinopKind>& GetBinopShifts();

// Represents a binary operation expression; e.g. `x + y`.
class Binop : public Expr {
 public:
  Binop(Module* owner, Span span, BinopKind kind, Expr* lhs, Expr* rhs);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleBinop(this);
  }
  void AcceptExpr(ExprVisitor* v) override { v->HandleBinop(this); }

  absl::string_view GetNodeTypeName() const override { return "Binop"; }
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
  Ternary(Module* owner, Span span, Expr* test, Expr* consequent,
          Expr* alternate);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleTernary(this);
  }
  void AcceptExpr(ExprVisitor* v) override { v->HandleTernary(this); }

  absl::string_view GetNodeTypeName() const override { return "Ternary"; }
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
//  fn f<X: u32, Y: u32 = X+X>(x: bits[X]) -> bits[Y] {
//      ^~~~~~~~~~~~~~~~~~~~~^
//    x ++ x
//  }
//
// There are two parametric bindings:
//
// * X is a u32.
// * Y is a value derived from the parametric binding of X (whose expression is
//   `X+X`)
class ParametricBinding : public AstNode {
 public:
  ParametricBinding(Module* owner, NameDef* name_def,
                    TypeAnnotation* type_annotation, Expr* expr);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleParametricBinding(this);
  }

  // TODO(leary): 2020-08-21 Fix this, the span is more than just the name def's
  // span, it must include the type/expr.
  const Span& span() const { return name_def_->span(); }
  absl::optional<Span> GetSpan() const override { return span(); }

  absl::string_view GetNodeTypeName() const override {
    return "ParametricBinding";
  }
  std::string ToString() const override;

  std::string ToReprString() const;

  std::vector<AstNode*> GetChildren(bool want_types) const override;

  NameDef* name_def() const { return name_def_; }
  TypeAnnotation* type_annotation() const { return type_annotation_; }
  Expr* expr() const { return expr_; }

  const std::string& identifier() const { return name_def_->identifier(); }

 private:
  NameDef* name_def_;
  TypeAnnotation* type_annotation_;

  // The "default" parametric expression (e.g. the expression to use for this
  // parametric name_def_ when there is no explicit binding provided by the
  // caller). May be null.
  Expr* expr_;
};

// Represents a function definition.
class Function : public AstNode {
 public:
  Function(Module* owner, Span span, NameDef* name_def,
           std::vector<ParametricBinding*> parametric_bindings,
           std::vector<Param*> params, TypeAnnotation* return_type, Expr* body,
           bool is_public);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleFunction(this);
  }
  absl::string_view GetNodeTypeName() const override { return "Function"; }
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  std::string Format(bool include_body = true) const;

  std::string ToString() const override {
    return Format(/*include_body=*/false);
  }

  NameDef* name_def() const { return name_def_; }
  Expr* body() const { return body_; }
  const Span& span() const { return span_; }
  absl::optional<Span> GetSpan() const override { return span_; }

  const std::string& identifier() const { return name_def_->identifier(); }
  const std::vector<Param*>& params() const { return params_; }
  const std::vector<ParametricBinding*>& parametric_bindings() const {
    return parametric_bindings_;
  }
  bool IsParametric() const { return !parametric_bindings_.empty(); }
  bool is_public() const { return is_public_; }

  TypeAnnotation* return_type() const { return return_type_; }

  std::vector<std::string> GetFreeParametricKeys() const;
  absl::btree_set<std::string> GetFreeParametricKeySet() const {
    std::vector<std::string> keys = GetFreeParametricKeys();
    return absl::btree_set<std::string>(keys.begin(), keys.end());
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
  MatchArm(Module* owner, Span span, std::vector<NameDefTree*> patterns,
           Expr* expr);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleMatchArm(this);
  }
  absl::string_view GetNodeTypeName() const override { return "MatchArm"; }
  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override;

  const std::vector<NameDefTree*>& patterns() const { return patterns_; }
  Expr* expr() const { return expr_; }
  const Span& span() const { return span_; }
  absl::optional<Span> GetSpan() const override { return span_; }

  // Returns the span from the start of the (first) pattern to the limit of the
  // (last) pattern.
  Span GetPatternSpan() const;

 private:
  Span span_;
  std::vector<NameDefTree*> patterns_;  // Note: never empty.
  Expr* expr_;  // Expression that is executed if one of the patterns matches.
};

// Represents a match (pattern match) expression.
//
// A match expression has zero or more *arms*.
// Each *arm* has a set of *patterns* that can cause a match, and a "right hand
// side" *expression* to yield the value of if any of the patterns match
// (prioritzed in sequential order from first arm to last arm).
class Match : public Expr {
 public:
  Match(Module* owner, Span span, Expr* matched, std::vector<MatchArm*> arms);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleMatch(this);
  }
  void AcceptExpr(ExprVisitor* v) override { v->HandleMatch(this); }

  absl::string_view GetNodeTypeName() const override { return "Match"; }
  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override;

  const std::vector<MatchArm*>& arms() const { return arms_; }
  Expr* matched() const { return matched_; }

 private:
  Expr* matched_;
  std::vector<MatchArm*> arms_;
};

// Represents an attribute access expression; e.g. `a.x`.
//                                                   ^
//                       (this dot makes an attr) ---+
class Attr : public Expr {
 public:
  Attr(Module* owner, Span span, Expr* lhs, NameDef* attr)
      : Expr(owner, std::move(span)), lhs_(lhs), attr_(attr) {}

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleAttr(this);
  }
  void AcceptExpr(ExprVisitor* v) override { v->HandleAttr(this); }

  absl::string_view GetNodeTypeName() const override { return "Attr"; }
  std::string ToString() const override {
    return absl::StrFormat("%s.%s", lhs_->ToString(), attr_->ToString());
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {lhs_, attr_};
  }

  Expr* lhs() const { return lhs_; }

  // TODO(leary): 2020-12-02 This probably can just be a string, because the
  // attribute access is not really defining a name.
  NameDef* attr() const { return attr_; }

 private:
  Expr* lhs_;
  NameDef* attr_;
};

// Represents an invocation expression; e.g. `f(a, b, c)`
class Invocation : public Expr {
 public:
  Invocation(Module* owner, Span span, Expr* callee, std::vector<Expr*> args,
             std::vector<Expr*> parametrics = std::vector<Expr*>({}));

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleInvocation(this);
  }
  void AcceptExpr(ExprVisitor* v) override { v->HandleInvocation(this); }

  absl::string_view GetNodeTypeName() const override { return "Invocation"; }
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  std::string FormatArgs() const;

  std::string FormatParametrics() const;

  std::string ToString() const override {
    return absl::StrFormat("%s%s(%s)", callee_->ToString(), FormatParametrics(),
                           FormatArgs());
  };

  const absl::Span<Expr* const> args() const { return args_; }
  Expr* callee() const { return callee_; }
  const std::vector<std::pair<std::string, int64_t>> symbolic_bindings() const {
    return symbolic_bindings_;
  }

  // Any parametric expressions given in this invocation; e.g. in:
  //
  //    f<a, b, c>()
  //
  // The expressions a, b, c would be in this sequence.
  const std::vector<Expr*>& parametrics() const { return parametrics_; }

 private:
  Expr* callee_;
  std::vector<Expr*> args_;
  std::vector<Expr*> parametrics_;
  std::vector<std::pair<std::string, int64_t>> symbolic_bindings_;
};

// Represents a slice in the AST.
//
// For example, we can have `x[-4:-2]`, where x is of bit width N.
// The start and limit Exprs must be constexpr.
class Slice : public AstNode {
 public:
  Slice(Module* owner, Span span, Expr* start, Expr* limit)
      : AstNode(owner), span_(std::move(span)), start_(start), limit_(limit) {}

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleSlice(this);
  }

  absl::string_view GetNodeTypeName() const override { return "Slice"; }
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  std::string ToString() const override;
  absl::optional<Span> GetSpan() const override { return span_; }

  Expr* start() const { return start_; }
  Expr* limit() const { return limit_; }

 private:
  Span span_;
  Expr* start_;  // May be nullptr.
  Expr* limit_;  // May be nullptr.
};

// Helper struct for members items defined inside of enums.
struct EnumMember {
  NameDef* name_def;  // The name being bound in the enum.
  Expr* value;  // The expression on the right hand side of `ENUM_VAL = $expr`
};

// Represents a user-defined enum definition; e.g.
//
//  type MyTypeDef = u2;
//  enum Foo : MyTypeDef {
//    A = 0,
//    B = 1,
//    C = 2,
//  }
class EnumDef : public AstNode {
 public:
  EnumDef(Module* owner, Span span, NameDef* name_def, TypeAnnotation* type,
          std::vector<EnumMember> values, bool is_public);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleEnumDef(this);
  }

  absl::string_view GetNodeTypeName() const override { return "EnumDef"; }

  // Returns whether this enum definition has a member named "name".
  bool HasValue(absl::string_view name) const;

  // Returns the value bound to the given enum definition name.
  //
  // These must be constexprs, which will be computed at type checking time.
  absl::StatusOr<Expr*> GetValue(absl::string_view name) const;

  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    std::vector<AstNode*> results = {name_def_, type_annotation_};
    for (const EnumMember& item : values_) {
      results.push_back(item.name_def);
      results.push_back(ToAstNode(item.value));
    }
    return results;
  }

  const std::string& identifier() const { return name_def_->identifier(); }

  const Span& span() const { return span_; }
  absl::optional<Span> GetSpan() const override { return span_; }
  NameDef* name_def() const { return name_def_; }
  const std::vector<EnumMember>& values() const { return values_; }
  TypeAnnotation* type_annotation() const { return type_annotation_; }
  bool is_public() const { return is_public_; }

  // The signedness of the enum is populated in the type inference phase, it is
  // not known at parse time (hence absl::optional<bool> / set_signedness).

  void set_signedness(bool is_signed) { is_signed_ = is_signed; }
  absl::optional<bool> signedness() const { return is_signed_; }

 private:
  Span span_;
  NameDef* name_def_;
  TypeAnnotation* type_annotation_;
  std::vector<EnumMember> values_;
  absl::optional<bool> is_signed_;
  bool is_public_;
};

// Represents a struct definition.
class StructDef : public AstNode {
 public:
  StructDef(Module* owner, Span span, NameDef* name_def,
            std::vector<ParametricBinding*> parametric_bindings,
            std::vector<std::pair<NameDef*, TypeAnnotation*>> members,
            bool is_public);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleStructDef(this);
  }

  bool IsParametric() const { return !parametric_bindings_.empty(); }

  const std::string& identifier() const { return name_def_->identifier(); }

  absl::string_view GetNodeTypeName() const override { return "Struct"; }
  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override;

  NameDef* name_def() const { return name_def_; }
  const std::vector<ParametricBinding*>& parametric_bindings() const {
    return parametric_bindings_;
  }
  const std::vector<std::pair<NameDef*, TypeAnnotation*>> members() const {
    return members_;
  }
  bool is_public() const { return public_; }
  const Span& span() const { return span_; }
  absl::optional<Span> GetSpan() const override { return span_; }

  const std::string& GetMemberName(int64_t i) const {
    return members_[i].first->identifier();
  }
  std::vector<std::string> GetMemberNames() const;

  // Returns the index at which the member name is "name".
  absl::optional<int64_t> GetMemberIndex(absl::string_view name) const;

 private:
  Span span_;
  NameDef* name_def_;
  std::vector<ParametricBinding*> parametric_bindings_;
  std::vector<std::pair<NameDef*, TypeAnnotation*>> members_;
  bool public_;
};

// Variant that either points at a struct definition or a module reference
// (which should be backed by a struct definition -- that property will be
// checked by the typechecker).
using StructRef = absl::variant<StructDef*, ColonRef*>;

std::string StructRefToText(const StructRef& struct_ref);

inline TypeDefinition ToTypeDefinition(const StructRef& struct_ref) {
  return ToTypeDefinition(ToAstNode(struct_ref)).value();
}

// Represents instantiation of a struct via member expressions.
//
// TODO(leary): 2020-09-08 Break out a StructMember type in lieu of the pair.
class StructInstance : public Expr {
 public:
  StructInstance(Module* owner, Span span, StructRef struct_ref,
                 std::vector<std::pair<std::string, Expr*>> members);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleStructInstance(this);
  }
  void AcceptExpr(ExprVisitor* v) override { v->HandleStructInstance(this); }

  absl::string_view GetNodeTypeName() const override {
    return "StructInstance";
  }
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  std::string ToString() const override;

  absl::Span<const std::pair<std::string, Expr*>> GetUnorderedMembers() const {
    return members_;
  }

  // Returns the members for the struct instance, ordered by the (resolved)
  // struct definition "struct_def".
  std::vector<std::pair<std::string, Expr*>> GetOrderedMembers(
      StructDef* struct_def) const;

  // Returns the expression associated with the member named "name", or a
  // NotFound error status if none exists.
  absl::StatusOr<Expr*> GetExpr(absl::string_view name) const;

  // An AST node that refers to the struct being instantiated.
  StructRef struct_def() const { return struct_ref_; }

 private:
  AstNode* GetStructNode() const { return ToAstNode(struct_ref_); }

  StructRef struct_ref_;
  std::vector<std::pair<std::string, Expr*>> members_;
};

// Represents a struct instantiation as a "delta" from a 'splatted' original;
// e.g.
//    Point { y: new_y, ..orig_p }
class SplatStructInstance : public Expr {
 public:
  SplatStructInstance(Module* owner, Span span, StructRef struct_ref,
                      std::vector<std::pair<std::string, Expr*>> members,
                      Expr* splatted);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleSplatStructInstance(this);
  }
  void AcceptExpr(ExprVisitor* v) override {
    v->HandleSplatStructInstance(this);
  }

  absl::string_view GetNodeTypeName() const override {
    return "SplatStructInstance";
  }
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  std::string ToString() const override;

  Expr* splatted() const { return splatted_; }
  StructRef struct_ref() const { return struct_ref_; }
  const std::vector<std::pair<std::string, Expr*>>& members() const {
    return members_;
  }

 private:
  // The struct being instantiated.
  StructRef struct_ref_;

  // Sequence of members being changed from the splatted original; e.g. in the
  // above example this is [('y', new_y)].
  std::vector<std::pair<std::string, Expr*>> members_;

  // Expression that's used as the original struct instance (that we're
  // instantiating a delta from); e.g. orig_p in the example above.
  Expr* splatted_;
};

// Represents a slice in the AST; e.g. `-4+:u2`
class WidthSlice : public AstNode {
 public:
  WidthSlice(Module* owner, Span span, Expr* start, TypeAnnotation* width)
      : AstNode(owner), span_(std::move(span)), start_(start), width_(width) {}

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleWidthSlice(this);
  }
  absl::string_view GetNodeTypeName() const override { return "WidthSlice"; }
  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {start_, width_};
  }

  Expr* start() const { return start_; }
  TypeAnnotation* width() const { return width_; }
  absl::optional<Span> GetSpan() const override { return span_; }

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
//
// * `lhs()` is the subject being indexed
// * `rhs()` is the index specifier, can be either an:
//   * expression (e.g. `i` in the `a[i]` example above)
//   * slice (from compile-time-constant index to compile-time-constant index)
//   * width slice (from start index a compile-time-constant number of bits)
class Index : public Expr {
 public:
  Index(Module* owner, Span span, Expr* lhs, IndexRhs rhs)
      : Expr(owner, std::move(span)), lhs_(lhs), rhs_(rhs) {}

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleIndex(this);
  }
  void AcceptExpr(ExprVisitor* v) override { v->HandleIndex(this); }

  absl::string_view GetNodeTypeName() const override { return "Index"; }
  std::string ToString() const override;

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
  Proc(Module* owner, Span span, NameDef* name_def,
       std::vector<Param*> proc_params, std::vector<Param*> iter_params,
       Expr* iter_body, bool is_public);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleProc(this);
  }
  absl::string_view GetNodeTypeName() const override { return "Proc"; }
  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override;

  NameDef* name_def() const { return name_def_; }
  bool is_public() const { return is_public_; }
  absl::optional<Span> GetSpan() const override { return span_; }

 private:
  Span span_;
  NameDef* name_def_;
  std::vector<Param*> proc_params_;
  std::vector<Param*> iter_params_;
  Expr* iter_body_;
  bool is_public_;
};

// Represents a unit test construct.
//
// These are specified with an annotation as follows:
//
// ```dslx
// #![test]
// fn test_foo() { ... }
// ```
class TestFunction : public AstNode {
 public:
  explicit TestFunction(Module* owner, Function* fn)
      : AstNode(owner), name_def_(fn->name_def()), body_(fn->body()), fn_(fn) {}

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleTestFunction(this);
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {name_def_, body_};
  }

  absl::string_view GetNodeTypeName() const override { return "TestFunction"; }
  std::string ToString() const override {
    return absl::StrFormat("#![test]\n%s", fn_->ToString());
  }

  Function* fn() const { return fn_; }
  absl::optional<Span> GetSpan() const override { return fn_->span(); }

  NameDef* name_def() const { return name_def_; }
  const std::string& identifier() const { return name_def_->identifier(); }
  Expr* body() const { return body_; }

 private:
  NameDef* name_def_;
  Expr* body_;
  Function* fn_;
};

// Represents a function to be quick-check'd.
class QuickCheck : public AstNode {
 public:
  static constexpr int64_t kDefaultTestCount = 1000;

  QuickCheck(Module* owner, Span span, Function* f,
             absl::optional<int64_t> test_count = absl::nullopt);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleQuickCheck(this);
  }

  absl::string_view GetNodeTypeName() const override { return "QuickCheck"; }
  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {f_};
  }

  const std::string& identifier() const { return f_->identifier(); }

  Function* f() const { return f_; }
  int64_t test_count() const {
    return test_count_ ? *test_count_ : kDefaultTestCount;
  }
  absl::optional<Span> GetSpan() const override { return f_->span(); }

 private:
  Span span_;
  Function* f_;
  absl::optional<int64_t> test_count_;
};

// Represents an XLS tuple expression.
class XlsTuple : public Expr {
 public:
  XlsTuple(Module* owner, Span span, std::vector<Expr*> members)
      : Expr(owner, std::move(span)), members_(members) {}

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleXlsTuple(this);
  }
  void AcceptExpr(ExprVisitor* v) override { v->HandleXlsTuple(this); }

  absl::string_view GetNodeTypeName() const override { return "XlsTuple"; }
  absl::Span<Expr* const> members() const { return members_; }
  int64_t size() const { return members_.size(); }
  bool empty() const { return members_.empty(); }

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
  For(Module* owner, Span span, NameDefTree* names, TypeAnnotation* type,
      Expr* iterable, Expr* body, Expr* init);

  absl::Status Accept(AstNodeVisitor* v) override { return v->HandleFor(this); }
  void AcceptExpr(ExprVisitor* v) override { v->HandleFor(this); }

  absl::string_view GetNodeTypeName() const override { return "For"; }
  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override;

  // Names bound in the body of the loop.
  NameDefTree* names() const { return names_; }

  // Annotation corresponding to "names".
  TypeAnnotation* type_annotation() const { return type_annotation_; }

  // Expression for "thing to iterate over".
  Expr* iterable() const { return iterable_; }

  // Expression for the loop body.
  Expr* body() const { return body_; }

  // Initial expression for the loop (start values expr).
  Expr* init() const { return init_; }

 private:
  NameDefTree* names_;
  TypeAnnotation* type_annotation_;
  Expr* iterable_;
  Expr* body_;
  Expr* init_;
};

// Represents a while-loop expression.
class While : public Expr {
 public:
  While(Module* owner, Span span) : Expr(owner, std::move(span)) {}

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleWhile(this);
  }
  void AcceptExpr(ExprVisitor* v) override { v->HandleWhile(this); }

  absl::string_view GetNodeTypeName() const override { return "While"; }
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
  Cast(Module* owner, Span span, Expr* expr, TypeAnnotation* type_annotation)
      : Expr(owner, std::move(span)),
        expr_(expr),
        type_annotation_(type_annotation) {}

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleCast(this);
  }
  void AcceptExpr(ExprVisitor* v) override { v->HandleCast(this); }

  absl::string_view GetNodeTypeName() const override { return "Cast"; }
  std::string ToString() const override {
    return absl::StrFormat("((%s) as %s)", expr_->ToString(),
                           type_annotation_->ToString());
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    if (want_types) {
      return {expr_, type_annotation_};
    }
    return {expr_};
  }

  Expr* expr() const { return expr_; }
  TypeAnnotation* type_annotation() const { return type_annotation_; }

 private:
  Expr* expr_;
  TypeAnnotation* type_annotation_;
};

// Represents `next` keyword, refers to the implicit loop-carry call in `Proc`.
class Next : public Expr {
 public:
  using Expr::Expr;

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleNext(this);
  }
  void AcceptExpr(ExprVisitor* v) override { v->HandleNext(this); }

  absl::string_view GetNodeTypeName() const override { return "Next"; }
  std::string ToString() const override { return "next"; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {};
  }
};

// Represents `carry` keyword, refers to the implicit loop-carry data in
// `While`.
class Carry : public Expr {
 public:
  Carry(Module* owner, Span span, While* loop)
      : Expr(owner, std::move(span)), loop_(loop) {}

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleCarry(this);
  }
  void AcceptExpr(ExprVisitor* v) override { v->HandleCarry(this); }

  absl::string_view GetNodeTypeName() const override { return "Carry"; }
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
//
//  is_public: Indicates whether the constant had a public annotation
//    (applicable to module level constant definitions only)
//  is_local: Indicates whether the constant was defined in a "local" function
//    scope (as opposed to module-level scope).
class ConstantDef : public AstNode {
 public:
  ConstantDef(Module* owner, Span span, NameDef* name_def, Expr* value,
              bool is_public, bool is_local);

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleConstantDef(this);
  }

  absl::string_view GetNodeTypeName() const override { return "ConstantDef"; }
  std::string ToString() const override;
  std::string ToReprString() const;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {name_def_, value_};
  }

  const std::string& identifier() const { return name_def_->identifier(); }
  NameDef* name_def() const { return name_def_; }
  Expr* value() const { return value_; }
  const Span& span() const { return span_; }
  absl::optional<Span> GetSpan() const override { return span_; }
  bool is_public() const { return is_public_; }
  bool is_local() const { return is_local_; }

 private:
  Span span_;
  NameDef* name_def_;
  Expr* value_;
  bool is_public_;
  bool is_local_;
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
  using Leaf =
      absl::variant<NameDef*, NameRef*, WildcardPattern*, Number*, ColonRef*>;

  NameDefTree(Module* owner, Span span, absl::variant<Nodes, Leaf> tree)
      : AstNode(owner), span_(std::move(span)), tree_(tree) {}

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleNameDefTree(this);
  }

  absl::string_view GetNodeTypeName() const override { return "NameDefTree"; }
  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override;

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
  std::vector<absl::variant<Leaf, NameDefTree*>> Flatten1();

  // Flattens the (recursive) NameDefTree into a list of leaves.
  std::vector<Leaf> Flatten() const;

  // Filters the values from Flatten() to just NameDef leaves.
  std::vector<NameDef*> GetNameDefs() const;

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

  // Performs a preorder traversal under this node in the NameDefTree.
  //
  // Args:
  //  f: Callback invoked as `f(NameDefTree*, level, branchno)`.
  //  level: Current level of the node.
  absl::Status DoPreorder(
      const std::function<absl::Status(NameDefTree*, int64_t, int64_t)>& f,
      int64_t level = 1) {
    if (is_leaf()) {
      return absl::OkStatus();
    }
    for (int64_t i = 0; i < nodes().size(); ++i) {
      NameDefTree* node = nodes()[i];
      XLS_RETURN_IF_ERROR(f(node, level, i));
      XLS_RETURN_IF_ERROR(node->DoPreorder(f, level + 1));
    }
    return absl::OkStatus();
  }

  const absl::variant<Nodes, Leaf>& tree() const { return tree_; }
  const Span& span() const { return span_; }
  absl::optional<Span> GetSpan() const override { return span_; }

 private:
  Span span_;
  absl::variant<Nodes, Leaf> tree_;
};

// Represents a let-binding expression.
class Let : public Expr {
 public:
  Let(Module* owner, Span span, NameDefTree* name_def_tree,
      TypeAnnotation* type, Expr* rhs, Expr* body, ConstantDef* const_def);

  void AcceptExpr(ExprVisitor* v) override { v->HandleLet(this); }
  absl::Status Accept(AstNodeVisitor* v) override { return v->HandleLet(this); }

  absl::string_view GetNodeTypeName() const override { return "Let"; }
  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override;

  NameDefTree* name_def_tree() const { return name_def_tree_; }
  TypeAnnotation* type_annotation() const { return type_annotation_; }
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
  TypeAnnotation* type_annotation_;

  // Right hand side of the let; e.g. in `let a = b; c` this is `b`.
  Expr* rhs_;

  // The body of the let: it has the expression to be evaluated with the let
  // bindings; e.g. in `let a = b; c` this is `c`.
  Expr* body_;

  // Whether or not this is a constant binding; constant bindings cannot be
  // shadowed. May be null.
  ConstantDef* constant_def_;
};

using ModuleMember =
    absl::variant<Function*, TestFunction*, QuickCheck*, TypeDef*, StructDef*,
                  ConstantDef*, EnumDef*, Import*>;

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
  explicit Module(std::string name) : AstNode(this), name_(std::move(name)) {
    XLS_VLOG(3) << "Created module \"" << name_ << "\" @ " << this;
  }

  ~Module() {
    XLS_VLOG(3) << "Destroying module \"" << name_ << "\" @ " << this;
  }

  absl::Status Accept(AstNodeVisitor* v) override {
    return v->HandleModule(this);
  }
  absl::optional<Span> GetSpan() const override { return absl::nullopt; }

  absl::string_view GetNodeTypeName() const override { return "Module"; }
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  std::string ToString() const override {
    return absl::StrJoin(top_, "\n",
                         [](std::string* out, const ModuleMember& member) {
                           absl::StrAppend(out, ToAstNode(member)->ToString());
                         });
  }
  // As in Python's "repr", a short code-like unique string representation.
  std::string ToRepr() const {
    return absl::StrFormat("Module(name='%s', id=%p)", name(), this);
  }

  template <typename T, typename... Args>
  T* Make(Args&&... args) {
    std::unique_ptr<T> node =
        absl::make_unique<T>(this, std::forward<Args>(args)...);
    T* ptr = node.get();
    nodes_.push_back(std::move(node));
    return ptr;
  }

  void AddTop(ModuleMember member) { top_.push_back(member); }

  // Gets a function in this module with the given "target_name", or returns a
  // NotFoundError.
  absl::optional<Function*> GetFunction(absl::string_view target_name);
  absl::StatusOr<Function*> GetFunctionOrError(absl::string_view target_name);

  // Gets a test construct in this module with the given "target_name", or
  // returns a NotFoundError.
  absl::StatusOr<TestFunction*> GetTest(absl::string_view target_name);

  absl::Span<ModuleMember const> top() const { return top_; }
  std::vector<ModuleMember>* mutable_top() { return &top_; }

  // Finds the first top-level member in top() with the given "target" name as
  // an identifier.
  absl::optional<ModuleMember*> FindMemberWithName(absl::string_view target);

  // Obtains all the type definition nodes in the module:
  //    TypeDef, Struct, Enum
  absl::flat_hash_map<std::string, TypeDefinition> GetTypeDefinitionByName()
      const;

  // Obtains all the type definition nodes in the module in module-member order.
  std::vector<TypeDefinition> GetTypeDefinitions() const;

  absl::StatusOr<TypeDefinition> GetTypeDefinition(
      absl::string_view name) const;

  // Retrieves a constant node from this module with the target name as its
  // identifier, or a NotFound error if none can be found.
  absl::StatusOr<ConstantDef*> GetConstantDef(absl::string_view target);

  absl::flat_hash_map<std::string, ConstantDef*> GetConstantByName() const {
    return GetTopWithTByName<ConstantDef>();
  }
  absl::flat_hash_map<std::string, Import*> GetImportByName() const {
    return GetTopWithTByName<Import>();
  }

  absl::flat_hash_map<std::string, Function*> GetFunctionByName() const {
    return GetTopWithTByName<Function>();
  }

  std::vector<TypeDef*> GetTypeDefs() const { return GetTopWithT<TypeDef>(); }
  std::vector<QuickCheck*> GetQuickChecks() const {
    return GetTopWithT<QuickCheck>();
  }
  std::vector<StructDef*> GetStructDefs() const {
    return GetTopWithT<StructDef>();
  }
  std::vector<Function*> GetFunctions() const {
    return GetTopWithT<Function>();
  }
  std::vector<TestFunction*> GetTests() const {
    return GetTopWithT<TestFunction>();
  }
  std::vector<ConstantDef*> GetConstantDefs() const {
    return GetTopWithT<ConstantDef>();
  }

  // Returns the identifiers for all functions within this module (in the order
  // in which they are defined).
  std::vector<std::string> GetFunctionNames() const;

  // Returns the identifiers for all tests within this module (in the order in
  // which they are defined).
  std::vector<std::string> GetTestNames() const;

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
