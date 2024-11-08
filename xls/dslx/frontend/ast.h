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

#ifndef XLS_DSLX_FRONTEND_AST_H_
#define XLS_DSLX_FRONTEND_AST_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/frontend/ast_node.h"  // IWYU pragma: export
#include "xls/dslx/frontend/pos.h"
#include "xls/ir/bits.h"
#include "xls/ir/foreign_function_data.pb.h"
#include "xls/ir/format_strings.h"

// Higher-order macro for all the Expr node leaf types (non-abstract).
#define XLS_DSLX_EXPR_NODE_EACH(X) \
  X(AllOnesMacro)                  \
  X(Array)                         \
  X(Attr)                          \
  X(Binop)                         \
  X(StatementBlock)                \
  X(Cast)                          \
  X(ChannelDecl)                   \
  X(ColonRef)                      \
  X(ConstantArray)                 \
  X(ConstRef)                      \
  X(For)                           \
  X(FormatMacro)                   \
  X(FunctionRef)                   \
  X(Index)                         \
  X(Invocation)                    \
  X(Let)                           \
  X(Match)                         \
  X(NameRef)                       \
  X(Number)                        \
  X(Range)                         \
  X(Spawn)                         \
  X(SplatStructInstance)           \
  X(String)                        \
  X(StructInstance)                \
  X(Conditional)                   \
  X(ConstAssert)                   \
  X(TupleIndex)                    \
  X(Unop)                          \
  X(UnrollFor)                     \
  X(XlsTuple)                      \
  X(ZeroMacro)

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
  X(Impl)                         \
  X(MatchArm)                     \
  X(Module)                       \
  X(NameDef)                      \
  X(NameDefTree)                  \
  X(Param)                        \
  X(ParametricBinding)            \
  X(Proc)                         \
  X(ProcDef)                      \
  X(ProcMember)                   \
  X(QuickCheck)                   \
  X(RestOfTuple)                  \
  X(Slice)                        \
  X(Statement)                    \
  X(StructDef)                    \
  X(TestFunction)                 \
  X(TestProc)                     \
  X(TypeAlias)                    \
  X(TypeRef)                      \
  X(VerbatimNode)                 \
  X(WidthSlice)                   \
  X(WildcardPattern)              \
  /* type annotations */          \
  X(ArrayTypeAnnotation)          \
  X(BuiltinTypeAnnotation)        \
  X(ChannelTypeAnnotation)        \
  X(TupleTypeAnnotation)          \
  X(TypeRefTypeAnnotation)        \
  XLS_DSLX_EXPR_NODE_EACH(X)

namespace xls::dslx {

inline constexpr int64_t kRustSpacesPerIndent = 4;

// Forward decls of all leaf types.
#define FORWARD_DECL(__type) class __type;
XLS_DSLX_AST_NODE_EACH(FORWARD_DECL)
#undef FORWARD_DECL

// Helper type (abstract base) for double dispatch on AST nodes.
class AstNodeVisitor {
 public:
  virtual ~AstNodeVisitor() = default;

#define DECLARE_HANDLER(__type) \
  virtual absl::Status Handle##__type(const __type* n) = 0;
  XLS_DSLX_AST_NODE_EACH(DECLARE_HANDLER)
#undef DECLARE_HANDLER
};

// Subtype of abstract AstNodeVisitor that returns ok status (does nothing) for
// every node type.
//
// Users can override the default behavior by overriding the DefaultHandler()
// method.
class AstNodeVisitorWithDefault : public AstNodeVisitor {
 public:
  ~AstNodeVisitorWithDefault() override = default;

  virtual absl::Status DefaultHandler() { return absl::OkStatus(); }

#define DECLARE_HANDLER(__type)                           \
  absl::Status Handle##__type(const __type* n) override { \
    return DefaultHandler();                              \
  }
  XLS_DSLX_AST_NODE_EACH(DECLARE_HANDLER)
#undef DECLARE_HANDLER
};

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

// Forward decl of non-leaf type.
class Expr;
class TypeAnnotation;

using ExprOrType = std::variant<Expr*, TypeAnnotation*>;
Span ExprOrTypeSpan(const ExprOrType& expr_or_type);

// Name definitions can be either built in (BuiltinNameDef, in which case they
// have no effective position) or defined in the user AST (NameDef).
using AnyNameDef = std::variant<const NameDef*, BuiltinNameDef*>;

// Holds a mapping {identifier: NameRefs} -- this is used for accumulating free
// variable references (the NameRefs) in the source program; see
// GetFreeVariables().
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
  void Add(std::string identifier, const NameRef* name_ref);

  // Returns the identifiers in this free variable set.
  absl::flat_hash_set<std::string> Keys() const;

  // Underlying data for this free variables set.
  const absl::flat_hash_map<std::string, std::vector<const NameRef*>>& values()
      const {
    return values_;
  }

  // Returns all of the free variable NameRefs that are references to constants
  // (as the ConstRef subtype of NameRef).
  std::vector<const ConstRef*> GetConstRefs();

  // Returns the number of unique free variables (note: not the number of
  // references, but the number of free variables).
  int64_t GetFreeVariableCount() const { return values_.size(); }

  // Returns the span of the first `NameRef` that is referring to `identifier`
  // in this free variables set.
  const Span& GetFirstNameRefSpan(std::string_view identifier) const;

 private:
  absl::flat_hash_map<std::string, std::vector<const NameRef*>> values_;
};

// Generalized form of `GetFreeVariablesByPos()` below -- takes a lambda that
// helps us determine if a `NameRef` present in the `node` should be considered
// free.
FreeVariables GetFreeVariablesByLambda(
    const AstNode* node,
    const std::function<bool(const NameRef&)>& consider_free = nullptr);

// Retrieves all the free variables (references to names that are defined
// prior to start_pos) that are transitively in this AST subtree.
//
// For example, if given the AST node for this function:
//
//    const FOO = u32:42;
//    fn main(x: u32) { FOO+x }
//
// And *using the starting point of the function* as the `start_pos`, the `FOO`
// will be flagged as a free variable and returned.
//
// Note: the start_pos given is a way to approximate "free variable with
// respect to this AST construct". i.e. all the references with defs that are
// defined before this start_pos point are considered free. This gives an easy
// way to say "everything defined inside the body we don't need to worry about
// -- only tell me about references to things before this lexical position in
// the file" -- "lexical position in the file" is an approximation for
// "everything defined outside of (before) this AST construct".
FreeVariables GetFreeVariablesByPos(const AstNode* node,
                                    const Pos* start_pos = nullptr);

// Analogous to ToAstNode(), but for Expr base.
template <typename... Types>
inline Expr* ToExprNode(const std::variant<Types...>& v) {
  return absl::ConvertVariantTo<Expr*>(v);
}

ExprOrType ToExprOrType(AstNode* n);

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
  TypeAnnotation(Module* owner, Span span) : AstNode(owner), span_(span) {}

  ~TypeAnnotation() override;

  AstNodeKind kind() const override { return AstNodeKind::kTypeAnnotation; }
  const Span& span() const { return span_; }
  std::optional<Span> GetSpan() const override { return span_; }

 private:
  Span span_;
};

#include "xls/dslx/frontend/ast_builtin_types.inc"

// Enumeration of types that are built-in keywords; e.g. `u32`, `bool`, etc.
enum class BuiltinType : uint8_t {
#define FIRST_COMMA(A, ...) A,
  XLS_DSLX_BUILTIN_TYPE_EACH(FIRST_COMMA)
#undef FIRST_COMMA
};

// All builtin types up to this limit have a concrete width and sign -- above
// this point are things like "bits", "uN", "sN" which need a corresponding
// array dimension to have a known bit count.
inline constexpr int64_t kConcreteBuiltinTypeLimit =
    static_cast<int64_t>(BuiltinType::kS64) + 1;

std::string BuiltinTypeToString(BuiltinType t);
absl::StatusOr<BuiltinType> BuiltinTypeFromString(std::string_view s);

absl::StatusOr<BuiltinType> GetBuiltinType(bool is_signed, int64_t width);
absl::StatusOr<bool> GetBuiltinTypeSignedness(BuiltinType type);
int64_t GetBuiltinTypeBitCount(BuiltinType type);

// Represents a built-in type annotation; e.g. `u32`, `bits`, etc.
class BuiltinTypeAnnotation : public TypeAnnotation {
 public:
  BuiltinTypeAnnotation(Module* owner, Span span, BuiltinType builtin_type,
                        BuiltinNameDef* builtin_name_def);

  ~BuiltinTypeAnnotation() override;

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleBuiltinTypeAnnotation(this);
  }

  std::string_view GetNodeTypeName() const override {
    return "BuiltinTypeAnnotation";
  }
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  std::string ToString() const override {
    return BuiltinTypeToString(builtin_type_);
  }

  // Note that types like `xN`/`uN`/`sN`/`bits` have a bit count of zero.
  int64_t GetBitCount() const;

  // Returns true if signed, false if unsigned.
  //
  // Note that a type like `xN` on its own does not have a signedness.
  absl::StatusOr<bool> GetSignedness() const;

  BuiltinType builtin_type() const { return builtin_type_; }

  BuiltinNameDef* builtin_name_def() const { return builtin_name_def_; }

 private:
  BuiltinType builtin_type_;
  BuiltinNameDef* builtin_name_def_;
};

// Represents a tuple type annotation; e.g. `(u32, s42)`.
class TupleTypeAnnotation : public TypeAnnotation {
 public:
  TupleTypeAnnotation(Module* owner, Span span,
                      std::vector<TypeAnnotation*> members);

  ~TupleTypeAnnotation() override;

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleTupleTypeAnnotation(this);
  }

  std::string_view GetNodeTypeName() const override {
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
                        std::vector<ExprOrType> parametrics);

  ~TypeRefTypeAnnotation() override;

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleTypeRefTypeAnnotation(this);
  }

  TypeRef* type_ref() const { return type_ref_; }

  std::string ToString() const override;

  std::string_view GetNodeTypeName() const override {
    return "TypeRefTypeAnnotation";
  }
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  const std::vector<ExprOrType>& parametrics() const { return parametrics_; }

 private:
  TypeRef* type_ref_;
  std::vector<ExprOrType> parametrics_;
};

// Represents an array type annotation; e.g. `u32[5]`.
class ArrayTypeAnnotation : public TypeAnnotation {
 public:
  ArrayTypeAnnotation(Module* owner, Span span, TypeAnnotation* element_type,
                      Expr* dim);

  ~ArrayTypeAnnotation() override;

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleArrayTypeAnnotation(this);
  }

  std::string_view GetNodeTypeName() const override {
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

  ~BuiltinNameDef() override;

  AstNodeKind kind() const override { return AstNodeKind::kBuiltinNameDef; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleBuiltinNameDef(this);
  }
  std::optional<Span> GetSpan() const override { return std::nullopt; }

  std::string_view GetNodeTypeName() const override { return "BuiltinNameDef"; }
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

  ~WildcardPattern() override;

  AstNodeKind kind() const override { return AstNodeKind::kWildcardPattern; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleWildcardPattern(this);
  }

  std::string_view GetNodeTypeName() const override {
    return "WildcardPattern";
  }
  std::string ToString() const override { return "_"; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {};
  }

  const Span& span() const { return span_; }
  std::optional<Span> GetSpan() const override { return span_; }

 private:
  Span span_;
};

// Represents the definition of a name (identifier).
class NameDef : public AstNode {
 public:
  NameDef(Module* owner, Span span, std::string identifier, AstNode* definer);

  ~NameDef() override;

  AstNodeKind kind() const override { return AstNodeKind::kNameDef; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleNameDef(this);
  }

  std::string_view GetNodeTypeName() const override { return "NameDef"; }
  const Span& span() const { return span_; }
  std::optional<Span> GetSpan() const override { return span_; }
  const std::string& identifier() const { return identifier_; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {};
  }
  std::string ToString() const override { return identifier_; }

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

#define DECLARE_HANDLER(__type) \
  virtual absl::Status Handle##__type(const __type* expr) = 0;
  XLS_DSLX_EXPR_NODE_EACH(DECLARE_HANDLER)
#undef DECLARE_HANDLER
};

// Subtype of abstract ExprVisitor that returns ok status (does nothing) for
// every node type.
class ExprVisitorWithDefault : public ExprVisitor {
 public:
  ~ExprVisitorWithDefault() override = default;

#define DECLARE_HANDLER(__type)                              \
  absl::Status Handle##__type(const __type* expr) override { \
    return absl::OkStatus();                                 \
  }
  XLS_DSLX_EXPR_NODE_EACH(DECLARE_HANDLER)
#undef DECLARE_HANDLER
};

// A la
// https://doc.rust-lang.org/reference/expressions.html#expression-precedence
//
// This is organized from strong to weak, so any time an expression has >
// precedence than its child nodes, we need to use parentheses.
enum class Precedence : uint8_t {
  kStrongest = 0,

  kPaths = 1,
  kMethodCall = 2,
  kFieldExpression = 3,
  kFunctionCallOrArrayIndex = 4,

  // Note: the DSL doesn't have a question mark operator, it's just here for
  // completeness with respect to the Rust precedence table.
  kQuestionMark = 5,

  kUnaryOp = 6,
  kAs = 7,
  kStrongArithmetic = 8,
  kWeakArithmetic = 9,
  kShift = 10,

  // Note: this is not present in Rust, but it seems the right level relative to
  // other constructs.
  kConcat = 11,

  kBitwiseAnd = 12,
  kBitwiseXor = 13,
  kBitwiseOr = 14,
  kComparison = 15,
  kLogicalAnd = 16,
  kLogicalOr = 17,
  kRange = 18,
  kEquals = 19,
  kReturn = 20,
  kWeakest = 21,
};

// Returns whether the given precedence level is for an infix operator -- if
// not (e.g the operator is in suffix form) we may not need to emit parentheses
// with respect to some outer precedence level that binds more weakly. E.g.
// consider the suffix operators:
//
//    f().g[h] + ...
//
// The plus is weaker than the operator precedences on the left hand side, but
// we still don't require parentheses, because those operators are suffix
// operators.
inline bool IsInfix(Precedence p) {
  switch (p) {
    case Precedence::kPaths:
    case Precedence::kMethodCall:
    case Precedence::kFieldExpression:
    case Precedence::kFunctionCallOrArrayIndex:
      return false;
    default:
      return true;
  }
}

std::string_view PrecedenceToString(Precedence p);

inline std::ostream& operator<<(std::ostream& os, Precedence p) {
  os << PrecedenceToString(p);
  return os;
}

// Returns whether x is weaker precedence (binds more loosely) than y.
//
// For example:
//
//    WeakerThan(kWeakArithmetic, kStrongArithmetic) == true
//
// Which would indicate that the weak arithmetic needs parens if it's inside the
// strong arithmetic operator.
inline bool WeakerThan(Precedence x, Precedence y) {
  return static_cast<int>(x) > static_cast<int>(y);
}

// Abstract base class for AST node that can appear in expression positions
// (i.e. can produce runtime values).
class Expr : public AstNode {
 public:
  Expr(Module* owner, Span span) : AstNode(owner), span_(span) {}

  ~Expr() override;

  const Span& span() const { return span_; }
  std::optional<Span> GetSpan() const override { return span_; }

  // Returns whether an expression contains blocks with a constant amount of
  // leading characters; e.g. a block or an array without a type.
  virtual bool IsBlockedExprNoLeader() const { return false; }

  virtual bool IsBlockedExprWithLeader() const { return false; }

  bool IsBlockedExprAnyLeader() const {
    return IsBlockedExprNoLeader() || IsBlockedExprWithLeader();
  }

  virtual absl::Status AcceptExpr(ExprVisitor* v) const = 0;

  // Implementation note: subtypes of Expr override `ToStringInternal()`
  // instead, so that we can consolidate parenthesization of the expression at
  // this level.
  std::string ToString() const final;

  // Returns the precedence of this expression, as observed by consumers; e.g.
  // if the expression is marked as being enclosed in parentheses, it returns
  // Precedence::kStrongest.
  //
  // To get the precedence of this operator directly (regardless of
  // parenthesization) call GetPrecedenceWithoutParens().
  Precedence GetPrecedence() const {
    if (in_parens()) {
      return Precedence::kStrongest;
    }
    return GetPrecedenceWithoutParens();
  }

  // Returns the precedence of this Expr node without considering whether it is
  // wrapped in parentheses. This is useful e.g. when determining whether
  // operands of the operator require parenthesization, see also WeakerThan().
  virtual Precedence GetPrecedenceWithoutParens() const = 0;

  // Note: this is one of the rare instances where we're ok with updating the
  // AST node after it has been formed just to note that it is enclosed in
  // parentheses. For example, we want to flag:
  //    x == y == z
  // But not:
  //    (x == y) == z
  bool in_parens() const { return in_parens_; }
  void set_in_parens(bool enabled) { in_parens_ = enabled; }

 protected:
  virtual std::string ToStringInternal() const = 0;

  void UpdateSpan(Span new_span) { span_ = new_span; }

 private:
  Span span_;
  bool in_parens_ = false;
};

// ChannelTypeAnnotation has to be placed after the definition of Expr, so it
// can convert `dims_` to a set of AstNodes.
class ChannelTypeAnnotation : public TypeAnnotation {
 public:
  // If this is a scalar channel, then `dims` will be nullopt.
  ChannelTypeAnnotation(Module* owner, Span span, ChannelDirection direction,
                        TypeAnnotation* payload,
                        std::optional<std::vector<Expr*>> dims);

  ~ChannelTypeAnnotation() override;

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleChannelTypeAnnotation(this);
  }

  std::string_view GetNodeTypeName() const override {
    return "ChannelTypeAnnotation";
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    std::vector<AstNode*> children{payload_};
    if (dims_.has_value()) {
      for (Expr* dim : dims_.value()) {
        children.push_back(dim);
      }
    }
    return children;
  }

  std::string ToString() const override;

  ChannelDirection direction() const { return direction_; }
  TypeAnnotation* payload() const { return payload_; }

  // A ChannelTypeAnnotation needs to keep its own dims (rather than being
  // enclosed in an ArrayTypeAnnotation simply because it prints itself in a
  // different manner than an array does - we want `chan<u32>[32] in` rather
  // than `chan in u32[32]` for a 32-channel declaration. The former declares 32
  // channels, each of which transmits a u32, whereas the latter declares a
  // single channel that transmits a 32-element array of u32s.
  const std::optional<std::vector<Expr*>>& dims() const { return dims_; }

 private:
  ChannelDirection direction_;
  TypeAnnotation* payload_;
  std::optional<std::vector<Expr*>> dims_;
};

// Represents an AST node that may-or-may-not be an expression. For example, in
// a function body we may have statements that set up local type aliases, or be
// "side effecting" operations:
//
//    fn f(x: u32) -> () {
//      type MyU32 = u32;
//      assert_eq(x, MyU32:42);
//    }
//
// Both of those lines are statements.
class Statement final : public AstNode {
 public:
  using Wrapped =
      std::variant<Expr*, TypeAlias*, Let*, ConstAssert*, VerbatimNode*>;

  static absl::StatusOr<Wrapped> NodeToWrapped(AstNode* n);

  Statement(Module* owner, Wrapped wrapped);

  AstNodeKind kind() const override { return AstNodeKind::kStatement; }
  std::string_view GetNodeTypeName() const override { return "Statement"; }
  std::string ToString() const override {
    return ToAstNode(wrapped_)->ToString();
  }
  std::optional<Span> GetSpan() const override;
  std::vector<AstNode*> GetChildren(bool want_types) const override;
  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleStatement(this);
  }

  const Wrapped& wrapped() const { return wrapped_; }

 private:
  Wrapped wrapped_;
};

// Represents a block of statements, which is itself an expression in the
// language (i.e. it results in its last expression-statement, if there is no
// trailing semicolon).
//
// This is also sometimes called a block expression, e.g.,
//
// ```dslx-snippet
// let i = {
//     let x = f();
//     x + u32:42
// };
// ```
//
// We also reuse this AST node in places where we just need a statement block
// construct, such as a function body:
//
// `fn f() { u32:42 }`
// --------^~~~~~~~~^ represented via a `StatementBlock`
//
// Even though that's not really a block expression it's a natural place to use
// the same construct.
//
// Attrs:
//  statements: Sequence of statements contained within the block.
//  trailing_semi: Whether the final statement had a trailing semicolon after
//    it.
//
// Invariant: in the degenerate case where there are no statements in the block,
// trailing_semi is always true.
class StatementBlock : public Expr {
 public:
  StatementBlock(Module* owner, Span span, std::vector<Statement*> statements,
                 bool trailing_semi);

  ~StatementBlock() override;

  bool IsBlockedExprNoLeader() const override { return true; }

  std::string ToInlineString() const override;

  AstNodeKind kind() const override { return AstNodeKind::kStatementBlock; }
  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleStatementBlock(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleStatementBlock(this);
  }
  std::string_view GetNodeTypeName() const override { return "StatementBlock"; }
  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return std::vector<AstNode*>(statements_.begin(), statements_.end());
  }

  absl::Span<Statement* const> statements() const { return statements_; }
  bool trailing_semi() const { return trailing_semi_; }
  bool empty() const { return statements_.empty(); }
  int64_t size() const { return statements_.size(); }

 private:
  Precedence GetPrecedenceWithoutParens() const final {
    return Precedence::kStrongest;
  }

  std::string ToStringInternal() const final;

  std::vector<Statement*> statements_;
  bool trailing_semi_;
};

// Represents a reference to a name (identifier).
//
// Every name reference has a link to its corresponding `name_def()`, which can
// either be defined in the module somewhere (NameDef) or defined as a built-in
// symbol that's implicitly available, e.g. built-in functions (BuiltinNameDef).
class NameRef : public Expr {
 public:
  NameRef(Module* owner, Span span, std::string identifier, AnyNameDef name_def)
      : Expr(owner, std::move(span)),
        name_def_(name_def),
        identifier_(std::move(identifier)) {
    CHECK_NE(identifier_, "_");
  }

  ~NameRef() override;

  AstNodeKind kind() const override { return AstNodeKind::kNameRef; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleNameRef(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleNameRef(this);
  }

  const std::string& identifier() const { return identifier_; }

  std::string_view GetNodeTypeName() const override { return "NameRef"; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {};
  }

  // Returns the node that defines the associated NameDef -- note: when the
  // corresponding NameDef for this reference is built-in this returns nullptr.
  const AstNode* GetDefiner() const {
    return IsBuiltin() ? nullptr
                       : std::get<const NameDef*>(name_def())->definer();
  }
  AstNode* GetDefiner() {
    const auto* self = this;
    return const_cast<AstNode*>(self->GetDefiner());
  }

  // Returns whether the definer node is of type `T`; e.g.
  //
  //  bool defined_by_fn = name_ref->DefinerIs<Function>();
  template <typename T>
  bool DefinerIs() {
    if (std::holds_alternative<BuiltinNameDef*>(name_def_)) {
      return false;
    }
    auto* name_def = std::get<const NameDef*>(name_def_);
    return dynamic_cast<T*>(name_def->definer()) != nullptr;
  }

  std::optional<Pos> GetNameDefStart() const {
    if (std::holds_alternative<const NameDef*>(name_def_)) {
      return std::get<const NameDef*>(name_def_)->span().start();
    }
    return std::nullopt;
  }

  std::variant<const NameDef*, BuiltinNameDef*> name_def() const {
    return name_def_;
  }

  bool IsBuiltin() const {
    return std::holds_alternative<BuiltinNameDef*>(name_def_);
  }

  Precedence GetPrecedenceWithoutParens() const final {
    return Precedence::kStrongest;
  }

 private:
  std::string ToStringInternal() const final { return identifier_; }

  AnyNameDef name_def_;
  std::string identifier_;
};

enum class NumberKind : uint8_t {
  kBool,
  kCharacter,
  kOther,
};

// Represents a literal number value.
class Number : public Expr {
 public:
  Number(Module* owner, Span span, std::string text, NumberKind kind,
         TypeAnnotation* type);

  ~Number() override;

  AstNodeKind kind() const override { return AstNodeKind::kNumber; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleNumber(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleNumber(this);
  }

  std::string_view GetNodeTypeName() const override { return "Number"; }
  std::vector<AstNode*> GetChildren(bool want_types) const override;

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
  void SetTypeAnnotation(TypeAnnotation* type_annotation);

  // Warning: be careful not to iterate over signed chars of the result, as they
  // may sign extend on platforms that compile with signed chars. Preferred
  // pattern is:
  //
  //    for (const uint8_t c : n->text()) { ... }
  const std::string& text() const { return text_; }

  // Determines whether the number fits in the given `bit_count`.
  absl::StatusOr<bool> FitsInType(int64_t bit_count) const;

  // Turns the text for this number into a Bits object with the given bit_count.
  absl::StatusOr<Bits> GetBits(int64_t bit_count,
                               const FileTable& file_table) const;

  // Note: fails if the value doesn't fit in 64 bits.
  absl::StatusOr<uint64_t> GetAsUint64(const FileTable& file_table) const {
    XLS_ASSIGN_OR_RETURN(Bits bits, GetBits(64, file_table));
    return bits.ToUint64();
  }

  NumberKind number_kind() const { return number_kind_; }

  Precedence GetPrecedenceWithoutParens() const final {
    return Precedence::kStrongest;
  }

 private:
  std::string ToStringInternal() const final;

  std::string text_;  // Will never be empty.
  NumberKind number_kind_;
  TypeAnnotation* type_annotation_;  // May be null.
};

// A literal string of u8s. Does not internally include opening and closing
// quotation marks.
class String : public Expr {
 public:
  String(Module* owner, Span span, std::string_view text)
      : Expr(owner, span), text_(text) {}

  ~String() override;

  AstNodeKind kind() const override { return AstNodeKind::kString; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleString(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleString(this);
  }
  std::string_view GetNodeTypeName() const override { return "String"; }
  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {};
  }

  const std::string& text() const { return text_; }

  Precedence GetPrecedenceWithoutParens() const final {
    return Precedence::kStrongest;
  }

 private:
  std::string ToStringInternal() const final;

  std::string text_;
};

// Represents a user-defined-type definition; e.g.
//    type Foo = (u32, u32);
//    type Bar = (u32, Foo);
class TypeAlias : public AstNode {
 public:
  static std::string_view GetDebugTypeName() { return "type alias"; }

  TypeAlias(Module* owner, Span span, NameDef& name_def, TypeAnnotation& type,
            bool is_public);

  ~TypeAlias() override;

  AstNodeKind kind() const override { return AstNodeKind::kTypeAlias; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleTypeAlias(this);
  }
  std::string_view GetNodeTypeName() const override { return "TypeAlias"; }
  const std::string& identifier() const { return name_def_.identifier(); }

  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {&name_def_, &type_annotation_};
  }

  NameDef& name_def() const { return name_def_; }
  TypeAnnotation& type_annotation() const { return type_annotation_; }
  bool is_public() const { return is_public_; }
  const Span& span() const { return span_; }
  std::optional<Span> GetSpan() const override { return span_; }
  void set_extern_type_name(std::string_view n) {
    extern_type_name_ = std::string(n);
  }
  const std::optional<std::string>& extern_type_name() const {
    return extern_type_name_;
  }

 private:
  Span span_;
  NameDef& name_def_;
  TypeAnnotation& type_annotation_;
  bool is_public_;
  // The external verilog type name
  std::optional<std::string> extern_type_name_;
};

// Represents an array expression; e.g. `[a, b, c]`.
class Array : public Expr {
 public:
  Array(Module* owner, Span span, std::vector<Expr*> members,
        bool has_ellipsis);

  ~Array() override;

  // The type annotation would be an arbitrary-length number of leading
  // characters.
  bool IsBlockedExprNoLeader() const override {
    return type_annotation_ == nullptr;
  }
  bool IsBlockedExprWithLeader() const override {
    return type_annotation_ != nullptr;
  }

  AstNodeKind kind() const override { return AstNodeKind::kArray; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleArray(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleArray(this);
  }

  std::string_view GetNodeTypeName() const override { return "Array"; }
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  const std::vector<Expr*>& members() const { return members_; }

  // TODO(leary): 2021-05-18 See TODO comment on Number::set_type_annotation for
  // the reason this exists (prefix types for literal values), but it should be
  // removed in favor of a decorator construct instead of using mutability.
  TypeAnnotation* type_annotation() const { return type_annotation_; }
  void set_type_annotation(TypeAnnotation* type_annotation) {
    type_annotation_ = type_annotation;
  }

  bool has_ellipsis() const { return has_ellipsis_; }

  Precedence GetPrecedenceWithoutParens() const final {
    return Precedence::kStrongest;
  }

 private:
  std::string ToStringInternal() const override;

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

  ~ConstantArray() override;

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleConstantArray(this);
  }
};

// Several different AST nodes define types that can be referred to by a
// TypeRef.
using TypeDefinition =
    std::variant<TypeAlias*, StructDef*, ProcDef*, EnumDef*, ColonRef*>;

// Returns the name definition that (most locally) defined this type definition
// AST node.
//
// In the case of a ColonRef the name definition given is the subject of the
// colon-reference, i.e. it does not traverse module boundaries to retrieve a
// name definition. That is:
//
//    fn f() -> foo::Bar { ... }
//    ----------^~~~~~~^
//
// The GetNameDef() on that ColonRef type definition node returns the subject
// "foo", which can be a built-in name.
AnyNameDef TypeDefinitionGetNameDef(const TypeDefinition& td);

absl::StatusOr<TypeDefinition> ToTypeDefinition(AstNode* node);

// AST construct that refers to a defined type.
//
// Attrs:
//  type_definition: The resolved type if it can be resolved locally, or a
//    ColonRef if the type lives in an external module.
class TypeRef : public AstNode {
 public:
  TypeRef(Module* owner, Span span, TypeDefinition type_definition);

  ~TypeRef() override;

  AstNodeKind kind() const override { return AstNodeKind::kTypeRef; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleTypeRef(this);
  }

  std::string_view GetNodeTypeName() const override { return "TypeRef"; }
  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {};
  }

  const TypeDefinition& type_definition() const { return type_definition_; }
  const Span& span() const { return span_; }
  std::optional<Span> GetSpan() const override { return span_; }

 private:
  Span span_;
  TypeDefinition type_definition_;
};

// Represents an import statement; e.g.
//  import std;
// Or:
//  import foo.bar.baz;
// Or:
//  import foo.bar.baz as my_alias;
//
// Attributes:
//  span: Span of the overall import statement in the text.
//  subject: The imported name; e.g. `foo.bar.baz`.
//  name_def: The name definition node for the binding this import creates.
//  alias: Alias that the imported name is bound to (if an `as` keyword was
//    present).
class Import : public AstNode {
 public:
  Import(Module* owner, Span span, std::vector<std::string> subject,
         NameDef& name_def, std::optional<std::string> alias);

  ~Import() override;

  AstNodeKind kind() const override { return AstNodeKind::kImport; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleImport(this);
  }
  std::string_view GetNodeTypeName() const override { return "Import"; }
  const std::string& identifier() const { return name_def_.identifier(); }

  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {&name_def_};
  }

  const std::vector<std::string>& subject() const { return subject_; }
  NameDef& name_def() const { return name_def_; }
  const Span& span() const { return span_; }
  std::optional<Span> GetSpan() const override { return span_; }
  const std::optional<std::string>& alias() const { return alias_; }

 private:
  Span span_;
  std::vector<std::string> subject_;
  NameDef& name_def_;
  std::optional<std::string> alias_;
};

// Represents a module-value or enum-value style reference when the LHS
// expression is unknown; e.g. when accessing a member in a module:
//
//    some_mod::SomeEnum::VALUE
//
// Then the ColonRef `some_mod::SomeEnum` is the LHS.
class ColonRef : public Expr {
 public:
  using Subject = std::variant<NameRef*, ColonRef*>;

  ColonRef(Module* owner, Span span, Subject subject, std::string attr);

  ~ColonRef() override;

  AstNodeKind kind() const override { return AstNodeKind::kColonRef; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleColonRef(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleColonRef(this);
  }

  std::string_view GetNodeTypeName() const override { return "ColonRef"; }

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
  std::optional<Import*> ResolveImportSubject() const;

  Precedence GetPrecedenceWithoutParens() const final {
    return Precedence::kPaths;
  }

 private:
  std::string ToStringInternal() const override {
    return absl::StrFormat("%s::%s", ToAstNode(subject_)->ToString(), attr_);
  }

  Subject subject_;
  std::string attr_;
};

// Represents a function parameter.
class Param : public AstNode {
 public:
  Param(Module* owner, NameDef* name_def, TypeAnnotation* type);

  ~Param() override;

  AstNodeKind kind() const override { return AstNodeKind::kParam; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleParam(this);
  }

  std::string_view GetNodeTypeName() const override { return "Param"; }
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
  std::optional<Span> GetSpan() const override { return span_; }

 private:
  NameDef* name_def_;
  TypeAnnotation* type_annotation_;
  Span span_;
};

enum class UnopKind : uint8_t {
  kInvert,  // one's complement inversion (bit flip)
  kNegate,  // two's complement aritmetic negation (~x+1)
};

std::string UnopKindToString(UnopKind k);

// Represents a unary operation expression; e.g. `!x`.
class Unop : public Expr {
 public:
  Unop(Module* owner, Span span, UnopKind unop_kind, Expr* operand)
      : Expr(owner, std::move(span)),
        unop_kind_(unop_kind),
        operand_(operand) {}

  ~Unop() override;

  AstNodeKind kind() const override { return AstNodeKind::kUnop; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleUnop(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleUnop(this);
  }

  std::string_view GetNodeTypeName() const override { return "Unop"; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {operand_};
  }

  UnopKind unop_kind() const { return unop_kind_; }
  Expr* operand() const { return operand_; }

  Precedence GetPrecedenceWithoutParens() const final {
    return Precedence::kUnaryOp;
  }

 private:
  std::string ToStringInternal() const final;

  UnopKind unop_kind_;
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
  X(kMod, "MOD", "%")                     \
  X(kLogicalAnd, "LOGICAL_AND", "&&")     \
  X(kLogicalOr, "LOGICAL_OR", "||")       \
  X(kConcat, "CONCAT", "++")

enum class BinopKind : uint8_t {
#define FIRST_COMMA(A, ...) A,
  XLS_DSLX_BINOP_KIND_EACH(FIRST_COMMA)
#undef FIRST_COMMA
};

absl::StatusOr<BinopKind> BinopKindFromString(std::string_view s);

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

// Binary operators that have the signature `(u1, u1) -> u1`.
const absl::btree_set<BinopKind>& GetBinopLogicalKinds();

// Binary operators that are shift operations.
const absl::btree_set<BinopKind>& GetBinopShifts();

// Represents a binary operation expression; e.g. `x + y`.
class Binop : public Expr {
 public:
  Binop(Module* owner, Span span, BinopKind kind, Expr* lhs, Expr* rhs);

  ~Binop() override;

  AstNodeKind kind() const override { return AstNodeKind::kBinop; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleBinop(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleBinop(this);
  }

  std::string_view GetNodeTypeName() const override { return "Binop"; }
  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {lhs_, rhs_};
  }

  BinopKind binop_kind() const { return binop_kind_; }
  Expr* lhs() const { return lhs_; }
  Expr* rhs() const { return rhs_; }

  Precedence GetPrecedenceWithoutParens() const final;

 private:
  std::string ToStringInternal() const final;

  BinopKind binop_kind_;
  Expr* lhs_;
  Expr* rhs_;
};

// Represents the conditional expression; e.g.
//
//  if test { consequent }
//  else { alternate }
//
// Note that, in Rust-like fashion, this operates as a ternary, i.e. it yields
// the last expression, unless a semicolon is given at the end that makes the
// result into the unit type.
//
//  let _: () = if { side_effect!(); } else { side_effect!(); };
//              note the semis ----^------------------------^
//
// To do laddered if / else if the `alternate` expression can itself be a
// `Conditional` expression instead of a `Block`.
class Conditional : public Expr {
 public:
  Conditional(Module* owner, Span span, Expr* test, StatementBlock* consequent,
              std::variant<StatementBlock*, Conditional*> alternate);

  ~Conditional() override;

  // The arbitrary length leading chars are the conditional test expression.
  bool IsBlockedExprWithLeader() const override { return true; }

  AstNodeKind kind() const override { return AstNodeKind::kConditional; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleConditional(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleConditional(this);
  }

  std::string_view GetNodeTypeName() const override { return "Conditional"; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {test_, consequent_, ToAstNode(alternate_)};
  }

  Expr* test() const { return test_; }
  StatementBlock* consequent() const { return consequent_; }
  std::variant<StatementBlock*, Conditional*> alternate() const {
    return alternate_;
  }

  bool HasElseIf() const {
    return std::holds_alternative<Conditional*>(alternate());
  }

  // Returns whether the blocks inside of this (potentially laddered)
  // conditional have multiple statements.
  bool HasMultiStatementBlocks() const;

  Precedence GetPrecedenceWithoutParens() const final {
    return Precedence::kStrongest;
  }

 private:
  std::string ToStringInternal() const final;

  Expr* test_;
  StatementBlock* consequent_;
  std::variant<StatementBlock*, Conditional*> alternate_;
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

  ~ParametricBinding() override;

  AstNodeKind kind() const override { return AstNodeKind::kParametricBinding; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleParametricBinding(this);
  }

  // TODO(leary): 2020-08-21 Fix this, the span is more than just the name def's
  // span, it must include the type/expr.
  const Span& span() const { return name_def_->span(); }
  std::optional<Span> GetSpan() const override { return span(); }

  std::string_view GetNodeTypeName() const override {
    return "ParametricBinding";
  }
  std::string ToString() const override;

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

class Proc;

// Indicates if a function is normal or is part of a proc instantiation.
enum class FunctionTag : uint8_t {
  kNormal,
  kProcConfig,
  kProcNext,
  kProcInit,
};

std::string_view FunctionTagToString(FunctionTag tag);

template <typename Sink>
inline void AbslStringify(Sink& sink, FunctionTag tag) {
  absl::Format(&sink, "%s", FunctionTagToString(tag));
}

// Attrs:
//  extern_verilog_module_name: Attribute that can be tagged on DSLX functions
//    to indicate that in the rest of the XLS toolchain the function can/should
//    be subsituted with this Verilog module instantiation. (The metadata for
//    this feature is expected to evolve over time beyond a simple name.)
class Function : public AstNode {
 public:
  static std::string_view GetDebugTypeName() { return "function"; }

  Function(Module* owner, Span span, NameDef* name_def,
           std::vector<ParametricBinding*> parametric_bindings,
           std::vector<Param*> params, TypeAnnotation* return_type,
           StatementBlock* body, FunctionTag tag, bool is_public);

  ~Function() override;
  AstNodeKind kind() const override { return AstNodeKind::kFunction; }
  std::optional<Span> GetSpan() const override { return span_; }
  Span span() const { return span_; }
  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleFunction(this);
  }
  std::string_view GetNodeTypeName() const override { return "Function"; }
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  std::string ToString() const override;

  // Returns a string representation of this function with the given identifier
  // and without parametrics. Used for printing Proc component functions, i.e.,
  // config and next.
  std::string ToUndecoratedString(std::string_view identifier) const;

  const std::string& identifier() const { return name_def_->identifier(); }
  const std::vector<ParametricBinding*>& parametric_bindings() const {
    return parametric_bindings_;
  }
  const std::vector<Param*>& params() const { return params_; }
  absl::StatusOr<Param*> GetParamByName(std::string_view param_name) const;

  // The body of the function is a block (sequence of statements that yields a
  // final expression).
  StatementBlock* body() const { return body_; }

  bool IsParametric() const { return !parametric_bindings_.empty(); }
  bool is_public() const { return is_public_; }

  // Returns all of the parametric identifiers that must be bound by the caller
  // in an invocation; i.e. they have no default expression.
  //
  // In a function like:
  //
  //    fn p<X: u32, Y: u32, Z: u32 = X+Y>()
  //
  // The free parametric keys are `["X", "Y"]`.
  std::vector<std::string> GetFreeParametricKeys() const;

  // As above but gives the result as a (order-stable) set.
  absl::btree_set<std::string> GetFreeParametricKeySet() const;

  // Returns all parametric binding identifiers; e.g. in the above example:
  //
  //    {"X", "Y", "Z"}
  //
  // This is useful when we want to check that a parametric environment is valid
  // to use as an environment a given function.
  const absl::flat_hash_set<std::string>& parametric_keys() const {
    return parametric_keys_;
  }

  NameDef* name_def() const { return name_def_; }

  TypeAnnotation* return_type() const { return return_type_; }
  void set_return_type(TypeAnnotation* return_type) {
    return_type_ = return_type;
  }

  void set_extern_verilog_module(xls::ForeignFunctionData data) {
    extern_verilog_module_ = std::move(data);
  }
  const std::optional<::xls::ForeignFunctionData>& extern_verilog_module()
      const {
    return extern_verilog_module_;
  }

  FunctionTag tag() const { return tag_; }
  std::optional<Proc*> proc() const { return proc_; }
  void set_proc(Proc* proc) { proc_ = proc; }
  bool IsInProc() const { return proc_.has_value(); }

  std::optional<Span> GetParametricBindingsSpan() const {
    if (parametric_bindings_.empty()) {
      return std::nullopt;
    }
    return Span(parametric_bindings_.front()->span().start(),
                parametric_bindings_.back()->span().limit());
  }

 private:
  Span span_;
  NameDef* name_def_;
  std::vector<ParametricBinding*> parametric_bindings_;
  absl::flat_hash_set<std::string> parametric_keys_;
  std::vector<Param*> params_;
  TypeAnnotation* return_type_;  // May be null.
  StatementBlock* body_;
  const FunctionTag tag_;
  std::optional<Proc*> proc_;

  const bool is_public_;
  std::optional<ForeignFunctionData> extern_verilog_module_;
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

  ~MatchArm() override;

  AstNodeKind kind() const override { return AstNodeKind::kMatchArm; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleMatchArm(this);
  }
  std::string_view GetNodeTypeName() const override { return "MatchArm"; }
  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override;

  const std::vector<NameDefTree*>& patterns() const { return patterns_; }
  Expr* expr() const { return expr_; }
  const Span& span() const { return span_; }
  std::optional<Span> GetSpan() const override { return span_; }

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
// (prioritized in sequential order from first arm to last arm).
class Match : public Expr {
 public:
  Match(Module* owner, Span span, Expr* matched, std::vector<MatchArm*> arms);

  ~Match() override;

  // Leading chars are the matched-on expression.
  bool IsBlockedExprWithLeader() const override { return true; }

  AstNodeKind kind() const override { return AstNodeKind::kMatch; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleMatch(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleMatch(this);
  }

  std::string_view GetNodeTypeName() const override { return "Match"; }

  std::vector<AstNode*> GetChildren(bool want_types) const override;

  const std::vector<MatchArm*>& arms() const { return arms_; }
  Expr* matched() const { return matched_; }

  Precedence GetPrecedenceWithoutParens() const final {
    return Precedence::kStrongest;
  }

 private:
  std::string ToStringInternal() const final;

  Expr* matched_;
  std::vector<MatchArm*> arms_;
};

// Represents an attribute access expression; e.g. `a.x`.
//                                                   ^
//                       (this dot makes an attr) ---+
class Attr : public Expr {
 public:
  Attr(Module* owner, Span span, Expr* lhs, std::string attr)
      : Expr(owner, span), lhs_(lhs), attr_(std::move(attr)) {}

  ~Attr() override;

  AstNodeKind kind() const override { return AstNodeKind::kAttr; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleAttr(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleAttr(this);
  }

  std::string_view GetNodeTypeName() const override { return "Attr"; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {lhs_};
  }

  Expr* lhs() const { return lhs_; }

  std::string_view attr() const { return attr_; }

  Precedence GetPrecedenceWithoutParens() const final {
    return Precedence::kFieldExpression;
  }

 private:
  std::string ToStringInternal() const final {
    return absl::StrFormat("%s.%s", lhs_->ToString(), attr_);
  }

  Expr* lhs_;
  const std::string attr_;
};

class Instantiation : public Expr {
 public:
  Instantiation(Module* owner, Span span, Expr* callee,
                std::vector<ExprOrType> explicit_parametrics);

  ~Instantiation() override;

  // Leading chars are the callee being invoked/instantiated.
  bool IsBlockedExprWithLeader() const override { return true; }

  AstNodeKind kind() const override { return AstNodeKind::kInstantiation; }

  Expr* callee() const { return callee_; }

  // Any explicit parametric expressions given in this invocation; e.g. in:
  //
  //    f<a, b, c>()
  //
  // The expressions a, b, c would be in this sequence.
  const std::vector<ExprOrType>& explicit_parametrics() const {
    return explicit_parametrics_;
  }

 protected:
  std::string FormatParametrics() const;

 private:
  Expr* callee_;
  std::vector<ExprOrType> explicit_parametrics_;
};

// A reference to a function, with possible explicit parametrics. Currently,
// this is only used for the function argument of a `map()` call, which is not a
// direct invocation context. `FunctionRef` is counterintuitively never used as
// the `callee` of an `Invocation`, but we may eventually retrofit `Invocation`
// so that the `callee` is a `FunctionRef`.
class FunctionRef : public Instantiation {
 public:
  FunctionRef(Module* owner, Span span, Expr* callee,
              std::vector<ExprOrType> explicit_parametrics);

  ~FunctionRef() override;

  AstNodeKind kind() const override { return AstNodeKind::kFunctionRef; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleFunctionRef(this);
  }

  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleFunctionRef(this);
  }

  std::string_view GetNodeTypeName() const override { return "FunctionRef"; }

  std::vector<AstNode*> GetChildren(bool want_types) const override;

  Precedence GetPrecedenceWithoutParens() const final {
    return Precedence::kStrongest;
  }

 private:
  std::string ToStringInternal() const final {
    return absl::StrCat(callee()->ToString(), FormatParametrics());
  }
};

// Represents an invocation expression; e.g. `f(a, b, c)` or an implicit
// invocation for the config & next members of a spawned Proc.
class Invocation : public Instantiation {
 public:
  Invocation(Module* owner, Span span, Expr* callee, std::vector<Expr*> args,
             std::vector<ExprOrType> explicit_parametrics = {});

  ~Invocation() override;

  AstNodeKind kind() const override { return AstNodeKind::kInvocation; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleInvocation(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleInvocation(this);
  }

  std::string_view GetNodeTypeName() const override { return "Invocation"; }
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  std::string FormatArgs() const;

  absl::Span<Expr* const> args() const { return args_; }

  Precedence GetPrecedenceWithoutParens() const final {
    return Precedence::kFunctionCallOrArrayIndex;
  }

 private:
  std::string ToStringInternal() const final {
    return absl::StrFormat("%s%s(%s)", callee()->ToString(),
                           FormatParametrics(), FormatArgs());
  }

  std::vector<Expr*> args_;
};

// Represents a call to spawn a proc, e.g.,
//   spawn foo(a, b)(c)
// TODO(rspringer): 2021-09-25: Post-new-proc-implementation, determine if
// Spawns need to still be Instantiation subclasses.
class Spawn : public Instantiation {
 public:
  // A Spawn's body can be nullopt if it's the last expr in an unroll_for body.
  Spawn(Module* owner, Span span, Expr* callee, Invocation* config,
        Invocation* next, std::vector<ExprOrType> explicit_parametrics);

  ~Spawn() override;

  AstNodeKind kind() const override { return AstNodeKind::kSpawn; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleSpawn(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleSpawn(this);
  }

  std::string_view GetNodeTypeName() const override { return "Spawn"; }
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  Invocation* config() const { return config_; }
  Invocation* next() const { return next_; }

  Precedence GetPrecedenceWithoutParens() const final {
    return Precedence::kStrongest;
  }

 private:
  std::string ToStringInternal() const override;

  Invocation* config_;
  Invocation* next_;
};

// A static assertion that is constexpr-evaluated at compile time (more
// precisely from an internals perspective: type-checking time); i.e.
//
//  const_assert!(u32:1 == u32:2);
class ConstAssert : public AstNode {
 public:
  ConstAssert(Module* owner, Span span, Expr* arg);

  ~ConstAssert() override;

  AstNodeKind kind() const override { return AstNodeKind::kConstAssert; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleConstAssert(this);
  }

  std::string_view GetNodeTypeName() const override { return "ConstAssert"; }
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  std::string ToString() const override;
  std::optional<Span> GetSpan() const override { return span_; }

  const Span& span() const { return span_; }
  Expr* arg() const { return arg_; }

 private:
  Span span_;
  Expr* arg_;
};

// Represents a call to a variable-argument formatting macro;
// `e.g. trace_fmt!("x is {}", x)`
class FormatMacro : public Expr {
 public:
  FormatMacro(Module* owner, Span span, std::string macro,
              std::vector<FormatStep> format, std::vector<Expr*> args,
              std::optional<Expr*> verbosity = std::nullopt);

  ~FormatMacro() override;

  AstNodeKind kind() const override { return AstNodeKind::kFormatMacro; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleFormatMacro(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleFormatMacro(this);
  }

  std::string_view GetNodeTypeName() const override { return "FormatMacro"; }
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  std::string FormatArgs() const;

  const std::string& macro() const { return macro_; }
  absl::Span<Expr* const> args() const { return args_; }
  absl::Span<const FormatStep> format() const { return format_; }
  std::optional<Expr*> verbosity() const { return verbosity_; }

  Precedence GetPrecedenceWithoutParens() const final {
    return Precedence::kStrongest;
  }

 private:
  std::string ToStringInternal() const final;

  std::string macro_;
  std::vector<FormatStep> format_;
  std::vector<Expr*> args_;
  std::optional<Expr*> verbosity_;
};

// Represents a call to a parametric "make a zero value" macro;
// e.g. `zero!<T>()`
//
// Note that the parametric arg is a type annotation or a type expression, which
// we currently represent as ExprOrType.
class ZeroMacro : public Expr {
 public:
  ZeroMacro(Module* owner, Span span, ExprOrType type);

  ~ZeroMacro() override;

  AstNodeKind kind() const override { return AstNodeKind::kZeroMacro; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleZeroMacro(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleZeroMacro(this);
  }

  std::string_view GetNodeTypeName() const override { return "ZeroMacro"; }
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  ExprOrType type() const { return type_; }

  Precedence GetPrecedenceWithoutParens() const final {
    return Precedence::kStrongest;
  }

 private:
  std::string ToStringInternal() const final;

  ExprOrType type_;
};

// Represents a call to a parametric "make an all-ones value" macro;
// e.g. `allones!<T>()`
//
// Note that the parametric arg is a type annotation or a type expression, which
// we currently represent as ExprOrType.
class AllOnesMacro : public Expr {
 public:
  AllOnesMacro(Module* owner, Span span, ExprOrType type);

  ~AllOnesMacro() override;

  AstNodeKind kind() const override { return AstNodeKind::kAllOnesMacro; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleAllOnesMacro(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleAllOnesMacro(this);
  }

  std::string_view GetNodeTypeName() const override { return "AllOnesMacro"; }
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  ExprOrType type() const { return type_; }

  Precedence GetPrecedenceWithoutParens() const final {
    return Precedence::kStrongest;
  }

 private:
  std::string ToStringInternal() const final;

  ExprOrType type_;
};

// Represents a slice in the AST.
//
// For example, we can have `x[-4:-2]`, where x is of bit width N.
// The start and limit Exprs must be constexpr.
class Slice : public AstNode {
 public:
  Slice(Module* owner, Span span, Expr* start, Expr* limit)
      : AstNode(owner), span_(span), start_(start), limit_(limit) {}

  ~Slice() override;

  AstNodeKind kind() const override { return AstNodeKind::kSlice; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleSlice(this);
  }

  std::string_view GetNodeTypeName() const override { return "Slice"; }
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  std::string ToString() const override;
  std::optional<Span> GetSpan() const override { return span_; }

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

  Span GetSpan() const {
    return Span(name_def->span().start(), value->span().limit());
  }
};

// Represents a user-defined enum definition; e.g.
//
//  type MyTypeAlias = u2;
//  enum Foo : MyTypeAlias {
//    A = 0,
//    B = 1,
//    C = 2,
//  }
class EnumDef : public AstNode {
 public:
  static std::string_view GetDebugTypeName() { return "enum definition"; }

  EnumDef(Module* owner, Span span, NameDef* name_def, TypeAnnotation* type,
          std::vector<EnumMember> values, bool is_public);

  ~EnumDef() override;

  AstNodeKind kind() const override { return AstNodeKind::kEnumDef; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleEnumDef(this);
  }

  std::string_view GetNodeTypeName() const override { return "EnumDef"; }

  // Returns whether this enum definition has a member named "name".
  bool HasValue(std::string_view name) const;

  // Returns the value bound to the given enum definition name.
  //
  // These must be constexprs, which will be computed at type checking time.
  absl::StatusOr<Expr*> GetValue(std::string_view name) const;

  NameDef* GetNameDef(std::string_view target);

  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    std::vector<AstNode*> results = {name_def_};
    if (want_types && type_annotation_ != nullptr) {
      results.push_back(type_annotation_);
    }
    for (const EnumMember& item : values_) {
      results.push_back(item.name_def);
      results.push_back(item.value);
    }
    return results;
  }

  const std::string& identifier() const { return name_def_->identifier(); }

  const Span& span() const { return span_; }
  std::optional<Span> GetSpan() const override { return span_; }
  NameDef* name_def() const { return name_def_; }

  const std::vector<EnumMember>& values() const { return values_; }
  std::vector<EnumMember>& mutable_values() { return values_; }

  // Type annotation that indicates the "underlying" bit type payload for the
  // enum.
  TypeAnnotation* type_annotation() const { return type_annotation_; }

  bool is_public() const { return is_public_; }

  const std::string& GetMemberName(int64_t i) const {
    return values_.at(i).name_def->identifier();
  }
  void set_extern_type_name(std::string_view n) {
    extern_type_name_ = std::string(n);
  }
  const std::optional<std::string>& extern_type_name() const {
    return extern_type_name_;
  }

 private:
  Span span_;
  NameDef* name_def_;

  // Underlying type that defines the width of this enum.
  TypeAnnotation* type_annotation_;

  // name / expr pairs that define the enumerated values offered.
  std::vector<EnumMember> values_;

  // Populated by typechecking as a memoized note on whether this enum type was
  // found to be signed when the underlying type_annotation_ was resoled to a
  // concrete type.
  //
  // TODO(leary): 2021-09-29 We should keep this in a supplemental data
  // structure instead of mutating AST nodes.
  std::optional<bool> is_signed_;

  // Whether or not this enum definition was marked as public.
  bool is_public_;

  // The external verilog type name
  std::optional<std::string> extern_type_name_;
};

// Helper struct for DSLX-struct items defined inside of DSLX-structs.
struct StructMember {
  Span name_span;
  std::string name;
  TypeAnnotation* type;

  Span GetSpan() const { return Span(name_span.start(), type->span().limit()); }
};

// Base class for a struct-like entity, which has a name and members, along with
// optional parametric bindings and an optional impl.
class StructDefBase : public AstNode {
 public:
  StructDefBase(Module* owner, Span span, NameDef* name_def,
                std::vector<ParametricBinding*> parametric_bindings,
                std::vector<StructMember> members, bool is_public);

  ~StructDefBase() override;

  // Returns a string for what to call the type of entity represented by this
  // `StructDefBase` in error messages such as "A %s can't have two members with
  // the same name."
  virtual std::string_view EntityTypeStringForErrorMessages() const = 0;

  bool IsParametric() const { return !parametric_bindings_.empty(); }

  const std::string& identifier() const { return name_def_->identifier(); }

  std::vector<AstNode*> GetChildren(bool want_types) const override;

  NameDef* name_def() const { return name_def_; }
  const std::vector<ParametricBinding*>& parametric_bindings() const {
    return parametric_bindings_;
  }

  const std::vector<StructMember>& members() const { return members_; }
  std::vector<StructMember>& mutable_members() { return members_; }

  bool is_public() const { return public_; }
  const Span& span() const { return span_; }
  std::optional<Span> GetSpan() const override { return span_; }

  const std::string& GetMemberName(int64_t i) const { return members_[i].name; }
  std::vector<std::string> GetMemberNames() const;

  int64_t size() const { return members_.size(); }

  std::optional<Span> GetParametricBindingsSpan() const {
    if (parametric_bindings_.empty()) {
      return std::nullopt;
    }
    return Span(parametric_bindings_.front()->span().start(),
                parametric_bindings_.back()->span().limit());
  }

  void set_impl(Impl* impl) { impl_ = impl; }

  std::optional<Impl*> impl() const { return impl_; }

  std::optional<ConstantDef*> GetImplConstant(std::string_view constant_name);

 protected:
  // Helper for a subclass to implement `ToString()`, given the entity keyword
  // for the subclass (like "struct" or "proc") and an optional DSLX attribute
  // string.
  std::string ToStringWithEntityKeywordAndAttribute(
      std::string_view keyword, std::string_view attribute = "") const;

 private:
  Span span_;
  NameDef* name_def_;
  std::vector<ParametricBinding*> parametric_bindings_;
  std::vector<StructMember> members_;
  bool public_;
  std::optional<Impl*> impl_;
};

// Represents a struct definition.
class StructDef : public StructDefBase {
 public:
  using StructDefBase::StructDefBase;

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleStructDef(this);
  }

  AstNodeKind kind() const override { return AstNodeKind::kStructDef; }

  std::string_view GetNodeTypeName() const override { return "Struct"; }

  std::string_view EntityTypeStringForErrorMessages() const override {
    return "struct";
  }

  std::string ToString() const override;

  void set_extern_type_name(std::string_view n) {
    extern_type_name_ = std::string(n);
  }
  const std::optional<std::string>& extern_type_name() const {
    return extern_type_name_;
  }

 private:
  // The external verilog type name
  std::optional<std::string> extern_type_name_;
};

// Represents a proc declared with struct-like syntax, with the functions in an
// impl.
class ProcDef : public StructDefBase {
 public:
  using StructDefBase::StructDefBase;

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleProcDef(this);
  }

  AstNodeKind kind() const override { return AstNodeKind::kProcDef; }

  std::string_view GetNodeTypeName() const override { return "ProcDef"; }

  std::string_view EntityTypeStringForErrorMessages() const override {
    return "proc";
  }

  std::string ToString() const override;
};

// Gets the `StructDefBase` contained in a `TypeDefinition` if it contains one.
inline std::optional<StructDefBase*> TypeDefinitionToStructDefBase(
    TypeDefinition def) {
  auto* result = dynamic_cast<StructDefBase*>(ToAstNode(def));
  return result == nullptr ? std::nullopt : std::make_optional(result);
}

using ImplMember = std::variant<ConstantDef*, Function*>;

// Represents an impl for a struct.
class Impl : public AstNode {
 public:
  Impl(Module* owner, Span span, TypeAnnotation* struct_ref,
       std::vector<ImplMember> members, bool is_public);

  ~Impl() override;

  AstNodeKind kind() const override { return AstNodeKind::kImpl; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleImpl(this);
  }

  std::string_view GetNodeTypeName() const override { return "Impl"; }

  // An AST node that refers to the struct being implemented.
  TypeAnnotation* struct_ref() const { return struct_ref_; }

  void set_struct_ref(TypeAnnotation* struct_ref) { struct_ref_ = struct_ref; }

  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override;

  bool is_public() const { return public_; }
  const Span& span() const { return span_; }
  std::optional<Span> GetSpan() const override { return span_; }

  const std::vector<ImplMember>& members() const { return members_; }

  std::vector<ConstantDef*> GetConstants() const;

  // Returns the constant with the given name if present.
  std::optional<ConstantDef*> GetConstant(std::string_view name) const;

 private:
  Span span_;
  TypeAnnotation* struct_ref_;
  std::vector<ImplMember> members_;
  bool public_;
};

// Represents instantiation of a struct via member expressions.
//
// TODO(leary): 2020-09-08 Break out a StructInstanceMember type in lieu of the
// pair.
class StructInstance : public Expr {
 public:
  StructInstance(Module* owner, Span span, TypeAnnotation* struct_ref,
                 std::vector<std::pair<std::string, Expr*>> members);

  ~StructInstance() override;

  // The leading chars are the struct reference.
  bool IsBlockedExprWithLeader() const override { return true; }

  AstNodeKind kind() const override { return AstNodeKind::kStructInstance; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleStructInstance(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleStructInstance(this);
  }

  std::string_view GetNodeTypeName() const override { return "StructInstance"; }
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  // These are the members in the order given in the instantiation, note that
  // this can be different from the order in the struct definition.
  absl::Span<const std::pair<std::string, Expr*>> GetUnorderedMembers() const {
    return members_;
  }

  // Returns the members for the struct instance, ordered by the (resolved)
  // struct definition "struct_def".
  std::vector<std::pair<std::string, Expr*>> GetOrderedMembers(
      const StructDef* struct_def) const;

  // Returns the expression associated with the member named "name", or a
  // NotFound error status if none exists.
  absl::StatusOr<Expr*> GetExpr(std::string_view name) const;

  // An AST node that refers to the struct being instantiated.
  TypeAnnotation* struct_ref() const { return struct_ref_; }

  Precedence GetPrecedenceWithoutParens() const final {
    return Precedence::kStrongest;
  }

 private:
  std::string ToStringInternal() const final;

  TypeAnnotation* struct_ref_;
  std::vector<std::pair<std::string, Expr*>> members_;
};

// Represents a struct instantiation as a "delta" from a 'splatted' original;
// e.g.
//    Point { y: new_y, ..orig_p }
class SplatStructInstance : public Expr {
 public:
  SplatStructInstance(Module* owner, Span span, TypeAnnotation* struct_ref,
                      std::vector<std::pair<std::string, Expr*>> members,
                      Expr* splatted);

  ~SplatStructInstance() override;

  // The leading chars are the struct reference.
  bool IsBlockedExprWithLeader() const override { return true; }

  AstNodeKind kind() const override {
    return AstNodeKind::kSplatStructInstance;
  }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleSplatStructInstance(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleSplatStructInstance(this);
  }

  std::string_view GetNodeTypeName() const override {
    return "SplatStructInstance";
  }
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  Expr* splatted() const { return splatted_; }
  TypeAnnotation* struct_ref() const { return struct_ref_; }
  const std::vector<std::pair<std::string, Expr*>>& members() const {
    return members_;
  }

  Precedence GetPrecedenceWithoutParens() const final {
    return Precedence::kStrongest;
  }

 private:
  std::string ToStringInternal() const final;

  // The struct being instantiated.
  TypeAnnotation* struct_ref_;

  // Sequence of members being changed from the splatted original; e.g. in the
  // above example this is [('y', new_y)].
  std::vector<std::pair<std::string, Expr*>> members_;

  // Expression that's used as the original struct instance (that we're
  // instantiating a delta from); e.g. orig_p in the example above.
  Expr* splatted_;
};

// Represents a tuple-destructuring instantiation that omits one or more values
// e.g.,
//    let (x, .., y) = (1, 2, 3, 4); // assigns x=1, y=4
class RestOfTuple : public AstNode {
 public:
  RestOfTuple(Module* owner, Span span)
      : AstNode(owner), span_(std::move(span)) {}

  ~RestOfTuple() override;

  AstNodeKind kind() const override { return AstNodeKind::kRestOfTuple; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleRestOfTuple(this);
  }

  std::string_view GetNodeTypeName() const override { return "RestOfTuple"; }

  std::string ToString() const override { return ".."; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {};
  }

  const Span& span() const { return span_; }
  std::optional<Span> GetSpan() const override { return span_; }

 private:
  Span span_;
};

// Represents a slice in the AST; e.g. `-4+:u2`
class WidthSlice : public AstNode {
 public:
  WidthSlice(Module* owner, Span span, Expr* start, TypeAnnotation* width)
      : AstNode(owner), span_(std::move(span)), start_(start), width_(width) {}

  ~WidthSlice() override;

  AstNodeKind kind() const override { return AstNodeKind::kWidthSlice; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleWidthSlice(this);
  }
  std::string_view GetNodeTypeName() const override { return "WidthSlice"; }
  std::string ToString() const final;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {start_, width_};
  }

  Expr* start() const { return start_; }
  TypeAnnotation* width() const { return width_; }
  std::optional<Span> GetSpan() const override { return span_; }

 private:
  Span span_;
  Expr* start_;
  TypeAnnotation* width_;
};

// Helper type that holds the allowed variations for an index AST node's right
// hand side.
using IndexRhs = std::variant<Expr*, Slice*, WidthSlice*>;

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

  ~Index() override;

  AstNodeKind kind() const override { return AstNodeKind::kIndex; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleIndex(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleIndex(this);
  }

  std::string_view GetNodeTypeName() const override { return "Index"; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {lhs_, ToAstNode(rhs_)};
  }

  Expr* lhs() const { return lhs_; }
  IndexRhs rhs() const { return rhs_; }

  Precedence GetPrecedenceWithoutParens() const final {
    return Precedence::kFunctionCallOrArrayIndex;
  }

 private:
  std::string ToStringInternal() const final;

  // Expression that yields the value being indexed into; e.g. `a` in `a[10]`.
  Expr* lhs_;
  // Index expression; e.g. `10` in `a[10]`.
  IndexRhs rhs_;
};

// Represents a range expression, e.g., a..b, which expands to the integral
// values [a, b). Currently, only the Rust RangeExpr form is supported
// (https://doc.rust-lang.org/reference/expressions/range-expr.html), i.e.,
// RangeFrom and other variants are not implemented.
class Range : public Expr {
 public:
  Range(Module* owner, Span span, Expr* start, Expr* end);
  ~Range() override;
  AstNodeKind kind() const override { return AstNodeKind::kRange; }
  std::string_view GetNodeTypeName() const override { return "Range"; }
  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleRange(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleRange(this);
  }
  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {start_, end_};
  }

  Expr* start() const { return start_; }
  Expr* end() const { return end_; }

  Precedence GetPrecedenceWithoutParens() const final {
    return Precedence::kRange;
  }

 private:
  std::string ToStringInternal() const final;

  Expr* start_;
  Expr* end_;
};

// Represents a unit test construct.
//
// These are specified with an annotation as follows:
//
// ```dslx
// #[test]
// fn test_foo() { ... }
// ```
class TestFunction : public AstNode {
 public:
  static std::string_view GetDebugTypeName() { return "test function"; }

  TestFunction(Module* owner, Span span, Function& fn)
      : AstNode(owner), span_(std::move(span)), fn_(fn) {}

  ~TestFunction() override;

  AstNodeKind kind() const override { return AstNodeKind::kTestFunction; }
  NameDef* name_def() const { return fn_.name_def(); }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleTestFunction(this);
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {&fn_};
  }

  std::string_view GetNodeTypeName() const override { return "TestFunction"; }
  std::string ToString() const override {
    return absl::StrFormat("#[test]\n%s", fn_.ToString());
  }

  Function& fn() const { return fn_; }
  std::optional<Span> GetSpan() const override { return span(); }
  const Span& span() const { return span_; }

  const std::string& identifier() const { return fn_.name_def()->identifier(); }

 private:
  const Span span_;
  Function& fn_;
};

// Represents a function to be quick-check'd.
class QuickCheck : public AstNode {
 public:
  static std::string_view GetDebugTypeName() { return "quickcheck"; }

  static constexpr int64_t kDefaultTestCount = 1000;

  QuickCheck(Module* owner, Span span, Function* fn,
             std::optional<int64_t> test_count = std::nullopt);

  ~QuickCheck() override;

  AstNodeKind kind() const override { return AstNodeKind::kQuickCheck; }
  NameDef* name_def() const { return fn_->name_def(); }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleQuickCheck(this);
  }

  std::string_view GetNodeTypeName() const override { return "QuickCheck"; }
  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {fn_};
  }

  const std::string& identifier() const { return fn_->identifier(); }

  Function* fn() const { return fn_; }
  int64_t GetTestCountOrDefault() const {
    return test_count_ ? *test_count_ : kDefaultTestCount;
  }
  std::optional<int64_t> test_count() const { return test_count_; }
  std::optional<Span> GetSpan() const override { return span_; }
  const Span& span() const { return span_; }

 private:
  Span span_;
  Function* fn_;
  std::optional<int64_t> test_count_;
};

// Represents an index into a tuple, e.g., "(u32:7, u32:8).1".
class TupleIndex : public Expr {
 public:
  TupleIndex(Module* owner, Span span, Expr* lhs, Number* index);
  ~TupleIndex() override;

  AstNodeKind kind() const override { return AstNodeKind::kTupleIndex; }
  absl::Status Accept(AstNodeVisitor* v) const override;
  absl::Status AcceptExpr(ExprVisitor* v) const override;
  std::string_view GetNodeTypeName() const override { return "TupleIndex"; }
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  Expr* lhs() const { return lhs_; }
  Number* index() const { return index_; }

  Precedence GetPrecedenceWithoutParens() const final {
    return Precedence::kFieldExpression;
  }

 private:
  std::string ToStringInternal() const final;

  Expr* lhs_;
  Number* index_;
};

// Represents an XLS tuple expression.
class XlsTuple : public Expr {
 public:
  XlsTuple(Module* owner, Span span, std::vector<Expr*> members,
           bool has_trailing_comma)
      : Expr(owner, std::move(span)),
        members_(std::move(members)),
        has_trailing_comma_(has_trailing_comma) {}

  ~XlsTuple() override;

  bool IsBlockedExprNoLeader() const override { return true; }

  AstNodeKind kind() const override { return AstNodeKind::kXlsTuple; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleXlsTuple(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleXlsTuple(this);
  }

  std::string_view GetNodeTypeName() const override { return "XlsTuple"; }
  absl::Span<Expr* const> members() const { return members_; }
  bool empty() const { return members_.empty(); }
  bool has_trailing_comma() const { return has_trailing_comma_; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return ToAstNodes<Expr>(members_);
  }

  Precedence GetPrecedenceWithoutParens() const final {
    return Precedence::kStrongest;
  }

 private:
  std::string ToStringInternal() const final;

  std::vector<Expr*> members_;
  bool has_trailing_comma_;
};

// Abstract base class for a `for` loop-like expression, which may be e.g. a
// regular `for` loop or an `unroll_for!`. The structure of these types of loops
// should be as much the same as possible.
class ForLoopBase : public Expr {
 public:
  ForLoopBase(Module* owner, Span span, NameDefTree* names,
              TypeAnnotation* type, Expr* iterable, StatementBlock* body,
              Expr* init);

  // Leader chars are the for loop bindings and iterable.
  bool IsBlockedExprWithLeader() const override { return true; }

  std::vector<AstNode*> GetChildren(bool want_types) const override;

  // Names bound in the body of the loop.
  NameDefTree* names() const { return names_; }

  // Annotation corresponding to "names".
  TypeAnnotation* type_annotation() const { return type_annotation_; }

  // Expression for "thing to iterate over".
  Expr* iterable() const { return iterable_; }

  // Expression for the loop body.
  StatementBlock* body() const { return body_; }

  // Initial expression for the loop (start values expr).
  Expr* init() const { return init_; }

  Precedence GetPrecedenceWithoutParens() const final {
    return Precedence::kStrongest;
  }

 protected:
  // Used by string conversion to determine the loop keyword for this loop.
  virtual std::string_view keyword() const { return "for"; }

 private:
  std::string ToStringInternal() const override;

  NameDefTree* names_;
  TypeAnnotation* type_annotation_;
  Expr* iterable_;
  StatementBlock* body_;
  Expr* init_;
};

// Represents a for-loop expression.
class For : public ForLoopBase {
 public:
  using ForLoopBase::ForLoopBase;

  ~For() override;

  AstNodeKind kind() const override { return AstNodeKind::kFor; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleFor(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleFor(this);
  }

  std::string_view GetNodeTypeName() const override { return "For"; }

 protected:
  std::string_view keyword() const override { return "for"; }
};

// Represents an operation to "unroll" the given for-like expression by the
// number of elements in the iterable.
class UnrollFor : public ForLoopBase {
 public:
  using ForLoopBase::ForLoopBase;

  ~UnrollFor() override;

  AstNodeKind kind() const override { return AstNodeKind::kUnrollFor; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleUnrollFor(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleUnrollFor(this);
  }
  std::string_view GetNodeTypeName() const override { return "unroll-for"; }

 protected:
  std::string_view keyword() const override { return "unroll_for!"; }
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

  ~Cast() override;

  AstNodeKind kind() const override { return AstNodeKind::kCast; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleCast(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleCast(this);
  }

  std::string_view GetNodeTypeName() const override { return "Cast"; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    if (want_types) {
      return {expr_, type_annotation_};
    }
    return {expr_};
  }

  Expr* expr() const { return expr_; }
  TypeAnnotation* type_annotation() const { return type_annotation_; }

  Precedence GetPrecedenceWithoutParens() const final {
    return Precedence::kAs;
  }

 private:
  std::string ToStringInternal() const final;

  Expr* expr_;
  TypeAnnotation* type_annotation_;
};

// Represents a constant definition.
//
//  is_public: Indicates whether the constant had a public annotation
//    (applicable to module level constant definitions only)
class ConstantDef : public AstNode {
 public:
  static std::string_view GetDebugTypeName() { return "constant definition"; }

  ConstantDef(Module* owner, Span span, NameDef* name_def,
              TypeAnnotation* type_annotation, Expr* value, bool is_public);

  ~ConstantDef() override;

  AstNodeKind kind() const override { return AstNodeKind::kConstantDef; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleConstantDef(this);
  }

  std::string_view GetNodeTypeName() const override { return "ConstantDef"; }
  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    if (want_types && type_annotation_ != nullptr) {
      return {name_def_, type_annotation_, value_};
    }
    return {name_def_, value_};
  }

  const std::string& identifier() const { return name_def_->identifier(); }
  NameDef* name_def() const { return name_def_; }
  TypeAnnotation* type_annotation() const { return type_annotation_; }
  Expr* value() const { return value_; }
  const Span& span() const { return span_; }
  std::optional<Span> GetSpan() const override { return span_; }
  bool is_public() const { return is_public_; }

 private:
  Span span_;
  NameDef* name_def_;
  TypeAnnotation* type_annotation_;
  Expr* value_;
  bool is_public_;
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
  using Leaf = std::variant<NameDef*, NameRef*, WildcardPattern*, Number*,
                            ColonRef*, Range*, RestOfTuple*>;

  NameDefTree(Module* owner, Span span, std::variant<Nodes, Leaf> tree)
      : AstNode(owner), span_(std::move(span)), tree_(std::move(tree)) {}

  ~NameDefTree() override;

  AstNodeKind kind() const override { return AstNodeKind::kNameDefTree; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleNameDefTree(this);
  }

  std::string_view GetNodeTypeName() const override { return "NameDefTree"; }
  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override;

  bool is_leaf() const { return std::holds_alternative<Leaf>(tree_); }
  Leaf leaf() const { return std::get<Leaf>(tree_); }

  const Nodes& nodes() const { return std::get<Nodes>(tree_); }

  // Flattens this NameDefTree a single level, unwrapping any leaf NDTs; e.g.
  //
  //    LEAF:a => [LEAF:a]
  //    LEAF:a, NODES:[NDT:b, NDT:c], LEAF:d => [LEAF:a, NDT:b, NDT:c, LEAF:d]
  //    NODES:[NDT:a, NDT:LEAF:c], NODES[NDT:c] => [NDT:a, NDT:b, LEAF:c, NDT:d]
  //
  // This is useful for flattening a tuple a single level; e.g. where a
  // NameDefTree is going to be used as variadic args in for-loop to function
  // conversion.
  std::vector<std::variant<Leaf, NameDefTree*>> Flatten1() const;

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
      return std::holds_alternative<NameDef*>(leaf) ||
             std::holds_alternative<WildcardPattern*>(leaf);
    });
  }

  bool IsWildcardLeaf() const {
    return is_leaf() && std::holds_alternative<WildcardPattern*>(leaf());
  }

  bool IsRestOfTupleLeaf() const {
    return is_leaf() && std::holds_alternative<RestOfTuple*>(leaf());
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

  [[maybe_unused]] const std::variant<Nodes, Leaf>& tree() const {
    return tree_;
  }
  const Span& span() const { return span_; }
  std::optional<Span> GetSpan() const override { return span_; }

 private:
  Span span_;
  std::variant<Nodes, Leaf> tree_;
};

// Represents a let-binding expression.
class Let : public AstNode {
 public:
  // A Let's body can be nullopt if it's the last expr in an unroll_for body.
  Let(Module* owner, Span span, NameDefTree* name_def_tree,
      TypeAnnotation* type, Expr* rhs, bool is_const);

  ~Let() override;

  AstNodeKind kind() const override { return AstNodeKind::kLet; }

  std::optional<Span> GetSpan() const override { return span_; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleLet(this);
  }

  std::string_view GetNodeTypeName() const override { return "Let"; }
  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override;

  NameDefTree* name_def_tree() const { return name_def_tree_; }
  TypeAnnotation* type_annotation() const { return type_annotation_; }
  Expr* rhs() const { return rhs_; }
  bool is_const() const { return is_const_; }
  const Span& span() const { return span_; }

 private:
  Span span_;

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

  // Whether or not this is a constant binding; constant bindings cannot be
  // shadowed.
  bool is_const_;
};

// Used to represent a named reference to a Constant name definition.
class ConstRef : public NameRef {
 public:
  using NameRef::NameRef;

  ~ConstRef() override;

  AstNodeKind kind() const override { return AstNodeKind::kConstRef; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleConstRef(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleConstRef(this);
  }

  std::string_view GetNodeTypeName() const override { return "ConstRef"; }

  // When holding a ConstRef we know that the corresponding NameDef cannot be
  // builtin (since consts are user constructs).
  const NameDef* name_def() const {
    return std::get<const NameDef*>(NameRef::name_def());
  }

  Expr* GetValue() const {
    AstNode* definer = name_def()->definer();
    CHECK(definer != nullptr);
    // Definer will only ever be a ConstantDef or a Let.
    if (ConstantDef* cd = dynamic_cast<ConstantDef*>(definer); cd != nullptr) {
      return cd->value();
    }

    Let* let = dynamic_cast<Let*>(definer);
    CHECK_NE(let, nullptr);
    return let->rhs();
  }
};

// A channel declaration, e.g., `let (p, c) = chan<u32>("my_chan");`
// -------------------------------------------^^^^^^^^^^^^^^^^^^^^ this part.
class ChannelDecl : public Expr {
 public:
  ChannelDecl(Module* owner, Span span, TypeAnnotation* type,
              std::optional<std::vector<Expr*>> dims,
              std::optional<Expr*> fifo_depth, Expr& channel_name_expr)
      : Expr(owner, std::move(span)),
        type_(type),
        dims_(std::move(dims)),
        fifo_depth_(fifo_depth),
        channel_name_expr_(channel_name_expr) {}

  ~ChannelDecl() override;

  AstNodeKind kind() const override { return AstNodeKind::kChannelDecl; }

  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleChannelDecl(this);
  }
  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleChannelDecl(this);
  }

  std::string_view GetNodeTypeName() const override { return "ChannelDecl"; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    if (want_types) {
      return {type_, &channel_name_expr_};
    }
    return {&channel_name_expr_};
  }

  TypeAnnotation* type() const { return type_; }
  const std::optional<std::vector<Expr*>>& dims() const { return dims_; }
  std::optional<Expr*> fifo_depth() const { return fifo_depth_; }
  Expr& channel_name_expr() const { return channel_name_expr_; }

  Precedence GetPrecedenceWithoutParens() const final {
    return Precedence::kStrongest;
  }

 private:
  std::string ToStringInternal() const final;

  TypeAnnotation* type_;
  std::optional<std::vector<Expr*>> dims_;
  std::optional<Expr*> fifo_depth_;
  Expr& channel_name_expr_;
};

// A node that contains original source text only; it is typically used by the
// formatter.
class VerbatimNode : public AstNode {
 public:
  VerbatimNode(Module* owner, Span span, const std::string text)
      : AstNode(owner), span_(std::move(span)), text_(std::move(text)) {}
  VerbatimNode(Module* owner, Span span)
      : AstNode(owner), span_(std::move(span)), text_("") {}

  ~VerbatimNode() override;

  std::string text() const { return text_; }
  bool IsEmpty() const { return text_.empty(); }

  AstNodeKind kind() const override { return AstNodeKind::kVerbatimNode; }

  std::string_view GetNodeTypeName() const override { return "VerbatimNode"; }

  std::string ToString() const override { return text_; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {};
  }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleVerbatimNode(this);
  }

  const Span& span() const { return span_; }
  std::optional<Span> GetSpan() const override { return span_; }

 private:
  Span span_;
  std::string text_;
};

// Helper for determining whether an AST node is constant (e.g. can be
// considered a constant value in a ConstantArray).
bool IsConstant(AstNode* n);

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_AST_H_
