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
#include "xls/ir/format_strings.h"

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
  X(Block)                         \
  X(Cast)                          \
  X(ChannelDecl)                   \
  X(ColonRef)                      \
  X(ConstantArray)                 \
  X(ConstRef)                      \
  X(For)                           \
  X(FormatMacro)                   \
  X(Index)                         \
  X(Invocation)                    \
  X(Join)                          \
  X(Let)                           \
  X(Match)                         \
  X(NameRef)                       \
  X(Number)                        \
  X(Range)                         \
  X(Recv)                          \
  X(RecvIf)                        \
  X(RecvIfNonBlocking)             \
  X(RecvNonBlocking)               \
  X(Send)                          \
  X(SendIf)                        \
  X(Spawn)                         \
  X(SplatStructInstance)           \
  X(String)                        \
  X(StructInstance)                \
  X(Ternary)                       \
  X(TupleIndex)                    \
  X(Unop)                          \
  X(UnrollFor)                     \
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
  X(TestProc)                     \
  X(TypeDef)                      \
  X(TypeRef)                      \
  X(WidthSlice)                   \
  X(WildcardPattern)              \
  /* type annotations */          \
  X(ArrayTypeAnnotation)          \
  X(BuiltinTypeAnnotation)        \
  X(ChannelTypeAnnotation)        \
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
  virtual absl::Status Handle##__type(const __type* n) = 0;
  XLS_DSLX_AST_NODE_EACH(DECLARE_HANDLER)
#undef DECLARE_HANDLER
};

// Subtype of abstract AstNodeVisitor that returns ok status (does nothing) for
// every node type.
class AstNodeVisitorWithDefault : public AstNodeVisitor {
 public:
  ~AstNodeVisitorWithDefault() override = default;

#define DECLARE_HANDLER(__type)                           \
  absl::Status Handle##__type(const __type* n) override { \
    return absl::OkStatus();                              \
  }
  XLS_DSLX_AST_NODE_EACH(DECLARE_HANDLER)
#undef DECLARE_HANDLER
};

// Name definitions can be either built in (BuiltinNameDef, in which case they
// have no effective position) or defined in the user AST (NameDef).
using AnyNameDef = std::variant<const NameDef*, BuiltinNameDef*>;

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

 private:
  absl::flat_hash_map<std::string, std::vector<const NameRef*>> values_;
};

// Enum with an entry for each leaf type in the AST class hierarchy -- this is
// primarily for convenience in tasks like serialization, for most purposes
// visitors should be used (e.g. AstNodeVisitor, ExprVisitor).
enum class AstNodeKind {
  kTypeAnnotation,
  kModule,
  kNameDef,
  kBuiltinNameDef,
  kTernary,
  kTypeDef,
  kNumber,
  kTypeRef,
  kImport,
  kUnop,
  kBinop,
  kColonRef,
  kParam,
  kFunction,
  kProc,
  kNameRef,
  kConstRef,
  kArray,
  kString,
  kStructInstance,
  kSplatStructInstance,
  kNameDefTree,
  kIndex,
  kRange,
  kRecv,
  kRecvIf,
  kRecvIfNonBlocking,
  kRecvNonBlocking,
  kSend,
  kSendIf,
  kJoin,
  kTestFunction,
  kTestProc,
  kWidthSlice,
  kWildcardPattern,
  kMatchArm,
  kMatch,
  kAttr,
  kInstantiation,
  kInvocation,
  kSpawn,
  kFormatMacro,
  kSlice,
  kEnumDef,
  kStructDef,
  kQuickCheck,
  kXlsTuple,
  kFor,
  kBlock,
  kCast,
  kConstantDef,
  kLet,
  kChannelDecl,
  kParametricBinding,
  kTupleIndex,
  kUnrollFor,
};

std::string_view AstNodeKindToString(AstNodeKind kind);

class AstNode;

// Abstract base class for AST nodes.
class AstNode {
 public:
  explicit AstNode(Module* owner) : owner_(owner) {}
  virtual ~AstNode();

  virtual AstNodeKind kind() const = 0;

  // Retrieves the name of the leafmost-derived class, suitable for debugging;
  // e.g. "NameDef", "BuiltinTypeAnnotation", etc.
  virtual std::string_view GetNodeTypeName() const = 0;
  virtual std::string ToString() const = 0;

  virtual std::optional<Span> GetSpan() const = 0;

  AstNode* parent() const { return parent_; }

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
  virtual absl::Status Accept(AstNodeVisitor* v) const = 0;

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
  FreeVariables GetFreeVariables(const Pos* start_pos = nullptr) const;

  Module* owner() const { return owner_; }

  // Marks this node as the parent of all its child nodes.
  void SetParentage();

 private:
  void set_parent(AstNode* parent) { parent_ = parent; }

  Module* owner_;
  AstNode* parent_ = nullptr;
};

// Visits transitively from the root down using post-order visitation (visit
// children, then node). want_types is as in AstNode::GetChildren().
absl::Status WalkPostOrder(AstNode* root, AstNodeVisitor* visitor,
                           bool want_types);

// Helpers for converting variants of "AstNode subtype" pointers and their
// variants to the base `AstNode*`.
template <typename... Types>
inline AstNode* ToAstNode(const std::variant<Types...>& v) {
  return absl::ConvertVariantTo<AstNode*>(v);
}
inline AstNode* ToAstNode(AstNode* n) { return n; }

// As above, but for Expr base.
template <typename... Types>
inline Expr* ToExprNode(const std::variant<Types...>& v) {
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

  ~TypeAnnotation() override;

  AstNodeKind kind() const override { return AstNodeKind::kTypeAnnotation; }
  const Span& span() const { return span_; }
  std::optional<Span> GetSpan() const override { return span_; }

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
absl::StatusOr<BuiltinType> BuiltinTypeFromString(std::string_view s);

absl::StatusOr<BuiltinType> GetBuiltinType(bool is_signed, int64_t width);
absl::StatusOr<bool> GetBuiltinTypeSignedness(BuiltinType type);
absl::StatusOr<int64_t> GetBuiltinTypeBitCount(BuiltinType type);

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

  int64_t GetBitCount() const;

  // Returns true if signed, false if unsigned.
  bool GetSignedness() const;

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
                        std::vector<Expr*> parametrics);

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
  std::optional<Span> GetSpan() const override { return absl::nullopt; }

  std::string_view GetNodeTypeName() const override {
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

// Abstract base class for AST node that can appear in expression positions
// (i.e. can produce runtime values).
class Expr : public AstNode {
 public:
  Expr(Module* owner, Span span) : AstNode(owner), span_(span) {}

  ~Expr() override;

  const Span& span() const { return span_; }
  void set_span(const Span& span) { span_ = span; }
  std::optional<Span> GetSpan() const override { return span_; }

  virtual absl::Status AcceptExpr(ExprVisitor* v) const = 0;

 private:
  Span span_;
};

// ChannelTypeAnnotation has to be placed after the definition of Expr, so it
// can convert `dims_` to a set of AstNodes.
class ChannelTypeAnnotation : public TypeAnnotation {
 public:
  enum Direction {
    kIn,
    kOut,
  };

  // If this is a scalar channel, then `dims` will be nullopt.
  ChannelTypeAnnotation(Module* owner, Span span, Direction direction,
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

  Direction direction() const { return direction_; }
  TypeAnnotation* payload() const { return payload_; }

  // A ChannelTypeAnnotation needs to keep its own dims (rather than being
  // enclosed in an ArrayTypeAnnotation simply because it prints itself in a
  // different manner than an array does - we want `chan<u32>[32] in` rather
  // than `chan in u32[32]` for a 32-channel declaration. The former declares 32
  // channels, each of which transmits a u32, whereas the latter declares a
  // single channel that transmits a 32-element array of u32s.
  std::optional<std::vector<Expr*>> dims() const { return dims_; }

 private:
  Direction direction_;
  TypeAnnotation* payload_;
  std::optional<std::vector<Expr*>> dims_;
};

// Represents a block expression, e.g.,
// let i = {
//   u32:5
// };
class Block : public Expr {
 public:
  Block(Module* owner, Span span, Expr* body)
      : Expr(owner, std::move(span)), body_(body) {}

  ~Block() override;

  AstNodeKind kind() const override { return AstNodeKind::kBlock; }
  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleBlock(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleBlock(this);
  }
  std::string_view GetNodeTypeName() const override { return "Block"; }
  std::string ToString() const override;
  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {body_};
  }

  Expr* body() const { return body_; }

 private:
  Expr* body_;
};

// Represents a reference to a name (identifier).
class NameRef : public Expr {
 public:
  NameRef(Module* owner, Span span, std::string identifier, AnyNameDef name_def)
      : Expr(owner, std::move(span)),
        name_def_(name_def),
        identifier_(std::move(identifier)) {}

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
  std::string ToString() const override { return identifier_; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {};
  }

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
    return absl::nullopt;
  }

  std::variant<const NameDef*, BuiltinNameDef*> name_def() const {
    return name_def_;
  }

 private:
  AnyNameDef name_def_;
  std::string identifier_;
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

  NumberKind number_kind() const { return number_kind_; }

 private:
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

  ~TypeDef() override;

  AstNodeKind kind() const override { return AstNodeKind::kTypeDef; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleTypeDef(this);
  }
  std::string_view GetNodeTypeName() const override { return "TypeDef"; }
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
  std::optional<Span> GetSpan() const override { return span_; }

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

  ~Array() override;

  AstNodeKind kind() const override { return AstNodeKind::kArray; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleArray(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleArray(this);
  }

  std::string_view GetNodeTypeName() const override { return "Array"; }
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

  ~ConstantArray() override;

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleConstantArray(this);
  }
};

// Several different AST nodes define types that can be referred to by a
// TypeRef.
using TypeDefinition = std::variant<TypeDef*, StructDef*, EnumDef*, ColonRef*>;

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

  ~TypeRef() override;

  AstNodeKind kind() const override { return AstNodeKind::kTypeRef; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleTypeRef(this);
  }

  std::string_view GetNodeTypeName() const override { return "TypeRef"; }
  std::string ToString() const override { return text_; }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {};
  }

  const std::string& text() const { return text_; }
  const TypeDefinition& type_definition() const { return type_definition_; }
  const Span& span() const { return span_; }
  std::optional<Span> GetSpan() const override { return span_; }

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
         NameDef* name_def, std::optional<std::string> alias);

  ~Import() override;

  AstNodeKind kind() const override { return AstNodeKind::kImport; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleImport(this);
  }
  std::string_view GetNodeTypeName() const override { return "Import"; }
  const std::string& identifier() const { return name_def_->identifier(); }

  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {name_def_};
  }

  const std::vector<std::string>& subject() const { return subject_; }
  NameDef* name_def() const { return name_def_; }
  const Span& span() const { return span_; }
  std::optional<Span> GetSpan() const override { return span_; }
  std::optional<std::string> alias() const { return alias_; }

 private:
  // Span of the import in the text.
  Span span_;
  // Name of the module being imported ("original" name before aliasing); e.g.
  // "std".
  std::vector<std::string> subject_;
  // The name definition we bind the import to.
  NameDef* name_def_;
  // The identifier text we bind the import to.
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
  std::optional<Import*> ResolveImportSubject() const;

 private:
  Subject subject_;
  std::string attr_;
};

absl::StatusOr<ColonRef::Subject> ToColonRefSubject(Expr* e);

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

enum class UnopKind {
  kInvert,  // one's complement inversion (bit flip)
  kNegate,  // two's complement aritmetic negation (~x+1)
};

absl::StatusOr<UnopKind> UnopKindFromString(std::string_view s);
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
  std::string ToString() const override {
    return absl::StrFormat("%s(%s)", UnopKindToString(unop_kind_),
                           operand_->ToString());
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {operand_};
  }

  UnopKind unop_kind() const { return unop_kind_; }
  Expr* operand() const { return operand_; }

 private:
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
  X(kLogicalAnd, "LOGICAL_AND", "&&")     \
  X(kLogicalOr, "LOGICAL_OR", "||")       \
  X(kConcat, "CONCAT", "++")

enum class BinopKind {
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

  std::string ToString() const override {
    return absl::StrFormat("(%s) %s (%s)", lhs_->ToString(),
                           BinopKindFormat(binop_kind_), rhs_->ToString());
  }

  BinopKind binop_kind() const { return binop_kind_; }
  Expr* lhs() const { return lhs_; }
  Expr* rhs() const { return rhs_; }

 private:
  BinopKind binop_kind_;
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

  ~Ternary() override;

  AstNodeKind kind() const override { return AstNodeKind::kTernary; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleTernary(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleTernary(this);
  }

  std::string_view GetNodeTypeName() const override { return "Ternary"; }
  std::string ToString() const override;

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

class Proc;

class Function : public AstNode {
 public:
  // Indicates if a function is normal or is part of a proc instantiation.
  enum class Tag {
    kNormal,
    kProcConfig,
    kProcNext,
    kProcInit,
  };

  Function(Module* owner, Span span, NameDef* name_def,
           std::vector<ParametricBinding*> parametric_bindings,
           std::vector<Param*> params, TypeAnnotation* return_type, Block* body,
           Tag tag, bool is_public);
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
  Block* body() const { return body_; }

  bool IsParametric() const { return !parametric_bindings_.empty(); }
  bool is_public() const { return is_public_; }
  std::vector<std::string> GetFreeParametricKeys() const;
  absl::btree_set<std::string> GetFreeParametricKeySet() const {
    std::vector<std::string> keys = GetFreeParametricKeys();
    return absl::btree_set<std::string>(keys.begin(), keys.end());
  }
  NameDef* name_def() const { return name_def_; }
  TypeAnnotation* return_type() const { return return_type_; }
  void set_return_type(TypeAnnotation* return_type) {
    return_type_ = return_type;
  }

  Tag tag() const { return tag_; }
  std::optional<Proc*> proc() const { return proc_; }
  void set_proc(Proc* proc) { proc_ = proc; }

 private:
  Span span_;
  NameDef* name_def_;
  std::vector<ParametricBinding*> parametric_bindings_;
  std::vector<Param*> params_;
  TypeAnnotation* return_type_;  // May be null.
  Block* body_;
  Tag tag_;
  std::optional<Proc*> proc_;

  bool is_public_;
};

// Represents a parsed 'process' specification in the DSL.
class Proc : public AstNode {
 public:
  Proc(Module* owner, Span span, NameDef* name_def, NameDef* config_name_def,
       NameDef* next_name_def,
       const std::vector<ParametricBinding*>& parametric_bindings,
       const std::vector<Param*>& members, Function* config, Function* next,
       Function* init, bool is_public);

  ~Proc() override;

  AstNodeKind kind() const override { return AstNodeKind::kProc; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleProc(this);
  }
  std::string_view GetNodeTypeName() const override { return "Proc"; }
  std::string ToString() const override;
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  NameDef* name_def() const { return name_def_; }
  NameDef* config_name_def() const { return config_name_def_; }
  NameDef* next_name_def() const { return next_name_def_; }
  const Span& span() const { return span_; }
  std::optional<Span> GetSpan() const override { return span_; }

  const std::string& identifier() const { return name_def_->identifier(); }
  const std::vector<ParametricBinding*>& parametric_bindings() const {
    return parametric_bindings_;
  }
  bool IsParametric() const { return !parametric_bindings_.empty(); }
  bool is_public() const { return is_public_; }

  std::vector<std::string> GetFreeParametricKeys() const;
  absl::btree_set<std::string> GetFreeParametricKeySet() const {
    std::vector<std::string> keys = GetFreeParametricKeys();
    return absl::btree_set<std::string>(keys.begin(), keys.end());
  }

  Function* config() const { return config_; }
  Function* next() const { return next_; }
  Function* init() const { return init_; }
  const std::vector<Param*>& members() const { return members_; }

 private:
  Span span_;
  NameDef* name_def_;
  NameDef* config_name_def_;
  NameDef* next_name_def_;
  std::vector<ParametricBinding*> parametric_bindings_;

  Function* config_;
  Function* next_;
  Function* init_;
  std::vector<Param*> members_;
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
// (prioritzed in sequential order from first arm to last arm).
class Match : public Expr {
 public:
  Match(Module* owner, Span span, Expr* matched, std::vector<MatchArm*> arms);

  ~Match() override;

  AstNodeKind kind() const override { return AstNodeKind::kMatch; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleMatch(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleMatch(this);
  }

  std::string_view GetNodeTypeName() const override { return "Match"; }
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

  ~Attr() override;

  AstNodeKind kind() const override { return AstNodeKind::kAttr; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleAttr(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleAttr(this);
  }

  std::string_view GetNodeTypeName() const override { return "Attr"; }
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

class Instantiation : public Expr {
 public:
  Instantiation(Module* owner, Span span, Expr* callee,
                const std::vector<Expr*>& explicit_parametrics);

  ~Instantiation();

  AstNodeKind kind() const override { return AstNodeKind::kInstantiation; }

  Expr* callee() const { return callee_; }

  // Any explicit parametric expressions given in this invocation; e.g. in:
  //
  //    f<a, b, c>()
  //
  // The expressions a, b, c would be in this sequence.
  const std::vector<Expr*>& explicit_parametrics() const {
    return explicit_parametrics_;
  }

 protected:
  std::string FormatParametrics() const;

 private:
  Expr* callee_;
  std::vector<Expr*> explicit_parametrics_;
};

// Represents an invocation expression; e.g. `f(a, b, c)` or an implicit
// invocation for the config & next members of a spawned Proc.
class Invocation : public Instantiation {
 public:
  Invocation(Module* owner, Span span, Expr* callee, std::vector<Expr*> args,
             std::vector<Expr*> explicit_parametrics = std::vector<Expr*>({}));

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

  std::string ToString() const override {
    return absl::StrFormat("%s%s(%s)", callee()->ToString(),
                           FormatParametrics(), FormatArgs());
  };

  const absl::Span<Expr* const> args() const { return args_; }

 private:
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
        Invocation* next, std::vector<Expr*> explicit_parametrics, Expr* body);

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
  std::string ToString() const override;

  Invocation* config() const { return config_; }
  Invocation* next() const { return next_; }
  bool IsParametric() { return !explicit_parametrics().empty(); }
  Expr* body() const { return body_; }

 private:
  Invocation* config_;
  Invocation* next_;
  Expr* body_;
};

// Represents a call to a variable-argument formatting macro; e.g. trace_fmt!("x
// is {}", x)
class FormatMacro : public Expr {
 public:
  FormatMacro(Module* owner, Span span, std::string macro,
              std::vector<FormatStep> format, std::vector<Expr*> args);

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

  std::string ToString() const override;

  const std::string macro() const { return macro_; }
  const absl::Span<Expr* const> args() const { return args_; }
  const std::vector<FormatStep> format() const { return format_; }

 private:
  std::string macro_;
  std::vector<FormatStep> format_;
  std::vector<Expr*> args_;
};

// Represents a slice in the AST.
//
// For example, we can have `x[-4:-2]`, where x is of bit width N.
// The start and limit Exprs must be constexpr.
class Slice : public AstNode {
 public:
  Slice(Module* owner, Span span, Expr* start, Expr* limit)
      : AstNode(owner), span_(std::move(span)), start_(start), limit_(limit) {}

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
  TypeAnnotation* type_annotation() const { return type_annotation_; }
  bool is_public() const { return is_public_; }

  const std::string& GetMemberName(int64_t i) const {
    return values_.at(i).name_def->identifier();
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
};

// Represents a struct definition.
class StructDef : public AstNode {
 public:
  StructDef(Module* owner, Span span, NameDef* name_def,
            std::vector<ParametricBinding*> parametric_bindings,
            std::vector<std::pair<NameDef*, TypeAnnotation*>> members,
            bool is_public);

  ~StructDef() override;

  AstNodeKind kind() const override { return AstNodeKind::kStructDef; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleStructDef(this);
  }

  bool IsParametric() const { return !parametric_bindings_.empty(); }

  const std::string& identifier() const { return name_def_->identifier(); }

  std::string_view GetNodeTypeName() const override { return "Struct"; }
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
  std::optional<Span> GetSpan() const override { return span_; }

  const std::string& GetMemberName(int64_t i) const {
    return members_[i].first->identifier();
  }
  std::vector<std::string> GetMemberNames() const;

  // Returns the index at which the member name is "name".
  std::optional<int64_t> GetMemberIndex(std::string_view name) const;

  int64_t size() const { return members_.size(); }

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
using StructRef = std::variant<StructDef*, ColonRef*>;

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

  ~StructInstance() override;

  AstNodeKind kind() const override { return AstNodeKind::kStructInstance; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleStructInstance(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleStructInstance(this);
  }

  std::string_view GetNodeTypeName() const override {
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
      const StructDef* struct_def) const;

  // Returns the expression associated with the member named "name", or a
  // NotFound error status if none exists.
  absl::StatusOr<Expr*> GetExpr(std::string_view name) const;

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

  ~SplatStructInstance() override;

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

  ~WidthSlice() override;

  AstNodeKind kind() const override { return AstNodeKind::kWidthSlice; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleWidthSlice(this);
  }
  std::string_view GetNodeTypeName() const override { return "WidthSlice"; }
  std::string ToString() const override;

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

  ~Index() override;

  AstNodeKind kind() const override { return AstNodeKind::kIndex; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleIndex(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleIndex(this);
  }

  std::string_view GetNodeTypeName() const override { return "Index"; }
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
  std::string ToString() const override;
  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {start_, end_};
  }

  Expr* start() const { return start_; }
  Expr* end() const { return end_; }

 private:
  Expr* start_;
  Expr* end_;
};

// Represents a recv node: the mechanism by which a proc gets info from another
// proc.
class Recv : public Expr {
 public:
  Recv(Module* owner, Span span, NameRef* token, Expr* channel);

  ~Recv() override;

  AstNodeKind kind() const override { return AstNodeKind::kRecv; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleRecv(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleRecv(this);
  }

  std::string_view GetNodeTypeName() const override { return "Recv"; }
  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {token_, channel_};
  }

  NameRef* token() const { return token_; }
  Expr* channel() const { return channel_; }

 private:
  NameRef* token_;
  Expr* channel_;
};

// Represents a non-blocking recv node: the mechanism by which a proc gets info
// from another proc.
class RecvNonBlocking : public Expr {
 public:
  RecvNonBlocking(Module* owner, Span span, NameRef* token, Expr* channel);

  ~RecvNonBlocking() override;

  AstNodeKind kind() const override { return AstNodeKind::kRecvNonBlocking; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleRecvNonBlocking(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleRecvNonBlocking(this);
  }

  std::string_view GetNodeTypeName() const override {
    return "RecvNonBlocking";
  }

  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {token_, channel_};
  }

  NameRef* token() const { return token_; }
  Expr* channel() const { return channel_; }

 private:
  NameRef* token_;
  Expr* channel_;
};

// A RecvIf is a recv node that's guarded by a condition: the send will be
// performed only if the condition is true.
class RecvIf : public Expr {
 public:
  RecvIf(Module* owner, Span span, NameRef* token, Expr* channel,
         Expr* condition);

  ~RecvIf() override;

  AstNodeKind kind() const override { return AstNodeKind::kRecvIf; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleRecvIf(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleRecvIf(this);
  }

  std::string_view GetNodeTypeName() const override { return "RecvIf"; }
  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {token_, channel_, condition_};
  }

  NameRef* token() const { return token_; }
  Expr* channel() const { return channel_; }
  Expr* condition() const { return condition_; }

 private:
  NameRef* token_;
  Expr* channel_;
  Expr* condition_;
};

// Represents a non-blocking recv node: the mechanism by which a proc gets info
// from another proc. Returns a three-tuple: (token, value, value_valid).
// If the condition is false or if no value is available,
// then a zero-valued result is produced. In both cases, value_valid will be
// false.
class RecvIfNonBlocking : public Expr {
 public:
  RecvIfNonBlocking(Module* owner, Span span, NameRef* token, Expr* channel,
                    Expr* condition);

  ~RecvIfNonBlocking() override;

  AstNodeKind kind() const override { return AstNodeKind::kRecvIfNonBlocking; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleRecvIfNonBlocking(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleRecvIfNonBlocking(this);
  }

  std::string_view GetNodeTypeName() const override {
    return "RecvIfNonBlocking";
  }

  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {token_, channel_, condition_};
  }

  NameRef* token() const { return token_; }
  Expr* channel() const { return channel_; }
  Expr* condition() const { return condition_; }

 private:
  NameRef* token_;
  Expr* channel_;
  Expr* condition_;
};

// Represents a send node: the mechanism by which a proc sends info to another
// proc.
// Sends are really _statements_, vs. expressions, but the language doesn't
// really make that distinction.
class Send : public Expr {
 public:
  Send(Module* owner, Span span, NameRef* token, Expr* channel, Expr* payload);

  ~Send() override;

  AstNodeKind kind() const override { return AstNodeKind::kSend; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleSend(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleSend(this);
  }

  std::string_view GetNodeTypeName() const override { return "Send"; }
  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {token_, channel_, payload_};
  }

  NameRef* token() const { return token_; }
  Expr* channel() const { return channel_; }
  Expr* payload() const { return payload_; }

 private:
  NameRef* token_;
  Expr* channel_;
  Expr* payload_;
};

// A SendIf is a send node that's guarded by a condition: the send will be
// performed only if the condition is true.
class SendIf : public Expr {
 public:
  SendIf(Module* owner, Span span, NameRef* token, Expr* channel,
         Expr* condition, Expr* payload);

  ~SendIf() override;

  AstNodeKind kind() const override { return AstNodeKind::kSendIf; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleSendIf(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleSendIf(this);
  }

  std::string_view GetNodeTypeName() const override { return "SendIf"; }
  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {token_, channel_, condition_, payload_};
  }

  NameRef* token() const { return token_; }
  Expr* channel() const { return channel_; }
  Expr* condition() const { return condition_; }
  Expr* payload() const { return payload_; }

 private:
  NameRef* token_;
  Expr* channel_;
  Expr* condition_;
  Expr* payload_;
};

// Represents a "join" expression: this "returns" a token that is only
// satisfied once all argument tokens have been satisfied. In this way, an
// operation can be configured to only act once multiple predecessor operations
// have completed.
class Join : public Expr {
 public:
  Join(Module* owner, Span span, const std::vector<Expr*>& tokens);
  ~Join() override;
  AstNodeKind kind() const override { return AstNodeKind::kJoin; }
  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleJoin(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleJoin(this);
  }
  std::string_view GetNodeTypeName() const override { return "Join"; }
  std::string ToString() const override;
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  const std::vector<Expr*>& tokens() const { return tokens_; }

 private:
  std::vector<Expr*> tokens_;
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
  explicit TestFunction(Module* owner, Function* fn)
      : AstNode(owner), name_def_(fn->name_def()), body_(fn->body()), fn_(fn) {}

  ~TestFunction() override;

  AstNodeKind kind() const override { return AstNodeKind::kTestFunction; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleTestFunction(this);
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {name_def_};
  }

  std::string_view GetNodeTypeName() const override { return "TestFunction"; }
  std::string ToString() const override {
    return absl::StrFormat("#[test]\n%s", fn_->ToString());
  }

  Function* fn() const { return fn_; }
  std::optional<Span> GetSpan() const override { return fn_->span(); }

  NameDef* name_def() const { return name_def_; }
  const std::string& identifier() const { return name_def_->identifier(); }
  Expr* body() const { return body_; }

 private:
  NameDef* name_def_;
  Expr* body_;
  Function* fn_;
};

// Represents a construct to unit test a Proc. Analogous to TestFunction, but
// for Procs.
//
// These are specified with an annotation as follows:
// ```dslx
// #[test_proc()]
// proc test_proc { ... }
// ```
class TestProc : public AstNode {
 public:
  TestProc(Module* owner, Proc* proc) : AstNode(owner), proc_(proc) {}
  ~TestProc() override;

  AstNodeKind kind() const override { return AstNodeKind::kTestProc; }
  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleTestProc(this);
  }
  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {proc_};
  }
  std::string_view GetNodeTypeName() const override { return "TestProc"; }
  std::string ToString() const override;

  Proc* proc() const { return proc_; }
  std::optional<Span> GetSpan() const override { return proc_->span(); }

  const std::string& identifier() const { return proc_->identifier(); }

 private:
  Proc* proc_;
};

// Represents a function to be quick-check'd.
class QuickCheck : public AstNode {
 public:
  static constexpr int64_t kDefaultTestCount = 1000;

  QuickCheck(Module* owner, Span span, Function* f,
             std::optional<int64_t> test_count = absl::nullopt);

  ~QuickCheck() override;

  AstNodeKind kind() const override { return AstNodeKind::kQuickCheck; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleQuickCheck(this);
  }

  std::string_view GetNodeTypeName() const override { return "QuickCheck"; }
  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {f_};
  }

  const std::string& identifier() const { return f_->identifier(); }

  Function* f() const { return f_; }
  int64_t test_count() const {
    return test_count_ ? *test_count_ : kDefaultTestCount;
  }
  std::optional<Span> GetSpan() const override { return f_->span(); }

 private:
  Span span_;
  Function* f_;
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
  std::string ToString() const override;
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  Expr* lhs() const { return lhs_; }
  Number* index() const { return index_; }

 private:
  Expr* lhs_;
  Number* index_;
};

// Represents an XLS tuple expression.
class XlsTuple : public Expr {
 public:
  XlsTuple(Module* owner, Span span, std::vector<Expr*> members)
      : Expr(owner, std::move(span)), members_(members) {}

  ~XlsTuple() override;

  AstNodeKind kind() const override { return AstNodeKind::kXlsTuple; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleXlsTuple(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleXlsTuple(this);
  }

  std::string_view GetNodeTypeName() const override { return "XlsTuple"; }
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
      Expr* iterable, Block* body, Expr* init);

  ~For() override;

  AstNodeKind kind() const override { return AstNodeKind::kFor; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleFor(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleFor(this);
  }

  std::string_view GetNodeTypeName() const override { return "For"; }
  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override;

  // Names bound in the body of the loop.
  NameDefTree* names() const { return names_; }

  // Annotation corresponding to "names".
  TypeAnnotation* type_annotation() const { return type_annotation_; }

  // Expression for "thing to iterate over".
  Expr* iterable() const { return iterable_; }

  // Expression for the loop body.
  Block* body() const { return body_; }

  // Initial expression for the loop (start values expr).
  Expr* init() const { return init_; }

 private:
  NameDefTree* names_;
  TypeAnnotation* type_annotation_;
  Expr* iterable_;
  Block* body_;
  Expr* init_;
};

// Represents an operation to "unroll" the given for-like expression by the
// number of elements in the given iterable.
class UnrollFor : public Expr {
 public:
  UnrollFor(Module* owner, Span span, NameDefTree* names, TypeAnnotation* types,
            Expr* iterable, Block* body, Expr* init);
  ~UnrollFor() override;
  AstNodeKind kind() const override { return AstNodeKind::kUnrollFor; }
  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleUnrollFor(this);
  }
  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleUnrollFor(this);
  }
  std::string_view GetNodeTypeName() const override { return "unroll-for"; }
  std::string ToString() const override;
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  NameDefTree* names() const { return names_; }
  TypeAnnotation* types() const { return types_; }
  Expr* iterable() const { return iterable_; }
  Block* body() const { return body_; }
  Expr* init() const { return init_; }

 private:
  NameDefTree* names_;
  TypeAnnotation* types_;
  Expr* iterable_;
  Block* body_;
  Expr* init_;
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

// Represents a constant definition.
//
//  is_public: Indicates whether the constant had a public annotation
//    (applicable to module level constant definitions only)
class ConstantDef : public AstNode {
 public:
  ConstantDef(Module* owner, Span span, NameDef* name_def, Expr* value,
              bool is_public);

  ~ConstantDef() override;

  AstNodeKind kind() const override { return AstNodeKind::kConstantDef; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleConstantDef(this);
  }

  std::string_view GetNodeTypeName() const override { return "ConstantDef"; }
  std::string ToString() const override;
  std::string ToReprString() const;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {name_def_, value_};
  }

  const std::string& identifier() const { return name_def_->identifier(); }
  NameDef* name_def() const { return name_def_; }
  Expr* value() const { return value_; }
  const Span& span() const { return span_; }
  std::optional<Span> GetSpan() const override { return span_; }
  bool is_public() const { return is_public_; }

 private:
  Span span_;
  NameDef* name_def_;
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
  using Leaf =
      std::variant<NameDef*, NameRef*, WildcardPattern*, Number*, ColonRef*>;

  NameDefTree(Module* owner, Span span, std::variant<Nodes, Leaf> tree)
      : AstNode(owner), span_(std::move(span)), tree_(tree) {}

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
  std::vector<std::variant<Leaf, NameDefTree*>> Flatten1();

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

  const std::variant<Nodes, Leaf>& tree() const { return tree_; }
  const Span& span() const { return span_; }
  std::optional<Span> GetSpan() const override { return span_; }

 private:
  Span span_;
  std::variant<Nodes, Leaf> tree_;
};

// Represents a let-binding expression.
class Let : public Expr {
 public:
  // A Let's body can be nullopt if it's the last expr in an unroll_for body.
  Let(Module* owner, Span span, NameDefTree* name_def_tree,
      TypeAnnotation* type, Expr* rhs, Expr* body, bool is_const);

  ~Let() override;

  AstNodeKind kind() const override { return AstNodeKind::kLet; }

  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleLet(this);
  }
  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleLet(this);
  }

  std::string_view GetNodeTypeName() const override { return "Let"; }
  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override;

  NameDefTree* name_def_tree() const { return name_def_tree_; }
  TypeAnnotation* type_annotation() const { return type_annotation_; }
  Expr* rhs() const { return rhs_; }
  Expr* body() const { return body_; }
  bool is_const() const { return is_const_; }

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

  // Returns the constant definition that this ConstRef is referring to.
  ConstantDef* GetConstantDef() const {
    AstNode* definer = name_def()->definer();
    XLS_CHECK(definer != nullptr);
    return down_cast<ConstantDef*>(definer);
  }

  Expr* GetValue() const {
    AstNode* definer = name_def()->definer();
    XLS_CHECK(definer != nullptr);
    // Definer will only ever be a ConstantDef or a Let.
    if (ConstantDef* cd = dynamic_cast<ConstantDef*>(definer); cd != nullptr) {
      return cd->value();
    }

    Let* let = dynamic_cast<Let*>(definer);
    XLS_CHECK_NE(let, nullptr);
    return let->rhs();
  }
};

// A channel declaration, e.g., let (p, c) = chan<u32>.
//                                           ^^^^^^^^^ this part.
class ChannelDecl : public Expr {
 public:
  ChannelDecl(Module* owner, Span span, TypeAnnotation* type,
              std::optional<std::vector<Expr*>> dims)
      : Expr(owner, span), type_(type), dims_(dims) {}

  ~ChannelDecl() override;

  AstNodeKind kind() const override { return AstNodeKind::kChannelDecl; }

  absl::Status AcceptExpr(ExprVisitor* v) const override {
    return v->HandleChannelDecl(this);
  }
  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleChannelDecl(this);
  }

  std::string_view GetNodeTypeName() const override { return "ChannelDecl"; }
  std::string ToString() const override;

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {ToAstNode(type_)};
  }

  TypeAnnotation* type() const { return type_; }
  std::optional<std::vector<Expr*>> dims() const { return dims_; }

 private:
  TypeAnnotation* type_;
  std::optional<std::vector<Expr*>> dims_;
};

using ModuleMember =
    std::variant<Function*, Proc*, TestFunction*, TestProc*, QuickCheck*,
                 TypeDef*, StructDef*, ConstantDef*, EnumDef*, Import*>;

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

  ~Module() override;

  AstNodeKind kind() const override { return AstNodeKind::kModule; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleModule(this);
  }
  std::optional<Span> GetSpan() const override { return absl::nullopt; }

  std::string_view GetNodeTypeName() const override { return "Module"; }
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  std::string ToString() const override {
    // Don't print Proc functions, as they'll be printed as part of the procs
    // themselves.
    std::vector<ModuleMember> print_top;
    for (const auto& member : top_) {
      if (std::holds_alternative<Function*>(member) &&
          std::get<Function*>(member)->proc().has_value()) {
        continue;
      }
      print_top.push_back(member);
    }
    return absl::StrJoin(print_top, "\n",
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
    static_assert(!std::is_same<T, BuiltinNameDef>::value,
                  "Use Module::GetOrCreateBuiltinNameDef()");
    return MakeInternal<T, Args...>(std::forward<Args>(args)...);
  }

  BuiltinNameDef* GetOrCreateBuiltinNameDef(std::string_view name) {
    auto it = builtin_name_defs_.find(name);
    if (it == builtin_name_defs_.end()) {
      BuiltinNameDef* bnd = MakeInternal<BuiltinNameDef>(std::string(name));
      builtin_name_defs_.emplace_hint(it, std::string(name), bnd);
      return bnd;
    }
    return it->second;
  }

  absl::Status AddTop(ModuleMember member);

  // Gets the element in this module with the given target_name, or returns a
  // NotFoundError.
  template <typename T>
  absl::StatusOr<T*> GetMemberOrError(std::string_view target_name) {
    for (ModuleMember& member : top_) {
      if (std::holds_alternative<T*>(member)) {
        T* t = std::get<T*>(member);
        if (t->identifier() == target_name) {
          return t;
        }
      }
    }

    return absl::NotFoundError(
        absl::StrFormat("No %s in module %s with name \"%s\"", typeid(T).name(),
                        name_, target_name));
  }

  std::optional<Function*> GetFunction(std::string_view target_name);
  std::optional<Proc*> GetProc(std::string_view target_name);

  // Gets a test construct in this module with the given "target_name", or
  // returns a NotFoundError.
  absl::StatusOr<TestFunction*> GetTest(std::string_view target_name);
  absl::StatusOr<TestProc*> GetTestProc(std::string_view target_name);

  absl::Span<ModuleMember const> top() const { return top_; }

  // Finds the first top-level member in top() with the given "target" name as
  // an identifier.
  std::optional<ModuleMember*> FindMemberWithName(std::string_view target);

  const StructDef* FindStructDef(const Span& span) const;

  const EnumDef* FindEnumDef(const Span& span) const;

  // Obtains all the type definition nodes in the module:
  //    TypeDef, Struct, Enum
  absl::flat_hash_map<std::string, TypeDefinition> GetTypeDefinitionByName()
      const;

  // Obtains all the type definition nodes in the module in module-member order.
  std::vector<TypeDefinition> GetTypeDefinitions() const;

  absl::StatusOr<TypeDefinition> GetTypeDefinition(
      std::string_view name) const;

  // Retrieves a constant node from this module with the target name as its
  // identifier, or a NotFound error if none can be found.
  absl::StatusOr<ConstantDef*> GetConstantDef(std::string_view target);

  absl::flat_hash_map<std::string, ConstantDef*> GetConstantByName() const {
    return GetTopWithTByName<ConstantDef>();
  }
  absl::flat_hash_map<std::string, Import*> GetImportByName() const {
    return GetTopWithTByName<Import>();
  }
  absl::flat_hash_map<std::string, Function*> GetFunctionByName() const {
    return GetTopWithTByName<Function>();
  }
  absl::flat_hash_map<std::string, Proc*> GetProcByName() const {
    return GetTopWithTByName<Proc>();
  }
  std::vector<TypeDef*> GetTypeDefs() const { return GetTopWithT<TypeDef>(); }
  std::vector<QuickCheck*> GetQuickChecks() const {
    return GetTopWithT<QuickCheck>();
  }
  std::vector<StructDef*> GetStructDefs() const {
    return GetTopWithT<StructDef>();
  }
  std::vector<Proc*> GetProcs() const { return GetTopWithT<Proc>(); }
  std::vector<TestProc*> GetProcTests() const {
    return GetTopWithT<TestProc>();
  }
  std::vector<Function*> GetFunctions() const {
    return GetTopWithT<Function>();
  }
  std::vector<TestFunction*> GetFunctionTests() const {
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

  const AstNode* FindNode(AstNodeKind kind, const Span& span) const {
    for (const auto& node : nodes_) {
      if (node->kind() == kind && node->GetSpan().has_value() &&
          node->GetSpan().value() == span) {
        return node.get();
      }
    }
    return nullptr;
  }

 private:
  template <typename T, typename... Args>
  T* MakeInternal(Args&&... args) {
    std::unique_ptr<T> node =
        std::make_unique<T>(this, std::forward<Args>(args)...);
    T* ptr = node.get();
    ptr->SetParentage();
    nodes_.push_back(std::move(node));
    return ptr;
  }

  // Returns all of the elements of top_ that have the given variant type T.
  template <typename T>
  std::vector<T*> GetTopWithT() const {
    std::vector<T*> result;
    for (auto& member : top_) {
      if (std::holds_alternative<T*>(member)) {
        result.push_back(std::get<T*>(member));
      }
    }
    return result;
  }

  // Returns all the elements of top_ that have the given variant type T, using
  // T's identifier as a key. (T must have a string identifier.)
  template <typename T>
  absl::flat_hash_map<std::string, T*> GetTopWithTByName() const {
    absl::flat_hash_map<std::string, T*> result;
    for (auto& member : top_) {
      if (std::holds_alternative<T*>(member)) {
        auto* c = std::get<T*>(member);
        result.insert({c->identifier(), c});
      }
    }
    return result;
  }

  std::string name_;               // Name of this module.
  std::vector<ModuleMember> top_;  // Top-level members of this module.
  std::vector<std::unique_ptr<AstNode>> nodes_;  // Lifetime-owned AST nodes.

  // Map of top-level module member name to the member itself.
  absl::flat_hash_map<std::string, ModuleMember> top_by_name_;

  // Builtin name definitions, which we common out on a per-module basis. Not
  // for any particular purpose at this time aside from cleanliness of not
  // having many definition nodes of the same builtin thing floating around.
  absl::flat_hash_map<std::string, BuiltinNameDef*> builtin_name_defs_;
};

// Helper for determining whether an AST node is constant (e.g. can be
// considered a constant value in a ConstantArray).
bool IsConstant(AstNode* n);

}  // namespace xls::dslx

#endif  // XLS_DSLX_AST_H_
