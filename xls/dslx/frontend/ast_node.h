// Copyright 2023 The XLS Authors
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

#ifndef XLS_DSLX_FRONTEND_AST_NODE_H_
#define XLS_DSLX_FRONTEND_AST_NODE_H_

#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "xls/dslx/frontend/pos.h"

namespace xls::dslx {

// Enum with an entry for each leaf type in the AST class hierarchy -- this is
// primarily for convenience in tasks like serialization, for most purposes
// visitors should be used (e.g. AstNodeVisitor, ExprVisitor).
enum class AstNodeKind {
  kArray,
  kAttr,
  kBinop,
  kBlock,
  kBuiltinNameDef,
  kCast,
  kChannelDecl,
  kColonRef,
  kConstantDef,
  kConstRef,
  kConstAssert,
  kEnumDef,
  kFor,
  kFormatMacro,
  kFunction,
  kImport,
  kIndex,
  kInstantiation,
  kInvocation,
  kJoin,
  kLet,
  kMatch,
  kMatchArm,
  kModule,
  kNameDef,
  kNameDefTree,
  kNameRef,
  kNumber,
  kParam,
  kParametricBinding,
  kProc,
  kProcMember,
  kQuickCheck,
  kRange,
  kRecv,
  kRecvIf,
  kRecvIfNonBlocking,
  kRecvNonBlocking,
  kSend,
  kSendIf,
  kSlice,
  kSpawn,
  kSplatStructInstance,
  kStatement,
  kString,
  kStructDef,
  kStructInstance,
  kConditional,
  kTestFunction,
  kTestProc,
  kTupleIndex,
  kTypeAlias,
  kTypeAnnotation,
  kTypeRef,
  kUnop,
  kUnrollFor,
  kWidthSlice,
  kWildcardPattern,
  kXlsTuple,
  kZeroMacro,
};

std::string_view AstNodeKindToString(AstNodeKind kind);

template <typename Sink>
void AbslStringify(Sink& sink, AstNodeKind kind) {
  sink.Append(AstNodeKindToString(kind));
}

inline std::ostream& operator<<(std::ostream& os, AstNodeKind kind) {
  os << AstNodeKindToString(kind);
  return os;
}

// Forward decls.
class Module;
class AstNodeVisitor;

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

  // Returns a string representation of this node and (if supported by this
  // node's actual type) attempts to return it as a single line.
  virtual std::string ToInlineString() const { return ToString(); }

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

  Module* owner() const { return owner_; }

  // Marks this node as the parent of all its child nodes.
  void SetParentage();

  // Warning: try to avoid using this in any new code!
  //
  // Sometimes the frontend currently desugars AST nodes into other AST node
  // constructs (which is not ideal, it's an AST and ideally shouldn't be
  // treated like an IR if that can be avoided). When we do this, we may need to
  // set a parent relationship that was not in the original source text, which
  // is why we call this "non lexical". (In the more typical case, as we parse
  // we can just call `SetParentage()` to have lexical parent relationships
  // arise.)
  void SetParentNonLexical(AstNode* parent) { parent_ = parent; }

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

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_AST_NODE_H_
