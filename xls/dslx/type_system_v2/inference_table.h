// Copyright 2024 The XLS Authors
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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_INFERENCE_TABLE_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_INFERENCE_TABLE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/pos.h"

namespace xls::dslx {

// The kinds of variables that can be defined in an `InferenceTable`.
enum class InferenceVariableKind : uint8_t { kInteger, kBool, kType };

// Identifies an invocation of a parametric function, with enough context to
// determine what its effective parametric value expressions must be. These are
// dealt out by an `InferenceTable`.
class ParametricInvocation {
 public:
  ParametricInvocation(
      uint64_t id, const Invocation& node, const Function& callee,
      const Function& caller,
      std::optional<const ParametricInvocation*> caller_invocation)
      : id_(id),
        node_(node),
        callee_(callee),
        caller_(caller),
        caller_invocation_(caller_invocation) {}

  const Invocation& node() const { return node_; }
  const Function& callee() const { return callee_; }
  const Function& caller() const { return caller_; }

  // Note: this is `nullopt` if the caller is not parametric.
  const std::optional<const ParametricInvocation*>& caller_invocation() const {
    return caller_invocation_;
  }

  template <typename H>
  friend H AbslHashValue(H h, const ParametricInvocation& invocation) {
    return H::combine(std::move(h), invocation.node_,
                      invocation.caller_invocation_);
  }

  std::string ToString() const {
    return absl::Substitute(
        "ParametricInvocation(id=$0, node=$1, caller=$2, caller_id=$3)", id_,
        node_.ToString(), caller_.identifier(),
        caller_invocation_.has_value()
            ? std::to_string((*caller_invocation_)->id_)
            : "none");
  }

 private:
  const uint64_t id_;  // Just for logging.
  const Invocation& node_;
  const Function& callee_;
  const Function& caller_;
  const std::optional<const ParametricInvocation*> caller_invocation_;
};

// An `Expr` paired with the `ParametricInvocation` in whose context any
// parametrics in it should be evaluated. This is useful for capturing the
// effective value expressions of parametrics, which may in turn refer to other
// parametrics. Defaulted values are scoped to the callee, while explicit values
// are scoped to the caller. In a scenario like:
//
//   fn foo<M: u32, N: u32 = {M * M}>(a: uN[M], b: uN[N]) { ... }
//   fn bar<M: u32> {
//     foo<M + u32:1>(...);
//   }
//
// - `M + u32:1`, as the value of `M`, is scoped to a `bar` invocation.
// - `M * M`, as the value of `N`, is scoped to a `foo` invocation.
class InvocationScopedExpr {
 public:
  InvocationScopedExpr(std::optional<const ParametricInvocation*> invocation,
                       const TypeAnnotation* type_annotation, const Expr* expr)
      : invocation_(invocation),
        type_annotation_(type_annotation),
        expr_(expr) {}

  const std::optional<const ParametricInvocation*>& invocation() const {
    return invocation_;
  }
  const TypeAnnotation* type_annotation() const { return type_annotation_; }
  const Expr* expr() const { return expr_; }

 private:
  const std::optional<const ParametricInvocation*> invocation_;
  const TypeAnnotation* const type_annotation_;
  const Expr* const expr_;
};

// A table that facilitates a type inference algorithm where unknowns during the
// course of inference are represented using variables (which we call "inference
// variables"). An inference variable may be internally fabricated by the
// inference system, or it may be a parametric named in the DSLX source code.
//
// The type inference system that uses the table defines variables via
// `DefineXXXVariable` functions, getting back `NameRef` objects for them, which
// it can then use in expressions if desired. The inference system also stores
// the inferred type of each processed AST node in the table, in the form of
// either a `TypeAnnotation` (when specified explicitly in the source code) or
// a defined inference variable.
//
// Once the inference system is finished populating the table, there should be
// enough information in it to concretize the stored type of every node, i.e.
// turn it into a `Type` object with concrete information. This can be done via
// the `InferenceTableToTypeInfo` utility.
class InferenceTable {
 public:
  virtual ~InferenceTable();

  // Creates an empty inference table for the given module.
  static std::unique_ptr<InferenceTable> Create(Module& module,
                                                const FileTable& file_table);

  // Defines an inference variable fabricated by the type inference system,
  // which has no direct representation in the DSLX source code that is being
  // analyzed. It is up to the inference system using the table to decide a
  // naming scheme for such variables.
  virtual absl::StatusOr<const NameRef*> DefineInternalVariable(
      InferenceVariableKind kind, AstNode* definer, std::string_view name) = 0;

  // Defines an inference variable corresponding to a parametric in the DSLX
  // source code. Unlike an internal variable, a parametric has a different
  // actual value or constraints per instantiation, so instantiations must be
  // created via `AddParametricInvocation`. In an example like:
  //    `fn foo<N: u32>(a: uN[N]) -> uN[N] { a + a }`
  //
  // N is a parametric variable, which is referenced by the type annotation for
  // `a`, the return type of `foo`, etc.
  //
  // The table will store different value expressions for `N` per invocation
  // context for `foo`, but it does not need distinct copies of the type
  // annotation for `a` for example. There is one copy of that in the table,
  // which uses a `NameRef` to the variable `N`.
  //
  // At the time of conversion of the table to `TypeInfo`, we distinctly resolve
  // `N` and its dependent types for each invocation context of `foo`.
  virtual absl::StatusOr<const NameRef*> DefineParametricVariable(
      const ParametricBinding& binding) = 0;

  // Defines an invocation context for a parametric function, giving its
  // associated parametric variables distinct value expression storage for that
  // context.
  virtual absl::StatusOr<const ParametricInvocation*> AddParametricInvocation(
      const Invocation& invocation, const Function& callee,
      const Function& caller,
      std::optional<const ParametricInvocation*> caller_invocation) = 0;

  // Retrieves all the parametric invocations that have been defined for all
  // parametric functions.
  virtual std::vector<const ParametricInvocation*> GetParametricInvocations()
      const = 0;

  // Returns the expression for the value of the given parametric in the given
  // invocation. Note that the return value may be scoped to either `invocation`
  // or its caller, depending on where the value expression originated from.
  virtual InvocationScopedExpr GetParametricValue(
      const NameDef& binding_name_def,
      const ParametricInvocation& invocation) const = 0;

  // Sets the type variable associated with `node`. The `type` must refer to a
  // type variable previously defined in this table. This can serve as a way to
  // constrain one or more nodes to match some unknown type, such as the left
  // and right sides of a `+` operation.
  virtual absl::Status SetTypeVariable(const AstNode* node,
                                       const NameRef* type) = 0;

  // Sets the explicit type annotation associated with `node`. Not all nodes
  // have one. For example, a `Let` node like `let x:u32 = something;` has a
  // type annotation, but `let x = something;` does not.
  virtual absl::Status SetTypeAnnotation(const AstNode* node,
                                         const TypeAnnotation* type) = 0;

  // Returns all the nodes that have information in the table and are not
  // dependent on parametric variables (i.e. their types, if any, should have
  // single concretizations). The nodes are returned in the order added to the
  // table.
  virtual std::vector<const AstNode*> GetStaticNodes() const = 0;

  // Returns all the nodes that have information in the table and whose types
  // may have distinct concretizations in the context of the given `invocation`.
  // The nodes are returned in the order added to the table.
  virtual std::vector<const AstNode*> GetNodesWithInvocationSpecificTypes(
      const ParametricInvocation* invocation) const = 0;

  // Returns the type annotation for `node` in the table, if any.
  virtual std::optional<const TypeAnnotation*> GetTypeAnnotation(
      const AstNode* node) const = 0;

  // Returns the type variable for `node` in the table, if any.
  virtual std::optional<const NameRef*> GetTypeVariable(
      const AstNode* node) const = 0;

  // Returns all type annotations that have been associated with the given
  // variable, in the order they were added to the table.
  virtual absl::StatusOr<std::vector<const TypeAnnotation*>>
  GetTypeAnnotationsForTypeVariable(const NameRef* variable) const = 0;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_INFERENCE_TABLE_H_
