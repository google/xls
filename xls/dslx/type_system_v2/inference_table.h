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
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/types/variant.h"
#include "xls/common/visitor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/type_system/parametric_env.h"

namespace xls::dslx {

// The kinds of variables that can be defined in an `InferenceTable`.
enum class InferenceVariableKind : uint8_t { kInteger, kBool, kType };

// The details for a `ParametricContext` that is for an invocation.
struct ParametricInvocationDetails {
  const Function* callee;
  std::optional<const Function*> caller;
};

// The details for a `ParametricContext` that is for a struct.
struct ParametricStructDetails {
  const StructDefBase* struct_or_proc_def;
  ParametricEnv env;
};

// Identifies either an invocation of a parametric function, or a
// parameterization of a parametric struct, with enough context to determine
// what its effective parametric value expressions must be. These are dealt out
// by an `InferenceTable`.
class ParametricContext {
 public:
  using Details =
      std::variant<ParametricInvocationDetails, ParametricStructDetails>;

  ParametricContext(uint64_t id, const AstNode* node, Details details,
                    std::optional<const ParametricContext*> parent_context)
      : id_(id),
        node_(node),
        details_(std::move(details)),
        parent_context_(parent_context) {}

  template <typename H>
  friend H AbslHashValue(H h, const ParametricContext& context) {
    return H::combine(std::move(h), context.node_, context.parent_context_);
  }

  // The node that motivated the creation of this context. For a parametric
  // invocation, it is an `Invocation` node. For a struct, it may be a
  // `StructInstance`, `ColonRef`, or other node that establishes the use of a
  // parameterization of the struct.
  const AstNode* node() const { return node_; }

  // The details about the context, which depend on whether it is for a function
  // or struct.
  const Details& details() const { return details_; }

  // Returns whether this context is for an invocation as opposed to a struct.
  bool is_invocation() const {
    return std::holds_alternative<ParametricInvocationDetails>(details_);
  }

  // The parent parametric context. In a scenario where `f` calls `g`, and they
  // are both parametric functions, a `g` context would have an `f` context as
  // its parent. An `f` context might then have no parent, if `f` is not called
  // from a parametric context (being called from a non-parametric context is
  // irrelevant to this). On the other hand, if `g` is called from the RHS of a
  // constant in a parametric impl, then the context for a parameterization of
  // the struct may be the parent of the `g` invocation.
  const std::optional<const ParametricContext*>& parent_context() const {
    return parent_context_;
  }

  // Returns the parametric bindings of the function or struct that this context
  // is for.
  std::vector<const ParametricBinding*> parametric_bindings() const {
    std::vector<const ParametricBinding*> result;
    // Note: this is due to the interface disparity between structs and
    // functions.
    absl::c_copy(is_invocation()
                     ? std::get<ParametricInvocationDetails>(details_)
                           .callee->parametric_bindings()
                     : std::get<ParametricStructDetails>(details_)
                           .struct_or_proc_def->parametric_bindings(),
                 std::back_inserter(result));
    return result;
  }

  // Converts this context to string for debugging and logging purposes.
  std::string ToString() const {
    return absl::Substitute(
        "ParametricContext(id=$0, parent_id=$1, node=$2, data=($3))", id_,
        parent_context_.has_value() ? std::to_string((*parent_context_)->id_)
                                    : "none",
        node_->ToString(), DetailsToString(details_));
  }

 private:
  static std::string DetailsToString(const Details& details) {
    return absl::visit(
        Visitor{[](const ParametricInvocationDetails& details) -> std::string {
                  return absl::StrCat(details.callee->ToString(), ", caller: ",
                                      details.caller.has_value()
                                          ? (*details.caller)->identifier()
                                          : "<standalone context>");
                },
                [](const ParametricStructDetails& details) -> std::string {
                  return absl::StrCat(
                      details.struct_or_proc_def->identifier(),
                      ", parametrics: ", details.env.ToString());
                }},
        details);
  }

  const uint64_t id_;  // Just for logging.
  const AstNode* node_;
  const Details details_;
  const std::optional<const ParametricContext*> parent_context_;
};

inline std::string ToString(std::optional<const ParametricContext*> context) {
  return context.has_value() ? (*context)->ToString() : "<standalone context>";
}

// An `Expr` paired with the `ParametricContext` in whose context any
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
class ParametricContextScopedExpr {
 public:
  ParametricContextScopedExpr(std::optional<const ParametricContext*> context,
                              const TypeAnnotation* type_annotation,
                              const Expr* expr)
      : context_(context), type_annotation_(type_annotation), expr_(expr) {}

  const std::optional<const ParametricContext*>& context() const {
    return context_;
  }
  const TypeAnnotation* type_annotation() const { return type_annotation_; }
  const Expr* expr() const { return expr_; }

 private:
  const std::optional<const ParametricContext*> context_;
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
  static std::unique_ptr<InferenceTable> Create(Module& module);

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
  // context. Note that the `caller` must only be `nullopt` if the invocation is
  // not in a function (e.g. it may be in the RHS of a free constant
  // declaration).
  virtual absl::StatusOr<const ParametricContext*> AddParametricInvocation(
      const Invocation& invocation, const Function& callee,
      std::optional<const Function*> caller,
      std::optional<const ParametricContext*> parent_context) = 0;

  // Retrieves all the parametric invocations that have been defined.
  virtual std::vector<const ParametricContext*> GetParametricInvocations()
      const = 0;

  // Defines a parametric struct context with the given parametric values, or
  // returns the existing one with the same values. The idea is to tie each
  // struct parameterization to a canonicalized env to avoid unnecessary
  // aliasing. The `parametric_env` does not need values for defaulted bindings.
  virtual const ParametricContext* GetOrCreateParametricStructContext(
      const StructDefBase* struct_def, const AstNode* node,
      ParametricEnv parametric_env) = 0;

  // Returns the expression for the value of the given parametric in the given
  // invocation, if the parametric has an explicit or default expression. If it
  // is implicit, then this returns `nullopt`. Note that the return value may be
  // scoped to either `invocation` or its caller, depending on where the value
  // expression originated from.
  virtual std::optional<ParametricContextScopedExpr> GetParametricValue(
      const NameDef& binding_name_def,
      const ParametricContext& context) const = 0;

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

  // Sets the explicit type annotation associated with `node`. If `invocation`
  // is specified, then the annotation is only valid within that parametric
  // context.
  virtual absl::Status AddTypeAnnotationToVariableForParametricContext(
      std::optional<const ParametricContext*> context, const NameRef* ref,
      const TypeAnnotation* type) = 0;

  // Returns the type annotation for `node` in the table, if any.
  virtual std::optional<const TypeAnnotation*> GetTypeAnnotation(
      const AstNode* node) const = 0;

  // Marks the given `annotation` as an auto-determined annotation for a
  // literal. These annotations have relaxed unification semantics, so that a
  // literal can become the type its context requires.
  virtual void MarkAsAutoLiteral(const TypeAnnotation* annotation) = 0;

  // Returns whether the given `annotation` has been marked as an auto literal
  // annotation.
  virtual bool IsAutoLiteral(const TypeAnnotation* annotation) = 0;

  // Sets the target of a `ColonRef`.
  virtual void SetColonRefTarget(const ColonRef* colon_ref,
                                 const AstNode* target) = 0;

  // Returns the stored target of a `ColonRef`.
  virtual std::optional<const AstNode*> GetColonRefTarget(
      const ColonRef* colon_ref) const = 0;

  // Returns the type variable for `node` in the table, if any.
  virtual std::optional<const NameRef*> GetTypeVariable(
      const AstNode* node) const = 0;

  // Returns all type annotations that have been associated with the given
  // variable, in the order they were added to the table.
  virtual absl::StatusOr<std::vector<const TypeAnnotation*>>
  GetTypeAnnotationsForTypeVariable(
      std::optional<const ParametricContext*> parametric_context,
      const NameRef* variable) const = 0;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_INFERENCE_TABLE_H_
