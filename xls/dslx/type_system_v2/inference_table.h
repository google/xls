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
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/types/variant.h"
#include "xls/common/visitor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"

namespace xls::dslx {

// The kinds of variables that can be defined in an `InferenceTable`.
enum class InferenceVariableKind : uint8_t { kInteger, kBool, kType };

// Flags for specifying a non-default interpretation of a type annotation in an
// `InferenceTable`. Certain flags may be combined.
class TypeInferenceFlag {
 public:
  TypeInferenceFlag() : flags_(0) {}

  static const TypeInferenceFlag kNone;

  // The minimum viable size of an integer. For example, a literal may have an
  // auto generated min annotation sized to fit the literal value. This
  // annotation can be unified with larger ones that do not take away
  // signedness, and the result is the larger type.
  static const TypeInferenceFlag kMinSize;

  // The standard type of an integer based on its use as an index or slice
  // bound. A type with these semantics can be unified with a mismatching
  // integer type, provided they agree on signedness. In such a case, a standard
  // size beats a min size, but a type with no flags beats a standard size.
  static const TypeInferenceFlag kStandardType;

  // For a literal without explicit type annotation, if it has a hexadecimal or
  // binary prefix, it can be interpreted as a negative number even if it does
  // not contain a negative sign when its unified type is sized and its most
  // significant bit is set for this literal. This is to be consistent with the
  // expectation that a hexadecimal or binary literal emphasizes the bit pattern
  // rather than the quantity of a number.
  static const TypeInferenceFlag kHasPrefix;

  bool HasFlag(const TypeInferenceFlag& value) const {
    if (value.flags_ == kNone.flags_) {
      return flags_ == kNone.flags_;
    }
    return flags_ & value.flags_;
  }

  void SetFlag(const TypeInferenceFlag& value) {
    flags_ |= value.flags_;
    // Only certain combinations of flags are allowed, all others result in an
    // internal error.
    // 1. Zero or one flag is set.
    // 2. Both kMinSize and kHasPrefix are set.
    CHECK((flags_ & (flags_ - 1)) == 0 ||
          flags_ == (kMinSize.flags_ | kHasPrefix.flags_));
  }

  bool operator==(const TypeInferenceFlag& other) const = default;

 private:
  // Only the named static const TypeInferenceFlag values are allowed to be used
  // externally.
  explicit constexpr TypeInferenceFlag(uint8_t flags) : flags_(flags) {}

  uint8_t flags_;
};

// Forward declaration.
class InferenceTable;

// The details for a `ParametricContext` that is for an invocation.
struct ParametricInvocationDetails {
  const Function* callee;
  std::optional<const Function*> caller;
  const FunctionTypeAnnotation* parametric_free_function_type = nullptr;
};

// The details for a `ParametricContext` that is for a struct.
struct ParametricStructDetails {
  const StructDefBase* struct_or_proc_def;
  const ParametricEnv env;
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
                    TypeInfo* type_info,
                    std::optional<const ParametricContext*> parent_context,
                    std::optional<const TypeAnnotation*> self_type)
      : id_(id),
        node_(node),
        details_(std::move(details)),
        parent_context_(parent_context),
        self_type_(self_type),
        type_info_(type_info) {}

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

  // Derived type info for this parametric context.
  TypeInfo* type_info() const { return type_info_; }

  // Returns whether this context is for an invocation as opposed to a struct.
  bool is_invocation() const {
    return std::holds_alternative<ParametricInvocationDetails>(details_);
  }

  // Returns whether this context is for a struct.
  bool is_struct() const {
    return std::holds_alternative<ParametricStructDetails>(details_);
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

  // The real type of `Self` in this context, including the parametric values.
  // Currently, we only populate this if this context is for a parametric struct
  // or a parametric method invocation on a parametric struct. In any other
  // case, the real type of `Self` can be determined by looking up the
  // `SelfTypeAnnotation` in the `InferenceTable`. This may change when we
  // support type parametrics.
  const std::optional<const TypeAnnotation*>& self_type() const {
    return self_type_;
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
        "ParametricContext(id=$0, parent_id=$1, self_type=$2, node=$3, "
        "data=($4), type_info_module=$5)",
        id_,
        parent_context_.has_value() ? std::to_string((*parent_context_)->id_)
                                    : "none",
        self_type_.has_value() ? (*self_type_)->ToString() : "none",
        node_->ToString(), DetailsToString(details_),
        type_info_ != nullptr ? type_info_->module()->name() : "none");
  }

  // Intended to be used only from the owning table, upon canonicalization.
  void SetInvocationEnv(ParametricEnv env) { invocation_env_ = std::move(env); }
  void SetTypeInfo(TypeInfo* type_info) { type_info_ = type_info; }

  // This is to enable reuse of a computed function type across compatible
  // parametric contexts. It is intended to be done as soon as feasible by the
  // owner of any non-canonicalized context, after the context is created. A
  // context for which `InferenceTable::MapToCanonicalInvocationTypeInfo`
  // returns true is guaranteed to already have this set in its details.
  void SetParametricFreeFunctionType(const FunctionTypeAnnotation* type) {
    auto& details = std::get<ParametricInvocationDetails>(details_);
    CHECK_EQ(details.parametric_free_function_type, nullptr);
    details.parametric_free_function_type = type;
  }

  // Returns the value of the given parametric binding in this context. If it is
  // not a relevant parametric binding, returns `nullopt`.
  std::optional<InterpValue> GetEnvValue(const NameDef* binding) const {
    if (invocation_env_.has_value()) {
      return invocation_env_->GetValue(binding);
    }
    if (std::holds_alternative<ParametricStructDetails>(details_)) {
      return std::get<ParametricStructDetails>(details_).env.GetValue(binding);
    }
    return std::nullopt;
  }

 private:
  static std::string DetailsToString(const Details& details) {
    return absl::visit(
        Visitor{
            [](const ParametricInvocationDetails& details) -> std::string {
              return absl::StrCat(details.callee->identifier(), ", caller: ",
                                  details.caller.has_value()
                                      ? (*details.caller)->identifier()
                                      : "<standalone context>");
            },
            [](const ParametricStructDetails& details) -> std::string {
              return absl::StrCat(details.struct_or_proc_def->identifier(),
                                  ", parametrics: ", details.env.ToString());
            }},
        details);
  }

  const uint64_t id_;  // Just for logging.
  const AstNode* node_;
  Details details_;
  const std::optional<const ParametricContext*> parent_context_;
  const std::optional<const TypeAnnotation*> self_type_;
  TypeInfo* type_info_;
  std::optional<ParametricEnv> invocation_env_;
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
// enough information in it to concretize the stored type of every node, i.e.,
// turn it into a `TypeInfo` object with concrete information. This is done via
// the `InferenceTableConverter` utility.
class InferenceTable {
 public:
  virtual ~InferenceTable();

  // Creates an empty inference table.
  static std::unique_ptr<InferenceTable> Create();

  // Defines an inference variable fabricated by the type inference system,
  // which has no direct representation in the DSLX source code that is being
  // analyzed. It is up to the inference system using the table to decide a
  // naming scheme for such variables. Optionally, if the user has provided a
  // TypeAnnotation at declaration time, a `declaration_annotation` can be
  // defined.
  virtual absl::StatusOr<const NameRef*> DefineInternalVariable(
      InferenceVariableKind kind, AstNode* definer, std::string_view name,
      std::optional<const TypeAnnotation*> declaration_annotation =
          std::nullopt) = 0;

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
  // annotation for `a`, for example. There is one copy of that in the table,
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
  virtual absl::StatusOr<ParametricContext*> AddParametricInvocation(
      const Invocation& invocation, const Function& callee,
      std::optional<const Function*> caller,
      std::optional<const ParametricContext*> parent_context,
      std::optional<const TypeAnnotation*> self_type,
      TypeInfo* invocation_type_info) = 0;

  // Finds an existing `ParametricContext` in the table that represents an
  // invocation of the same function with the same parametrics as `context` and
  // `env`.
  //
  // If an existing context is found, updates the `TypeInfo` pointer associated
  // with `context` to be one from the existing context, and returns true. The
  // implication is that there is then no point in spending effort to populate
  // the original `TypeInfo` that was associated with `context`. Whether or not
  // the caller subsequently uses this information, the table will make use of
  // it for internal cache optimization.
  //
  // If an existing matching context is not found, then this function captures
  // the `env`, makes `context` eligible to serve as a canonical context for
  // that `env` in the future, and returns false.
  virtual bool MapToCanonicalInvocationTypeInfo(ParametricContext* context,
                                                ParametricEnv env) = 0;

  // Retrieves all the parametric invocations that have been defined.
  virtual std::vector<const ParametricContext*> GetParametricInvocations()
      const = 0;

  // The result of `GetOrCreateParametricStructContext()`.
  struct StructContextResult {
    const ParametricContext* context;
    bool created_new;
  };

  // Defines a parametric struct context with the given parametric values, or
  // returns the existing one with the same values. The idea is to tie each
  // struct parameterization to a canonicalized env to avoid unnecessary
  // aliasing. The `parametric_env` does not need values for defaulted bindings.
  virtual absl::StatusOr<StructContextResult>
  GetOrCreateParametricStructContext(
      const StructDefBase* struct_def, const AstNode* node,
      ParametricEnv parametric_env, const TypeAnnotation* self_type,
      absl::FunctionRef<absl::StatusOr<TypeInfo*>()> type_info_factory) = 0;

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

  // Sets the explicit type annotation associated with `node`. If `context`
  // is specified, then the annotation is only valid within that parametric
  // context.
  virtual absl::Status AddTypeAnnotationToVariableForParametricContext(
      std::optional<const ParametricContext*> context, const NameRef* ref,
      const TypeAnnotation* type) = 0;

  // Convenience variant that takes the `ParametricBinding` associated with a
  // variable instead of a `NameRef`, with no difference in effect.
  virtual absl::Status AddTypeAnnotationToVariableForParametricContext(
      std::optional<const ParametricContext*> context,
      const ParametricBinding* binding, const TypeAnnotation* type) = 0;

  // Removes annotations matching the given predicate from the general list of
  // annotations for the variable referred to by `ref`.
  virtual absl::Status RemoveTypeAnnotationsFromTypeVariable(
      const NameRef* ref,
      absl::FunctionRef<bool(const TypeAnnotation*)> remove_predicate) = 0;

  // Returns the type annotation for `node` in the table, if any.
  virtual std::optional<const TypeAnnotation*> GetTypeAnnotation(
      const AstNode* node) const = 0;

  // Sets the given flag which modifies how to interpret the given annotation in
  // this table. Currently we only support at most one flag per annotation.
  virtual void SetAnnotationFlag(const TypeAnnotation* annotation,
                                 TypeInferenceFlag flag) = 0;

  // Returns the flag that has been set for the given `annotation`, or `kNone`
  // if one hasn't been set.
  virtual TypeInferenceFlag GetAnnotationFlag(
      const TypeAnnotation* annotation) const = 0;

  // Sets the target of a `ColonRef`.
  virtual void SetColonRefTarget(const ColonRef* colon_ref,
                                 const AstNode* target) = 0;

  // Returns the stored target of a `ColonRef`.
  virtual std::optional<const AstNode*> GetColonRefTarget(
      const ColonRef* colon_ref) const = 0;

  // Returns the type variable for `node` in the table, if any.
  virtual std::optional<const NameRef*> GetTypeVariable(
      const AstNode* node) const = 0;

  // Returns the type annotation declared by the user, if any, for the type
  // variable associated with `ref`.
  virtual absl::StatusOr<std::optional<const TypeAnnotation*>>
  GetDeclarationTypeAnnotation(const NameRef* ref) const = 0;

  // Returns all type annotations that have been associated with the given
  // variable, in the order they were added to the table.
  virtual absl::StatusOr<std::vector<const TypeAnnotation*>>
  GetTypeAnnotationsForTypeVariable(
      std::optional<const ParametricContext*> parametric_context,
      const NameRef* variable) const = 0;

  // Clones the given `input` subtree and the table data for each node.
  virtual absl::StatusOr<AstNode*> Clone(const AstNode* input,
                                         CloneReplacer replacer) = 0;

  // Stores the expanded, absolute start and width expressions for a slice,
  // which need to eventually be concretized and added to `TypeInfo`.
  virtual absl::Status SetSliceStartAndWidthExprs(
      const AstNode* node, StartAndWidthExprs start_and_width) = 0;

  // Retrieves the previously stored start and width expressions for a slice.
  virtual std::optional<StartAndWidthExprs> GetSliceStartAndWidthExprs(
      const AstNode* node) = 0;

  // Caches the given unified type for the given type variable in the given
  // context. The `transitive_variable_dependencies` must include all the
  // inference variables that factored into the unified type. The cached unified
  // type will then automatically be invalidated if any of those variables
  // change.
  virtual void SetCachedUnifiedTypeForVariable(
      std::optional<const ParametricContext*> parametric_context,
      const NameRef* variable,
      const absl::flat_hash_set<const NameRef*>&
          transitive_variable_dependencies,
      const TypeAnnotation* unified_type) = 0;

  // Retrieves the cached unified type for the given variable in the given
  // context, if one has been cached and it is still valid.
  virtual std::optional<const TypeAnnotation*> GetCachedUnifiedTypeForVariable(
      std::optional<const ParametricContext*> parametric_context,
      const NameRef* variable) = 0;

  // Sets the parametric env for the given context. This is only necessary to do
  // for invocation contexts, since `GetOrCreateParametricStructContext()`
  // captures it for struct contexts at creation time.
  virtual void SetParametricEnv(const ParametricContext* parametric_context,
                                ParametricEnv env) = 0;

  // Retrieves the `ParametricEnv` for the given context. An empty
  // `ParametricEnv` will be returned for a `nullopt` context or an invocation
  // context without an env set. This allows this function to be used for an
  // invocation context whose env is still in the process of being computed. It
  // will deal out an empty caller env for `clog2` in a situation like
  // `fn foo<A: u32, B: u32 = {clog2(A)}>`, which is what downstream logic
  // expects.
  virtual ParametricEnv GetParametricEnv(
      std::optional<const ParametricContext*> parametric_context) = 0;

  // Stores the parametric value expressions for the given context. Unlike the
  // `ParametricEnv`, this is not set automatically at construction time for any
  // type of parametric context.
  virtual void SetParametricValueExprs(
      const ParametricContext* parametric_context,
      absl::flat_hash_map<const NameDef*, ExprOrType> value_exprs) = 0;

  // Retrieves the parametric value expressions that have been set for the given
  // context.
  virtual absl::StatusOr<absl::flat_hash_map<const NameDef*, ExprOrType>>
  GetParametricValueExprs(const ParametricContext* parametric_context) = 0;

  // Converts the table to string for debugging purposes.
  virtual std::string ToString() const = 0;
};

// Fabricates a `Number` node and sets the given type annotation for it in the
// inference table.
absl::StatusOr<Number*> MakeTypeCheckedNumber(
    Module& module, InferenceTable& table, const Span& span,
    const InterpValue& value, const TypeAnnotation* type_annotation);

// Variant that takes a raw `int64_t` value for the number.
absl::StatusOr<Number*> MakeTypeCheckedNumber(
    Module& module, InferenceTable& table, const Span& span, int64_t value,
    const TypeAnnotation* type_annotation);

// Returns whether the given `expr` is a `ColonRef` to a type as opposed to a
// value. The determination is based on table data for `expr`; this function
// will not actually resolve and analyze the `ColonRef` itself.
bool IsColonRefWithTypeTarget(const InferenceTable& table, const Expr* expr);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_INFERENCE_TABLE_H_
