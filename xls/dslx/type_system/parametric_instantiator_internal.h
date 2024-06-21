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

#ifndef XLS_DSLX_TYPE_SYSTEM_PARAMETRIC_INSTANTIATOR_INTERNAL_H_
#define XLS_DSLX_TYPE_SYSTEM_PARAMETRIC_INSTANTIATOR_INTERNAL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/parametric_with_type.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_and_parametric_env.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {
namespace internal {

// Abstract base class, encapsulates reusable functionality for a) instantiating
// parametric functions and b) instantiating parametric structs. In many ways
// those tasks are quite similar, taking values with 'actual' argument types
// provided and binding them against a parametric specification of 'formal'
// parameter types.
//
// As 'actual' argument types are observed they create known bindings for the
// 'formal' parameters' parametric symbols. Consistency with bindings created by
// other parametric expressions have to be checked to see if they hold.
//
// e.g. consider:
//
//    f<N: u32>(x: bits[N], y: bits[N])
//
// invoked via:
//
//    f(u32:42, u8:64)  // would imply inconsistent inferred value of N
//
// If the case of any inconsistency, type errors must be raised.
class ParametricInstantiator {
 public:
  // See `InstantiateFunction` for details on
  // typed_parametrics/explicit_parametrics and member comments for other
  // arguments.
  ParametricInstantiator(
      Span span, std::optional<Span> parametrics_span,
      absl::Span<const InstantiateArg> args, DeduceCtx* ctx,
      absl::Span<const ParametricWithType> typed_parametrics,
      const absl::flat_hash_map<std::string, InterpValue>& explicit_parametrics,
      absl::Span<absl::Nonnull<const ParametricBinding*> const>
          parametric_bindings);

  // Non-movable as the destructor performs meaningful work -- the class
  // lifetime is used as a scope for parametric type information (used in
  // finding the concrete signature).
  ParametricInstantiator(ParametricInstantiator&& other) = delete;
  ParametricInstantiator(const ParametricInstantiator& other) = delete;
  ParametricInstantiator& operator=(const ParametricInstantiator& other) =
      delete;

  virtual ~ParametricInstantiator();

  virtual absl::StatusOr<TypeAndParametricEnv> Instantiate() = 0;

  // e.g. "struct" or "function"
  virtual std::string_view GetKindName() = 0;

 protected:
  // Binds param_type via arg_type, updating 'parametric_env_map_'.
  absl::Status InstantiateOneArg(int64_t i, const Type& param_type,
                                 const Type& arg_type);

  const absl::flat_hash_map<std::string, InterpValue>& parametric_env_map()
      const {
    return parametric_env_map_;
  }
  absl::flat_hash_map<std::string, InterpValue>& parametric_env_map() {
    return parametric_env_map_;
  }

  const absl::flat_hash_map<std::string, Expr*>& parametric_default_exprs()
      const {
    return parametric_default_exprs_;
  }

  absl::Span<const InstantiateArg> args() const { return args_; }
  absl::Span<const ParametricWithType> typed_parametrics() const {
    return typed_parametrics_;
  }

  // Returns the span of the parametrics; e.g.
  //
  //   f<A: u32, B: u32>(...)
  //    ^~~~~~~~~~~~~~~^ this
  //
  // Nullopt when we're (presumably uselessly) instantiating a function that
  // has no parametrics, which does seem to happen in the type system at
  // present.
  const std::optional<Span>& parametrics_span() const {
    return parametrics_span_;
  }

  const Span& span() const { return span_; }
  DeduceCtx& ctx() { return *ctx_; }

 private:
  // Span for the instantiation; e.g. of the invocation AST node being
  // instantiated.
  Span span_;

  // Span of parametric bindings being instantiated; see `parametrics_span()`.
  std::optional<Span> parametrics_span_;

  // Arguments driving the instantiation, see `InstantiateArg` for more details.
  absl::Span<const InstantiateArg> args_;

  // The type deduction context. This is used to determine what function is
  // currently being instantiated / what parametric bindings it has, but it also
  // provides context we need to use when we do constexpr evaluation of
  // parametric exprs (e.g. type info, import data).
  DeduceCtx* ctx_;

  // The "derived type information" mapping we use in evaluating the parametric
  // types in this instantiation.
  TypeInfo* derived_type_info_ = nullptr;

  absl::Span<const ParametricWithType> typed_parametrics_;

  // Note: the expressions may be null (e.g. when the parametric binding has no
  // "default" expression and must be provided by the user).
  absl::flat_hash_map<std::string, Expr*> parametric_default_exprs_;

  absl::flat_hash_map<std::string, std::unique_ptr<Type>>
      parametric_binding_types_;
  absl::flat_hash_map<std::string, InterpValue> parametric_env_map_;

  absl::Span<absl::Nonnull<const ParametricBinding*> const>
      parametric_bindings_;
};

// Instantiates a parametric function invocation.
class FunctionInstantiator : public ParametricInstantiator {
 public:
  static absl::StatusOr<std::unique_ptr<FunctionInstantiator>> Make(
      Span span, Function& callee_fn, const FunctionType& function_type,
      absl::Span<const InstantiateArg> args, DeduceCtx* ctx,
      absl::Span<const ParametricWithType> typed_parametrics,
      const absl::flat_hash_map<std::string, InterpValue>& explicit_parametrics,
      absl::Span<absl::Nonnull<const ParametricBinding*> const>
          parametric_bindings);

  // Updates symbolic bindings for the parameter types according to args_'s
  // types.
  //
  // Instantiates the parameters of function_type_ according to the presented
  // args_' types.
  absl::StatusOr<TypeAndParametricEnv> Instantiate() override;

  std::string_view GetKindName() override { return "function"; }

 private:
  FunctionInstantiator(
      Span span, Function& callee_fn, const FunctionType& function_type,
      absl::Span<const InstantiateArg> args, DeduceCtx* ctx,
      absl::Span<const ParametricWithType> typed_parametrics,
      const absl::flat_hash_map<std::string, InterpValue>& explicit_parametrics,
      absl::Span<absl::Nonnull<const ParametricBinding*> const>
          parametric_bindings)
      : ParametricInstantiator(
            std::move(span), callee_fn.GetParametricBindingsSpan(), args, ctx,
            typed_parametrics, explicit_parametrics, parametric_bindings),
        callee_fn_(callee_fn),
        function_type_(CloneToUnique(function_type)),
        param_types_(function_type_->params()) {}

  Function& callee_fn_;
  std::unique_ptr<FunctionType> function_type_;
  absl::Span<std::unique_ptr<Type> const> param_types_;
};

// Instantiates a parametric struct.
class StructInstantiator : public ParametricInstantiator {
 public:
  static absl::StatusOr<std::unique_ptr<StructInstantiator>> Make(
      Span span, const StructType& struct_type,
      absl::Span<const InstantiateArg> args,
      absl::Span<std::unique_ptr<Type> const> member_types, DeduceCtx* ctx,
      absl::Span<const ParametricWithType> typed_parametrics,
      absl::Span<absl::Nonnull<const ParametricBinding*> const>
          parametric_bindings);

  absl::StatusOr<TypeAndParametricEnv> Instantiate() override;

  std::string_view GetKindName() override { return "struct"; }

 private:
  StructInstantiator(Span span, const StructType& struct_type,
                     absl::Span<const InstantiateArg> args,
                     absl::Span<std::unique_ptr<Type> const> member_types,
                     DeduceCtx* ctx,
                     absl::Span<const ParametricWithType> typed_parametrics,
                     absl::Span<absl::Nonnull<const ParametricBinding*> const>
                         parametric_bindings)
      : ParametricInstantiator(
            std::move(span),
            struct_type.nominal_type().GetParametricBindingsSpan(), args, ctx,
            /*typed_parametrics=*/typed_parametrics,
            /*explicit_parametrics=*/{}, parametric_bindings),
        struct_type_(CloneToUnique(struct_type)),
        member_types_(member_types) {}

  std::unique_ptr<Type> struct_type_;
  absl::Span<std::unique_ptr<Type> const> member_types_;
};

}  // namespace internal
}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_PARAMETRIC_INSTANTIATOR_INTERNAL_H_
