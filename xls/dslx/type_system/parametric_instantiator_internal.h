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

#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/concrete_type.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/parametric_constraint.h"

namespace xls::dslx {
namespace internal {

// Abstract base class, encapsulates reusable functionality for a) instantiating
// parametric functions and b) instantiating parametric structs. In many ways
// those tasks are quite similar, taking values with 'actual' argument types
// provided and binding them against a parametric specification of 'formal'
// parameter types.
//
// As 'actual' argument types are observed they create known bindings for the
// 'formal' parameters' parametric symbols, and constraints created by
// parametric expressions have to be checked to see if they hold. If the case of
// any inconsistency, type errors must be raised.
class ParametricInstantiator {
 public:
  // See `InstantiateFunction` for details on
  // parametric_constraints/explicit_constraints and member comments for other
  // arguments.
  ParametricInstantiator(Span span, absl::Span<const InstantiateArg> args,
                         DeduceCtx* ctx,
                         std::optional<absl::Span<const ParametricConstraint>>
                             parametric_constraints,
                         const absl::flat_hash_map<std::string, InterpValue>*
                             explicit_constraints);

  // Non-movable as the destructor performs meaningful work -- the class
  // lifetime is used as a scope for parametric type information (used in
  // finding the concrete signature).
  ParametricInstantiator(ParametricInstantiator&& other) = delete;
  ParametricInstantiator(const ParametricInstantiator& other) = delete;
  ParametricInstantiator& operator=(const ParametricInstantiator& other) =
      delete;

  virtual ~ParametricInstantiator();

  virtual absl::StatusOr<TypeAndBindings> Instantiate() = 0;

 protected:
  // Binds param_type via arg_type, updating 'parametric_env_'.
  absl::StatusOr<std::unique_ptr<ConcreteType>> InstantiateOneArg(
      int64_t i, const ConcreteType& param_type, const ConcreteType& arg_type);

  // Verifies all constraints and then resolves possibly-parametric type
  // 'annotated' via 'parametric_env_'.
  absl::StatusOr<std::unique_ptr<ConcreteType>> Resolve(
      const ConcreteType& annotated);

  const absl::flat_hash_map<std::string, InterpValue>& parametric_env() const {
    return parametric_env_;
  }

  absl::Span<const InstantiateArg> args() const { return args_; }

  const Span& span() const { return span_; }

 private:
  // Verifies that all parametrics adhere to signature constraints.
  //
  // Take the following function signature for example:
  //
  //  fn [X: u32, Y: u32 = X + X] f(x: bits[X], y: bits[Y]) -> bits[Y]
  //
  // The parametric Y has two constraints based only off the signature:
  // * it must match the bitwidth of the argument y
  // * it must be equal to X+X
  //
  // This function is responsible for computing any derived parametrics and
  // asserting that their values are consistent with other constraints (argument
  // types).
  absl::Status VerifyConstraints();

  // The following is a group of helpers that bind parametric symbols in
  // param_type according to arg_type.
  //
  // e.g. for SymbolicBindBits, if the formal argument is `x: uN[N]` and the
  // actual argument is a u32, we'll bind `N=32` in the `parametric_env_`
  // map.
  template <typename T>
  absl::Status SymbolicBindDims(const T& param_type, const T& arg_type);

  absl::Status SymbolicBindBits(const ConcreteType& param_type,
                                const ConcreteType& arg_type);
  absl::Status SymbolicBindTuple(const TupleType& param_type,
                                 const TupleType& arg_type);
  absl::Status SymbolicBindStruct(const StructType& param_type,
                                  const StructType& arg_type);
  absl::Status SymbolicBindArray(const ArrayType& param_type,
                                 const ArrayType& arg_type);
  absl::Status SymbolicBindFunction(const FunctionType& param_type,
                                    const FunctionType& arg_type);

  // Binds symbols present in 'param_type' according to value of 'arg_type'.
  absl::Status SymbolicBind(const ConcreteType& param_type,
                            const ConcreteType& arg_type);

  // Span for the instantiation; e.g. of the invocation AST node being
  // instantiated.
  Span span_;

  // Arguments driving the instantiation, see `InstantiateArg` for more details.
  absl::Span<const InstantiateArg> args_;

  // The type deduction context. This is used to determine what function is
  // currently being instantiated / what parametric bindings it has, but it also
  // provides context we need to use when we do constexpr evaluation of
  // parametric constraints (e.g. type info, import data).
  DeduceCtx* ctx_;

  // Notes the iteration order in the original parametric bindings.
  std::vector<std::string> constraint_order_;

  // Note: the expressions may be null (e.g. when the parametric binding has no
  // "default" expression and must be provided by the user).
  absl::flat_hash_map<std::string, Expr*> parametric_default_exprs_;

  absl::flat_hash_map<std::string, std::unique_ptr<ConcreteType>>
      parametric_binding_types_;
  absl::flat_hash_map<std::string, InterpValue> parametric_env_;
};

// Instantiates a parametric function invocation.
class FunctionInstantiator : public ParametricInstantiator {
 public:
  static absl::StatusOr<std::unique_ptr<FunctionInstantiator>> Make(
      Span span, const FunctionType& function_type,
      absl::Span<const InstantiateArg> args, DeduceCtx* ctx,
      std::optional<absl::Span<const ParametricConstraint>>
          parametric_constraints,
      const absl::flat_hash_map<std::string, InterpValue>*
          explicit_constraints = nullptr);

  // Updates symbolic bindings for the parameter types according to args_'s
  // types.
  //
  // Instantiates the parameters of function_type_ according to the presented
  // args_' types.
  absl::StatusOr<TypeAndBindings> Instantiate() override;

 private:
  FunctionInstantiator(Span span, const FunctionType& function_type,
                       absl::Span<const InstantiateArg> args, DeduceCtx* ctx,
                       std::optional<absl::Span<const ParametricConstraint>>
                           parametric_constraints,
                       const absl::flat_hash_map<std::string, InterpValue>*
                           explicit_constraints = nullptr)
      : ParametricInstantiator(std::move(span), args, ctx,
                               parametric_constraints, explicit_constraints),
        function_type_(CloneToUnique(function_type)),
        param_types_(function_type_->params()) {}

  std::unique_ptr<FunctionType> function_type_;
  absl::Span<std::unique_ptr<ConcreteType> const> param_types_;
};

// Instantiates a parametric struct.
class StructInstantiator : public ParametricInstantiator {
 public:
  static absl::StatusOr<std::unique_ptr<StructInstantiator>> Make(
      Span span, const StructType& struct_type,
      absl::Span<const InstantiateArg> args,
      absl::Span<std::unique_ptr<ConcreteType> const> member_types,
      DeduceCtx* ctx,
      std::optional<absl::Span<const ParametricConstraint>>
          parametric_bindings);

  absl::StatusOr<TypeAndBindings> Instantiate() override;

 private:
  StructInstantiator(
      Span span, const StructType& struct_type,
      absl::Span<const InstantiateArg> args,
      absl::Span<std::unique_ptr<ConcreteType> const> member_types,
      DeduceCtx* ctx,
      std::optional<absl::Span<const ParametricConstraint>> parametric_bindings)
      : ParametricInstantiator(std::move(span), args, ctx,
                               /*parametric_constraints=*/parametric_bindings,
                               /*explicit_constraints=*/nullptr),
        struct_type_(CloneToUnique(struct_type)),
        member_types_(member_types) {}

  std::unique_ptr<ConcreteType> struct_type_;
  absl::Span<std::unique_ptr<ConcreteType> const> member_types_;
};

}  // namespace internal
}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_PARAMETRIC_INSTANTIATOR_INTERNAL_H_
