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

#ifndef XLS_DSLX_CPP_PARAMETRIC_INSTANTIATOR_H_
#define XLS_DSLX_CPP_PARAMETRIC_INSTANTIATOR_H_

#include "xls/dslx/concrete_type.h"
#include "xls/dslx/deduce_ctx.h"
#include "xls/dslx/type_and_bindings.h"
#include "xls/dslx/type_info.h"

namespace xls::dslx {

// Describes an argument being presented for instantiation (of a parametric
// function or struct) -- these argument expressions have types and come from
// some originating span, which is used for error reporting.
//
// Note that both *function* instantiation and *struct* instantiation
// conceptually have "argument" values given, with the 'actual' types, that are
// filling in the (possibly parametric) slots of the formal (declared) types --
// the formal types may be parametric.
struct InstantiateArg {
  const ConcreteType& type;
  const dslx::Span span;
};

// Decorates a parametric binding with its (deduced) ConcreteType.
//
// These are provided as inputs to parametric instantiation functions below.
class ParametricConstraint {
 public:
  // Decorates the given "binding" with the provided type information.
  ParametricConstraint(const ParametricBinding& binding,
                       std::unique_ptr<ConcreteType> type);

  // Decorates the given "binding" with the type information as above, but
  // exposes the (replacement) expression "expr".
  ParametricConstraint(const ParametricBinding& binding,
                       std::unique_ptr<ConcreteType> type, Expr* expr);

  const std::string& identifier() const { return binding_->identifier(); }
  const ConcreteType& type() const { return *type_; }
  Expr* expr() const { return expr_; }

  std::string ToString() const;

 private:
  const ParametricBinding* binding_;
  std::unique_ptr<ConcreteType> type_;

  // Expression that the parametric value should take on (e.g. when there are
  // "derived parametrics" that are computed from other parametric values). Note
  // that this may be null.
  Expr* expr_;
};

// Instantiates a function invocation using the bindings derived from args'
// types.
//
// Args:
//  span: Invocation span causing the instantiation to occur.
//  function_type: Type (possibly parametric) of the function being
//    instantiated.
//  args: Arguments driving the instantiation of the function signature.
//  ctx: Type deduction context, e.g. used in constexpr evaluation.
//  parametric_constraints: Contains expressions being given as parametrics that
//    must be evaluated. They are called "constraints" because they may be
//    in conflict as a result of deductive inference; e.g. for
//    `f<N: u32, R: u32 = N+N>(x: bits[N]) -> bits[R] { x }` we'll find the
//    "constraint" on R of being `N+N` is incorrect/infeasible (when N != 0).
//  explicit_constraints: Environment to use for evaluating the
//    parametric_constraints expressions; e.g. for the example above if the
//    caller invoked `const M: u32 = 42; f<M>(x)`, this environment would
//    be `{N: u32:42}` (since M is passed as the N value for the callee).
absl::StatusOr<TypeAndBindings> InstantiateFunction(
    Span span, const FunctionType& function_type,
    absl::Span<const InstantiateArg> args, DeduceCtx* ctx,
    absl::optional<absl::Span<const ParametricConstraint>>
        parametric_constraints = absl::nullopt,
    const absl::flat_hash_map<std::string, InterpValue>* explicit_constraints =
        nullptr);

// Instantiates a struct using the bindings derived from args' types.
//
// See InstantiateFunction() above.
absl::StatusOr<TypeAndBindings> InstantiateStruct(
    Span span, const StructType& struct_type,
    absl::Span<const InstantiateArg> args,
    absl::Span<std::unique_ptr<ConcreteType> const> member_types,
    DeduceCtx* ctx,
    absl::optional<absl::Span<const ParametricConstraint>> parametric_bindings =
        absl::nullopt);

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
                         absl::optional<absl::Span<const ParametricConstraint>>
                             parametric_constraints,
                         const absl::flat_hash_map<std::string, InterpValue>*
                             explicit_constraints);

  ParametricInstantiator(ParametricInstantiator&& other) = default;

  ParametricInstantiator(const ParametricInstantiator& other) = delete;
  ParametricInstantiator& operator=(const ParametricInstantiator& other) =
      delete;

  virtual ~ParametricInstantiator() = default;

  virtual absl::StatusOr<TypeAndBindings> Instantiate() = 0;

 protected:
  // Binds param_type via arg_type, updating 'symbolic_bindings_'.
  absl::StatusOr<std::unique_ptr<ConcreteType>> InstantiateOneArg(
      int64_t i, const ConcreteType& param_type, const ConcreteType& arg_type);

  // Verifies all constraints and then resolves possibly-parametric type
  // 'annotated' via 'symbolic_bindings_'.
  absl::StatusOr<std::unique_ptr<ConcreteType>> Resolve(
      const ConcreteType& annotated);

  const absl::flat_hash_map<std::string, InterpValue>& symbolic_bindings()
      const {
    return symbolic_bindings_;
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
  // actual argument is a u32, we'll bind `N=32` in the `symbolic_bindings_`
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
  absl::flat_hash_map<std::string, Expr*> constraints_;

  absl::flat_hash_map<std::string, InterpValue> bit_widths_;
  absl::flat_hash_map<std::string, InterpValue> symbolic_bindings_;
};

// Instantiates a parametric function invocation.
class FunctionInstantiator : public ParametricInstantiator {
 public:
  static absl::StatusOr<FunctionInstantiator> Make(
      Span span, const FunctionType& function_type,
      absl::Span<const InstantiateArg> args, DeduceCtx* ctx,
      absl::optional<absl::Span<const ParametricConstraint>>
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
                       absl::optional<absl::Span<const ParametricConstraint>>
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
  static absl::StatusOr<StructInstantiator> Make(
      Span span, const StructType& struct_type,
      absl::Span<const InstantiateArg> args,
      absl::Span<std::unique_ptr<ConcreteType> const> member_types,
      DeduceCtx* ctx,
      absl::optional<absl::Span<const ParametricConstraint>>
          parametric_bindings);

  absl::StatusOr<TypeAndBindings> Instantiate() override;

 private:
  StructInstantiator(
      Span span, const StructType& struct_type,
      absl::Span<const InstantiateArg> args,
      absl::Span<std::unique_ptr<ConcreteType> const> member_types,
      DeduceCtx* ctx,
      absl::optional<absl::Span<const ParametricConstraint>>
          parametric_bindings)
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

#endif  // XLS_DSLX_CPP_PARAMETRIC_INSTANTIATOR_H_
