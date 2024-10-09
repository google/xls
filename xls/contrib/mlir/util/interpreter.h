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

#ifndef GDM_HW_MLIR_XLS_UTIL_INTERPRETER_H_
#define GDM_HW_MLIR_XLS_UTIL_INTERPRETER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/include/llvm/ADT/DenseMap.h"
#include "llvm/include/llvm/ADT/ScopedHashTable.h"
#include "mlir/include/mlir/Analysis/Liveness.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/OperationSupport.h"
#include "mlir/include/mlir/IR/OwningOpRef.h"
#include "mlir/include/mlir/IR/Region.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/Support/DebugStringHelper.h"
#include "xls/common/status/status_macros.h"

namespace mlir::xls {

// A generic interpreter over an MLIR function. Each Value is represented
// by a ValueType. The interpreter deals with control flow and the lifetime
// of Values.
//
// An Interpreter is subclassed via CRTP to provide support for interpreting
// custom ops. It is provided an InterpeterContext CRTP subclass that handles
// value ownership and conversion of values to bool/int for control flow.
//
// Example:
//  class MyContext : public
//     InterpreterContext<MyContext, std::shared_ptr<int>> {
//   public:
//    absl::StatusOr<bool> GetValueAsPred(std::shared_ptr<int> value_type) {
//      ... used for mhlo::IfOp ...
//    }
//  };
//  class MyInterpreter : public Interpreter<MyInterpreter, MyContext,
//                                           std::shared_ptr<int>,
//                                           MyOp> {
//   public:
//    using Interpreter::Interpret;
//    absl::Status Interpret(MyOp op, MyContext& ctx) {
//     ...
//    }
//  };
//
// The representation type (std::shared_ptr<int> in this case) must be
// shared_ptr-like:
//   * Moveable, copyable.
//   * Default-constructible.
//   * Assignment to the default constructor (x = {}) deallocates any resources.
//
// Arguments:
//  Context: A subclass of InterpreterContext. Contains implementation-specific
//    op implementations and implementation-defined sdate.
//  ValueType: The type of the representation of a Value. This should
//    have value semantics and be copyable and movable. std::shared_ptr<T>
//    is an example ValueType that fulfills these requirements.
//  DerivedOps...: The concrete ops that the derived interpreter implements
//    support for. Interpret(OpT, Context) will be called for each OpT in
//    DerivedOps.
template <typename Derived, typename Context, typename ValueType,
          typename... DerivedOps>
class Interpreter {
 public:
  // The representation of values internally.
  using value_type = ValueType;

  // Interpreter entry point. Interprets `func` from beginning to end.
  // REQUIRES: each region within `func` has only one block.
  //
  // Takes a value_type for each function argument as input and returns
  // a value_type for each function result.
  //
  // The trailing arguments are used to construct a Context.
  template <typename... T>
  absl::StatusOr<std::vector<value_type>> InterpretFunc(
      mlir::func::FuncOp func, absl::Span<value_type const> arguments,
      T&&... context_args);

 protected:
  Derived& derived() { return *static_cast<Derived*>(this); }

  // Interpret implementation for builtin ops. Subclasses should implement
  // Interpret() for op types within the ...DerivedOps type parameter.
  absl::Status Interpret(mlir::func::CallOp op, Context& ctx);
  absl::Status Interpret(mlir::func::ReturnOp ret, Context& ctx);

  // Interprets all ops within `region` from start to terminator.
  absl::StatusOr<std::vector<value_type>> Interpret(
      Region& region, absl::Span<value_type const> values, Context& ctx);

  template <typename OpTy, typename... Rest>
  absl::Status InterpretImpl(Operation* op, Context& ctx) {
    if (OpTy o = dyn_cast<OpTy>(op)) {
      return derived().Interpret(o, ctx);
    }
    if constexpr (sizeof...(Rest) > 0) {
      return InterpretImpl<Rest...>(op, ctx);
    } else {
      return absl::UnimplementedError("Interpreter op unimplemented:" +
                                      op->getName().getStringRef().str());
    }
  }

  absl::Status Interpret(Operation* op, Context& ctx) {
    return InterpretImpl<mlir::func::CallOp, mlir::func::ReturnOp,
                         DerivedOps...>(op, ctx);
  }

  // Obtains a Liveness for `func`. This object will never be invalidated.
  // Thread-safe.
  mlir::Liveness* GetOrCreateLiveness(Operation* op);

 private:
  // TODO(jpienaar): Not currently thread safe.
  llvm::DenseMap<Operation*, std::shared_ptr<mlir::Liveness>> liveness_;
};

// The InterpreterContext holds information specific to a particular execution
// of the interpreter. It is passed to all op implementations and is the storage
// for ValueTypes.
//
// Arguments:
//  Derived: The CRTP subclass defined by the user.
//  ValueType: The type of the representation of a Value. This should
//    have value semantics and be copyable and movable. std::shared_ptr<T>
//    is an example ValueType that fulfills these requirements.
template <typename Derived, typename ValueType>
class InterpreterContext {
 public:
  using value_type = ValueType;

  // Sets the representation for a range of values to `contents`.
  void Set(ValueRange values, absl::Span<value_type const> contents);

  void Set(Value value, value_type content) {
    value_type contents[1] = {std::move(content)};
    Set(ValueRange{value}, absl::MakeConstSpan(contents));
  }

  // Obtains the representation for a range of values. If any value
  // does not have a representation (as called by Set), returns a
  // FAILED_PRECONDITION error.
  absl::StatusOr<std::vector<value_type>> Get(ValueRange values);

  absl::StatusOr<value_type> Get(Value value) {
    absl::StatusOr<std::vector<ValueType>> results = Get(ValueRange{value});
    if (!results.ok()) {
      return results.status();
    }
    return (*results)[0];
  }

  // Clears (sets to {}) the representation for `value`.
  void Clear(Value value) { map_.insert(value, {}); }

  // Pushes a new scope. The returned object is RAII and will pop the scope
  // when destroyed.
  llvm::ScopedHashTableScope<Value, value_type> Scope() { return {map_}; }

  // Returns true if the representation for `value`, which must be an operand
  // to `op`, may be reused or modified in computing the output of `op`.
  //
  // If this returns true, the value is dead after computation of `op`.
  bool InputMayBeReused(Value value, Operation* op);

  // Given a value_type, returns it as a predicate.
  absl::StatusOr<bool> GetValueAsPred(const value_type&) {
    return absl::UnimplementedError("GetValueAsPred unimplemented");
  }

  // Given a value_type, returns it as an integer.
  absl::StatusOr<int64_t> GetValueAsSwitchCase(const value_type&) {
    return absl::UnimplementedError("GetValueAsSwitchCase unimplemented");
  }

  // Returns the current liveness information. This will always be non-nullptr.
  mlir::Liveness* liveness() { return liveness_stack_.back(); }

  void PushLiveness(mlir::Liveness* liveness) {
    liveness_stack_.push_back(liveness);
  }
  void PopLiveness() { liveness_stack_.pop_back(); }

 private:
  llvm::ScopedHashTable<Value, value_type> map_;
  std::vector<mlir::Liveness*> liveness_stack_;
};

///////////////////////////////////////////////////////////////////////////////
// Implementation details

template <typename Derived, typename ValueType>
void InterpreterContext<Derived, ValueType>::Set(
    ValueRange values, absl::Span<ValueType const> contents) {
  CHECK_EQ(values.size(), contents.size());
  for (std::size_t i = 0; i < contents.size(); ++i) {
    map_.insert(values[i], std::move(contents[i]));
  }
}

template <typename Derived, typename ValueType>
absl::StatusOr<std::vector<ValueType>>
InterpreterContext<Derived, ValueType>::Get(ValueRange values) {
  std::vector<value_type> contents;
  for (Value value : values) {
    // We perform a count() + lookup() because ScopedHashTable has no find().
    if (map_.count(value) == 0) {
      return absl::NotFoundError("Representation for value " +
                                 mlir::debugString(value));
    }
    contents.push_back(map_.lookup(value));
  }
  return contents;
}

template <typename Derived, typename ValueType>
bool InterpreterContext<Derived, ValueType>::InputMayBeReused(Value value,
                                                              Operation* op) {
  return liveness()->isDeadAfter(value, op);
}

template <typename Derived, typename Context, typename ValueType,
          typename... DerivedOps>
absl::StatusOr<std::vector<ValueType>>
Interpreter<Derived, Context, ValueType, DerivedOps...>::Interpret(
    Region& region, absl::Span<value_type const> values, Context& ctx) {
  if (region.getBlocks().size() != 1) {
    return absl::InvalidArgumentError("Only single block regions are allowed");
  }
  Block& block = region.getBlocks().front();

  auto scope = ctx.Scope();
  ctx.Set(block.getArguments(), values);
  for (Operation& op : block.getOperations()) {
    XLS_RETURN_IF_ERROR(Interpret(&op, ctx));

    // Delete any now-dead values, but don't do this if this is a terminator
    // (like ReturnOp), because those are live-out.
    if (!block.mightHaveTerminator() || block.getTerminator() != &op) {
      for (Value operand : op.getOperands()) {
        if (ctx.liveness()->isDeadAfter(operand, &op)) {
          ctx.Clear(operand);
        }
      }
    }
  }

  // We only support ReturnOp-like terminators.
  if (!block.mightHaveTerminator()) {
    return std::vector<ValueType>{};
  }
  return ctx.Get(block.getTerminator()->getOperands());
}

template <typename Derived, typename Context, typename ValueType,
          typename... DerivedOps>
absl::Status Interpreter<Derived, Context, ValueType, DerivedOps...>::Interpret(
    mlir::func::ReturnOp /*ret*/, Context& /*ctx*/) {
  return absl::OkStatus();
}

template <typename Derived, typename Context, typename ValueType,
          typename... DerivedOps>
absl::Status Interpreter<Derived, Context, ValueType, DerivedOps...>::Interpret(
    mlir::func::CallOp op, Context& ctx) {
  mlir::func::FuncOp callee =
      op->getParentOfType<mlir::ModuleOp>().lookupSymbol<mlir::func::FuncOp>(
          op.getCallee());
  if (!callee) {
    return absl::NotFoundError("Function callee not found for " +
                               mlir::debugString(op));
  }

  ctx.PushLiveness(GetOrCreateLiveness(callee));
  absl::Cleanup popper = [&] { ctx.PopLiveness(); };

  XLS_ASSIGN_OR_RETURN(auto arguments, ctx.Get(op.getArgOperands()));
  if (arguments.size() != callee.getNumArguments()) {
    return absl::InternalError(absl::StrFormat(
        "Call to %s requires %d arguments but got %d", op.getCallee(),
        callee.getNumArguments(), arguments.size()));
  }
  XLS_ASSIGN_OR_RETURN(auto results,
                       Interpret(callee.getBody(), arguments, ctx));

  ctx.Set(op.getResults(), std::move(results));
  return absl::OkStatus();
}

template <typename Derived, typename Context, typename ValueType,
          typename... DerivedOps>
template <typename... T>
absl::StatusOr<std::vector<ValueType>>
Interpreter<Derived, Context, ValueType, DerivedOps...>::InterpretFunc(
    mlir::func::FuncOp func, absl::Span<value_type const> arguments,
    T&&... context_args) {
  Context ctx(std::forward<T>(context_args)...);
  ctx.PushLiveness(GetOrCreateLiveness(func));
  return Interpret(func.getBody(), arguments, ctx);
}

template <typename Derived, typename Context, typename ValueType,
          typename... DerivedOps>
mlir::Liveness* Interpreter<Derived, Context, ValueType,
                            DerivedOps...>::GetOrCreateLiveness(Operation* op) {
  auto it = liveness_.find(op);
  if (it != liveness_.end()) {
    return it->second.get();
  }
  return (liveness_[op] = std::make_shared<mlir::Liveness>(op)).get();
}

}  // namespace mlir::xls

#endif  // GDM_HW_MLIR_XLS_UTIL_INTERPRETER_H_
