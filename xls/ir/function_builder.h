// Copyright 2020 Google LLC
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

#ifndef XLS_IR_FUNCTION_BUILDER_H_
#define XLS_IR_FUNCTION_BUILDER_H_

#include "absl/strings/string_view.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/ir/proc.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"

namespace xls {

class BuilderBase;
class FunctionBuilder;
class Node;

// Represents a value for use in the function-definition building process,
// supports some basic C++ operations that have a natural correspondence to the
// staged versions.
class BValue {
 public:
  // The vacuous constructor lets us use values in vectors and as values that
  // are subsequently initialized. Check valid() to ensure proper
  // initialization.
  BValue() : BValue(nullptr, nullptr) {}

  BValue(Node* node, BuilderBase* builder) : node_(node), builder_(builder) {}

  BuilderBase* builder() const { return builder_; }

  Node* node() const { return node_; }
  Type* GetType() const { return node()->GetType(); }
  int64 BitCountOrDie() const { return node()->BitCountOrDie(); }
  absl::optional<SourceLocation> loc() const { return node_->loc(); }
  std::string ToString() const {
    return node_ == nullptr ? std::string("<null BValue>") : node_->ToString();
  }

  bool valid() const {
    XLS_CHECK_EQ(node_ == nullptr, builder_ == nullptr);
    return node_ != nullptr;
  }

  BValue operator>>(BValue rhs);
  BValue operator<<(BValue rhs);
  BValue operator|(BValue rhs);
  BValue operator-(BValue rhs);
  BValue operator+(BValue rhs);
  BValue operator*(BValue rhs);
  BValue operator^(BValue rhs);

  // Note: unary negation.
  BValue operator-();

 private:
  Node* node_;
  BuilderBase* builder_;
};

// Base class for function and proc. Provides interface for adding nodes to a
// function proc. Example usage (for derived FunctionBuilder class):
//
//  Package p("my_package");
//  FunctionBuilder b("my_or_function_32b", &p);
//  auto bits_32 = m.GetBitsType(32);
//  auto x = b.Param("x", bits_32);
//  auto y = b.Param("y", bits_32);
//  XLS_ASSIGN_OR_RETURN(Function* f, b.BuildWithReturnValue(x | y));
//
// Note that the BValues returned by builder routines are specific to that
// function, and attempt to use those BValues with FunctionBuilders that did not
// originate them will cause fatal errors.
class BuilderBase {
 public:
  // The given argument 'function' can contain either a Function or Proc.
  explicit BuilderBase(std::unique_ptr<Function> function)
      : function_(std::move(function)), error_pending_(false) {}

  const std::string& name() const { return function_->name(); }

  // Get access to currently built up function (or proc).
  Function* function() const { return function_.get(); }

  // Declares a parameter to the function being built of type "type".
  BValue Param(absl::string_view name, Type* type,
               absl::optional<SourceLocation> loc = absl::nullopt);

  // Shift right arithmetic.
  BValue Shra(BValue operand, BValue amount,
              absl::optional<SourceLocation> loc = absl::nullopt);

  // Shift right logical.
  BValue Shrl(BValue operand, BValue amount,
              absl::optional<SourceLocation> loc = absl::nullopt);

  // Shift left (logical).
  BValue Shll(BValue operand, BValue amount,
              absl::optional<SourceLocation> loc = absl::nullopt);

  // Bitwise or.
  BValue Or(BValue lhs, BValue rhs,
            absl::optional<SourceLocation> loc = absl::nullopt);
  BValue Or(absl::Span<const BValue> operands,
            absl::optional<SourceLocation> loc = absl::nullopt);

  // Bitwise xor.
  BValue Xor(BValue lhs, BValue rhs,
             absl::optional<SourceLocation> loc = absl::nullopt);

  // Bitwise and.
  BValue And(BValue lhs, BValue rhs,
             absl::optional<SourceLocation> loc = absl::nullopt);

  // Unary and-reduction.
  BValue AndReduce(BValue operand,
                   absl::optional<SourceLocation> loc = absl::nullopt);

  // Unary or-reduction.
  BValue OrReduce(BValue operand,
                  absl::optional<SourceLocation> loc = absl::nullopt);

  // Unary xor-reduction.
  BValue XorReduce(BValue operand,
                   absl::optional<SourceLocation> loc = absl::nullopt);

  // Signed/unsigned multiply. Result width is the same as the operand
  // width. Operand widths must be the same.
  BValue UMul(BValue lhs, BValue rhs,
              absl::optional<SourceLocation> loc = absl::nullopt);
  BValue SMul(BValue lhs, BValue rhs,
              absl::optional<SourceLocation> loc = absl::nullopt);

  // Signed/unsigned multiply with explicitly specified result width. Operand
  // widths can be arbitrary.
  BValue UMul(BValue lhs, BValue rhs, int64 result_width,
              absl::optional<SourceLocation> loc = absl::nullopt);
  BValue SMul(BValue lhs, BValue rhs, int64 result_width,
              absl::optional<SourceLocation> loc = absl::nullopt);

  // Unsigned division.
  BValue UDiv(BValue lhs, BValue rhs,
              absl::optional<SourceLocation> loc = absl::nullopt);

  // Two's complement subtraction/addition.
  BValue Subtract(BValue lhs, BValue rhs,
                  absl::optional<SourceLocation> loc = absl::nullopt);
  BValue Add(BValue lhs, BValue rhs,
             absl::optional<SourceLocation> loc = absl::nullopt);

  // Concatenates the operands (zero-th operand are the most significant bits in
  // the result).
  BValue Concat(absl::Span<const BValue> operands,
                absl::optional<SourceLocation> loc = absl::nullopt);

  // Unsigned less-or-equal.
  BValue ULe(BValue lhs, BValue rhs,
             absl::optional<SourceLocation> loc = absl::nullopt);
  // Unsigned less-than.
  BValue ULt(BValue lhs, BValue rhs,
             absl::optional<SourceLocation> loc = absl::nullopt);
  // Unsigned greater-or-equal.
  BValue UGe(BValue lhs, BValue rhs,
             absl::optional<SourceLocation> loc = absl::nullopt);
  // Unsigned greater-than.
  BValue UGt(BValue lhs, BValue rhs,
             absl::optional<SourceLocation> loc = absl::nullopt);

  // Signed less-or-equal.
  BValue SLe(BValue lhs, BValue rhs,
             absl::optional<SourceLocation> loc = absl::nullopt);
  // Signed less-than.
  BValue SLt(BValue lhs, BValue rhs,
             absl::optional<SourceLocation> loc = absl::nullopt);
  // Signed greater-or-equal.
  BValue SGe(BValue lhs, BValue rhs,
             absl::optional<SourceLocation> loc = absl::nullopt);
  // Signed greater-than.
  BValue SGt(BValue lhs, BValue rhs,
             absl::optional<SourceLocation> loc = absl::nullopt);

  // Equal.
  BValue Eq(BValue lhs, BValue rhs,
            absl::optional<SourceLocation> loc = absl::nullopt);
  // Not Equal.
  BValue Ne(BValue lhs, BValue rhs,
            absl::optional<SourceLocation> loc = absl::nullopt);

  // Two's complement (arithmetic) negation.
  BValue Negate(BValue x, absl::optional<SourceLocation> loc = absl::nullopt);

  // One's complement negation (bitwise not).
  BValue Not(BValue x, absl::optional<SourceLocation> loc = absl::nullopt);

  // Turns a literal value into a handle that can be used in this function being
  // built.
  BValue Literal(Value value,
                 absl::optional<SourceLocation> loc = absl::nullopt);
  BValue Literal(Bits bits,
                 absl::optional<SourceLocation> loc = absl::nullopt) {
    return Literal(Value(bits), loc);
  }

  // An n-ary select which selects among an arbitrary number of cases based on
  // the selector value. If the number of cases is less than 2**selector_width
  // then a default value must be specified.
  BValue Select(BValue selector, absl::Span<const BValue> cases,
                absl::optional<BValue> default_value = absl::nullopt,
                absl::optional<SourceLocation> loc = absl::nullopt);

  // An overload for binary select: selects on_true when selector is true,
  // on_false otherwise.
  // TODO(meheff): switch positions of on_true and on_false to match the
  // ordering of the cases span in the n-ary select.
  BValue Select(BValue selector, BValue on_true, BValue on_false,
                absl::optional<SourceLocation> loc = absl::nullopt);

  // Creates a one-hot operation which generates a one-hot bits value from its
  // inputs.
  BValue OneHot(BValue input, LsbOrMsb priority,
                absl::optional<SourceLocation> loc = absl::nullopt);

  // Creates a select operation which uses a one-hot selector (rather than a
  // binary encoded selector as used in Select).
  BValue OneHotSelect(BValue selector, absl::Span<const BValue> cases,
                      absl::optional<SourceLocation> loc = absl::nullopt);

  // Counts the number of leading zeros in the value 'x'.
  //
  // x must be of bits type. This function returns a value of the same type that
  // it takes as an argument.
  BValue Clz(BValue x, absl::optional<SourceLocation> loc = absl::nullopt);

  // Counts the number of trailing zeros in the value 'x'.
  //
  // x must be of bits type. This function returns a value of the same type that
  // it takes as an argument.
  BValue Ctz(BValue x, absl::optional<SourceLocation> loc = absl::nullopt);

  // Creates an match operation which compares 'condition' against each of the
  // case clauses and produces the respective case value of the first match. If
  // 'condition' matches no cases 'default_value' is produced. 'condition' and
  // each case clause must be of the same type. 'default_value' and each case
  // value must be of the same type. The match operation is composed of multiple
  // IR nodes including one-hot and one-hot-select.
  struct Case {
    BValue clause;
    BValue value;
  };
  BValue Match(BValue condition, absl::Span<const Case> cases,
               BValue default_value,
               absl::optional<SourceLocation> loc = absl::nullopt);

  // Creates a match operation which matches each of the case clauses against
  // the single-bit literal one. Each case clause must be a single-bit Bits
  // value.
  BValue MatchTrue(absl::Span<const Case> cases, BValue default_value,
                   absl::optional<SourceLocation> loc = absl::nullopt);
  // Overload which is easier to expose to Python.
  BValue MatchTrue(absl::Span<const BValue> case_clauses,
                   absl::Span<const BValue> case_values, BValue default_value,
                   absl::optional<SourceLocation> loc = absl::nullopt);

  // Creates a tuple of values.
  //
  // Note that "loc" would generally refer to the source location whether the
  // tuple expression begins; e.g. in DSLX:
  //
  //    let x = (1, 2, 3, 4) in ...
  //  ~~~~~~~~~~^
  //
  //  The arrow indicates the source location for the tuple expression.
  BValue Tuple(absl::Span<const BValue> elements,
               absl::optional<SourceLocation> loc = absl::nullopt);

  // Creates an AfterAll ordering operation.
  BValue AfterAll(absl::Span<const BValue> dependencies,
                  absl::optional<SourceLocation> loc = absl::nullopt);

  // Creates an array of values. Each value in element must be the same type
  // which is given by element_type.
  BValue Array(absl::Span<const BValue> elements, Type* element_type,
               absl::optional<SourceLocation> loc = absl::nullopt);

  // Adds an tuple index expression.
  BValue TupleIndex(BValue arg, int64 idx,
                    absl::optional<SourceLocation> loc = absl::nullopt);

  // Adds a "counted for-loop" to the computation, having a known-constant
  // number of loop iterations.
  //
  // Args:
  //  trip_count: Number of iterations to execute "body".
  //  init_value: Of type "T", the starting value for the loop carry data.
  //  body: Of type "(u32, T) -> T": the body that is invoked on the loop carry
  //    data decorated with the iteration number, producing new loop carry data
  //    (of the same type) each iteration.
  //  invariant_args: Arguments that are passed to the body function in a
  //    loop-invariant fashion (for each of these arguments, the same value is
  //    passed as an argument on every trip and the value does not change).
  //  loc: Source location for this counted for-loop.
  //
  // Returns the value that results from this counted for loop after it has
  // completed all of its trips.
  BValue CountedFor(BValue init_value, int64 trip_count, int64 stride,
                    Function* body,
                    absl::Span<const BValue> invariant_args = {},
                    absl::optional<SourceLocation> loc = absl::nullopt);

  // Adds a map to the computation.
  // Applies the function to_apply to each element of the array-typed operand
  // and returns the result as an array
  //
  // Args:
  //   operand: Of type "Array<T, N>" containing N elements of type T.
  //   to_apply: Of type "T -> U".
  //   loc: Source location for this map.
  //
  // Returns a value of type "Array<U, N>".
  BValue Map(BValue operand, Function* to_apply,
             absl::optional<SourceLocation> loc = absl::nullopt);

  // Adds a function call with parameters.
  BValue Invoke(absl::Span<const BValue> args, Function* to_apply,
                absl::optional<SourceLocation> loc = absl::nullopt);

  // Adds an array index expression.
  BValue ArrayIndex(BValue arg, BValue idx,
                    absl::optional<SourceLocation> loc = absl::nullopt);

  // Updates the array element at index "idx" to update_value.
  BValue ArrayUpdate(BValue arg, BValue idx, BValue update_value,
                     absl::optional<SourceLocation> loc = absl::nullopt);

  // Reverses the order of the bits of the argument.
  BValue Reverse(BValue arg,
                 absl::optional<SourceLocation> loc = absl::nullopt);

  // Adds an identity expression.
  BValue Identity(BValue var,
                  absl::optional<SourceLocation> loc = absl::nullopt);

  // Sign-extends arg to the new_bit_count.
  BValue SignExtend(BValue arg, int64 new_bit_count,
                    absl::optional<SourceLocation> loc = absl::nullopt);

  // Zero-extends arg to the new_bit_count.
  BValue ZeroExtend(BValue arg, int64 new_bit_count,
                    absl::optional<SourceLocation> loc = absl::nullopt);

  // Extracts a slice from the bits-typed arg. 'start' is the first bit of the
  // slice and is zero-indexed where zero is the LSb of arg.
  BValue BitSlice(BValue arg, int64 start, int64 width,
                  absl::optional<SourceLocation> loc = absl::nullopt);

  // Same as BitSlice, but allows for dynamic 'start' offsets
  BValue DynamicBitSlice(BValue arg, BValue start, int64 width,
                         absl::optional<SourceLocation> loc = absl::nullopt);

  // Binary encodes the n-bit input to a log_2(n) bit output.
  BValue Encode(BValue arg, absl::optional<SourceLocation> loc = absl::nullopt);

  // Binary decodes the n-bit input to a one-hot output. 'width' can be at most
  // 2**n where n is the bit width of the operand. If 'width' is not specified
  // the output is 2**n bits wide.
  BValue Decode(BValue arg, absl::optional<int64> width = absl::nullopt,
                absl::optional<SourceLocation> loc = absl::nullopt);

  // Retrieves the type of "value" and returns it.
  Type* GetType(BValue value) { return value.node()->GetType(); }

  // Adds a Arith/UnOp/BinOp/CompareOp to the function. Exposed for
  // programmatically adding these ops using with variable Op values.
  BValue AddUnOp(Op op, BValue x,
                 absl::optional<SourceLocation> loc = absl::nullopt);
  BValue AddBinOp(Op op, BValue lhs, BValue rhs,
                  absl::optional<SourceLocation> loc = absl::nullopt);
  BValue AddCompareOp(Op op, BValue lhs, BValue rhs,
                      absl::optional<SourceLocation> loc = absl::nullopt);
  BValue AddNaryOp(Op op, absl::Span<const BValue> args,
                   absl::optional<SourceLocation> loc = absl::nullopt);
  // If result width is not given the result width set to the width of the
  // arguments lhs and rhs which must have the same width.
  BValue AddArithOp(Op op, BValue lhs, BValue rhs,
                    absl::optional<int64> result_width,
                    absl::optional<SourceLocation> loc = absl::nullopt);
  BValue AddBitwiseReductionOp(
      Op op, BValue arg, absl::optional<SourceLocation> loc = absl::nullopt);

  Package* package() const { return function_->package(); }

 protected:
  BValue SetError(std::string msg, absl::optional<SourceLocation> loc);
  bool ErrorPending() { return error_pending_; }

  // Constructs and adds a node to the function and returns a corresponding
  // BValue.
  template <typename NodeT, typename... Args>
  BValue AddNode(Args&&... args) {
    last_node_ = function_->AddNode<NodeT>(
        absl::make_unique<NodeT>(std::forward<Args>(args)..., function_.get()));
    return BValue(last_node_, this);
  }

  // The most recently added node to the function.
  Node* last_node_ = nullptr;

  // The function being built by this builder.
  std::unique_ptr<Function> function_;

  bool error_pending_;
  std::string error_msg_;
  std::string error_stacktrace_;
  absl::optional<SourceLocation> error_loc_;
};

// Class for building an XLS Function.
class FunctionBuilder : public BuilderBase {
 public:
  explicit FunctionBuilder(absl::string_view name, Package* package)
      : BuilderBase(absl::make_unique<Function>(std::string(name), package)) {}

  // Adds the function internally being built-up by this builder to the package
  // given at construction time, and returns a pointer to it (the function is
  // subsequently owned by the package and this builder should be discarded).
  //
  // The return value of the function is the most recently added operation.
  xabsl::StatusOr<Function*> Build();

  // Build function using given return value.
  xabsl::StatusOr<Function*> BuildWithReturnValue(BValue return_value);
};

// Class for building an XLS Proc (a communicating sequential process).
class ProcBuilder : public BuilderBase {
 public:
  explicit ProcBuilder(absl::string_view name, const Value& init_value,
                       absl::string_view state_name,
                       absl::string_view token_name, Package* package)
      : BuilderBase(absl::make_unique<Proc>(name, init_value, state_name,
                                            token_name, package)),
        // The parameter nodes are added at construction time. Create a BValue
        // for each param node so they may be used in construction of
        // expressions.
        state_param_(proc()->StateParam(), this),
        token_param_(proc()->TokenParam(), this) {}

  // Returns the Proc being constructed.
  Proc* proc() const { return down_cast<Proc*>(function()); }

  // Returns the Param BValue for the state or token parameter. Unlike
  // BuilderBase::Param this does add a Param node to the Proc. Rather the state
  // and token parameters are added to the Proc at construction time and these
  // methods return references to the these parameters.
  BValue GetStateParam() const { return state_param_; }
  BValue GetTokenParam() const { return token_param_; }

  // Build function using given return value.
  xabsl::StatusOr<Proc*> BuildWithReturnValue(BValue return_value);

 private:
  // The BValue of the state parameter (parameter 0).
  BValue state_param_;

  // The BValue of the token parameter (parameter 1).
  BValue token_param_;
};

}  // namespace xls

#endif  // XLS_IR_FUNCTION_BUILDER_H_
