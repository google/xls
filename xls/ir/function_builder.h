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

#ifndef XLS_IR_FUNCTION_BUILDER_H_
#define XLS_IR_FUNCTION_BUILDER_H_

// NOTE: We're trying to keep the API minimal here to not drag too much into
// publicly-exposed APIs (FunctionBuilder/ProcBuilder are publicly exposed), so
// forward decls are used.
//
// To this end, DO NOT add node/function headers here.

#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/foreign_function_data.pb.h"
#include "xls/ir/format_strings.h"
#include "xls/ir/function.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/ir/value_builder.h"

namespace xls {

class BuilderBase;
class Channel;
class Function;
class FunctionBase;
class Node;
class Proc;
class Type;

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
  Type* GetType() const;
  int64_t BitCountOrDie() const;
  const SourceInfo& loc() const;
  std::string ToString() const;

  bool valid() const {
    CHECK_EQ(node_ == nullptr, builder_ == nullptr);
    return node_ != nullptr;
  }

  // Sets the name of the node.
  std::string SetName(std::string_view name);

  // Returns the name of the node.
  std::string GetName() const;

  // Returns whether the node has been assigned a name. Nodes without assigned
  // names have names generated from the opcode and unique id.
  bool HasAssignedName() const;

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

std::ostream& operator<<(std::ostream& os, const BValue& bv);

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
  // 'should_verify' is a test-only argument which can be set to false in tests
  // that wish to build malformed IR.
  explicit BuilderBase(std::unique_ptr<FunctionBase> function,
                       bool should_verify = true);

  // Builders are neither copyable or movable- builder values contain references
  // to their builder that can't be updated.
  BuilderBase(const BuilderBase&) = delete;
  BuilderBase& operator=(const BuilderBase&) = delete;
  BuilderBase(BuilderBase&&) = delete;
  BuilderBase& operator=(BuilderBase&&) = delete;

  virtual ~BuilderBase();

  const std::string& name() const;

  // Set function as top to package.
  absl::Status SetAsTop();

  bool IsFunction() const { return function_->IsFunction(); }
  bool IsBlock() const { return function_->IsBlock(); }
  bool IsProc() const { return function_->IsProc(); }

  // Set information about foreign function if the underlying function is
  // a foreign function.
  void SetForeignFunctionData(const std::optional<ForeignFunctionData>& ff);

  // Get access to currently built up function (or proc).
  FunctionBase* function() const { return function_.get(); }

  // Declares a parameter to the function being built of type "type".
  virtual BValue Param(std::string_view name, Type* type,
                       const SourceInfo& loc = SourceInfo()) = 0;

  // Shift right arithmetic.
  BValue Shra(BValue operand, BValue amount,
              const SourceInfo& loc = SourceInfo(), std::string_view name = "");

  // Shift right logical.
  BValue Shrl(BValue operand, BValue amount,
              const SourceInfo& loc = SourceInfo(), std::string_view name = "");

  // Shift left (logical).
  BValue Shll(BValue operand, BValue amount,
              const SourceInfo& loc = SourceInfo(), std::string_view name = "");

  // Bitwise or.
  BValue Or(BValue lhs, BValue rhs, const SourceInfo& loc = SourceInfo(),
            std::string_view name = "");
  BValue Or(absl::Span<const BValue> operands,
            const SourceInfo& loc = SourceInfo(), std::string_view name = "");

  // Bitwise nor.
  BValue Nor(BValue lhs, BValue rhs, const SourceInfo& loc = SourceInfo(),
             std::string_view name = "");
  BValue Nor(absl::Span<const BValue> operands,
             const SourceInfo& loc = SourceInfo(), std::string_view name = "");

  // Bitwise xor.
  BValue Xor(BValue lhs, BValue rhs, const SourceInfo& loc = SourceInfo(),
             std::string_view name = "");
  BValue Xor(absl::Span<const BValue> operands,
             const SourceInfo& loc = SourceInfo(), std::string_view name = "");

  // Bitwise and.
  BValue And(BValue lhs, BValue rhs, const SourceInfo& loc = SourceInfo(),
             std::string_view name = "");
  BValue And(absl::Span<const BValue> operands,
             const SourceInfo& loc = SourceInfo(), std::string_view name = "");

  // Bitwise nand.
  BValue Nand(BValue lhs, BValue rhs, const SourceInfo& loc = SourceInfo(),
              std::string_view name = "");
  BValue Nand(absl::Span<const BValue> operands,
              const SourceInfo& loc = SourceInfo(), std::string_view name = "");

  // Unary and-reduction.
  BValue AndReduce(BValue operand, const SourceInfo& loc = SourceInfo(),
                   std::string_view name = "");

  // Unary or-reduction.
  BValue OrReduce(BValue operand, const SourceInfo& loc = SourceInfo(),
                  std::string_view name = "");

  // Unary xor-reduction.
  BValue XorReduce(BValue operand, const SourceInfo& loc = SourceInfo(),
                   std::string_view name = "");

  // Unsigned/signed multiply. Result width is the same as the operand
  // width. Operand widths must be the same.
  BValue UMul(BValue lhs, BValue rhs, const SourceInfo& loc = SourceInfo(),
              std::string_view name = "");
  BValue SMul(BValue lhs, BValue rhs, const SourceInfo& loc = SourceInfo(),
              std::string_view name = "");

  // Unsigned/signed multiply with explicitly specified result width. Operand
  // widths can be arbitrary.
  BValue UMul(BValue lhs, BValue rhs, int64_t result_width,
              const SourceInfo& loc = SourceInfo(), std::string_view name = "");
  BValue SMul(BValue lhs, BValue rhs, int64_t result_width,
              const SourceInfo& loc = SourceInfo(), std::string_view name = "");

  // Unsigned/signed partial product multiply. Result width is the same as the
  // operand width. Operand widths must be the same.
  BValue UMulp(BValue lhs, BValue rhs, const SourceInfo& loc = SourceInfo(),
               std::string_view name = "");
  BValue SMulp(BValue lhs, BValue rhs, const SourceInfo& loc = SourceInfo(),
               std::string_view name = "");

  // Unsigned/signed partial product multiply with explicitly specified result
  // width. Operand widths can be arbitrary.
  BValue UMulp(BValue lhs, BValue rhs, int64_t result_width,
               const SourceInfo& loc = SourceInfo(),
               std::string_view name = "");
  BValue SMulp(BValue lhs, BValue rhs, int64_t result_width,
               const SourceInfo& loc = SourceInfo(),
               std::string_view name = "");

  // Unsigned/signed division. Result width is the same as the operand
  // width. Operand widths must be the same.
  BValue UDiv(BValue lhs, BValue rhs, const SourceInfo& loc = SourceInfo(),
              std::string_view name = "");
  BValue SDiv(BValue lhs, BValue rhs, const SourceInfo& loc = SourceInfo(),
              std::string_view name = "");
  BValue UMod(BValue lhs, BValue rhs, const SourceInfo& loc = SourceInfo(),
              std::string_view name = "");
  BValue SMod(BValue lhs, BValue rhs, const SourceInfo& loc = SourceInfo(),
              std::string_view name = "");

  // Two's complement subtraction/addition.
  BValue Subtract(BValue lhs, BValue rhs, const SourceInfo& loc = SourceInfo(),
                  std::string_view name = "");
  BValue Add(BValue lhs, BValue rhs, const SourceInfo& loc = SourceInfo(),
             std::string_view name = "");

  // Concatenates the operands (zero-th operand are the most significant bits in
  // the result).
  BValue Concat(absl::Span<const BValue> operands,
                const SourceInfo& loc = SourceInfo(),
                std::string_view name = "");

  // Unsigned less-or-equal.
  BValue ULe(BValue lhs, BValue rhs, const SourceInfo& loc = SourceInfo(),
             std::string_view name = "");
  // Unsigned less-than.
  BValue ULt(BValue lhs, BValue rhs, const SourceInfo& loc = SourceInfo(),
             std::string_view name = "");
  // Unsigned greater-or-equal.
  BValue UGe(BValue lhs, BValue rhs, const SourceInfo& loc = SourceInfo(),
             std::string_view name = "");
  // Unsigned greater-than.
  BValue UGt(BValue lhs, BValue rhs, const SourceInfo& loc = SourceInfo(),
             std::string_view name = "");

  // Signed less-or-equal.
  BValue SLe(BValue lhs, BValue rhs, const SourceInfo& loc = SourceInfo(),
             std::string_view name = "");
  // Signed less-than.
  BValue SLt(BValue lhs, BValue rhs, const SourceInfo& loc = SourceInfo(),
             std::string_view name = "");
  // Signed greater-or-equal.
  BValue SGe(BValue lhs, BValue rhs, const SourceInfo& loc = SourceInfo(),
             std::string_view name = "");
  // Signed greater-than.
  BValue SGt(BValue lhs, BValue rhs, const SourceInfo& loc = SourceInfo(),
             std::string_view name = "");

  // Equal.
  BValue Eq(BValue lhs, BValue rhs, const SourceInfo& loc = SourceInfo(),
            std::string_view name = "");
  // Not Equal.
  BValue Ne(BValue lhs, BValue rhs, const SourceInfo& loc = SourceInfo(),
            std::string_view name = "");

  // Two's complement (arithmetic) negation.
  BValue Negate(BValue x, const SourceInfo& loc = SourceInfo(),
                std::string_view name = "");

  // One's complement negation (bitwise not).
  BValue Not(BValue x, const SourceInfo& loc = SourceInfo(),
             std::string_view name = "");

  // Turns a literal value into a handle that can be used in this function being
  // built.
  BValue Literal(Value value, const SourceInfo& loc = SourceInfo(),
                 std::string_view name = "");
  BValue Literal(ValueBuilder value, const SourceInfo& loc = SourceInfo(),
                 std::string_view name = "");
  BValue Literal(Bits bits, const SourceInfo& loc = SourceInfo(),
                 std::string_view name = "") {
    return Literal(Value(bits), loc, name);
  }

  // An n-ary select which selects among an arbitrary number of cases based on
  // the selector value. If the number of cases is less than 2**selector_width
  // then a default value must be specified.
  BValue Select(BValue selector, absl::Span<const BValue> cases,
                std::optional<BValue> default_value = std::nullopt,
                const SourceInfo& loc = SourceInfo(),
                std::string_view name = "");

  // An overload for binary select: selects on_true when selector is true,
  // on_false otherwise.
  // TODO(meheff): switch positions of on_true and on_false to match the
  // ordering of the cases span in the n-ary select.
  BValue Select(BValue selector, BValue on_true, BValue on_false,
                const SourceInfo& loc = SourceInfo(),
                std::string_view name = "");

  // Creates a one-hot operation which generates a one-hot bits value from its
  // inputs.
  BValue OneHot(BValue input, LsbOrMsb priority,
                const SourceInfo& loc = SourceInfo(),
                std::string_view name = "");

  // Creates a select operation which uses a one-hot selector (rather than a
  // binary encoded selector as used in Select).
  BValue OneHotSelect(BValue selector, absl::Span<const BValue> cases,
                      const SourceInfo& loc = SourceInfo(),
                      std::string_view name = "");

  // Creates a select operation which uses a one-hot selector (rather than a
  // binary encoded selector as used in Select).
  BValue PrioritySelect(BValue selector, absl::Span<const BValue> cases,
                        BValue default_value,
                        const SourceInfo& loc = SourceInfo(),
                        std::string_view name = "");

  // Counts the number of leading zeros in the value 'x'.
  //
  // x must be of bits type. This function returns a value of the same type that
  // it takes as an argument.
  BValue Clz(BValue x, const SourceInfo& loc = SourceInfo(),
             std::string_view name = "");

  // Counts the number of trailing zeros in the value 'x'.
  //
  // x must be of bits type. This function returns a value of the same type that
  // it takes as an argument.
  BValue Ctz(BValue x, const SourceInfo& loc = SourceInfo(),
             std::string_view name = "");

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
               BValue default_value, const SourceInfo& loc = SourceInfo(),
               std::string_view name = "");

  // Creates a match operation which matches each of the case clauses against
  // the single-bit literal one. Each case clause must be a single-bit Bits
  // value.
  BValue MatchTrue(absl::Span<const Case> cases, BValue default_value,
                   const SourceInfo& loc = SourceInfo(),
                   std::string_view name = "");
  // Overload which is easier to expose to Python.
  BValue MatchTrue(absl::Span<const BValue> case_clauses,
                   absl::Span<const BValue> case_values, BValue default_value,
                   const SourceInfo& loc = SourceInfo(),
                   std::string_view name = "");

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
               const SourceInfo& loc = SourceInfo(),
               std::string_view name = "");

  // Creates an AfterAll ordering operation.
  BValue AfterAll(absl::Span<const BValue> dependencies,
                  const SourceInfo& loc = SourceInfo(),
                  std::string_view name = "");

  // Creates a MinDelay constraint operation.
  BValue MinDelay(BValue token, int64_t delay,
                  const SourceInfo& loc = SourceInfo(),
                  std::string_view name = "");

  // Creates an array of values. Each value in element must be the same type
  // which is given by element_type.
  BValue Array(absl::Span<const BValue> elements, Type* element_type,
               const SourceInfo& loc = SourceInfo(),
               std::string_view name = "");

  // Adds an tuple index expression.
  BValue TupleIndex(BValue arg, int64_t idx,
                    const SourceInfo& loc = SourceInfo(),
                    std::string_view name = "");

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
  BValue CountedFor(BValue init_value, int64_t trip_count, int64_t stride,
                    Function* body,
                    absl::Span<const BValue> invariant_args = {},
                    const SourceInfo& loc = SourceInfo(),
                    std::string_view name = "");

  // Adds a "dynamic counted for-loop" to the computation, having a
  // dynamically determined number of loop iterations and stride.
  //
  // Args:
  //  trip_count: Of bits type, number of iterations to execute "body".
  //  stride: Of bits type, count by which to increment induction variable.
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
  BValue DynamicCountedFor(BValue init_value, BValue trip_count, BValue stride,
                           Function* body,
                           absl::Span<const BValue> invariant_args = {},
                           const SourceInfo& loc = SourceInfo(),
                           std::string_view name = "");

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
             const SourceInfo& loc = SourceInfo(), std::string_view name = "");

  // Adds a function call with parameters.
  BValue Invoke(absl::Span<const BValue> args, Function* to_apply,
                const SourceInfo& loc = SourceInfo(),
                std::string_view name = "");

  // Adds an multi-dimensional array index expression. The indices should be all
  // bits types.
  BValue ArrayIndex(BValue arg, absl::Span<const BValue> indices,
                    bool known_in_bounds, const SourceInfo& loc = SourceInfo(),
                    std::string_view name = "");

  // Adds an multi-dimensional array index expression. The indices should be all
  // bits types.
  BValue ArrayIndex(BValue arg, absl::Span<const BValue> indices,
                    const SourceInfo& loc = SourceInfo(),
                    std::string_view name = "") {
    return ArrayIndex(arg, indices, /*known_in_bounds=*/false, loc, name);
  }

  // Slices an array with a given start and end position.
  BValue ArraySlice(BValue array, BValue start, int64_t width,
                    const SourceInfo& loc = SourceInfo(),
                    std::string_view name = "");

  // Updates the array element at index "idx" to update_value. The indices
  // should be all bits types.
  BValue ArrayUpdate(BValue arg, BValue update_value,
                     absl::Span<const BValue> indices, bool known_in_bounds,
                     const SourceInfo& loc = SourceInfo(),
                     std::string_view name = "");

  // Updates the array element at index "idx" to update_value. The indices
  // should be all bits types.
  BValue ArrayUpdate(BValue arg, BValue update_value,
                     absl::Span<const BValue> indices,
                     const SourceInfo& loc = SourceInfo(),
                     std::string_view name = "") {
    return ArrayUpdate(arg, update_value, indices, /*known_in_bounds=*/false,
                       loc, name);
  }

  // Concatenates array operands into a single array.  zero-th
  // element is the zero-th element of the zero-th (left-most) array.
  BValue ArrayConcat(absl::Span<const BValue> operands,
                     const SourceInfo& loc = SourceInfo(),
                     std::string_view name = "");

  // Reverses the order of the bits of the argument.
  BValue Reverse(BValue arg, const SourceInfo& loc = SourceInfo(),
                 std::string_view name = "");

  // Adds an identity expression.
  BValue Identity(BValue var, const SourceInfo& loc = SourceInfo(),
                  std::string_view name = "");

  // Sign-extends arg to the new_bit_count.
  BValue SignExtend(BValue arg, int64_t new_bit_count,
                    const SourceInfo& loc = SourceInfo(),
                    std::string_view name = "");

  // Zero-extends arg to the new_bit_count.
  BValue ZeroExtend(BValue arg, int64_t new_bit_count,
                    const SourceInfo& loc = SourceInfo(),
                    std::string_view name = "");

  // Extracts a slice from the bits-typed arg. 'start' is the first bit of the
  // slice and is zero-indexed where zero is the LSb of arg.
  BValue BitSlice(BValue arg, int64_t start, int64_t width,
                  const SourceInfo& loc = SourceInfo(),
                  std::string_view name = "");

  // Updates a slice of the given bits-typed 'arg' starting at index 'start'
  // with 'update_value'.
  BValue BitSliceUpdate(BValue arg, BValue start, BValue update_value,
                        const SourceInfo& loc = SourceInfo(),
                        std::string_view name = "");

  // Same as BitSlice, but allows for dynamic 'start' offsets
  BValue DynamicBitSlice(BValue arg, BValue start, int64_t width,
                         const SourceInfo& loc = SourceInfo(),
                         std::string_view name = "");

  // Binary encodes the n-bit input to a log_2(n) bit output.
  BValue Encode(BValue arg, const SourceInfo& loc = SourceInfo(),
                std::string_view name = "");

  // Binary decodes the n-bit input to a one-hot output. 'width' can be at most
  // 2**n where n is the bit width of the operand. If 'width' is not specified
  // the output is 2**n bits wide.
  BValue Decode(BValue arg, std::optional<int64_t> width = std::nullopt,
                const SourceInfo& loc = SourceInfo(),
                std::string_view name = "");

  // Retrieves the type of "value" and returns it.
  Type* GetType(const BValue& value) { return value.GetType(); }

  // Adds a Arith/UnOp/BinOp/CompareOp to the function. Exposed for
  // programmatically adding these ops using with variable Op values.
  BValue AddUnOp(Op op, BValue x, const SourceInfo& loc = SourceInfo(),
                 std::string_view name = "");
  BValue AddBinOp(Op op, BValue lhs, BValue rhs,
                  const SourceInfo& loc = SourceInfo(),
                  std::string_view name = "");
  BValue AddCompareOp(Op op, BValue lhs, BValue rhs,
                      const SourceInfo& loc = SourceInfo(),
                      std::string_view name = "");
  BValue AddNaryOp(Op op, absl::Span<const BValue> args,
                   const SourceInfo& loc = SourceInfo(),
                   std::string_view name = "");
  // If result width is not given the result width set to the width of the
  // arguments lhs and rhs which must have the same width.
  BValue AddArithOp(Op op, BValue lhs, BValue rhs,
                    std::optional<int64_t> result_width,
                    const SourceInfo& loc = SourceInfo(),
                    std::string_view name = "");
  BValue AddPartialProductOp(Op op, BValue lhs, BValue rhs,
                             std::optional<int64_t> result_width,
                             const SourceInfo& loc = SourceInfo(),
                             std::string_view name = "");
  BValue AddBitwiseReductionOp(Op op, BValue arg,
                               const SourceInfo& loc = SourceInfo(),
                               std::string_view name = "");

  // Adds an assert op to the function. Assert raises an error containing the
  // given message if the given condition evaluates to false.
  BValue Assert(BValue token, BValue condition, std::string_view message,
                std::optional<std::string> label = std::nullopt,
                const SourceInfo& loc = SourceInfo(),
                std::string_view name = "");

  // Adds a trace op to the function. In simulation, when the condition is true
  // the traced data will be printed.
  BValue Trace(BValue token, BValue condition, absl::Span<const BValue> args,
               absl::Span<const FormatStep> format, int64_t verbosity = 0,
               const SourceInfo& loc = SourceInfo(),
               std::string_view name = "");

  // Overloaded version of Trace that parses a format string argument instead of
  // directly requiring the parsed form.
  BValue Trace(BValue token, BValue condition, absl::Span<const BValue> args,
               std::string_view format_string, int64_t verbosity = 0,
               const SourceInfo& loc = SourceInfo(),
               std::string_view name = "");

  // Adds a coverpoint to the function that records every time the associated
  // condition evaluates to true.
  BValue Cover(BValue condition, std::string_view label,
               const SourceInfo& loc = SourceInfo(),
               std::string_view name = "");

  // Adds a gate operation. The output of the operation is `data` if `cond` is
  // true and zero-valued otherwise. Gates are side-effecting.
  BValue Gate(BValue condition, BValue data,
              const SourceInfo& loc = SourceInfo(), std::string_view name = "");

  Package* package() const;

  // Returns the last node enqueued onto this builder -- when Build() is called
  // this is what is used as the return value.
  absl::StatusOr<BValue> GetLastValue() {
    XLS_RET_CHECK(last_node_ != nullptr);
    return BValue(last_node_, this);
  }

  // Returns a detailed pending error, or absl::OkStatus() if no error
  // is pending.
  absl::Status GetError() const;

 protected:
  BValue SetError(std::string_view msg, const SourceInfo& loc);
  bool ErrorPending() const { return error_pending_; }

  // Constructs and adds a node to the function and returns a corresponding
  // BValue.
  template <typename NodeT, typename... Args>
  BValue AddNode(const SourceInfo& loc, Args&&... args);

  BValue CreateBValue(Node* node, const SourceInfo& loc);

  // The most recently added node to the function.
  Node* last_node_ = nullptr;

  // The function being built by this builder.
  std::unique_ptr<FunctionBase> function_;

  bool error_pending_;

  // Whether nodes and the built function should be verified. Only false in
  // tests.
  bool should_verify_;

  std::string error_msg_;
  std::string error_stacktrace_;
  SourceInfo error_loc_;
};

// Class for building an XLS Function.
class FunctionBuilder : public BuilderBase {
 public:
  // Builder for xls::Functions. 'should_verify' is a test-only argument which
  // can be set to false in tests that wish to build malformed IR.
  FunctionBuilder(std::string_view name, Package* package,
                  bool should_verify = true);
  ~FunctionBuilder() override = default;

  // Builders are neither copyable or movable- builder values contain references
  // to their builder that can't be updated.
  FunctionBuilder(const FunctionBuilder&) = delete;
  FunctionBuilder& operator=(const FunctionBuilder&) = delete;
  FunctionBuilder(FunctionBuilder&&) = delete;
  FunctionBuilder& operator=(FunctionBuilder&&) = delete;

  BValue Param(std::string_view name, Type* type,
               const SourceInfo& loc = SourceInfo()) override;

  // Adds the function internally being built-up by this builder to the package
  // given at construction time, and returns a pointer to it (the function is
  // subsequently owned by the package and this builder should be discarded).
  //
  // The return value of the function is the most recently added operation.
  absl::StatusOr<Function*> Build();

  // Build function using given return value.
  absl::StatusOr<Function*> BuildWithReturnValue(BValue return_value);
};

// Type used as special argument to ProcBuilder constructor to indicate that the
// proc to be built is a new style proc (ie, has proc-scoped channels). This
// makes ProcBuilder definitions self-documenting:
//
//   ProcBuilder pb(NewStyleProc(), ...);
//
// TODO(https://github.com/google/xls/issues/869): Remove this when all procs
// are new style.
struct NewStyleProc {};

// Class for building an XLS Proc (a communicating sequential process).
class ProcBuilder : public BuilderBase {
 public:
  // Builder for xls::Procs. 'should_verify' is a test-only argument which can
  // be set to false in tests that wish to build malformed IR. Proc starts with
  // no state elements.
  ProcBuilder(std::string_view name, Package* package,
              bool should_verify = true);
  // Constructor for new-style procs which have proc-scoped channels.
  ProcBuilder(NewStyleProc tag, std::string_view name, Package* package,
              bool should_verify = true);

  ~ProcBuilder() override = default;

  // Builders are neither copyable or movable- builder values contain references
  // to their builder that can't be updated.
  ProcBuilder(const ProcBuilder&) = delete;
  ProcBuilder& operator=(const ProcBuilder&) = delete;
  ProcBuilder(ProcBuilder&&) = delete;
  ProcBuilder& operator=(ProcBuilder&&) = delete;

  // Returns the Proc being constructed.
  Proc* proc() const;

  // Add an internal channel scoped to the proc. Only can be called for new
  // style procs.
  absl::StatusOr<ChannelReferences> AddChannel(
      std::string_view name, Type* type,
      ChannelKind kind = ChannelKind::kStreaming,
      absl::Span<const Value> initial_values = {});

  // Add an interface channel to the proc. Only can be called for new style
  // procs.
  absl::StatusOr<ReceiveChannelReference*> AddInputChannel(
      std::string_view name, Type* type,
      ChannelKind kind = ChannelKind::kStreaming,
      std::optional<ChannelStrictness> strictness = std::nullopt);
  absl::StatusOr<SendChannelReference*> AddOutputChannel(
      std::string_view name, Type* type,
      ChannelKind kind = ChannelKind::kStreaming,
      std::optional<ChannelStrictness> strictness = std::nullopt);

  // Returns true if there is an channel entity (a xls::Channel object for
  // old-style proc or a proc-scoped channel or interface channel for new style
  // procs).
  // TODO(https://github.com/google/xls/issues/869): Rename this Has.*End when
  // all procs are new style.
  bool HasSendChannelRef(std::string_view name) const;
  bool HasReceiveChannelRef(std::string_view name) const;

  // Returns the receive/send channel end with the given name. Only can be
  // called for new style procs.
  absl::StatusOr<ReceiveChannelReference*> GetReceiveChannelReference(
      std::string_view name);
  absl::StatusOr<SendChannelReference*> GetSendChannelReference(
      std::string_view name);

  // Instantiates the specified proc in this proc. `channel_references` must
  // match the number, type, and direction of channels on the interface of the
  // instantiated proc.
  absl::Status InstantiateProc(
      std::string_view name, Proc* instantiated_proc,
      absl::Span<ChannelReference* const> channel_references);

  // Returns the Param BValue for the state parameters. Unlike
  // BuilderBase::Param this doesn't add a Param node to the Proc. Rather the
  // state parameters are added to the Proc at construction time and these
  // methods return references to these parameters.
  virtual BValue GetStateParam(int64_t index) const {
    return state_params_.at(index);
  }

  // Build the proc using the given BValues as the next state values. If
  // `next_state` is not empty, the number of recurrent state elements in
  // `next_state` must match the number of state parameters.
  absl::StatusOr<Proc*> Build(absl::Span<const BValue> next_state = {});

  // Adds a state element to the proc with the given initial value. Returns the
  // newly added state parameter.
  BValue StateElement(std::string_view name, const Value& initial_value,
                      const SourceInfo& loc = SourceInfo());

  // Adds a state element to the proc with the given initial value. Returns the
  // newly added state parameter.
  BValue StateElement(std::string_view name, const ValueBuilder& initial_value,
                      const SourceInfo& loc = SourceInfo());

  // Adds a state element to the proc with the given initial value. Returns the
  // newly added state parameter.
  BValue StateElement(std::string_view name, const Bits& initial_value,
                      const SourceInfo& loc = SourceInfo()) {
    return StateElement(name, Value(initial_value), loc);
  }

  // Adds a (conditional) next value for the named state element. Returns an
  // empty tuple.
  BValue Next(BValue param, BValue value,
              std::optional<BValue> pred = std::nullopt,
              const SourceInfo& loc = SourceInfo(), std::string_view name = "");

  // Overriden Param method is explicitly disabled (returns an error). Use
  // StateElement method to add state elements.
  BValue Param(std::string_view name, Type* type,
               const SourceInfo& loc = SourceInfo()) override;

  // Add a receive operation. The type of the data value received is
  // determined by the channel.
  BValue Receive(ReceiveChannelRef channel, BValue token,
                 const SourceInfo& loc = SourceInfo(),
                 std::string_view name = "");

  // Add a conditional receive operation. The receive executes conditionally on
  // the value of the predicate "pred". The type of the data value received is
  // determined by the channel.
  BValue ReceiveIf(ReceiveChannelRef channel, BValue token, BValue pred,
                   const SourceInfo& loc = SourceInfo(),
                   std::string_view name = "");

  // Add a non-blocking receive operation. The type of the data value received
  // is determined by the channel.
  BValue ReceiveIfNonBlocking(ReceiveChannelRef channel, BValue token,
                              BValue pred, const SourceInfo& loc = SourceInfo(),
                              std::string_view name = "");

  // Add a non-blocking receive operation. The type of the data value received
  // is determined by the channel.
  BValue ReceiveNonBlocking(ReceiveChannelRef channel, BValue token,
                            const SourceInfo& loc = SourceInfo(),
                            std::string_view name = "");

  // Add a send operation.
  BValue Send(SendChannelRef channel, BValue token, BValue data,
              const SourceInfo& loc = SourceInfo(), std::string_view name = "");

  // Add a conditional send operation. The send executes conditionally on the
  // value of the predicate "pred".
  BValue SendIf(SendChannelRef channel, BValue token, BValue pred, BValue data,
                const SourceInfo& loc = SourceInfo(),
                std::string_view name = "");

 private:
  std::string_view GetChannelName(ReceiveChannelRef channel) const;
  std::string_view GetChannelName(SendChannelRef channel) const;

  // The BValues of the state parameters.
  std::vector<BValue> state_params_;
};

// A derived ProcBuilder which automatically manages tokens internally.  This
// makes it much less verbose to construct procs with sends, receives, or other
// token-using operations. In the TokenlessProcBuilder, token-using
// operations are totally ordered by threading of a single token. The limitation
// of the TokenlessProcBuilder is that it cannot be used if non-trivial ordering
// of these operations is required, or if there are requirements on
// cross-activation ordering between nodes (enforced via token data
// dependencies).
//
// Note: a proc built with the TokenlessProcBuilder still has token types
// internally. "Tokenless" refers to the fact that token values are hidden from
// the builder interface.
class TokenlessProcBuilder : public ProcBuilder {
 public:
  // Builder for xls::Procs. 'should_verify' is a test-only argument which can
  // be set to false in tests that wish to build malformed IR. Proc starts with
  // no state elements.
  TokenlessProcBuilder(std::string_view name, std::string_view token_name,
                       Package* package, bool should_verify = true)
      : ProcBuilder(name, package, should_verify),
        orig_token_(Literal(Value::Token(), SourceInfo(), token_name)),
        last_token_(orig_token_) {}
  // Constructor for new-style procs which have proc-scoped channels.
  TokenlessProcBuilder(NewStyleProc tag, std::string_view name,
                       std::string_view token_name, Package* package,
                       bool should_verify = true)
      : ProcBuilder(tag, name, package, should_verify),
        orig_token_(Literal(Value::Token(), SourceInfo(), token_name)),
        last_token_(orig_token_) {}

  ~TokenlessProcBuilder() override = default;

  BValue InitialToken() const { return orig_token_; }

  BValue CurrentToken() const { return last_token_; }

  // Add a MinDelay constraint operation.
  using ProcBuilder::MinDelay;
  BValue MinDelay(int64_t delay, const SourceInfo& loc = SourceInfo(),
                  std::string_view name = "");

  // Add a receive operation. The type of the data value received is determined
  // by the channel. The returned BValue is the received data itself (*not* the
  // receive operation itself which produces a tuple containing a token and the
  // data).
  using ProcBuilder::Receive;
  BValue Receive(ReceiveChannelRef channel,
                 const SourceInfo& loc = SourceInfo(),
                 std::string_view name = "");

  // Add a non-blocking receive operation. The type of the data value received
  // is determined by the channel. The returned BValue is a pair of
  // the received data itself along with a valid bit.
  using ProcBuilder::ReceiveNonBlocking;
  std::pair<BValue, BValue> ReceiveNonBlocking(
      ReceiveChannelRef channel, const SourceInfo& loc = SourceInfo(),
      std::string_view name = "");

  // Add a conditinal receive operation. The receive executes conditionally on
  // the value of the predicate "pred". The type of the data value received is
  // determined by the channel.  The returned BValue is the received data itself
  // (*not* the receiveif operation itself which produces a tuple containing a
  // token and the data).
  using ProcBuilder::ReceiveIf;
  BValue ReceiveIf(ReceiveChannelRef channel, BValue pred,
                   const SourceInfo& loc = SourceInfo(),
                   std::string_view name = "");

  // Add a conditional, non-blocking receive operation. The type of the data
  // value received is determined by the channel. The returned BValue is a pair
  // of the received data itself along with a valid bit.
  using ProcBuilder::ReceiveIfNonBlocking;
  std::pair<BValue, BValue> ReceiveIfNonBlocking(
      ReceiveChannelRef channel, BValue pred,
      const SourceInfo& loc = SourceInfo(), std::string_view name = "");

  // Add a send operation. Returns the token-typed BValue of the send node.
  using ProcBuilder::Send;
  BValue Send(SendChannelRef channel, BValue data,
              const SourceInfo& loc = SourceInfo(), std::string_view name = "");

  // Add a conditional send operation. The send executes conditionally on the
  // value of the predicate "pred". Returns the token-typed BValue of the send
  // node.
  using ProcBuilder::SendIf;
  BValue SendIf(SendChannelRef channel, BValue pred, BValue data,
                const SourceInfo& loc = SourceInfo(),
                std::string_view name = "");

  // Add an assert operation. Returns the token-typed assert operation.
  using BuilderBase::Assert;
  BValue Assert(BValue condition, std::string_view message,
                std::optional<std::string> label = std::nullopt,
                const SourceInfo& loc = SourceInfo(),
                std::string_view name = "");

 private:
  // The token created at the start of each activation.
  BValue orig_token_;

  // The token of the most recently added token-producing operation (send,
  // receive, etc).
  BValue last_token_;
};

// Class for building an XLS Block.
class BlockBuilder : public BuilderBase {
 public:
  // Builder for xls::Blocks. 'should_verify' is a test-only argument which can
  // be set to false in tests that wish to build malformed IR.
  BlockBuilder(std::string_view name, Package* package,
               bool should_verify = true)
      : BuilderBase(std::make_unique<Block>(name, package), should_verify) {}
  ~BlockBuilder() override = default;

  // Builders are neither copyable or movable- builder values contain references
  // to their builder that can't be updated.
  BlockBuilder(const BlockBuilder&) = delete;
  BlockBuilder& operator=(const BlockBuilder&) = delete;
  BlockBuilder(BlockBuilder&&) = delete;
  BlockBuilder& operator=(BlockBuilder&&) = delete;

  // Returns the Block being constructed.
  Block* block() const { return down_cast<Block*>(function()); }

  // Build the block.
  absl::StatusOr<Block*> Build();

  absl::Status AddClockPort(std::string_view name) {
    return block()->AddClockPort(name);
  }

  BValue Param(std::string_view name, Type* type,
               const SourceInfo& loc = SourceInfo()) override;

  // Add a reset port.
  BValue ResetPort(std::string_view name);
  // Add an input/output port.
  BValue InputPort(std::string_view name, Type* type,
                   const SourceInfo& loc = SourceInfo());
  BValue OutputPort(std::string_view name, BValue operand,
                    const SourceInfo& loc = SourceInfo());

  // Add a register read operation. The register argument comes from a
  // Block::AddRegister.
  BValue RegisterRead(Register* reg, const SourceInfo& loc = SourceInfo(),
                      std::string_view name = "");

  // Add a register write operation. The register argument comes from a
  // Block::AddRegister. If the register being writen has a reset value then
  // `reset` must be specified.
  BValue RegisterWrite(Register* reg, BValue data,
                       std::optional<BValue> load_enable = std::nullopt,
                       std::optional<BValue> reset = std::nullopt,
                       const SourceInfo& loc = SourceInfo(),
                       std::string_view name = "");

  // Add a register named 'name' with an input of 'data'. Returns the output
  // value of the register. Equivalent to creating a register with
  // Block::AddRegister, adding a RegisterWrite operation with operand 'data'
  // and adding a RegisterRead operation. Returned BValue is the RegisterRead
  // operation.
  BValue InsertRegister(std::string_view name, BValue data,
                        std::optional<BValue> load_enable = std::nullopt,
                        const SourceInfo& loc = SourceInfo());

  // As InsertRegister above but with a reset value.
  BValue InsertRegister(std::string_view name, BValue data, BValue reset_signal,
                        Reset reset,
                        std::optional<BValue> load_enable = std::nullopt,
                        const SourceInfo& loc = SourceInfo());

  // Add an instantiation input/output to the block.
  BValue InstantiationInput(Instantiation* instantiation,
                            std::string_view port_name, BValue data,
                            const SourceInfo& loc = SourceInfo(),
                            std::string_view name = "");
  BValue InstantiationOutput(Instantiation* instantiation,
                             std::string_view port_name,
                             const SourceInfo& loc = SourceInfo(),
                             std::string_view name = "");
};

}  // namespace xls

#endif  // XLS_IR_FUNCTION_BUILDER_H_
