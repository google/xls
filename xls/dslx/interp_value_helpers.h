// Copyright 2021 The XLS Authors
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
#ifndef XLS_DSLX_INTERP_VALUE_HELPERS_H_
#define XLS_DSLX_INTERP_VALUE_HELPERS_H_

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/concrete_type.h"

namespace xls::dslx {

// Converts the given (Bits-typed) InterpValue to an array of equal- or
// smaller-sized Bits-typed values.
absl::StatusOr<InterpValue> CastBitsToArray(const InterpValue& bits_value,
                                            const ArrayType& array_type);

// Converts the given Bits-typed value into an enum-typed value.
absl::StatusOr<InterpValue> CastBitsToEnum(const InterpValue& bits_value,
                                           const EnumType& enum_type);

// Creates a zero-valued InterpValue with the same structure as the input.
absl::StatusOr<InterpValue> CreateZeroValue(const InterpValue& value);

// Creates a zero-valued InterpValue from the given ConcreteType.
absl::StatusOr<InterpValue> CreateZeroValueFromType(
    const ConcreteType& concrete_type);

// Places a "flat" representation of the input value (if it's a tuple) in
// `result`. Converts, e.g., (a, (b, c), d) into {a, b, c, d}.
absl::Status FlattenTuple(const InterpValue& value,
                          std::vector<InterpValue>* result);

// Finds the first index in the LHS and RHS sequences at which values differ or
// nullopt if the two are equal.
absl::StatusOr<std::optional<int64_t>> FindFirstDifferingIndex(
    absl::Span<const InterpValue> lhs, absl::Span<const InterpValue> rhs);

// Converts the values to matched the signedness of the concrete type.
//
// Converts bits-typed Values contained within the given Value to match the
// signedness of the ConcreteType. Examples:
//
// invocation: sign_convert_value(s8, u8:64)
// returns: s8:64
//
// invocation: sign_convert_value(s3, u8:7)
// returns: s3:-1
//
// invocation: sign_convert_value((s8, u8), (u8:42, u8:10))
// returns: (s8:42, u8:10)
//
// This conversion functionality is required because the Values used in the DSLX
// may be signed while Values in IR interpretation and Verilog simulation are
// always unsigned.
//
// Args:
//   concrete_type: ConcreteType to match.
//   value: Input value.
//
// Returns:
//   Sign-converted value.
absl::StatusOr<InterpValue> SignConvertValue(const ConcreteType& concrete_type,
                                             const InterpValue& value);

// As above, but a handy vectorized form for application on parameters of a
// function.
absl::StatusOr<std::vector<InterpValue>> SignConvertArgs(
    const FunctionType& fn_type, absl::Span<const InterpValue> args);

// Converts an (IR) value to an interpreter value.
absl::StatusOr<InterpValue> ValueToInterpValue(
    const Value& v, const ConcreteType* type = nullptr);

// Parses a semicolon-delimited list of values.
//
// Example input:
//  bits[32]:6; (bits[8]:2, bits[16]:4)
//
// Returned bits values are always unsigned.
//
// Note: these values are parsed to InterpValues, but really they are just IR
// values that we're converting into InterpValues. Things like enums or structs
// (via named tuples) can't be parsed via this mechanism, it's fairly
// specialized for the scenario we've created in our fuzzing process.
absl::StatusOr<std::vector<InterpValue>> ParseArgs(std::string_view args_text);

// Does the above, but for a series of argument strings, one per line of input.
absl::StatusOr<std::vector<std::vector<InterpValue>>> ParseArgsBatch(
    std::string_view args_text);

}  // namespace xls::dslx

#endif  // XLS_DSLX_INTERP_VALUE_HELPERS_H_
