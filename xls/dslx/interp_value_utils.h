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
#ifndef XLS_DSLX_INTERP_VALUE_UTILS_H_
#define XLS_DSLX_INTERP_VALUE_UTILS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type.h"
#include "xls/ir/value.h"

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

// Creates a zero-valued InterpValue from the given Type.
absl::StatusOr<InterpValue> CreateZeroValueFromType(const Type& type);

// Finds the first index in the LHS and RHS sequences at which values differ or
// nullopt if the two are equal.
absl::StatusOr<std::optional<int64_t>> FindFirstDifferingIndex(
    absl::Span<const InterpValue> lhs, absl::Span<const InterpValue> rhs);

// Converts the values to matched the signedness of the concrete type.
//
// Converts bits-typed Values contained within the given Value to match the
// signedness of the Type. Examples:
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
//   type: Type to match.
//   value: Input value.
//
// Returns:
//   Sign-converted value.
absl::StatusOr<InterpValue> SignConvertValue(const Type& type,
                                             const InterpValue& value);

// As above, but a handy vectorized form for application on parameters of a
// function.
absl::StatusOr<std::vector<InterpValue>> SignConvertArgs(
    const FunctionType& fn_type, absl::Span<const InterpValue> args);

// Converts an (IR) value to an interpreter value.
absl::StatusOr<InterpValue> ValueToInterpValue(const Value& v,
                                               const Type* type = nullptr);

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

// Converts an InterpValue of type u8[len] to a string. Assumes the InterpValue
// is utf8 encoded like everything else.
absl::StatusOr<std::string> InterpValueAsString(const InterpValue& v);

// Creates a ChannelReference InterpValue. `type` is the type of the channel
// node not the payload type. `type` may be an array in which case an array of
// ChannelReferences is returned. `channel_instance_allocator`, if specified, is
// called to set the instance ID of each ChannelReference as they are created.
absl::StatusOr<InterpValue> CreateChannelReference(
    ChannelDirection direction, const Type* type,
    std::optional<absl::FunctionRef<int64_t()>> channel_instance_allocator =
        std::nullopt);

// Creates a pair of ChannelReference InterpValues. The first element has
// channel direction "out" while the second element has channel direction
// "in". As with `CreateChannelReference` this function can produce arrays of
// channel references (or arrays of arrays, etc). Corresponding
// ChannelReferences in the first and second elements will have the same channel
// instance id (if any). This is similar in form to what a DSLX channel
// declaration produces. For example,
//
//   let (foo_s, foo_r) = chan<u32>("foo");
absl::StatusOr<std::pair<InterpValue, InterpValue>> CreateChannelReferencePair(
    const Type* type,
    std::optional<absl::FunctionRef<int64_t()>> channel_instance_allocator =
        std::nullopt);

}  // namespace xls::dslx

#endif  // XLS_DSLX_INTERP_VALUE_UTILS_H_
