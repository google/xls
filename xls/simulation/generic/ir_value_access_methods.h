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

#ifndef XLS_SIMULATION_GENERIC_IR_VALUE_ACCESS_METHODS_H_
#define XLS_SIMULATION_GENERIC_IR_VALUE_ACCESS_METHODS_H_

#include <vector>

#include "absl/status/statusor.h"
#include "xls/ir/value.h"

namespace xls::simulation::generic {

// ValueArrayReadUInt64 adds a byte level interface to read up to 8 bytes from
// an array of XLS::IR::Values using byte offsets and number of bytes to be
// read. It returns a value read from the array. Each value in the array is
// treated as a bit vector and the whole array is treated as a concatenation of
// its values. This virtual bit vector is then sliced into virtual bytes that
// can be addressed. Arguments:
// 1. array of XLS::IR::Values
// 2. name of the underlying XLS::IR:Channel
// 3. byte offset in the array
// 4. maximal number of bytes to read from the array
absl::StatusOr<uint64_t> ValueArrayReadUInt64(
    absl::Span<const Value> data_array, std::string_view channel_name,
    uint64_t byte_offset, uint64_t byte_count);

// ValueArrayWriteUInt64 adds a byte level interface to modify up to 8 bytes
// from an array of XLS::IR::Values using byte offsets and number of bytes to be
// modified and payload. It returns a new vector with modified values. Each
// value in the array is treated as a bit vector and the whole array is treated
// as a concatenation of its values. This virtual bit vector is then sliced into
// virtual bytes that can be addressed and modified. Arguments:
// 1. array of XLS::IR::Values
// 2. name of the underlying XLS::IR:Channel
// 3. byte offset in the array
// 4. maximal number of bytes to modify in the array
// 5. payload with new value
absl::StatusOr<std::vector<Value>> ValueArrayWriteUInt64(
    absl::Span<const Value> data_array, std::string_view channel_name,
    uint64_t byte_offset, uint64_t byte_count, uint64_t const payload);

}  // namespace xls::simulation::generic
#endif  // XLS_SIMULATION_GENERIC_IR_VALUE_ACCESS_METHODS_H_
