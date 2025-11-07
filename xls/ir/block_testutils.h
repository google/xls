// Copyright 2025 The XLS Authors
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

#ifndef XLS_IR_BLOCK_TESTUTILS_H_
#define XLS_IR_BLOCK_TESTUTILS_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "xls/ir/block.h"
#include "xls/ir/function.h"

namespace xls {

// Helper to convert a block into a function which performs 'activation_count'
// ticks.
//
// This function adds the function version to the input blocks package.
//
// The registers are considered to start at their reset values.
//
// The return type of the function depends on the lexicographic ordering of
// ports and registers so this should not be messed with.
//
// This is only intended for use with testing tools such as z3.
//
// If `zero_invalid_outputs` is true then the channel metadata will be used to
// zero out output ports which don't have their 'valid' bit set.
absl::StatusOr<Function*> UnrollBlockToFunction(Block* b,
                                                int64_t activation_count,
                                                bool include_state,
                                                bool zero_invalid_outputs);

}  // namespace xls

#endif  // XLS_IR_BLOCK_TESTUTILS_H_
