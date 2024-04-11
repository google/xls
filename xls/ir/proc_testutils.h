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

#ifndef XLS_IR_PROC_TESTUTILS_H_
#define XLS_IR_PROC_TESTUTILS_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "xls/ir/bits.h"
#include "xls/ir/nodes.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"

namespace xls {

// Helper to convert a proc into a function which performs 'activation_count'
// activations. All channels are assumed to always be ready and to provide
// unconstrained input.
//
// Tokens are replaced with literals of the given 'token_value'. NB this should
// be a non-zero-length value because z3 doesn't like zero-length values with
// uses.
//
// This function adds the function version to the input procs package.
//
// The state elements are considered to start at their initial values.
//
// Each channel may only be sent on once per activation.
// TODO(allight): Support sending on a single channel multiple times (with
// predicates).
//
// The return type of the function depends on the lexicographic ordering of
// channels so this should not be messed with.
//
// This is only intended for use with testing tools such as z3.
absl::StatusOr<Function*> UnrollProcToFunction(
    Proc* p, int64_t activation_count, bool include_state,
    const Value& token_value = Value::Tuple({Value(UBits(0xdeadbeef, 32))}));

}  // namespace xls

#endif  // XLS_IR_PROC_TESTUTILS_H_
