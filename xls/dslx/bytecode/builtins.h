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

#ifndef XLS_DSLX_BYTECODE_BUILTINS_H_
#define XLS_DSLX_BYTECODE_BUILTINS_H_

#include "absl/status/status.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/interpreter_stack.h"

namespace xls::dslx {

absl::Status RunBuiltinSlice(const Bytecode& bytecode, InterpreterStack& stack);

absl::Status RunBuiltinUpdate(const Bytecode& bytecode,
                              InterpreterStack& stack);

absl::Status RunBuiltinBitSlice(const Bytecode& bytecode,
                                InterpreterStack& stack);

absl::Status RunBuiltinBitSliceUpdate(const Bytecode& bytecode,
                                      InterpreterStack& stack);

}  // namespace xls::dslx

#endif  // XLS_DSLX_BYTECODE_BUILTINS_H_
