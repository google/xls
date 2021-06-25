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

#ifndef XLS_INTERPRETER_FUNCTION_INTERPRETER_H_
#define XLS_INTERPRETER_FUNCTION_INTERPRETER_H_

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/ir/function.h"
#include "xls/ir/value.h"

namespace xls {

// An interpreter for XLS functions.
class FunctionInterpreter : public IrInterpreter {
 public:
  explicit FunctionInterpreter(absl::Span<const Value> args)
      : args_(args.begin(), args.end()) {}

  // Runs the interpreter on the given function. 'args' are the argument values
  // indexed by parameter name.
  static absl::StatusOr<Value> Run(Function* function,
                                   absl::Span<const Value> args);

  // Runs the interpreter on the function where the arguments are given by name.
  static absl::StatusOr<Value> RunKwargs(
      Function* function, const absl::flat_hash_map<std::string, Value>& args);

  absl::Status HandleParam(Param* param) override;

 private:
  // The arguments to the Function being evaluated indexed by parameter name.
  std::vector<Value> args_;
};

}  // namespace xls

#endif  // XLS_INTERPRETER_FUNCTION_INTERPRETER_H_
