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

#ifndef XLS_IR_PROC_H_
#define XLS_IR_PROC_H_

#include "absl/strings/string_view.h"
#include "xls/ir/function.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"

namespace xls {

// Abstraction representing an XLS Proc. Procs (from "processes") are stateful
// blocks which iterate indefinitely over mutable state of a fixed type. Procs
// communicate to other components via channels.
// TODO(meheff): Add link to documentation when we have some.
class Proc : public Function {
 public:
  Proc(absl::string_view name, const Value& init_value,
       absl::string_view token_param_name, absl::string_view state_param_name,
       Package* package)
      : Function(name, package),
        init_value_(init_value),
        state_type_(package->GetTypeForValue(init_value_)),
        return_type_(
            package->GetTupleType({package->GetTokenType(), state_type_})),
        token_param_(AddNode(absl::make_unique<Param>(
            absl::nullopt, token_param_name, package->GetTokenType(), this))),
        state_param_(AddNode(absl::make_unique<Param>(
            absl::nullopt, state_param_name, state_type_, this))) {}

  virtual ~Proc() = default;

  absl::Status set_return_value(Node* n) override;

  // Returns the initial value of the state variable.
  const Value& InitValue() const { return init_value_; }

  // Returns the type of the recurrent state variable.
  Type* StateType() const { return state_type_; }

  // Returns the return type of the Proc. The return type is a 2-tuple
  // containing the state type and a token.
  Type* ReturnType() const { return return_type_; }

  // Returns the state (or token) parameter node. These are the only Params of
  // the Proc.
  Param* StateParam() const { return state_param_; }
  Param* TokenParam() const { return token_param_; }

  std::string DumpIr(bool recursive = false) const override;

 private:
  Value init_value_;
  Type* state_type_;
  Type* return_type_;

  // State and token parameters. Procs have fixed set of parameters (state data
  // and an input token) which are added at construction time.
  Param* token_param_;
  Param* state_param_;
};

}  // namespace xls

#endif  // XLS_IR_PROC_H_
