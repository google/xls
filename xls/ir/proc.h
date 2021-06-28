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
class Proc : public FunctionBase {
 public:
  Proc(absl::string_view name, const Value& init_value,
       absl::string_view token_param_name, absl::string_view state_param_name,
       Package* package)
      : FunctionBase(name, package),
        init_value_(init_value),
        token_param_(AddNode(absl::make_unique<Param>(
            absl::nullopt, token_param_name, package->GetTokenType(), this))),
        state_param_(AddNode(absl::make_unique<Param>(
            absl::nullopt, state_param_name,
            package->GetTypeForValue(init_value_), this))),
        next_token_(token_param_),
        next_state_(state_param_) {}

  virtual ~Proc() = default;

  // Returns the initial value of the state variable.
  const Value& InitValue() const { return init_value_; }

  // Returns the type of the recurrent state variable.
  Type* StateType() const { return StateParam()->GetType(); }

  // Returns the state (or token) parameter node. These are the only Params of
  // the Proc.
  Param* StateParam() const { return state_param_; }
  Param* TokenParam() const { return token_param_; }

  // Returns the node holding the next recurrent token/state value.
  Node* NextToken() const { return next_token_; }
  Node* NextState() const { return next_state_; }

  // Sets the next recurrent token value. Node must be token typed.
  absl::Status SetNextToken(Node* next);

  // Sets the next recurrent state value. Node type must match the type of the
  // state of the proc.
  absl::Status SetNextState(Node* next);

  // Removes the existing state param node and creates a new one with the given
  // name and type matching next_state (which may be any type). Replaces the
  // proc's next recurrent state with the given next_state. The existing state
  // param must have no uses.
  absl::Status ReplaceState(absl::string_view state_param_name,
                            Node* next_state, const Value& init_value);

  bool HasImplicitUse(Node* node) const override {
    return node == NextToken() || node == NextState();
  }

  std::string DumpIr(bool recursive = false) const override;

 private:
  Value init_value_;

  // State and token parameters. Procs have fixed set of parameters (state data
  // and an input token) which are added at construction time.
  Param* token_param_;
  Param* state_param_;

  // The nodes representing the token/state values for the next iteration of the
  // proc.
  Node* next_token_;
  Node* next_state_;
};

}  // namespace xls

#endif  // XLS_IR_PROC_H_
