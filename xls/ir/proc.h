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

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"

namespace xls {

// Abstraction representing an XLS Proc. Procs (from "processes") are stateful
// blocks which iterate indefinitely over mutable state of a fixed type. Procs
// communicate to other components via channels.
// TODO(meheff): Add link to documentation when we have some.
class Proc : public FunctionBase {
 public:
  // Creates a proc with no state elements.
  Proc(std::string_view name, std::string_view token_param_name,
       Package* package)
      : FunctionBase(name, package),
        next_token_(AddNode(std::make_unique<Param>(
            SourceInfo(), token_param_name, package->GetTokenType(), this))) {}

  virtual ~Proc() = default;

  // Returns the initial values of the state variables.
  absl::Span<const Value> InitValues() const { return init_values_; }
  const Value& GetInitValueElement(int64_t index) const {
    return init_values_.at(index);
  }

  // Returns the token parameter node.
  Param* TokenParam() const { return params().at(0); }

  int64_t GetStateElementCount() const { return StateParams().size(); }

  // Returns the total number of bits in the proc state.
  int64_t GetStateFlatBitCount() const;

  // Returns the state parameter node(s).
  absl::Span<Param* const> StateParams() const { return params().subspan(1); }
  Param* GetStateParam(int64_t index) const { return StateParams().at(index); }

  // Returns the element index (in the vector of state parameters) of the given
  // state parameter.
  absl::StatusOr<int64_t> GetStateParamIndex(Param* param) const;

  // Returns the node holding the next recurrent token value.
  Node* NextToken() const { return next_token_; }

  // Returns the nodes holding the next recurrent state value.
  absl::Span<Node* const> NextState() const { return next_state_; }
  Node* GetNextStateElement(int64_t index) const {
    return NextState().at(index);
  }

  // Return the state element indices for which the given `node` is the next
  // recurrent state value for that element.
  std::vector<int64_t> GetNextStateIndices(Node* node) const;

  // Returns the type of the given state element.
  Type* GetStateElementType(int64_t index) const {
    return StateParams().at(index)->GetType();
  }

  // Sets the next token value.
  absl::Status SetNextToken(Node* next);

  // Sets the next token value of the proc to the existing next token value
  // joined with `tokens` using an after all node.
  absl::Status JoinNextTokenWith(absl::Span<Node* const> tokens);

  // Sets the next recurrent state value for the state element of the given
  // index. Node type must match the type of the state element.
  absl::Status SetNextStateElement(int64_t index, Node* next);

  // Replace all state elements with new state parameters and the given initial
  // values. The next state nodes are set to the newly created state parameter
  // nodes.
  absl::Status ReplaceState(absl::Span<const std::string> state_param_names,
                            absl::Span<const Value> init_values);

  // Replace all state elements with new state parameters and the given initial
  // values, and the next state values. This is defined as an overload rather
  // than as a std::optional `next_state` argument because initializer lists do
  // not explicitly convert to std::optional<absl::Span> making callsites
  // verbose.
  absl::Status ReplaceState(absl::Span<const std::string> state_param_names,
                            absl::Span<const Value> init_values,
                            absl::Span<Node* const> next_state);

  // Replace the state element at the given index with a new state parameter,
  // initial value, and next state value. If `next_state` is not given then the
  // next state node for this state element is set to the newly created state
  // parameter node. Returns the newly created parameter node.
  absl::StatusOr<Param*> ReplaceStateElement(
      int64_t index, std::string_view state_param_name,
      const Value& init_value, std::optional<Node*> next_state = std::nullopt);

  // Remove the state element at the given index. All state elements higher than
  // `index` are shifted down one to fill the hole. The state parameter at the
  // index must have no uses.
  absl::Status RemoveStateElement(int64_t index);

  // Appends a state element with the given parameter name, next state value,
  // and initial value. If `next_state` is not given then the next state node
  // for this state element is set to the newly created state parameter node.
  // Returns the newly created parameter node.
  absl::StatusOr<Param*> AppendStateElement(
      std::string_view state_param_name, const Value& init_value,
      std::optional<Node*> next_state = std::nullopt);

  // Adds a state element at the given index. Current state elements at the
  // given index or higher will be shifted up. Returns the newly created
  // parameter node.
  absl::StatusOr<Param*> InsertStateElement(
      int64_t index, std::string_view state_param_name,
      const Value& init_value, std::optional<Node*> next_state = std::nullopt);

  bool HasImplicitUse(Node* node) const override {
    return node == NextToken() ||
           std::find(next_state_.begin(), next_state_.end(), node) !=
               next_state_.end();
  }

  // Creates a clone of the proc with the new name `new_name`. Proc is
  // owned by `target_package`. `channel_remapping` dictates how to map channel
  // IDs to new channel IDs in the cloned version; if a key is unavailable in
  // `channel_remapping` it is assumed to be the identity mapping at that key.
  absl::StatusOr<Proc*> Clone(
      std::string_view new_name, Package* target_package = nullptr,
      absl::flat_hash_map<int64_t, int64_t> channel_remapping = {}) const;

  std::string DumpIr() const override;

 private:
  std::vector<Value> init_values_;

  // The nodes representing the token/state values for the next iteration of the
  // proc.
  Node* next_token_;
  std::vector<Node*> next_state_;
};

}  // namespace xls

#endif  // XLS_IR_PROC_H_
