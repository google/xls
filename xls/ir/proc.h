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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/proc_instantiation.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {

// Abstraction representing an XLS Proc. Procs (from "processes") are stateful
// blocks which iterate indefinitely over mutable state of a fixed type. Procs
// communicate to other components via channels.
// TODO(meheff): Add link to documentation when we have some.
class Proc : public FunctionBase {
 public:
  // Creates a proc with no state elements.
  Proc(std::string_view name, Package* package)
      : FunctionBase(name, package), is_new_style_proc_(false) {}

  // Creates a new-style proc which supports proc-scoped channels.
  Proc(std::string_view name,
       absl::Span<std::unique_ptr<ChannelReference>> interface,
       Package* package)
      : FunctionBase(name, package), is_new_style_proc_(true) {
    for (std::unique_ptr<ChannelReference>& channel_reference : interface) {
      channel_references_.push_back(std::move(channel_reference));
      interface_.push_back(channel_references_.back().get());
    }
  }

  ~Proc() override = default;

  // Returns the initial values of the state variables.
  absl::Span<const Value> InitValues() const { return init_values_; }
  absl::StatusOr<Value> GetInitValue(Param* p);
  const Value& GetInitValueElement(int64_t index) const {
    return init_values_.at(index);
  }

  int64_t GetStateElementCount() const { return StateParams().size(); }

  // Returns the total number of bits in the proc state.
  int64_t GetStateFlatBitCount() const;

  // Returns the state parameter node(s).
  absl::Span<Param* const> StateParams() const { return params(); }
  Param* GetStateParam(int64_t index) const { return StateParams().at(index); }

  // Returns the element index (in the vector of state parameters) of the given
  // state parameter.
  absl::StatusOr<int64_t> GetStateParamIndex(Param* param) const;

  // Returns the nodes holding the next recurrent state value.
  //
  // TODO: Remove this once fully transitioned over to `next_value` nodes.
  absl::Span<Node* const> NextState() const { return next_state_; }
  Node* GetNextStateElement(int64_t index) const {
    return NextState().at(index);
  }

  // Return the state element indices for which the given `node` is the next
  // recurrent state value for that element.
  //
  // TODO: Remove this once fully transitioned over to `next_value` nodes.
  absl::btree_set<int64_t> GetNextStateIndices(Node* node) const;

  // Returns the type of the given state element.
  Type* GetStateElementType(int64_t index) const {
    return StateParams().at(index)->GetType();
  }

  // Sets the next recurrent state value for the state element of the given
  // index. Node type must match the type of the state element.
  //
  // TODO: Remove this once fully transitioned over to `next_value` nodes.
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
  //
  // TODO: Remove this once fully transitioned over to `next_value` nodes.
  absl::Status ReplaceState(absl::Span<const std::string> state_param_names,
                            absl::Span<const Value> init_values,
                            absl::Span<Node* const> next_state);

  // Replace the state element at the given index with a new state parameter,
  // initial value, and next state value. If `next_state` is not given then the
  // next state node for this state element is set to the newly created state
  // parameter node. Returns the newly created parameter node.
  absl::StatusOr<Param*> ReplaceStateElement(
      int64_t index, std::string_view state_param_name, const Value& init_value,
      std::optional<Node*> next_state = std::nullopt);

  // A set of callbacks to help one replace a state element with one of a
  // different type.
  class StateElementTransformer {
   public:
    virtual ~StateElementTransformer() = default;
    // Called with the new_param node and the old param_node. Must return a node
    // which adapts the new_param's type to the old_params type.
    virtual absl::StatusOr<Node*> TransformParamRead(Proc* proc,
                                                     Param* new_param,
                                                     Param* old_param) {
      XLS_RET_CHECK(new_param->GetType() == old_param->GetType());
      return new_param;
    }
    // Called with the new_param node and the next-node (Without any updates
    // applied to it). Must return a node which adapts the old_next's value()
    // node to the value of the corresponding next on new_param.
    virtual absl::StatusOr<Node*> TransformNextValue(Proc* proc,
                                                     Param* new_param,
                                                     Next* old_next) {
      XLS_RET_CHECK(old_next->value()->GetType() == new_param->GetType());
      return old_next->value();
    }
    // Caled with the new_param node and the next-node (Without any updates
    // applied to it). Must return a node which will be the new 'predicate' for
    // the corresponding 'next' on the new_param.
    virtual absl::StatusOr<std::optional<Node*>> TransformNextPredicate(
        Proc* proc, Param* new_param, Next* old_next) {
      return old_next->predicate();
    }
  };

  // Transform the given state element using the 'transformer'. The new state
  // element will have the same name (though a different index) as the old one
  // and the type of 'init_value'. The callbacks in 'transform' will be called
  // to adapt everything to the new type.
  //
  // Once the transformer has fixed up the types ReplaceUsesWith will be used to
  // switch users of the old param to the new one.
  //
  // The old state element will continue to exist with a new name and all
  // identity next nodes. It should be cleaned up using the
  // NextValueOptimizationPass.
  //
  // The proc must only use 'next' nodes to call this function.
  absl::StatusOr<Param*> TransformStateElement(
      Param* old_param, const Value& init_value,
      StateElementTransformer& transform);

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
      int64_t index, std::string_view state_param_name, const Value& init_value,
      std::optional<Node*> next_state = std::nullopt);

  bool HasImplicitUse(Node* node) const override;

  // Creates a clone of the proc with the new name `new_name`. Proc is
  // owned by `target_package`. `channel_remapping` dictates how to map channel
  // names to new channel names in the cloned version; if a key is unavailable
  // in `channel_remapping` it is assumed to be the identity mapping at that
  // key.
  absl::StatusOr<Proc*> Clone(
      std::string_view new_name, Package* target_package = nullptr,
      const absl::flat_hash_map<std::string, std::string>& channel_remapping =
          {},
      const absl::flat_hash_map<const FunctionBase*, FunctionBase*>&
          call_remapping = {},
      const absl::flat_hash_map<std::string, std::string>&
          state_name_remapping = {}) const;

  std::string DumpIr() const override;

  // Returns true if this is a new-style proc which has proc-scoped channels.
  bool is_new_style_proc() const { return is_new_style_proc_; }

  // Returns the type of the channel reference (Channel or ChannelReference)
  // with the given name.
  absl::StatusOr<Type*> GetChannelReferenceType(std::string_view name) const;

  // Return the ordered list of the channel references which form the interface
  // of the proc. Only can be called for new style procs.
  absl::Span<ChannelReference* const> interface() const {
    CHECK(is_new_style_proc());
    return interface_;
  }

  // Return the channels defined in this proc. Only can be called for new style
  // procs.
  absl::Span<Channel* const> channels() const {
    CHECK(is_new_style_proc());
    return channel_vec_;
  }

  // Add a channel definition to the proc.  Only can be called for new style
  // procs. Returns a data structure holding pointers to the references to the
  // two sides of the channel.
  absl::StatusOr<ChannelReferences> AddChannel(
      std::unique_ptr<Channel> channel);

  // Returns the channel with the given name defined in the proc.
  absl::StatusOr<Channel*> GetChannel(std::string_view name);

  bool ChannelIsOwnedByProc(Channel* channel);

  // Add input/output channels to the interfacce of the proc.
  absl::StatusOr<ReceiveChannelReference*> AddInputChannel(
      std::string_view name, Type* type, ChannelKind kind,
      std::optional<ChannelStrictness> strictness = std::nullopt);
  absl::StatusOr<SendChannelReference*> AddOutputChannel(
      std::string_view name, Type* type, ChannelKind kind,
      std::optional<ChannelStrictness> strictness = std::nullopt);
  absl::StatusOr<ChannelReference*> AddInterfaceChannel(
      std::string_view name, Direction direction, Type* type, ChannelKind kind,
      std::optional<ChannelStrictness> strictness = std::nullopt);

  // Remove a channel from the interface of the proc. ChannelReferences later
  // than `channel_ref` in the interface are shifted down.
  absl::Status RemoveInterfaceChannel(ChannelReference* channel_ref);

  // Add an input/output channel to the interface of the proc. Only can be
  // called for new style procs.
  absl::StatusOr<ReceiveChannelReference*> AddInputChannelReference(
      std::unique_ptr<ReceiveChannelReference> channel_ref);
  absl::StatusOr<SendChannelReference*> AddOutputChannelReference(
      std::unique_ptr<SendChannelReference> channel_ref);
  absl::StatusOr<ChannelReference*> AddInterfaceChannelReference(
      std::unique_ptr<ChannelReference> channel_ref);

  // Create and add a proc instantiation to the proc.
  absl::StatusOr<ProcInstantiation*> AddProcInstantiation(
      std::string_view name, absl::Span<ChannelReference* const> channel_args,
      Proc* proc);

  // Returns whether this proc has a channel reference of the given name. Only
  // can be called for new style procs.
  bool HasChannelReference(std::string_view name, Direction direction) const;

  // Returns the (Send/Receive) channel reference with the given name.
  absl::StatusOr<ChannelReference*> GetChannelReference(
      std::string_view name, Direction direction) const;
  absl::StatusOr<SendChannelReference*> GetSendChannelReference(
      std::string_view name) const;
  absl::StatusOr<ReceiveChannelReference*> GetReceiveChannelReference(
      std::string_view name) const;

  // Returns all the channel references in the proc. This includes references to
  // interface channels and declared channels.
  absl::Span<const std::unique_ptr<ChannelReference>> channel_references()
      const {
    return channel_references_;
  }

  // Returns the list of instantiations of other procs in this proc.
  absl::Span<const std::unique_ptr<ProcInstantiation>> proc_instantiations()
      const {
    return proc_instantiations_;
  }

  absl::StatusOr<ProcInstantiation*> GetProcInstantiation(
      std::string_view instantiation_name) const;

  // Sets the new-style proc bit to true. Proc may be malformed until the
  // interface and channel declarations are added.
  // TODO(https://github.com/google/xls/issues/869): Remove when all procs are
  // new-style.
  absl::Status ConvertToNewStyle();

 private:
  std::vector<Value> init_values_;

  bool is_new_style_proc_;

  // TODO: Remove this once fully transitioned over to `next_value` nodes.
  std::vector<Node*> next_state_;

  // A map from the `next_state_` nodes back to the indices they control.
  //
  // TODO: Remove this once fully transitioned over to `next_value` nodes.
  absl::flat_hash_map<Node*, absl::btree_set<int64_t>> next_state_indices_;

  // All channel references in this proc. Channel references can be part of the
  // interface or the references of channels declared in this proc.
  std::vector<std::unique_ptr<ChannelReference>> channel_references_;

  // Channel references which form interface of the proc.
  std::vector<ChannelReference*> interface_;

  // Instantiations of other procs within this proc.
  std::vector<std::unique_ptr<ProcInstantiation>> proc_instantiations_;

  // Channels declared in this proc indexed by channel name.
  absl::flat_hash_map<std::string, std::unique_ptr<Channel>> channels_;
  std::vector<Channel*> channel_vec_;
};

}  // namespace xls

#endif  // XLS_IR_PROC_H_
