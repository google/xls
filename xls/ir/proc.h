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

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_base.h"
#include "xls/ir/name_uniquer.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/proc_instantiation.h"
#include "xls/ir/state_element.h"
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
       absl::Span<std::unique_ptr<ChannelInterface>> interface,
       Package* package)
      : FunctionBase(name, package), is_new_style_proc_(true) {
    for (std::unique_ptr<ChannelInterface>& channel_interface : interface) {
      channel_interfaces_.push_back(std::move(channel_interface));
      interface_.push_back(channel_interfaces_.back().get());
    }
  }

  ~Proc() override = default;

  int64_t GetStateElementCount() const { return StateElements().size(); }

  // Returns the total number of bits in the proc state.
  int64_t GetStateFlatBitCount() const;

  // Returns the state element(s).
  absl::Span<StateElement* const> StateElements() const { return state_vec_; }
  StateElement* GetStateElement(int64_t index) const {
    return StateElements().at(index);
  }
  absl::StatusOr<StateElement*> GetStateElement(std::string_view name) const;
  std::optional<StateElement*> MaybeGetStateElement(
      std::string_view name) const;

  bool HasStateElement(std::string_view name) const {
    return state_elements_.contains(name);
  }

  StateRead* GetStateRead(int64_t index) const {
    return state_reads_.at(GetStateElement(index));
  }
  StateRead* GetStateRead(StateElement* state_element) const {
    return state_reads_.at(state_element);
  }

  // Returns the index of the given state element in the vector of state
  // elements.
  absl::StatusOr<int64_t> GetStateElementIndex(
      StateElement* state_element) const;
  std::optional<int64_t> MaybeGetStateElementIndex(
      StateElement* state_element) const;

  // Returns true if the given proc-scoped construct (state element) is owned by
  // this block.
  bool IsOwned(StateElement* reg) const {
    return state_elements_.contains(reg->name()) &&
           state_elements_.at(reg->name()).get() == reg;
  }

  // Sanitizes and uniquifies the given name using the proc's name uniquer.
  // Registers the uniquified name in the state uniquer so it is not handed out
  // again.
  std::string UniquifyStateName(std::string_view name) {
    return state_name_uniquer_.GetSanitizedUniqueName(name);
  }

  // Returns the type of the given state element.
  Type* GetStateElementType(int64_t index) const {
    return StateElements().at(index)->type();
  }

  // Replace all state elements with new state parameters and the given initial
  // values. The next state nodes are set to the newly created state parameter
  // nodes.
  absl::Status ReplaceState(absl::Span<const std::string> requested_state_names,
                            absl::Span<const Value> init_values);

  // Replace all state elements with new state parameters and the given initial
  // values, using the given read predicates. The next state nodes are set to
  // the newly created state parameter nodes. This is defined as an overload
  // rather than as a std::optional `read_predicates` argument because
  // initializer lists do not explicitly convert to std::optional<absl::Span>
  // making callsites verbose.
  absl::Status ReplaceState(
      absl::Span<const std::string> requested_state_names,
      absl::Span<const Value> init_values,
      absl::Span<const std::optional<Node*>> read_predicates);

  // Replace all state elements with new state parameters and the given initial
  // values, using the given read predicates and the given next state values.
  // This is defined as an overload rather than as a std::optional `next_state`
  // argument because initializer lists do not explicitly convert to
  // std::optional<absl::Span> making callsites verbose.
  //
  // Provided as a convenience function for the common case of replacing each
  // state element with a single next value.
  absl::Status ReplaceState(
      absl::Span<const std::string> requested_state_names,
      absl::Span<const Value> init_values,
      absl::Span<const std::optional<Node*>> read_predicates,
      absl::Span<Node* const> next_state);

  // Replace the state element at the given index with a new state parameter,
  // initial value, and next state value. If `next_state` is not given then the
  // next state node for this state element is set to the newly created state
  // parameter node. Returns the newly created parameter node.
  absl::StatusOr<StateRead*> ReplaceStateElement(
      int64_t index, std::string_view requested_state_name,
      const Value& init_value, std::optional<Node*> next_state = std::nullopt);

  // A set of callbacks to help one replace a state element with one of a
  // different type.
  class StateElementTransformer {
   public:
    virtual ~StateElementTransformer() = default;
    // Called with the new_state_read node and the old state_read_node. Must
    // return a node which adapts the new_state_read's type to the
    // old_state_read's type.
    virtual absl::StatusOr<Node*> TransformStateRead(
        Proc* proc, StateRead* new_state_read, StateRead* old_state_read) {
      XLS_RET_CHECK(new_state_read->GetType() == old_state_read->GetType());
      return new_state_read;
    }
    // Called with the old state_read node. Must return a node which will be the
    // new 'predicate' for the new state_read. old_state_read's predicate.
    virtual absl::StatusOr<std::optional<Node*>> TransformReadPredicate(
        Proc* proc, StateRead* old_state_read) {
      return old_state_read->predicate();
    }
    // Called with the new_state_read node and the next-node (Without any
    // updates applied to it). Must return a node which adapts the old_next's
    // value() node to the value of the corresponding next on new_param.
    virtual absl::StatusOr<Node*> TransformNextValue(Proc* proc,
                                                     StateRead* new_state_read,
                                                     Next* old_next) {
      XLS_RET_CHECK(old_next->value()->GetType() == new_state_read->GetType());
      return old_next->value();
    }
    // Caled with the new_state_read node and the next-node (Without any updates
    // applied to it). Must return a node which will be the new 'predicate' for
    // the corresponding 'next' on the new_state_read.
    virtual absl::StatusOr<std::optional<Node*>> TransformNextPredicate(
        Proc* proc, StateRead* new_state_read, Next* old_next) {
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
  absl::StatusOr<StateRead*> TransformStateElement(
      StateRead* old_state_read, const Value& init_value,
      StateElementTransformer& transform);

  // Remove the state element at the given index. All state elements higher than
  // `index` are shifted down one to fill the hole. The state parameter at the
  // index must have no uses.
  absl::Status RemoveStateElement(int64_t index);

  // Appends a state element with the given name (if possible), next state
  // value, and initial value. If `next_state` is not given then the next state
  // node for this state element is set to the newly created state parameter
  // node. Returns the newly created state read node.
  absl::StatusOr<StateRead*> AppendStateElement(
      std::string_view requested_state_name, const Value& init_value,
      std::optional<Node*> read_predicate, std::optional<Node*> next_state);
  absl::StatusOr<StateRead*> AppendStateElement(
      std::string_view requested_state_name, const Value& init_value) {
    return AppendStateElement(requested_state_name, init_value,
                              /*read_predicate=*/std::nullopt,
                              /*next_state=*/std::nullopt);
  }

  // Adds a state element at the given index. Current state elements at the
  // given index or higher will be shifted up. Returns the newly created
  // parameter node.
  absl::StatusOr<StateRead*> InsertStateElement(
      int64_t index, std::string_view requested_state_name,
      const Value& init_value, std::optional<Node*> read_predicate,
      std::optional<Node*> next_state);
  absl::StatusOr<StateRead*> InsertStateElement(
      int64_t index, std::string_view requested_state_name,
      const Value& init_value) {
    return InsertStateElement(index, requested_state_name, init_value,
                              /*read_predicate=*/std::nullopt,
                              /*next_state=*/std::nullopt);
  }

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

  // Returns the type of the channel reference (Channel or ChannelInterface)
  // with the given name.
  absl::StatusOr<Type*> GetChannelReferenceType(std::string_view name) const;

  // Return the ordered list of the channel interfaces which form the interface
  // of the proc. Only can be called for new style procs.
  absl::Span<ChannelInterface* const> interface() const {
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
  // procs. Returns a data structure holding pointers to the interfaces to the
  // two sides of the channel.
  absl::StatusOr<ChannelWithInterfaces> AddChannel(
      std::unique_ptr<Channel> channel);

  // Returns the channel with the given name defined in the proc. Only can be
  // called for new style procs.
  absl::StatusOr<Channel*> GetChannel(std::string_view name);

  // Returns whether a channel with the given name is declared in the proc. Only
  // can be called for new style procs.
  bool HasChannelWithName(std::string_view name);

  // Returns the ChannelRef referring to the global channel or proc-scoped
  // channel (new-style procs) with the given name.
  absl::StatusOr<ChannelRef> GetChannelRef(std::string_view name,
                                           ChannelDirection direction);

  bool ChannelIsOwnedByProc(Channel* channel);

  // Add input/output channels to the interface of the proc.
  absl::StatusOr<ReceiveChannelInterface*> AddInputChannel(
      std::string_view name, Type* type, ChannelKind kind,
      std::optional<ChannelStrictness> strictness = std::nullopt);
  absl::StatusOr<SendChannelInterface*> AddOutputChannel(
      std::string_view name, Type* type, ChannelKind kind,
      std::optional<ChannelStrictness> strictness = std::nullopt);
  absl::StatusOr<ChannelInterface*> AddInterfaceChannel(
      std::string_view name, ChannelDirection direction, Type* type,
      ChannelKind kind,
      std::optional<ChannelStrictness> strictness = std::nullopt);

  // Remove a channel from the interface of the proc. ChannelInterfaceslater
  // than `channel_interface` in the interface are shifted down.
  absl::Status RemoveChannelInterface(ChannelInterface* channel_interface);

  // Returns true if the given ChannelInterface refers to an element on the
  // interface of the proc.
  bool IsOnProcInterface(ChannelInterface* channel_interface);

  // Add an input/output channel to the interface of the proc. Only can be
  // called for new style procs.
  absl::StatusOr<ReceiveChannelInterface*> AddInputChannelInterface(
      std::unique_ptr<ReceiveChannelInterface> channel_interface);
  absl::StatusOr<SendChannelInterface*> AddOutputChannelInterface(
      std::unique_ptr<SendChannelInterface> channel_interface);
  absl::StatusOr<ChannelInterface*> AddChannelInterface(
      std::unique_ptr<ChannelInterface> channel_interface);

  // Create and add a proc instantiation to the proc.
  absl::StatusOr<ProcInstantiation*> AddProcInstantiation(
      std::string_view name, absl::Span<ChannelInterface* const> channel_args,
      Proc* proc);

  // Returns whether this proc has a channel interface of the given name. Only
  // can be called for new style procs.
  bool HasChannelInterface(std::string_view name,
                           ChannelDirection direction) const;

  // Returns the (Send/Receive) channel interface with the given name.
  absl::StatusOr<ChannelInterface*> GetChannelInterface(
      std::string_view name, ChannelDirection direction) const;
  absl::StatusOr<SendChannelInterface*> GetSendChannelInterface(
      std::string_view name) const;
  absl::StatusOr<ReceiveChannelInterface*> GetReceiveChannelInterface(
      std::string_view name) const;

  // Returns all the channel interfaces in the proc. This includes interfaces on
  // the interface of the proc and of channels declared in the proc.
  absl::Span<const std::unique_ptr<ChannelInterface>> channel_interfaces()
      const {
    return channel_interfaces_;
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
  bool is_new_style_proc_;

  NameUniquer state_name_uniquer_ =
      NameUniquer(/*separator=*/"__", GetIrReservedWords());

  // State elements within this proc. Indexed by state element name. Stored as
  // std::unique_ptrs for pointer stability.
  absl::flat_hash_map<std::string, std::unique_ptr<StateElement>>
      state_elements_;

  // Map of the unique StateRead node for each state element.
  absl::flat_hash_map<StateElement*, StateRead*> state_reads_;

  // Vector of state element pointers. Kept in sync with the state_elements_
  // map. Enables easy, stable iteration over state elements. With this vector,
  // deletion of a state element is O(n) with the number of state elements. If
  // this is a problem, a linked list might be used instead.
  std::vector<StateElement*> state_vec_;

  // All channel interfaces in this proc. Channel interfaces can be part of the
  // proc interface or for channels declared in this proc.
  std::vector<std::unique_ptr<ChannelInterface>> channel_interfaces_;

  // Channel interfaces which form interface of the proc.
  std::vector<ChannelInterface*> interface_;

  // Instantiations of other procs within this proc.
  std::vector<std::unique_ptr<ProcInstantiation>> proc_instantiations_;

  // Channels declared in this proc indexed by channel name.
  absl::flat_hash_map<std::string, std::unique_ptr<Channel>> channels_;
  std::vector<Channel*> channel_vec_;
};

}  // namespace xls

#endif  // XLS_IR_PROC_H_
