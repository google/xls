// Copyright 2022 The XLS Authors
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

#include "xls/passes/proc_inlining_pass.h"

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/node_util.h"
#include "xls/ir/op.h"
#include "xls/ir/value_helpers.h"
#include "xls/passes/new_proc_inlining_pass.h"

namespace xls {
namespace {

// Analysis which identifies invariant elements of each node where "element" is
// defined as a leaf element in the type (as in LeafTypeTree). Literals and data
// received over a single-value channel are invariant. Also, any element
// computed exclusively from invariant elements is also invariant. All other
// elements are not invariant. The analysis traces through tuple and tuple-index
// operations.
class InvariantAnalysis : public DfsVisitorWithDefault {
 public:
  static absl::StatusOr<InvariantAnalysis> Run(FunctionBase* f) {
    InvariantAnalysis nt(f);
    XLS_RETURN_IF_ERROR(f->Accept(&nt));
    XLS_VLOG_LINES(3, absl::StrFormat("InvariantAnalysis:\n%s", nt.ToString()));
    return std::move(nt);
  }

  // Returns the linear index of each variant element in the given node. The
  // linear index is the in a DFS traversal of the type of `node`.
  std::vector<int64_t> GetVariantLinearIndices(Node* node) {
    std::vector<int64_t> result;
    int64_t linear_index = 0;
    invariant_elements_.at(node)
        .ForEach([&](Type* element_type, bool is_invariant,
                     absl::Span<const int64_t> index) {
          if (!is_invariant) {
            result.push_back(linear_index);
          }
          linear_index++;
          return absl::OkStatus();
        })
        .IgnoreError();
    return result;
  }

  std::string ToString() const {
    std::vector<std::string> lines;
    lines.push_back(absl::StrFormat("InvariantAnalysis(%s):", f_->name()));
    for (Node* node : TopoSort(f_)) {
      lines.push_back(absl::StrFormat("  %s:", node->ToString()));
      invariant_elements_.at(node)
          .ForEach([&](Type* element_type, bool is_invariant,
                       absl::Span<const int64_t> index) {
            lines.push_back(absl::StrFormat(
                "    {%s}: %d", absl::StrJoin(index, ", "), is_invariant));
            return absl::OkStatus();
          })
          .IgnoreError();
    }
    return absl::StrJoin(lines, "\n");
  }

 private:
  explicit InvariantAnalysis(FunctionBase* f) : f_(f) {}

  bool IsInvariant(Node* node) {
    // If no token types involved and all inputs invariant then output is
    // invariant.
    if (TypeHasToken(node->GetType()) || OpIsSideEffecting(node->op())) {
      return false;
    }
    for (bool element_is_invariant : invariant_elements_.at(node).elements()) {
      if (!element_is_invariant) {
        return false;
      }
    }
    return true;
  }

  absl::Status DefaultHandler(Node* node) override {
    bool node_is_invariant =
        !OpIsSideEffecting(node->op()) &&
        std::all_of(node->operands().begin(), node->operands().end(),
                    [&](Node* operand) { return IsInvariant(operand); });
    invariant_elements_[node] =
        LeafTypeTree<bool>(node->GetType(), /*init_value=*/node_is_invariant);
    return absl::OkStatus();
  }

  absl::Status HandleReceive(Receive* receive) override {
    XLS_ASSIGN_OR_RETURN(Channel * ch, GetChannelUsedByNode(receive));
    if (ch->kind() == ChannelKind::kSingleValue) {
      // The data received over a single-value channel is invariant.
      invariant_elements_[receive] =
          LeafTypeTree<bool>(receive->GetType(), /*init_value=*/true);
      // The token is not invariant.
      invariant_elements_.at(receive).Set(/*index=*/{0}, false);
    } else {
      // The data and token from a receive over a streaming channel is not
      // invariant.
      invariant_elements_[receive] =
          LeafTypeTree<bool>(receive->GetType(), /*init_value=*/false);
    }
    return absl::OkStatus();
  }

  absl::Status HandleTuple(Tuple* tuple) override {
    // Use inlined vector to avoid std::vector bool specialization abomination.
    absl::InlinedVector<bool, 1> elements;
    for (Node* operand : tuple->operands()) {
      elements.insert(elements.end(),
                      invariant_elements_.at(operand).elements().begin(),
                      invariant_elements_.at(operand).elements().end());
    }
    invariant_elements_[tuple] = LeafTypeTree<bool>(tuple->GetType(), elements);
    return absl::OkStatus();
  }

  absl::Status HandleTupleIndex(TupleIndex* tuple_index) override {
    invariant_elements_[tuple_index] =
        invariant_elements_.at(tuple_index->operand(0))
            .CopySubtree(/*index=*/{tuple_index->index()});
    return absl::OkStatus();
  }

  absl::Status HandleLiteral(Literal* literal) override {
    invariant_elements_[literal] =
        LeafTypeTree<bool>(literal->GetType(), /*init_value=*/true);
    return absl::OkStatus();
  }

  FunctionBase* f_;
  absl::flat_hash_map<Node*, LeafTypeTree<bool>> invariant_elements_;
};

// Returns the set of procs to inline. Effectively this is all of the procs in
// the package except the top-level proc.
std::vector<Proc*> GetProcsToInline(Proc* top) {
  std::vector<Proc*> procs;
  for (const std::unique_ptr<Proc>& proc : top->package()->procs()) {
    if (proc.get() == top) {
      continue;
    }
    procs.push_back(proc.get());
  }
  return procs;
}

// Verifies that the token network from the token parameter to the proc's
// next-token is a linear chain. Also verifies that the only side-effecting
// operations with tokens are sends, receives, and params.
absl::Status VerifyTokenNetwork(Proc* proc) {
  struct TokenNode {
    Node* node;
    std::vector<Node*> sources;
  };
  std::vector<TokenNode> token_nodes;
  absl::flat_hash_map<Node*, std::vector<Node*>> token_sources;
  for (Node* node : TopoSort(proc)) {
    std::vector<Node*> operand_sources;
    for (Node* operand : node->operands()) {
      if (TypeHasToken(operand->GetType())) {
        for (Node* token_source : token_sources.at(operand)) {
          if (std::find(operand_sources.begin(), operand_sources.end(),
                        token_source) == operand_sources.end()) {
            operand_sources.push_back(token_source);
          }
        }
      }
    }
    // No fan-in (or fan-out) is supported.
    if (operand_sources.size() > 1) {
      return absl::UnimplementedError(absl::StrFormat(
          "Proc inlining does not support ops with multiple token operands: %s",
          node->GetName()));
    }
    if (OpIsSideEffecting(node->op()) && TypeHasToken(node->GetType())) {
      if (!node->Is<Param>() && !node->Is<Send>() && !node->Is<Receive>()) {
        return absl::UnimplementedError(absl::StrFormat(
            "Proc inlining does not support %s ops", OpToString(node->op())));
      }

      // Only a linear chain of tokens is supported.
      if (operand_sources.size() == 1 &&
          operand_sources.front() != token_nodes.back().node) {
        return absl::UnimplementedError(
            "For proc inlining, tokens must form a linear chain.");
      }

      token_nodes.push_back(TokenNode{node, std::move(operand_sources)});
      token_sources[node] = {node};
    } else {
      token_sources[node] = std::move(operand_sources);
    }
  }
  XLS_VLOG(3) << absl::StreamFormat("Token network for proc %s:", proc->name());
  for (const TokenNode& token_node : token_nodes) {
    XLS_VLOG(3) << absl::StreamFormat(
        "%s (%s)", token_node.node->GetName(),
        absl::StrJoin(token_node.sources, ", ", NodeFormatter));
  }
  return absl::OkStatus();
}

// Makes and returns a node computing Not(node).
absl::StatusOr<Node*> Not(
    Node* node, std::optional<absl::string_view> name = absl::nullopt) {
  if (name.has_value()) {
    return node->function_base()->MakeNodeWithName<UnOp>(
        node->loc(), node, Op::kNot, name.value());
  }
  return node->function_base()->MakeNode<UnOp>(node->loc(), node, Op::kNot);
}

// Makes and returns a node computing Identity(node).
absl::StatusOr<Node*> Identity(
    Node* node, std::optional<absl::string_view> name = absl::nullopt) {
  if (name.has_value()) {
    return node->function_base()->MakeNodeWithName<UnOp>(
        node->loc(), node, Op::kIdentity, name.value());
  }
  return node->function_base()->MakeNode<UnOp>(node->loc(), node,
                                               Op::kIdentity);
}

// Makes and returns a node computing And(a, b).
absl::StatusOr<Node*> And(
    Node* a, Node* b, std::optional<absl::string_view> name = absl::nullopt) {
  if (name.has_value()) {
    return a->function_base()->MakeNodeWithName<NaryOp>(
        a->loc(), std::vector<Node*>{a, b}, Op::kAnd, name.value());
  }
  return a->function_base()->MakeNode<NaryOp>(
      a->loc(), std::vector<Node*>{a, b}, Op::kAnd);
}

// Makes and returns a node computing Or(a, b).
absl::StatusOr<Node*> Or(
    Node* a, Node* b, std::optional<absl::string_view> name = absl::nullopt) {
  if (name.has_value()) {
    return a->function_base()->MakeNodeWithName<NaryOp>(
        a->loc(), std::vector<Node*>{a, b}, Op::kOr, name.value());
  }
  return a->function_base()->MakeNode<NaryOp>(
      a->loc(), std::vector<Node*>{a, b}, Op::kOr);
}

// Makes and returns a node computing And(a, !b).
absl::StatusOr<Node*> AndNot(
    Node* a, Node* b, std::optional<absl::string_view> name = absl::nullopt) {
  XLS_ASSIGN_OR_RETURN(
      Node * not_b, a->function_base()->MakeNode<UnOp>(b->loc(), b, Op::kNot));
  return And(a, not_b, name);
}

absl::Status DecomposeNodeHelper(Node* node, std::vector<Node*>& elements) {
  if (node->GetType()->IsTuple()) {
    for (int64_t i = 0; i < node->GetType()->AsTupleOrDie()->size(); ++i) {
      XLS_ASSIGN_OR_RETURN(
          Node * tuple_index,
          node->function_base()->MakeNode<TupleIndex>(SourceInfo(), node, i));
      XLS_RETURN_IF_ERROR(DecomposeNodeHelper(tuple_index, elements));
    }
  } else if (node->GetType()->IsArray()) {
    for (int64_t i = 0; i < node->GetType()->AsArrayOrDie()->size(); ++i) {
      XLS_ASSIGN_OR_RETURN(Node * index,
                           node->function_base()->MakeNode<Literal>(
                               SourceInfo(), Value(UBits(i, 64))));
      XLS_ASSIGN_OR_RETURN(
          Node * array_index,
          node->function_base()->MakeNode<ArrayIndex>(
              SourceInfo(), node, std::vector<Node*>({index})));
      XLS_RETURN_IF_ERROR(DecomposeNodeHelper(array_index, elements));
    }
  } else {
    elements.push_back(node);
  }
  return absl::OkStatus();
}

// Decomposes and returns the elements of the given node as a vector. An
// "element" is a leaf in the type tree of the node. The elements are extracted
// using TupleIndex and ArrayIndex operations which are added to the graph as
// necessary. Example vectors returned for different types:
//
//   x: bits[32] ->
//         {x}
//   x: (bits[32], bits[32], bits[32]) ->
//         {TupleIndex(x, 0), TupleIndex(x, 1), TupleIndex(x, 2)}
//   x: (bits[32], (bits[32])) ->
//         {TupleIndex(x, 0), TupleIndex(TupleIndex(x, 1), 0)}
//   x: bits[32][2] ->
//         {ArrayIndex(x, {0}), ArrayIndex(x, {1})}
absl::StatusOr<std::vector<Node*>> DecomposeNode(Node* node) {
  std::vector<Node*> elements;
  XLS_RETURN_IF_ERROR(DecomposeNodeHelper(node, elements));
  return std::move(elements);
}

absl::StatusOr<Node*> ComposeNodeHelper(Type* type,
                                        absl::Span<Node* const> elements,
                                        int64_t& linear_index,
                                        FunctionBase* f) {
  if (type->IsTuple()) {
    std::vector<Node*> tuple_elements;
    for (int64_t i = 0; i < type->AsTupleOrDie()->size(); ++i) {
      XLS_ASSIGN_OR_RETURN(
          Node * element,
          ComposeNodeHelper(type->AsTupleOrDie()->element_type(i), elements,
                            linear_index, f));
      tuple_elements.push_back(element);
    }
    return f->MakeNode<Tuple>(SourceInfo(), tuple_elements);
  }
  if (type->IsArray()) {
    std::vector<Node*> array_elements;
    for (int64_t i = 0; i < type->AsArrayOrDie()->size(); ++i) {
      XLS_ASSIGN_OR_RETURN(
          Node * element,
          ComposeNodeHelper(type->AsArrayOrDie()->element_type(), elements,
                            linear_index, f));
      array_elements.push_back(element);
    }
    return f->MakeNode<Array>(SourceInfo(), array_elements,
                              type->AsArrayOrDie()->element_type());
  }
  return elements[linear_index++];
}

// Constructs a value of the given type using the given leaf elements. Array and
// Tuple operations are added to the graph as necessary. Example expressions
// return for given type and leaf_elements vector:
//
//   bits[32] {x} -> x
//   (bits[32], bits[32]) {x, y} -> Tuple(x, y)
//   (bits[32], (bits[32])) {x, y} -> Tuple(x, Tuple(y))
//   bits[32][2] {x, y} -> Array(x,y)
absl::StatusOr<Node*> ComposeNode(Type* type, absl::Span<Node* const> elements,
                                  FunctionBase* f) {
  int64_t linear_index = 0;
  return ComposeNodeHelper(type, elements, linear_index, f);
}

// Adds the given predicate to the send. If the send already has a predicate
// then the new predicate will be `old_predicate` AND `predicate`, otherwise the
// send is replaced with a new send with `predicate` as the predicate.
absl::StatusOr<Send*> AddSendPredicate(Send* send, Node* predicate) {
  if (send->predicate().has_value()) {
    XLS_ASSIGN_OR_RETURN(Node * new_predicate,
                         And(send->predicate().value(), predicate));
    XLS_RETURN_IF_ERROR(send->ReplaceOperandNumber(2, new_predicate));
    return send;
  }
  XLS_ASSIGN_OR_RETURN(Send * new_send, send->ReplaceUsesWithNew<Send>(
                                            send->token(), send->data(),
                                            predicate, send->channel_id()));
  XLS_RETURN_IF_ERROR(send->function_base()->RemoveNode(send));
  return new_send;
}

// Adds the given predicate to the receive. If the receive already has a
// predicate then the new predicate will be `old_predicate` AND `predicate`,
// otherwise the receive is replaced with a new receive with `predicate` as the
// predicate.
absl::StatusOr<Receive*> AddReceivePredicate(Receive* receive,
                                             Node* predicate) {
  if (receive->predicate().has_value()) {
    XLS_ASSIGN_OR_RETURN(Node * new_predicate,
                         And(receive->predicate().value(), predicate));
    XLS_RETURN_IF_ERROR(receive->ReplaceOperandNumber(1, new_predicate));
    return receive;
  }
  XLS_ASSIGN_OR_RETURN(Receive * new_receive,
                       receive->ReplaceUsesWithNew<Receive>(
                           receive->token(), predicate, receive->channel_id()));
  XLS_RETURN_IF_ERROR(receive->function_base()->RemoveNode(receive));
  return new_receive;
}

// An abstraction defining a element of the proc state. These elements are
// gathered together in a tuple to define the state.
struct StateElement {
  // Name of the element.
  std::string name;

  // Initial value of the element.
  Value initial_value;

  // Dummy placeholder for the state element. When the element is added to the
  // proc state all uses of this dummy node will be replaced by the state
  // element.
  Node* dummy;

  // The next value of the state for the next proc tick.
  Node* next;
};

// Abstraction representing a node in the activation network of a proc
// thread. An activation bit is passed along through the network to execute the
// operations corresponding to the activation node.
struct ActivationNode {
  std::string name;

  // A placeholder node representing the incoming activation bit for this send
  // operation. It will eventually be replaced by the activation passed from the
  // previous operation in the activation chain on the proc thread.
  Node* dummy_activation_in;

  // The outgoing activation bit. This will be wired to the next operation
  // in the activation chain on the proc thread.
  Node* activation_out;

  // The state element which holds the activation bit. Only present for
  // operations which can stall and hold the activation bit for more than one
  // tick.
  absl::optional<StateElement*> activation_state;
};

// Abstraction representing a send operation on a proc thread. A virtual send
// is created by decomposing a Op::kSend node into logic operations which manage
// the data and its flow-control. In a virtual send, data is "sent" over a
// dependency edge in the proc graph rather than over a channel. This data
// structure gathers together the inputs and outputs of this logic.
struct VirtualSend {
  // The channel which the originating send communicated over.
  Channel* channel;

  // The data to be sent.
  Node* data;

  // Whether `data` is valid. Only present in streaming channels. Single-value
  // channels have no flow control.
  absl::optional<Node*> data_valid;
};

// Abstraction representing a receive operation on a proc thread. A virtual
// receive is created by decomposing a Op::kReceive node into logic operations
// which manage the data and its flow-control. In a virtual receive, data is
// "sent" over a dependency edge in the proc graph rather than over a channel.
// This data structure gathers together the inputs and outputs of this logic.
struct VirtualReceive {
  // The channel which the originating receive communicated over.
  Channel* channel;

  // A placeholder node representing the data. It will eventually be replaced
  // with the data node from the corresponding virtual send.
  Node* dummy_data;

  // The output of the virtual receive node. Matches the type signature of a
  // receive node, a tuple (token, data).
  Node* receive_output;

  // The following optional fields are only present for streaming channels.
  // Single-value channel have no flow control. This means no data valid signal
  // and no holding on to the activation bit.

  // A placeholder node representing whether the data is valid. It will
  // eventually be replaced with the data valid node from the corresponding
  // virtual send.
  absl::optional<Node*> dummy_data_valid;
};

// Abstraction representing a proc thread. A proc thread virtually evaluates a
// particular proc within the top-level proc. An activation bit is threaded
// through the proc's virtual send/receive operations and keeps track of
// progress through the proc thread.
class ProcThread {
 public:
  // Creates and returns a proc thread which executes the given proc. `top` is
  // the FunctionBase which will contain the proc thread.
  static absl::StatusOr<ProcThread> Create(Proc* proc, Proc* top) {
    ProcThread proc_thread;
    proc_thread.proc_ = proc;
    proc_thread.top_ = top;

    // Create a dummy state node.
    XLS_ASSIGN_OR_RETURN(
        proc_thread.proc_state_,
        proc_thread.AllocateState(absl::StrFormat("%s_state", proc->name()),
                                  proc->GetUniqueInitValue()));

    // Create a dummy state node.
    XLS_ASSIGN_OR_RETURN(proc_thread.activation_state_,
                         proc_thread.AllocateState(
                             absl::StrFormat("%s_activation", proc->name()),
                             Value(UBits(1, 1))));

    return std::move(proc_thread);
  }

  // Allocate and return a state element. The `next` field of the element will
  // be set to nullptr. This field must be filled in prior to calling
  // ProcThread::Finalize.
  absl::StatusOr<StateElement*> AllocateState(absl::string_view name,
                                              Value initial_value);

  // Allocates an activation node on the proc threads activation
  // chain. `activation_condition` is a condition which must be met for the node
  // to pass the activation bit along.
  absl::StatusOr<ActivationNode*> AllocateActivationNode(
      absl::string_view name, absl::optional<Node*> activation_condition);

  // Completes construction of the proc thread. Weaves the activation bit
  // through the activation network and adds logic for updating the threaded
  // proc state.
  absl::Status Finalize(Node* next_state) {
    // Thread an activation bit through the activation node network. The
    // topology of this activation network mirrors the topology of the token
    // network in the original proc.
    Node* activation = activation_state_->dummy;
    for (const ActivationNode& activation_node : activation_chain_) {
      XLS_RETURN_IF_ERROR(
          activation_node.dummy_activation_in->ReplaceUsesWith(activation));
      activation = activation_node.activation_out;
    }
    activation_state_->next = activation;

    // Add selector to commit state if the activation token is present.
    XLS_ASSIGN_OR_RETURN(proc_state_->next,
                         top_->MakeNodeWithName<Select>(
                             SourceInfo(), /*selector=*/activation, /*cases=*/
                             std::vector<Node*>{GetDummyState(), next_state},
                             /*default_case=*/absl::nullopt,
                             absl::StrFormat("%s_next_state", proc_->name())));

    return absl::OkStatus();
  }

  // Creates a virtual receive corresponding to receiving the given data on the
  // given channel with an optional predicate.
  absl::StatusOr<VirtualReceive> CreateVirtualReceive(
      Channel* channel, Node* token, absl::optional<Node*> receive_predicate);

  // Creates a virtual send corresponding to sending the given data on the given
  // channel with an optional predicate.
  absl::StatusOr<VirtualSend> CreateVirtualSend(
      Channel* channel, Node* data, absl::optional<Node*> send_predicate,
      absl::Span<const int64_t> variant_linear_indices);

  // Converts `send` to a send operation which is connected to the proc thread's
  // activation network. The activation bit is included in the predicate of the
  // new send. Returns the new send.
  absl::StatusOr<Send*> ConvertToActivatedSend(Send* send);

  // Converts `receive` to a receive operation which is connected to the proc
  // thread's activation network. The activation bit is included in the
  // predicate of the new receive. Returns the new receive.
  absl::StatusOr<Receive*> ConvertToActivatedReceive(Receive* receive);

  const std::vector<std::unique_ptr<StateElement>>& GetStateElements() const {
    return state_elements_;
  }

  // Returns the dummy node representing the proc state in the proc thread.
  Node* GetDummyState() const { return proc_state_->dummy; }

 private:
  // The proc which this proc thread evaluates.
  Proc* proc_;

  // The proc in which this proc thread evaluates.
  Proc* top_;

  // The state elements required to by this proc thread. These elements are
  // later added to the top proc state.
  std::vector<std::unique_ptr<StateElement>> state_elements_;

  // The state element representing the state of the proc. This points to an
  // element in `state_elements_`.
  StateElement* proc_state_;

  // The single bit of state representing the activation bit. This points to an
  // element in `state_elements_`.
  StateElement* activation_state_;

  // The chain of activation nodes representing side-effecting operations
  // including virtual send/receives through with the activation bit will be
  // threaded. Stored as a list for pointer stability.
  std::list<ActivationNode> activation_chain_;
};

absl::StatusOr<StateElement*> ProcThread::AllocateState(absl::string_view name,
                                                        Value initial_value) {
  XLS_VLOG(3) << absl::StreamFormat(
      "AllocateState: %s, size: %d, initial value %s", name,
      initial_value.GetFlatBitCount(), initial_value.ToString());
  auto element = std::make_unique<StateElement>();
  element->name = name;
  element->initial_value = initial_value;
  XLS_ASSIGN_OR_RETURN(
      element->dummy,
      top_->MakeNodeWithName<Literal>(
          SourceInfo(),
          ZeroOfType(top_->package()->GetTypeForValue(initial_value)),
          absl::StrFormat("%s_dummy_state", name)));
  // Next element to be filled in by caller.
  element->next = nullptr;
  state_elements_.push_back(std::move(element));
  return state_elements_.back().get();
}

absl::StatusOr<ActivationNode*> ProcThread::AllocateActivationNode(
    absl::string_view name, absl::optional<Node*> activation_condition) {
  ActivationNode node;
  node.name = name;

  XLS_ASSIGN_OR_RETURN(node.dummy_activation_in,
                       top_->MakeNodeWithName<Literal>(
                           SourceInfo(), Value(UBits(1, 1)),
                           absl::StrFormat("%s_dummy_activation_in", name)));
  if (!activation_condition.has_value()) {
    // Activation node unconditionally fires when it has the activation
    // bit. The activation bit is passed along unconditionally.
    XLS_ASSIGN_OR_RETURN(node.activation_out,
                         Identity(node.dummy_activation_in,
                                  absl::StrFormat("%s_activation_out", name)));
    activation_chain_.push_back(node);
    return &activation_chain_.back();
  }

  // Activation node has a condition which guards firing. The activation node
  // may stall so it needs to store a bit on the state indicating that it
  // holds the activation bit.
  XLS_ASSIGN_OR_RETURN(
      node.activation_state,
      AllocateState(absl::StrFormat("%s_holds_activation", name),
                    Value(UBits(0, 1))));
  XLS_ASSIGN_OR_RETURN(
      Node * has_activation,
      Or(node.dummy_activation_in, node.activation_state.value()->dummy,
         absl::StrFormat("%s_has_activation", name)));

  XLS_ASSIGN_OR_RETURN(node.activation_out,
                       And(has_activation, activation_condition.value(),
                           absl::StrFormat("%s_activation_out", name)));

  XLS_ASSIGN_OR_RETURN(
      node.activation_state.value()->next,
      AndNot(has_activation, activation_condition.value(),
             absl::StrFormat("%s_next_holds_activation", name)));

  activation_chain_.push_back(node);
  return &activation_chain_.back();
}

absl::StatusOr<VirtualSend> ProcThread::CreateVirtualSend(
    Channel* channel, Node* data, absl::optional<Node*> send_predicate,
    absl::Span<const int64_t> variant_linear_indices) {
  if (channel->kind() == ChannelKind::kSingleValue &&
      send_predicate.has_value()) {
    return absl::UnimplementedError(
        "Conditional sends on single-value channels are not supported");
  }

  VirtualSend virtual_send;
  virtual_send.channel = channel;

  XLS_ASSIGN_OR_RETURN(
      ActivationNode * activation_node,
      AllocateActivationNode(absl::StrFormat("%s_send", channel->name()),
                             /*activation_condition=*/absl::nullopt));

  // Only streaming channels have flow control (valid signal).
  if (channel->kind() == ChannelKind::kStreaming) {
    virtual_send.data = data;
    if (send_predicate.has_value()) {
      // The send is conditional. Valid logic:
      //
      //   data_valid = activation_in && cond
      XLS_ASSIGN_OR_RETURN(
          virtual_send.data_valid,
          And(activation_node->dummy_activation_in, send_predicate.value(),
              absl::StrFormat("%s_data_valid", channel->name())));
    } else {
      // The send is unconditional. The channel has data if the send is
      // activated.
      XLS_ASSIGN_OR_RETURN(
          virtual_send.data_valid,
          Identity(activation_node->dummy_activation_in,
                   absl::StrFormat("%s_data_valid", channel->name())));
    }
  } else if (variant_linear_indices.empty()) {
    // This is a single-value channel with no variant elements. The virtual
    // send simply passes through the data.
    XLS_RET_CHECK_EQ(channel->kind(), ChannelKind::kSingleValue);
    XLS_VLOG(3) << absl::StreamFormat(
        "Generating virtual send for single-value channel %s of type %s with "
        "no variants",
        channel->name(), channel->type()->ToString());
    virtual_send.data = data;
  } else {
    // This is a single-value channel with variant elements being sent over
    // the channel. The variant values must be saved on the state.
    XLS_RET_CHECK_EQ(channel->kind(), ChannelKind::kSingleValue);
    XLS_VLOG(3) << absl::StreamFormat(
        "Generating virtual send for single-value channel %s of type %s with "
        "variants",
        channel->name(), channel->type()->ToString());

    // Completely decompose the sent data into a std::vector<Node*> of
    // elements. These elements are exactly the leaf elements of the type as
    // defined in LeafTypeTree.
    XLS_ASSIGN_OR_RETURN(std::vector<Node*> data_elements, DecomposeNode(data));

    // Construct a tuple-type containing the type of each variant element.
    XLS_VLOG(3) << absl::StreamFormat(
        "Variant linear indices (out of %d leaves):", data_elements.size());
    std::vector<Type*> variant_element_types;
    for (int64_t i = 0; i < variant_linear_indices.size(); ++i) {
      int64_t linear_index = variant_linear_indices[i];
      XLS_VLOG(3) << absl::StreamFormat(
          "  %d: %s", linear_index,
          data_elements[linear_index]->GetType()->ToString());
      variant_element_types.push_back(data_elements[linear_index]->GetType());
    }
    Type* variant_tuple_type =
        top_->package()->GetTupleType(variant_element_types);

    XLS_VLOG(3) << absl::StreamFormat(
        "Total size of saved data for channel %s: %d", channel->name(),
        variant_tuple_type->GetFlatBitCount());

    // Construct a dummy state variable for the variant element tuple type.
    XLS_ASSIGN_OR_RETURN(
        StateElement * saved_data,
        AllocateState(absl::StrFormat("%s_send_saved_data", channel->name()),
                      ZeroOfType(variant_tuple_type)));

    // Replace the variant elements of the sent data element vector with a
    // select between the state and the data value. The selector is the
    // activation bit of the send.
    std::vector<Node*> next_saved_data_elements;
    for (int64_t i = 0; i < variant_linear_indices.size(); ++i) {
      int64_t linear_index = variant_linear_indices[i];
      XLS_ASSIGN_OR_RETURN(
          Node * state_element,
          top_->MakeNode<TupleIndex>(SourceInfo(), saved_data->dummy, i));
      XLS_ASSIGN_OR_RETURN(
          Node * selected_data,
          top_->MakeNodeWithName<Select>(
              SourceInfo(),
              /*selector=*/activation_node->dummy_activation_in,
              /*cases=*/
              std::vector<Node*>{state_element, data_elements[linear_index]},
              /*default_case=*/absl::nullopt,
              absl::StrFormat("%s_data", channel->name())));
      next_saved_data_elements.push_back(selected_data);
      data_elements[linear_index] = selected_data;
    }

    // Reconstruct the data to send from the constituent elements.
    XLS_ASSIGN_OR_RETURN(virtual_send.data,
                         ComposeNode(data->GetType(), data_elements, top_));

    // Construct the tuple for the next state. It consists of the selects rom
    // each variant element.
    XLS_ASSIGN_OR_RETURN(
        saved_data->next,
        top_->MakeNodeWithName<Tuple>(
            SourceInfo(), next_saved_data_elements,
            absl::StrFormat("%s_send_hold_next", channel->name())));
  }

  return virtual_send;
}

absl::StatusOr<VirtualReceive> ProcThread::CreateVirtualReceive(
    Channel* channel, Node* token, absl::optional<Node*> receive_predicate) {
  if (channel->kind() == ChannelKind::kSingleValue &&
      receive_predicate.has_value()) {
    return absl::UnimplementedError(
        "Conditional receives on single-value channels are not supported");
  }

  VirtualReceive virtual_receive;
  virtual_receive.channel = channel;
  SourceInfo loc;

  // Create a dummy channel data node. Later this will be replaced with
  // the data sent from the the corresponding send node.
  XLS_ASSIGN_OR_RETURN(virtual_receive.dummy_data,
                       top_->MakeNodeWithName<Literal>(
                           loc, ZeroOfType(channel->type()),
                           absl::StrFormat("%s_dummy_data", channel->name())));

  ActivationNode* activation_node;
  if (channel->kind() == ChannelKind::kStreaming) {
    // Create a dummy data valid node. Later this will be replaced with
    // signal generated from the corresponding send.
    XLS_ASSIGN_OR_RETURN(
        virtual_receive.dummy_data_valid,
        top_->MakeNodeWithName<Literal>(
            loc, Value(UBits(0, 1)),
            absl::StrFormat("%s_dummy_data_valid", channel->name())));

    // Logic indicating whether the activation will be passed along (if the
    // receive has the activation):
    //
    //   pass_activation_along = (!predicate || data_valid)
    Node* pass_activation_along;
    if (receive_predicate.has_value()) {
      XLS_ASSIGN_OR_RETURN(Node * not_predicate,
                           Not(receive_predicate.value()));
      XLS_ASSIGN_OR_RETURN(
          pass_activation_along,
          Or(not_predicate, virtual_receive.dummy_data_valid.value(),
             absl::StrFormat("%s_rcv_pass_activation", channel->name())));
    } else {
      // Receive is unconditional. The activation is passed along iff the
      // channel has data.
      XLS_ASSIGN_OR_RETURN(
          pass_activation_along,
          Identity(virtual_receive.dummy_data_valid.value(),
                   absl::StrFormat("%s_rcv_pass_activation", channel->name())));
    }

    XLS_ASSIGN_OR_RETURN(
        activation_node,
        AllocateActivationNode(absl::StrFormat("%s_rcv", channel->name()),
                               pass_activation_along));

    // Add assert which fails when data is lost. That is, data_valid is true
    // and the receive did not fire.
    //
    //   receive_fired = has_activation && predicate
    //   data_loss = data_valid && !activation_out
    //   assert(!data_loss)
    XLS_ASSIGN_OR_RETURN(
        Node * has_activation,
        Or(activation_node->dummy_activation_in,
           activation_node->activation_state.value()->dummy,
           absl::StrFormat("%s_rcv_has_activation", channel->name())));

    Node* receive_fired;
    if (receive_predicate.has_value()) {
      XLS_ASSIGN_OR_RETURN(
          receive_fired, And(has_activation, receive_predicate.value(),
                             absl::StrFormat("%s_rcv_fired", channel->name())));
    } else {
      receive_fired = activation_node->activation_out;
    }
    XLS_ASSIGN_OR_RETURN(
        Node * data_loss,
        AndNot(virtual_receive.dummy_data_valid.value(), receive_fired,
               absl::StrFormat("%s_data_loss", channel->name())));
    XLS_ASSIGN_OR_RETURN(Node * no_data_loss, Not(data_loss));
    XLS_ASSIGN_OR_RETURN(
        Node * asrt,
        top_->MakeNode<Assert>(
            loc, token, no_data_loss,
            /*message=*/
            absl::StrFormat("Channel %s lost data", channel->name()),
            /*label=*/
            absl::StrFormat("%s_data_loss_assert", channel->name())));

    // Ensure assert is threaded into token network
    token = asrt;
  } else {
    // Nothing to do for receives on single-value channels.
    XLS_RET_CHECK(channel->kind() == ChannelKind::kSingleValue);
  }

  // TODO(meheff): 2022/02/11 For conditional receives, add a select between
  // the data value and zero to preserve the zero-value semantics when the
  // predicate is false.

  // The output of a receive operation is a tuple of (token, data).
  XLS_ASSIGN_OR_RETURN(
      virtual_receive.receive_output,
      top_->MakeNodeWithName<Tuple>(
          loc, std::vector<Node*>{token, virtual_receive.dummy_data},
          absl::StrFormat("%s_rcv", channel->name())));

  return virtual_receive;
}

absl::StatusOr<Send*> ProcThread::ConvertToActivatedSend(Send* send) {
  XLS_ASSIGN_OR_RETURN(Channel * ch, GetChannelUsedByNode(send));
  XLS_ASSIGN_OR_RETURN(
      ActivationNode * activation_node,
      AllocateActivationNode(absl::StrFormat("%s_send", ch->name()),
                             /*activation_condition=*/absl::nullopt));
  if (send->predicate().has_value()) {
    XLS_ASSIGN_OR_RETURN(
        Node * new_predicate,
        And(activation_node->dummy_activation_in, send->predicate().value()));
    return AddSendPredicate(send, new_predicate);
  }
  return AddSendPredicate(send, activation_node->dummy_activation_in);
}

absl::StatusOr<Receive*> ProcThread::ConvertToActivatedReceive(
    Receive* receive) {
  XLS_ASSIGN_OR_RETURN(Channel * ch, GetChannelUsedByNode(receive));
  XLS_ASSIGN_OR_RETURN(
      ActivationNode * activation_node,
      AllocateActivationNode(absl::StrFormat("%s_receive", ch->name()),
                             /*activation_condition=*/absl::nullopt));
  if (receive->predicate().has_value()) {
    XLS_ASSIGN_OR_RETURN(Node * new_predicate,
                         And(activation_node->dummy_activation_in,
                             receive->predicate().value()));
    return AddReceivePredicate(receive, new_predicate);
  }
  return AddReceivePredicate(receive, activation_node->dummy_activation_in);
}

// Converts the logic of the side-effecting operations of `proc` into a proc
// thread. The proc thread executes inside of `proc` itself.  Newly created
// virtual send/receieves are inserted into the `virtual_send` and
// `virtual_receive` maps.
absl::StatusOr<ProcThread> ConvertToProcThread(
    Proc* proc, absl::flat_hash_map<Channel*, VirtualSend>& virtual_sends,
    absl::flat_hash_map<Channel*, VirtualReceive>& virtual_receives) {
  XLS_ASSIGN_OR_RETURN(ProcThread proc_thread, ProcThread::Create(proc, proc));

  // Before transforming the proc at all verify the token network of the proc.
  XLS_RETURN_IF_ERROR(VerifyTokenNetwork(proc));

  XLS_RETURN_IF_ERROR(proc->GetUniqueStateParam()->ReplaceUsesWith(
      proc_thread.GetDummyState()));

  XLS_ASSIGN_OR_RETURN(InvariantAnalysis invariant_analysis,
                       InvariantAnalysis::Run(proc));

  for (Node* node : TopoSort(proc)) {
    if (node->Is<Send>()) {
      Send* send = node->As<Send>();
      XLS_ASSIGN_OR_RETURN(Channel * ch, GetChannelUsedByNode(send));
      if (ch->supported_ops() == ChannelOps::kSendReceive) {
        // Non-external send. Convert send into a virtual send.
        XLS_ASSIGN_OR_RETURN(
            VirtualSend virtual_send,
            proc_thread.CreateVirtualSend(
                ch, send->data(), send->predicate(),
                invariant_analysis.GetVariantLinearIndices(send->data())));
        XLS_RETURN_IF_ERROR(send->ReplaceUsesWith(send->token()));
        virtual_sends[ch] = virtual_send;
        continue;
      }
      if (ch->kind() == ChannelKind::kStreaming) {
        // A streaming external channel needs to be added to the activation
        // chain. The send can fire only when it is activated.
        XLS_RETURN_IF_ERROR(proc_thread.ConvertToActivatedSend(send).status());
        continue;
      }
      // Nothing to do for external single value channels.
      XLS_RET_CHECK_EQ(ch->kind(), ChannelKind::kSingleValue);
    } else if (node->Is<Receive>()) {
      Receive* receive = node->As<Receive>();
      XLS_ASSIGN_OR_RETURN(Channel * ch, GetChannelUsedByNode(receive));
      if (ch->supported_ops() == ChannelOps::kSendReceive) {
        // Non-external receive. Convert receive into a virtual receive.
        XLS_ASSIGN_OR_RETURN(VirtualReceive virtual_receive,
                             proc_thread.CreateVirtualReceive(
                                 ch, receive->token(), receive->predicate()));
        XLS_RETURN_IF_ERROR(
            receive->ReplaceUsesWith(virtual_receive.receive_output));
        virtual_receives[ch] = virtual_receive;
        continue;
      }
      if (ch->kind() == ChannelKind::kStreaming) {
        // A streaming external channel needs to be added to the activation
        // chain. The receive can fire only when it is activated.
        XLS_RETURN_IF_ERROR(
            proc_thread.ConvertToActivatedReceive(receive).status());
        continue;
      }
      // Nothing to do for external single value channels.
      XLS_RET_CHECK_EQ(ch->kind(), ChannelKind::kSingleValue);
    }
  }

  XLS_RETURN_IF_ERROR(proc_thread.Finalize(
      /*next_state=*/proc->GetUniqueNextState()));

  return std::move(proc_thread);
}

// Inlines the given proc into `top` as a proc thread. Sends and receives in
// `proc` are replaced with virtual sends/receives which execute via the
// activation chain. Newly created virtual send/receieves are inserted into the
// `virtual_send` and `virtual_receive` maps.
absl::StatusOr<ProcThread> InlineProcAsProcThread(
    Proc* proc, Proc* top,
    absl::flat_hash_map<Channel*, VirtualSend>& virtual_sends,
    absl::flat_hash_map<Channel*, VirtualReceive>& virtual_receives) {
  XLS_ASSIGN_OR_RETURN(ProcThread proc_thread, ProcThread::Create(proc, top));
  absl::flat_hash_map<Node*, Node*> node_map;

  // Before transforming the proc at all verify the token network of the proc.
  XLS_RETURN_IF_ERROR(VerifyTokenNetwork(proc));

  XLS_ASSIGN_OR_RETURN(InvariantAnalysis invariant_analysis,
                       InvariantAnalysis::Run(proc));

  auto clone_node = [&](Node* node) -> absl::StatusOr<Node*> {
    std::vector<Node*> new_operands;
    for (Node* operand : node->operands()) {
      new_operands.push_back(node_map.at(operand));
    }
    return node->CloneInNewFunction(new_operands, top);
  };

  for (Node* node : TopoSort(proc)) {
    if (node->Is<Param>()) {
      if (node == proc->GetUniqueStateParam()) {
        // The dummy state value will later be replaced with an element from the
        // top-level proc state.
        node_map[node] = proc_thread.GetDummyState();
      } else {
        // Connect the inlined token network from `proc` to the token parameter
        // of `top`.
        XLS_RET_CHECK_EQ(node, proc->TokenParam());
        node_map[node] = top->TokenParam();
      }
    } else if (node->Is<Send>()) {
      Send* send = node->As<Send>();
      XLS_ASSIGN_OR_RETURN(Channel * ch, GetChannelUsedByNode(send));
      // Only convert sends over internal channels into virtual sends. Sends
      // over external channels are inlined as is.
      if (ch->supported_ops() == ChannelOps::kSendReceive) {
        absl::optional<Node*> predicate;
        if (send->predicate().has_value()) {
          predicate = node_map.at(send->predicate().value());
        }
        XLS_ASSIGN_OR_RETURN(
            VirtualSend virtual_send,
            proc_thread.CreateVirtualSend(
                ch, node_map.at(send->data()), predicate,
                invariant_analysis.GetVariantLinearIndices(send->data())));

        // The output of the send operation itself is token-typed. Just use the
        // token operand of the send.
        node_map[node] = node_map.at(send->token());
        virtual_sends[ch] = virtual_send;
      } else if (ch->kind() == ChannelKind::kStreaming) {
        // A streaming external channel needs to be added to the activation
        // chain. The send can fire only when it is activated.
        XLS_ASSIGN_OR_RETURN(Node * new_send, clone_node(send));
        XLS_ASSIGN_OR_RETURN(
            Send * activated_send,
            proc_thread.ConvertToActivatedSend(new_send->As<Send>()));
        node_map[node] = activated_send;
      } else {
        // Nothing special to do for external single value channels. Simply
        // clone the node into the top.
        XLS_RET_CHECK_EQ(ch->kind(), ChannelKind::kSingleValue);
        XLS_ASSIGN_OR_RETURN(node_map[node], clone_node(node));
      }
    } else if (node->Is<Receive>()) {
      Receive* receive = node->As<Receive>();
      XLS_ASSIGN_OR_RETURN(Channel * ch, GetChannelUsedByNode(receive));
      // Only convert receives over internal channels into virtual receives.
      // Receives over external channels are inlined as is.
      if (ch->supported_ops() == ChannelOps::kSendReceive) {
        absl::optional<Node*> predicate;
        if (receive->predicate().has_value()) {
          predicate = node_map.at(receive->predicate().value());
        }
        XLS_ASSIGN_OR_RETURN(VirtualReceive virtual_receive,
                             proc_thread.CreateVirtualReceive(
                                 ch, node_map.at(receive->token()), predicate));

        node_map[node] = virtual_receive.receive_output;
        virtual_receives[ch] = virtual_receive;
      } else if (ch->kind() == ChannelKind::kStreaming) {
        // A streaming external channel needs to be added to the activation
        // chain. The receive can fire only when it is activated.
        XLS_ASSIGN_OR_RETURN(Node * new_receive, clone_node(receive));
        XLS_ASSIGN_OR_RETURN(
            Receive * activated_receive,
            proc_thread.ConvertToActivatedReceive(new_receive->As<Receive>()));
        node_map[node] = activated_receive;
      } else {
        // Nothing special to do for external single value channels. Simply
        // clone the node into the top.
        XLS_RET_CHECK_EQ(ch->kind(), ChannelKind::kSingleValue);
        XLS_ASSIGN_OR_RETURN(node_map[node], clone_node(node));
      }
    } else {
      // Other operations are just cloned into `top`.
      XLS_ASSIGN_OR_RETURN(node_map[node], clone_node(node));
    }
  }

  XLS_RETURN_IF_ERROR(proc_thread.Finalize(
      /*next_state=*/node_map.at(proc->GetUniqueNextState())));

  // Wire in the next-token value from the inlined proc into the next-token of
  // the top proc. Use an afterall to join the tokens. If the top next-token is
  // already an afterall, replace it with an afterall with the proc's next-token
  // added in.
  Node* proc_next_token = node_map.at(proc->NextToken());
  if (top->NextToken()->Is<AfterAll>()) {
    std::vector<Node*> operands(top->NextToken()->operands().begin(),
                                top->NextToken()->operands().end());
    operands.push_back(proc_next_token);
    Node* old_after_all = top->NextToken();
    XLS_RETURN_IF_ERROR(
        old_after_all->ReplaceUsesWithNew<AfterAll>(operands).status());
    XLS_RETURN_IF_ERROR(top->RemoveNode(old_after_all));
  } else {
    XLS_RETURN_IF_ERROR(top->NextToken()
                            ->ReplaceUsesWithNew<AfterAll>(std::vector<Node*>(
                                {top->NextToken(), proc_next_token}))
                            .status());
  }

  return std::move(proc_thread);
}

// Replace the state of `proc` with a tuple defined by the given StateElements.
absl::Status ReplaceProcState(Proc* proc,
                              absl::Span<const StateElement> elements) {
  std::vector<Value> initial_values;
  std::vector<Type*> types;
  std::vector<Node*> nexts;
  for (const StateElement& element : elements) {
    initial_values.push_back(element.initial_value);
    types.push_back(element.dummy->GetType());
    nexts.push_back(element.next);
  }
  Value initial_value = Value::Tuple(initial_values);
  XLS_ASSIGN_OR_RETURN(
      Node * next,
      proc->MakeNodeWithName<Tuple>(
          SourceInfo(), nexts,
          absl::StrFormat("%s_next", proc->GetUniqueStateParam()->GetName())));
  XLS_RETURN_IF_ERROR(proc->ReplaceUniqueState(
      proc->GetUniqueStateParam()->name(), next, initial_value));

  for (int64_t i = 0; i < elements.size(); ++i) {
    XLS_ASSIGN_OR_RETURN(
        Node * state_element,
        proc->MakeNodeWithName<TupleIndex>(
            SourceInfo(), proc->GetUniqueStateParam(), i, elements[i].name));
    XLS_RETURN_IF_ERROR(elements[i].dummy->ReplaceUsesWith(state_element));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> ProcInliningPass::RunInternal(Package* p,
                                                   const PassOptions& options,
                                                   PassResults* results) const {
  // TODO(meheff): 2022/4/4 Remove when proc state optimization lands and flop
  // count with new proc inliner is comparable to the old one.
  if (options.use_new_proc_inliner) {
    NewProcInliningPass new_proc_inlining_pass;
    return new_proc_inlining_pass.RunInternal(p, options, results);
  }

  if (!options.inline_procs || p->procs().empty()) {
    return false;
  }

  if (!p->HasTop()) {
    return absl::InvalidArgumentError(
        "Must specify top-level proc name when running proc inlining");
  }

  FunctionBase* top_func_base = p->GetTop().value();

  if (!top_func_base->IsProc()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Top level %s should be a proc when running proc inlining",
        top_func_base->name()));
  }

  Proc* top = top_func_base->AsProcOrDie();

  std::vector<Proc*> procs_to_inline = GetProcsToInline(top);
  if (procs_to_inline.empty()) {
    return false;
  }

  std::vector<ProcThread> proc_threads;
  absl::flat_hash_map<Channel*, VirtualSend> virtual_sends;
  absl::flat_hash_map<Channel*, VirtualReceive> virtual_receives;

  // Convert send/receives in top proc which communicate with other procs into
  // virtual send/receives.
  XLS_ASSIGN_OR_RETURN(
      ProcThread top_thread,
      ConvertToProcThread(top, virtual_sends, virtual_receives));
  proc_threads.push_back(std::move(top_thread));

  // Inline each proc into `top`. Sends/receives are converted to virtual
  // send/receives.
  // TODO(meheff): 2022/02/11 Add analysis which determines whether inlining is
  // a legal transformation.
  for (Proc* proc : procs_to_inline) {
    XLS_ASSIGN_OR_RETURN(
        ProcThread proc_thread,
        InlineProcAsProcThread(proc, top, virtual_sends, virtual_receives));
    proc_threads.push_back(std::move(proc_thread));
  }

  XLS_VLOG(3) << "After inlining procs:\n" << p->DumpIr();

  // Connect data and data_valid signals between the virtual send and the
  // respective virtual receive nodes.
  for (Channel* ch : p->channels()) {
    if (ch->supported_ops() != ChannelOps::kSendReceive) {
      continue;
    }
    const VirtualSend& virtual_send = virtual_sends.at(ch);
    const VirtualReceive& virtual_receive = virtual_receives.at(ch);

    XLS_VLOG(3) << absl::StreamFormat(
        "Connecting channel %s data: replacing %s with %s", ch->name(),
        virtual_receive.dummy_data->GetName(), virtual_send.data->GetName());
    XLS_RETURN_IF_ERROR(
        virtual_receive.dummy_data->ReplaceUsesWith(virtual_send.data));

    // Only streaming channels have flow control (valid signal).
    if (ch->kind() == ChannelKind::kStreaming) {
      XLS_RET_CHECK(virtual_send.data_valid.has_value());
      XLS_RET_CHECK(virtual_receive.dummy_data_valid.has_value());
      XLS_VLOG(3) << absl::StreamFormat(
          "Connecting channel %s data_valid: replacing %s with %s", ch->name(),
          virtual_receive.dummy_data_valid.value()->GetName(),
          virtual_send.data_valid.value()->GetName());
      XLS_RETURN_IF_ERROR(
          virtual_receive.dummy_data_valid.value()->ReplaceUsesWith(
              virtual_send.data_valid.value()));
    } else {
      XLS_RET_CHECK(ch->kind() == ChannelKind::kSingleValue);
      XLS_RET_CHECK(!virtual_send.data_valid.has_value());
      XLS_RET_CHECK(!virtual_receive.dummy_data_valid.has_value());
    }
  }

  XLS_VLOG(3) << "After connecting data channels:\n" << p->DumpIr();

  // Gather all inlined proc state and proc thread book-keeping bits and add to
  // the top-level proc state.
  std::vector<StateElement> state_elements;

  // Add the inlined (and top) proc state and activation bits.
  for (const ProcThread& proc_thread : proc_threads) {
    for (const auto& state_element : proc_thread.GetStateElements()) {
      state_elements.push_back(*state_element);
    }
  }
  XLS_RETURN_IF_ERROR(ReplaceProcState(top, state_elements));

  XLS_VLOG(3) << "After transforming proc state:\n" << p->DumpIr();

  // Delete inlined procs.
  for (Proc* proc : procs_to_inline) {
    XLS_RETURN_IF_ERROR(p->RemoveProc(proc));
  }

  // Delete send and receive nodes in top which were used for communicating with
  // the inlined procs.
  std::vector<Node*> to_remove;
  for (Node* node : top->nodes()) {
    if (!IsChannelNode(node)) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(Channel * ch, GetChannelUsedByNode(node));
    if (ch->supported_ops() == ChannelOps::kSendReceive) {
      to_remove.push_back(node);
    }
  }
  for (Node* node : to_remove) {
    XLS_RETURN_IF_ERROR(top->RemoveNode(node));
  }

  // Delete channels used for communicating with the inlined procs.
  std::vector<Channel*> channels(p->channels().begin(), p->channels().end());
  for (Channel* ch : channels) {
    if (ch->supported_ops() == ChannelOps::kSendReceive) {
      XLS_RETURN_IF_ERROR(p->RemoveChannel(ch));
    }
  }

  XLS_VLOG(3) << "After deleting inlined procs:\n" << p->DumpIr();

  return true;
}

}  // namespace xls
