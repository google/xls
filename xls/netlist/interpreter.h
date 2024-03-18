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
#ifndef XLS_NETLIST_INTERPRETER_H_
#define XLS_NETLIST_INTERPRETER_H_

#include <atomic>
#include <deque>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/thread.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/netlist/function_parser.h"
#include "xls/netlist/netlist.h"

namespace xls {
namespace netlist {

template <typename EvalT>
using AbstractNetRef2Value =
    absl::flat_hash_map<const rtl::AbstractNetRef<EvalT>, EvalT>;

using NetRef2Value = AbstractNetRef2Value<bool>;

// Interprets Netlists/Modules given a set of input values and returns the
// resulting value.
template <typename EvalT = bool>
class AbstractInterpreter {
 public:
  explicit AbstractInterpreter(rtl::AbstractNetlist<EvalT>* netlist, EvalT zero,
                               EvalT one, size_t num_threads = 0)
      : netlist_(netlist),
        zero_(std::move(zero)),
        one_(std::move(one)),
        num_available_threads_(0),
        threads_should_exit_(false) {
    for (int c = 0; c < num_threads; ++c) {
      threads_.push_back(std::move(
          std::make_unique<xls::Thread>([this]() { CHECK_OK(ThreadBody()); })));
    }
  }

  ~AbstractInterpreter() {
    if (threads_.empty() == false) {
      // Wake up threads
      input_queue_guard_.Lock();
      threads_should_exit_.store(true);
      input_queue_cond_.SignalAll();
      input_queue_guard_.Unlock();
      // Wait for exit
      for (auto& t : threads_) {
        t->Join();
      }
    }
  }

  template <typename = std::is_constructible<EvalT, bool>>
  explicit AbstractInterpreter(rtl::AbstractNetlist<EvalT>* netlist)
      : AbstractInterpreter(netlist, EvalT{false}, EvalT{true}) {}

  // Interprets the given module with the given input mapping.
  //  - inputs: Mapping of module input wire to value. Must have the same size
  //    as module->inputs();
  //  - dump_cells: List of cells whose inputs and outputs should be dumped
  //    on evaluation.
  absl::StatusOr<AbstractNetRef2Value<EvalT>> InterpretModule(
      const rtl::AbstractModule<EvalT>* module,
      const AbstractNetRef2Value<EvalT>& inputs,
      absl::Span<const std::string> dump_cells = {});

 private:
  // Returns true if the specified AbstractNetRef is an output of the given
  // cell.

  absl::StatusOr<AbstractNetRef2Value<EvalT>> InterpretCell(
      const rtl::AbstractCell<EvalT>* cell,
      const AbstractNetRef2Value<EvalT>& inputs);

  // A struct to contain the state of a cell as we are processing the netlist.
  // Initially, all cells are unsatisfied, meaning none of their input wires are
  // ready.  A cell is satisfied when all of its inputs are marked active.  When
  // a wire is marked active, its value is added to the relevant cell's entry's
  // ProcessedCellState::inputs.  When all of a cell's inputs become ready,  the
  // cell's ProcessedCellState::inputs is sent to InterpretCell.
  struct ProcessedCellState {
    size_t missing_wires = 0;
    AbstractNetRef2Value<EvalT> inputs;
  };

  // Called after InterpretCell finishes, to update InterpretModule()'s local
  // state (map of cells to ProcessedCellState, and active_wires).
  void UpdateProcessedState(
      absl::flat_hash_map<rtl::AbstractCell<EvalT>*,
                          std::unique_ptr<ProcessedCellState>>& processed_cells,
      std::deque<rtl::AbstractNetRef<EvalT>>& active_wires,
      AbstractNetRef2Value<EvalT>& module_outputs,
      const rtl::AbstractModule<EvalT>* module,
      const absl::flat_hash_set<std::string>& dump_cell_set,
      const rtl::AbstractCell<EvalT>* cell, AbstractNetRef2Value<EvalT>& wires);

  absl::StatusOr<EvalT> InterpretFunction(
      const rtl::AbstractCell<EvalT>& cell, const function::Ast& ast,
      const AbstractNetRef2Value<EvalT>& inputs);

  // Returns the value of the internal/output pin from the cell (defined by a
  // "statetable" attribute under the conditions defined in "inputs".
  absl::StatusOr<EvalT> InterpretStateTable(
      const rtl::AbstractCell<EvalT>& cell, const std::string& pin_name,
      const AbstractNetRef2Value<EvalT>& inputs);

  absl::Status ThreadBody();

  rtl::AbstractNetlist<EvalT>* netlist_;
  EvalT zero_;
  EvalT one_;

  // Queue entries for both the input and output queues.  Going in, wires
  // holds the inputs.  Going out, wires holds the results of InterpretCell.
  struct QueueEntry {
    const rtl::AbstractCell<EvalT>* cell;
    AbstractNetRef2Value<EvalT> wires;
  };

  // The absl::Mutex input_queue_guard is used to both protect the input queue
  // shared by all the worker threads, as well as a conditional variable for the
  // threads to block on while waiting for input.
  absl::Mutex input_queue_guard_;  // protects input_queue_
  absl::CondVar input_queue_cond_;
  std::queue<QueueEntry> input_queue_ ABSL_GUARDED_BY(input_queue_guard_);

  static bool queue_has_data(std::queue<QueueEntry>* queue) {
    return !queue->empty();
  }

  // The absl::Mutex output_queue_guard is used to both protect the output queue
  // shared by all the worker threads, as well as a conditional variable for the
  // InterpretModule() to wait on for processed cells.
  absl::Mutex output_queue_guard_;  // protects output_queue_
  std::queue<QueueEntry> output_queue_ ABSL_GUARDED_BY(output_queue_guard_);

  // Thread pool.
  std::vector<std::unique_ptr<xls::Thread>> threads_;

  // Keeps track of threads blocked on the input_queue_, ready to get a
  // dispatch.  The counter is atomic by itself, but it also needs to be updated
  // in sync with a thread blocking on the input queue, which is why it's also
  // guarded by a mutex.  The atomicity is to ensure memory ordering.
  std::atomic_size_t num_available_threads_ ABSL_GUARDED_BY(input_queue_guard_);
  // Set to shut down thread pool.
  std::atomic_bool threads_should_exit_ ABSL_GUARDED_BY(input_queue_guard_);
};

using Interpreter = AbstractInterpreter<>;

template <typename EvalT>
void AbstractInterpreter<EvalT>::UpdateProcessedState(
    absl::flat_hash_map<rtl::AbstractCell<EvalT>*,
                        std::unique_ptr<ProcessedCellState>>& processed_cells,
    std::deque<rtl::AbstractNetRef<EvalT>>& active_wires,
    AbstractNetRef2Value<EvalT>& module_outputs,
    const rtl::AbstractModule<EvalT>* module,
    const absl::flat_hash_set<std::string>& dump_cell_set,
    const rtl::AbstractCell<EvalT>* cell, AbstractNetRef2Value<EvalT>& wires) {
  // The NetRefs in cell->outputs() are also in wires, which contains the result
  // of that cell's evaluation.  We could use either one.

  for (auto wire_val : wires) {
    rtl::AbstractNetRef<EvalT> wire = wire_val.first;
    active_wires.push_back(wire);
    // Is this wire connected to any cells?  If so, add its value to these cells
    // inputs.  If not connected to any cells, then it's either floating or a
    // module output.
    for (const auto connected_cell : wire->connected_input_cells()) {
      // We want to copy, not move, the value in wire_val.second.
      processed_cells[connected_cell]->inputs.insert({wire, wire_val.second});
    }

    if (wire->kind() == rtl::NetDeclKind::kOutput) {
      for (const rtl::AbstractNetRef<EvalT> output : module->outputs()) {
        if (output == wire) {
          module_outputs.try_emplace(wire, wire_val.second);
        }
      }
    }
  }

  if (dump_cell_set.contains(cell->name())) {
    XLS_LOG(INFO) << "Cell " << cell->name() << " inputs:";
    if constexpr (std::is_convertible<EvalT, int>()) {
      for (const auto& input : cell->inputs()) {
        XLS_LOG(INFO) << "   " << input.netref->name() << " : "
                      << static_cast<int>(
                             processed_cells.at(cell)->inputs.at(input.netref));
      }

      XLS_LOG(INFO) << "Cell " << cell->name() << " outputs:";
      for (const auto& output : cell->outputs()) {
        XLS_LOG(INFO) << "   " << output.netref->name() << " : "
                      << static_cast<int>(wires[output.netref]);
      }
    } else {
      XLS_LOG(INFO) << "Cell " << cell->name() << " inputs are not printable.";
    }
  }
}

template <typename EvalT>
absl::StatusOr<AbstractNetRef2Value<EvalT>>
AbstractInterpreter<EvalT>::InterpretModule(
    const rtl::AbstractModule<EvalT>* module,
    const AbstractNetRef2Value<EvalT>& inputs,
    absl::Span<const std::string> dump_cells) {
  // Reserve space in the outputs map.
  AbstractNetRef2Value<EvalT> outputs;
  outputs.reserve(module->outputs().size());

  // Do a topological sort through all cells, evaluating each as its inputs are
  // fully satisfied, and store those results with each output wire.

  absl::flat_hash_map<rtl::AbstractCell<EvalT>*,
                      std::unique_ptr<ProcessedCellState>>
      processed_cells;

  // The set of wires that have been "activated" (whose source cells have been
  // processed) but not yet processed.
  std::deque<rtl::AbstractNetRef<EvalT>> active_wires;

  absl::flat_hash_set<std::string> dump_cell_set(dump_cells.begin(),
                                                 dump_cells.end());

  // First, populate the unsatisfied cell list.
  for (const auto& cell : module->cells()) {
    // if a cell has no inputs, it's active, so process it now.
    auto pcs = std::make_unique<ProcessedCellState>();
    if (cell->inputs().empty()) {
      XLS_ASSIGN_OR_RETURN(AbstractNetRef2Value<EvalT> results,
                           InterpretCell(cell.get(), {}));
      processed_cells[cell.get()] = std::move(pcs);
      UpdateProcessedState(processed_cells, active_wires, outputs, module,
                           dump_cell_set, cell.get(), results);
    } else {
      pcs->missing_wires = cell->inputs().size();
      processed_cells[cell.get()] = std::move(pcs);
    }
  }

  // Set all inputs as "active".
  for (const rtl::AbstractNetRef<EvalT> ref : module->inputs()) {
    active_wires.push_back(ref);
  }

  for (const auto& input : inputs) {
    rtl::AbstractNetRef<EvalT> wire = input.first;
    for (const auto cell : wire->connected_cells()) {
      CHECK(processed_cells.contains(cell));
      processed_cells[cell]->inputs.insert({wire, std::move(input.second)});
    }
    if constexpr (std::is_convertible<EvalT, int>()) {
      XLS_VLOG(2) << "Input : " << input.first->name() << " : "
                  << static_cast<int>(input.second);
    }
  }

  // Process all active wires : see if this wire satisfies all of a cell's
  // inputs. If so, interpret the cell, and place its outputs on the active wire
  // list.  When running multi-threaded, active_wires may be empty, but as long
  // as num_pending_outputs > 0, it means we expect to get that many
  // additional wires activated.  Thus we exit the loop only when active_wires
  // is empty and there are no pending outputs.
  size_t num_pending_outputs = 0;
  while (!active_wires.empty() || num_pending_outputs > 0) {
    // Drain the output queue as much as we can until we get some active wires,
    // so that we can proceed.
    while (active_wires.empty()) {
      output_queue_guard_.LockWhen(
          absl::Condition(queue_has_data, &output_queue_));
      while (!output_queue_.empty()) {
        QueueEntry entry = std::move(output_queue_.front());
        output_queue_.pop();
        output_queue_guard_.Unlock();
        num_pending_outputs--;
        UpdateProcessedState(processed_cells, active_wires, outputs, module,
                             dump_cell_set, entry.cell, entry.wires);
        output_queue_guard_.Lock();
      }
      output_queue_guard_.Unlock();
    }

    rtl::AbstractNetRef<EvalT> wire = active_wires.front();
    active_wires.pop_front();
    XLS_VLOG(2) << "Processing wire: " << wire->name();

    for (const auto cell : wire->connected_input_cells()) {
      auto processed_cell_state = processed_cells[cell].get();
      CHECK_GT(processed_cell_state->missing_wires, 0);
      processed_cell_state->missing_wires--;
      if (processed_cell_state->missing_wires == 0) {
        // TODO: Only fall back to this thread if no available threads and next
        // cell is a module.  Right now we fall back to the main thread only
        // when the workers are all busy, but that may be suboptimal if
        // InterpretCell() is slow.  Ideally, we want to use this thread to
        // process InterpretCell only if the cell is itself a Module, there is
        // only one worker thread left, and all other threads are processing
        // modules as well.
        if (!threads_.empty() && num_available_threads_.load() > 0) {
          input_queue_guard_.Lock();
          QueueEntry entry;
          entry.cell = cell;
          entry.wires = std::move(processed_cell_state->inputs);
          input_queue_.push(std::move(entry));
          input_queue_cond_.Signal();
          num_available_threads_--;
          num_pending_outputs++;
          input_queue_guard_.Unlock();
          XLS_VLOG(2) << "Dispatched cell: " << cell->name();
        } else {
          XLS_VLOG(2) << "Processing locally cell: " << cell->name();
          XLS_ASSIGN_OR_RETURN(
              auto results, InterpretCell(cell, processed_cell_state->inputs));
          UpdateProcessedState(processed_cells, active_wires, outputs, module,
                               dump_cell_set, cell, results);
        }
      }
    }
  }

  // Soundness check that we've processed all cells (i.e., that there aren't
  // unsatisfiable cells).
  for (const auto& cell : module->cells()) {
    if (processed_cells[cell.get()]->missing_wires > 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Netlist contains unconnected subgraphs and cannot be translated. "
          "Example: cell %s",
          cell->name()));
    }
  }

  // Handle assigns.
  const auto& assigns = module->assigns();
  for (const rtl::AbstractNetRef<EvalT> output : module->outputs()) {
    if (!outputs.contains(output)) {
      // If the check below fails, it means the output wire is undefined;
      // there's no value assignment for it.  This is allowed, but when
      // interpreting netlists it indicates a bug in the verilog more often than
      // not. We err on the side of caution.
      XLS_RET_CHECK(assigns.contains(output));
      rtl::AbstractNetRef<EvalT> net_value = output;
      // Follow the chain of assignments until we get to a 0, 1, or an input
      // wire.
      while (assigns.contains(net_value)) {
        net_value = assigns.at(net_value);
      }
      if (net_value == module->zero()) {
        outputs.insert({output, zero_});
      } else if (net_value == module->one()) {
        outputs.insert({output, one_});
      } else {
        XLS_RET_CHECK(inputs.contains(net_value));
        outputs.insert({output, inputs.at(net_value)});
      }
    }
  }

  return outputs;
}

template <typename EvalT>
absl::StatusOr<AbstractNetRef2Value<EvalT>>
AbstractInterpreter<EvalT>::InterpretCell(
    const rtl::AbstractCell<EvalT>* cell,
    const AbstractNetRef2Value<EvalT>& inputs) {
  const AbstractCellLibraryEntry<EvalT>* entry = cell->cell_library_entry();
  std::optional<const rtl::AbstractModule<EvalT>*> opt_module =
      netlist_->MaybeGetModule(entry->name());

  AbstractNetRef2Value<EvalT> results;

  if (opt_module.has_value()) {
    // If this "cell" is actually a module defined in the netlist,
    // then recursively evaluate it.
    AbstractNetRef2Value<EvalT> module_inputs;
    // who's input/output name - needs to be internal
    // need to map cell inputs to module inputs?
    auto module = opt_module.value();
    const std::vector<rtl::AbstractNetRef<EvalT>>& module_input_refs =
        module->inputs();
    const absl::Span<const std::string> module_input_names =
        module->AsCellLibraryEntry()->input_names();

    for (const auto& input : cell->inputs()) {
      // We need to match the inputs - from the AbstractNetRefs in this module
      // to the AbstractNetRefs in the child module. In AbstractModule, the
      // order of inputs (as AbstractNetRefs) is the same as the input names in
      // its AbstractCellLibraryEntry. That means, for each input (in this
      // module):
      //  - Find the child module input pin/AbstractNetRef with the same name.
      //  - Assign the corresponding child module input AbstractNetRef to have
      //  the value
      //    of the wire in this module.
      // If ever an input isn't found, that's bad. Abort.
      bool input_found = false;
      for (int i = 0; i < module_input_names.size(); i++) {
        if (module_input_names[i] == input.name) {
          module_inputs.emplace(module_input_refs[i],
                                std::move(inputs.at(input.netref)));
          input_found = true;
          break;
        }
      }

      XLS_RET_CHECK(input_found) << absl::StrFormat(
          "Could not find input pin \"%s\" in module \"%s\", referenced in "
          "cell \"%s\"!",
          input.name, module->name(), cell->name());
    }

    XLS_ASSIGN_OR_RETURN(AbstractNetRef2Value<EvalT> child_outputs,
                         InterpretModule(module, module_inputs));
    // We need to do the same here - map the AbstractNetRefs in the module's
    // output to the AbstractNetRefs in this module, using pin names as the
    // matching keys.
    for (const auto& child_output : child_outputs) {
      bool output_found = false;
      for (const auto& cell_output : cell->outputs()) {
        if (child_output.first->name() == cell_output.name) {
          results.insert({cell_output.netref, child_output.second});
          output_found = true;
          break;
        }
      }
      XLS_RET_CHECK(output_found);
      XLS_RET_CHECK(output_found) << absl::StrFormat(
          "Could not find cell output pin \"%s\" in cell \"%s\", referenced in "
          "child module \"%s\"!",
          child_output.first->name(), cell->name(), module->name());
    }

    return results;
  }

  const auto& pins = entry->output_pin_to_function();
  for (int i = 0; i < cell->outputs().size(); i++) {
    if (cell->outputs()[i].eval != nullptr) {
      // The order of values in cell->inputs() is the same as the order of
      // inputs in the cell declaration.  Extract the values from that list and
      // supply them to the eval function.
      std::vector<EvalT> args;
      for (const auto& input : cell->inputs()) {
        args.push_back(inputs.at(input.netref));
      }
      XLS_ASSIGN_OR_RETURN(EvalT value, cell->outputs()[i].eval(args));
      results.insert({cell->outputs()[i].netref, value});
    } else {
      XLS_ASSIGN_OR_RETURN(
          function::Ast ast,
          function::Parser::ParseFunction(pins.at(cell->outputs()[i].name)));
      XLS_ASSIGN_OR_RETURN(EvalT value, InterpretFunction(*cell, ast, inputs));
      results.insert({cell->outputs()[i].netref, value});
    }
  }

  return results;
}

template <typename EvalT>
absl::Status AbstractInterpreter<EvalT>::ThreadBody() {
  while (true) {
    input_queue_guard_.Lock();
    num_available_threads_++;
    while (input_queue_.empty() && threads_should_exit_ == false) {
      input_queue_cond_.Wait(&input_queue_guard_);
    }
    if (threads_should_exit_) {
      input_queue_guard_.Unlock();
      break;
    }
    QueueEntry entry = input_queue_.front();
    input_queue_.pop();
    input_queue_guard_.Unlock();

    XLS_ASSIGN_OR_RETURN(auto results,
                         InterpretCell(entry.cell, std::move(entry.wires)));

    entry.wires = std::move(results);
    output_queue_guard_.Lock();
    output_queue_.push(entry);
    output_queue_guard_.Unlock();
  }

  return absl::OkStatus();
}

template <typename EvalT>
absl::StatusOr<EvalT> AbstractInterpreter<EvalT>::InterpretFunction(
    const rtl::AbstractCell<EvalT>& cell, const function::Ast& ast,
    const AbstractNetRef2Value<EvalT>& inputs) {
  switch (ast.kind()) {
    case function::Ast::Kind::kAnd: {
      XLS_ASSIGN_OR_RETURN(EvalT lhs,
                           InterpretFunction(cell, ast.children()[0], inputs));
      XLS_ASSIGN_OR_RETURN(EvalT rhs,
                           InterpretFunction(cell, ast.children()[1], inputs));
      return lhs & rhs;
    }
    case function::Ast::Kind::kIdentifier: {
      rtl::AbstractNetRef<EvalT> ref = nullptr;
      for (const auto& input : cell.inputs()) {
        if (input.name == ast.name()) {
          ref = input.netref;
        }
      }

      if (ref == nullptr) {
        for (const auto& internal : cell.internal_pins()) {
          if (internal.name == ast.name()) {
            return InterpretStateTable(cell, internal.name, inputs);
          }
        }
      }

      if (ref == nullptr) {
        return absl::NotFoundError(
            absl::StrFormat("Identifier \"%s\" not found in cell %s's inputs "
                            "or internal signals.",
                            ast.name(), cell.name()));
      }

      return inputs.at(ref);
    }
    case function::Ast::Kind::kLiteralOne:
      return one_;
    case function::Ast::Kind::kLiteralZero:
      return zero_;
    case function::Ast::Kind::kNot: {
      XLS_ASSIGN_OR_RETURN(EvalT value,
                           InterpretFunction(cell, ast.children()[0], inputs));
      return !value;
    }
    case function::Ast::Kind::kOr: {
      XLS_ASSIGN_OR_RETURN(EvalT lhs,
                           InterpretFunction(cell, ast.children()[0], inputs));
      XLS_ASSIGN_OR_RETURN(EvalT rhs,
                           InterpretFunction(cell, ast.children()[1], inputs));
      return lhs | rhs;
    }
    case function::Ast::Kind::kXor: {
      XLS_ASSIGN_OR_RETURN(EvalT lhs,
                           InterpretFunction(cell, ast.children()[0], inputs));
      XLS_ASSIGN_OR_RETURN(EvalT rhs,
                           InterpretFunction(cell, ast.children()[1], inputs));
      return lhs ^ rhs;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unknown AST element type: ", ast.kind()));
  }
}

template <typename EvalT>
absl::StatusOr<EvalT> AbstractInterpreter<EvalT>::InterpretStateTable(
    const rtl::AbstractCell<EvalT>& cell, const std::string& pin_name,
    const AbstractNetRef2Value<EvalT>& inputs) {
  XLS_RET_CHECK(cell.cell_library_entry()->state_table());
  const AbstractStateTable<EvalT>& state_table =
      cell.cell_library_entry()->state_table().value();

  typename AbstractStateTable<EvalT>::InputStimulus stimulus;
  for (const auto& input : cell.inputs()) {
    stimulus.emplace(input.name, std::move(inputs.at(input.netref)));
  }

  for (const auto& pin : cell.internal_pins()) {
    if (pin.name == pin_name) {
      return state_table.GetSignalValue(stimulus, pin.name);
    }
  }

  return absl::NotFoundError(
      absl::StrFormat("Signal %s not found in state table!", pin_name));
}

}  // namespace netlist
}  // namespace xls

#endif  // XLS_NETLIST_INTERPRETER_H_
