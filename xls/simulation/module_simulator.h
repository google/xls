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

#ifndef XLS_SIMULATION_MODULE_SIMULATOR_H_
#define XLS_SIMULATION_MODULE_SIMULATOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/vast/vast.h"
#include "xls/ir/bits.h"
#include "xls/ir/value.h"
#include "xls/simulation/module_testbench.h"
#include "xls/simulation/module_testbench_thread.h"
#include "xls/simulation/testbench_signal_capture.h"
#include "xls/simulation/verilog_simulator.h"
#include "xls/tools/verilog_include.h"

namespace xls {
namespace verilog {

// Data structure describing the holdoff behavior of the valid signal for an
// input to a ready/valid channel. This enables deasserting the valid signal for
// a specified number of cycles between inputs.
struct ValidHoldoff {
  // Number of cycles to deassert valid.
  int64_t cycles;
  // Optional values to drive on the data lines when valid is deasserted. If
  // non-empty, the vector must have `cycles` elements. If empty, X is driven
  // on the data lines when valid is deasserted.
  std::vector<BitsOrX> driven_values;
};

// Data structure describing the holdoff behavior of the ready and valid signals
// of all ready/valid channels of a module.
struct ReadyValidHoldoffs {
  // Indexed by input channel name. Each vector element describes the valid
  // holdoff behavior for a single input on the channel.
  absl::flat_hash_map<std::string, std::vector<ValidHoldoff>> valid_holdoffs;

  // Indexed by output channel name. Each element defines the number of cycles
  // that ready is deasserted. Each deassertion is followed by a single-cycle
  // assertion of ready. After the sequence is exhausted, ready is asserted
  // indefinitely. For example, the sequence {1,2,0,3,0,0,1} results in the
  // following pattern of ready assertions: 010011000111011111111...
  //
  // Transactions may occur at any point when ready is asserted. This does not
  // affect the subsequent pattern of assertion/deassertions.
  //
  // Note: the different encoding of ready and valid holdoffs reflects the
  // different allowed behavior between ready and valid. Once valid is asserted
  // it must remain asserted until the transaction occurs. This means a single
  // holdoff value for each transaction is sufficient to encode any legal valid
  // holdoff behavior. Ready, however, may be deasserted prior to a transaction
  // so the encoding mechanism must be flexible enough to encode an arbitrary
  // wave form.
  absl::flat_hash_map<std::string, std::vector<int64_t>> ready_holdoffs;
};

// Abstraction for simulating a module described by a SignatureProto using a
// testbench run under the Verilog simulator.
class ModuleSimulator {
 public:
  // Type alias for passing named Bits value to and from module simulation.
  using BitsMap = absl::flat_hash_map<std::string, Bits>;

  // Constructor for the simulator. Arguments:
  //  signature: Proto describing the interface of the module.
  //  verilog_text: Verilog text containing the module to test.
  //  simulator: Verilog simulator to use.
  ModuleSimulator(const ModuleSignature& signature,
                  std::string_view verilog_text, FileType file_type,
                  const VerilogSimulator* simulator,
                  absl::Span<const VerilogInclude> includes = {})
      : signature_(signature),
        verilog_text_(verilog_text),
        file_type_(file_type),
        simulator_(simulator),
        includes_(includes) {}

  // Simulates the module with the given inputs as Bits types. Returns a
  // map containing the outputs by port name.
  absl::StatusOr<BitsMap> RunFunction(const BitsMap& inputs) const;

  // As above but expects that there is a single data output of the module
  // otherwise an error is returned. Returns the single output value.
  absl::StatusOr<Bits> RunAndReturnSingleOutput(const BitsMap& inputs) const;

  // Runs the given batch of argument values through the module with a single
  // invocation of the Verilog simulator. Generally, this is much faster than
  // running via separate calls to Run.
  absl::StatusOr<std::vector<BitsMap>> RunBatched(
      absl::Span<const BitsMap> inputs) const;

  // Overloads which accept Values rather than Bits.
  absl::StatusOr<Value> RunFunction(
      const absl::flat_hash_map<std::string, Value>& inputs) const;
  absl::StatusOr<std::vector<Value>> RunBatched(
      absl::Span<const absl::flat_hash_map<std::string, Value>> inputs) const;

  // Runs the given channel inputs and expects a number of values at an output
  // channel on the a design under test (DUT) derived from a proc.
  absl::StatusOr<absl::flat_hash_map<std::string, std::vector<Bits>>>
  RunInputSeriesProc(
      const absl::flat_hash_map<std::string, std::vector<Bits>>& channel_inputs,
      const absl::flat_hash_map<std::string, int64_t>& output_channel_counts,
      std::optional<ReadyValidHoldoffs> holdoffs = std::nullopt) const;
  // Overload of the above function that accepts Values rather than Bits.
  absl::StatusOr<absl::flat_hash_map<std::string, std::vector<Value>>>
  RunInputSeriesProc(
      const absl::flat_hash_map<std::string, std::vector<Value>>&
          channel_inputs,
      const absl::flat_hash_map<std::string, int64_t>& output_channel_counts,
      std::optional<ReadyValidHoldoffs> holdoffs = std::nullopt) const;

  // Runs a function with arguments as a Span.
  absl::StatusOr<Value> RunFunction(absl::Span<const Value> inputs) const;

  // Returns the (System)Verilog testbench for testing the module with the given
  // inputs and expected outputs counts.
  absl::StatusOr<std::string> GenerateProcTestbenchVerilog(
      const absl::flat_hash_map<std::string, std::vector<Bits>>& channel_inputs,
      const absl::flat_hash_map<std::string, int64_t>& output_channel_counts,
      std::optional<ReadyValidHoldoffs> holdoffs = std::nullopt) const;

 private:
  // Returns the control input ports and their deasserted values.
  std::vector<DutInput> DeassertControlSignals() const;

  struct ProcTestbench {
    std::unique_ptr<ModuleTestbench> testbench;

    // Containers of the Bits objects which will be filled in with output
    // channel values when the testbench is run. Indexed by output channel name.
    absl::flat_hash_map<std::string, std::vector<std::unique_ptr<Bits>>>
        outputs;
  };
  absl::StatusOr<ProcTestbench> CreateProcTestbench(
      const absl::flat_hash_map<std::string, std::vector<Bits>>& channel_inputs,
      const absl::flat_hash_map<std::string, int64_t>& output_channel_counts,
      std::optional<ReadyValidHoldoffs> holdoffs) const;

  ModuleSignature signature_;
  std::string verilog_text_;
  FileType file_type_;
  const VerilogSimulator* simulator_;
  absl::Span<const VerilogInclude> includes_;
};

}  // namespace verilog
}  // namespace xls

#endif  // XLS_SIMULATION_MODULE_SIMULATOR_H_
