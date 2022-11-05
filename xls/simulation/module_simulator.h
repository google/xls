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

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/vast.h"
#include "xls/ir/value.h"
#include "xls/simulation/verilog_simulator.h"

namespace xls {
namespace verilog {

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
      const absl::flat_hash_map<std::string, int64_t>& output_channel_counts)
      const;
  // Overload of the above function that accepts Values rather than Bits.
  absl::StatusOr<absl::flat_hash_map<std::string, std::vector<Value>>>
  RunInputSeriesProc(const absl::flat_hash_map<std::string, std::vector<Value>>&
                         channel_inputs,
                     const absl::flat_hash_map<std::string, int64_t>&
                         output_channel_counts) const;

  // Runs a function with arguments as a Span.
  absl::StatusOr<Value> RunFunction(absl::Span<const Value> inputs) const;

 private:
  // Deassert all control inputs on the module. Returns a map of the signal name
  // to its deasserted value.
  absl::flat_hash_map<std::string, Bits> DeassertControlSignals() const;

  ModuleSignature signature_;
  std::string verilog_text_;
  FileType file_type_;
  const VerilogSimulator* simulator_;
  absl::Span<const VerilogInclude> includes_;
};

}  // namespace verilog
}  // namespace xls

#endif  // XLS_SIMULATION_MODULE_SIMULATOR_H_
