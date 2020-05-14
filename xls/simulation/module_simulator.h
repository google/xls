// Copyright 2020 Google LLC
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

#ifndef THIRD_PARTY_XLS_SIMULATION_MODULE_SIMULATOR_H_
#define THIRD_PARTY_XLS_SIMULATION_MODULE_SIMULATOR_H_

#include "absl/container/flat_hash_map.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/vast.h"
#include "xls/common/status/statusor.h"
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
                  absl::string_view verilog_text,
                  const VerilogSimulator* simulator)
      : signature_(signature),
        verilog_text_(verilog_text),
        simulator_(simulator) {}

  // Simulates the module with the given inputs as Bits types. Returns a
  // map containing the outputs by port name.
  xabsl::StatusOr<BitsMap> Run(const BitsMap& inputs) const;

  // As above but expects that there is a single data output of the module
  // otherwise an error is returned. Returns the single output value.
  xabsl::StatusOr<Bits> RunAndReturnSingleOutput(const BitsMap& inputs) const;

  // Runs the given batch of argument values through the module with a single
  // invocation of the Verilog simulator. Generally, this is much faster than
  // running via separate calls to Run.
  xabsl::StatusOr<std::vector<BitsMap>> RunBatched(
      absl::Span<const BitsMap> inputs) const;

  // Overloads which accept Values rather than Bits.
  xabsl::StatusOr<Value> Run(
      const absl::flat_hash_map<std::string, Value>& inputs) const;
  xabsl::StatusOr<std::vector<Value>> RunBatched(
      absl::Span<const absl::flat_hash_map<std::string, Value>> inputs) const;

  // Overload which accepts arguments as a Span.
  xabsl::StatusOr<Value> Run(absl::Span<const Value> inputs) const;

 private:
  ModuleSignature signature_;
  std::string verilog_text_;
  const VerilogSimulator* simulator_;
};

}  // namespace verilog
}  // namespace xls

#endif  // THIRD_PARTY_XLS_SIMULATION_MODULE_SIMULATOR_H_
