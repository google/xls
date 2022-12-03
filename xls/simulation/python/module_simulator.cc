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

#include "xls/simulation/module_simulator.h"

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "pybind11/pybind11.h"
#include "pybind11_abseil/absl_casters.h"
#include "pybind11_abseil/statusor_caster.h"
#include "xls/codegen/vast.h"
#include "xls/common/status/import_status_module.h"
#include "xls/ir/value.h"
#include "xls/simulation/verilog_simulators.h"

namespace py = pybind11;

namespace xls {
namespace verilog {
namespace {

// Wrapper class around ModuleSimulator which uses the default Verilog
// simulator.
class ModuleSimulatorWrapper : public ModuleSimulator {
 public:
  ModuleSimulatorWrapper(const ModuleSignature& signature,
                         std::string_view verilog_text, bool is_system_verilog)
      : ModuleSimulator(
            signature, verilog_text,
            is_system_verilog ? FileType::kSystemVerilog : FileType::kVerilog,
            &GetDefaultVerilogSimulator()) {}
};

}  // namespace

PYBIND11_MODULE(module_simulator, m) {
  ImportStatusModule();

  py::module::import("xls.codegen.python.module_signature");
  py::module::import("xls.ir.python.value");

  absl::StatusOr<Value> (ModuleSimulator::*run_kwargs)(
      const absl::flat_hash_map<std::string, Value>&) const =
      &ModuleSimulator::RunFunction;
  absl::StatusOr<Value> (ModuleSimulator::*run_function)(
      absl::Span<const Value> inputs) const = &ModuleSimulator::RunFunction;

  py::class_<ModuleSimulatorWrapper>(m, "ModuleSimulator")
      .def(py::init<const ModuleSignature&, std::string_view, bool>(),
           py::arg("signature"), py::arg("verilog_text"),
           py::arg("is_system_verilog"))
      .def("run_kwargs", run_kwargs, py::arg("args"))
      .def("run_function", run_function, py::arg("args"));
}

}  // namespace verilog
}  // namespace xls
