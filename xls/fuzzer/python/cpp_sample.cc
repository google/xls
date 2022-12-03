// Copyright 2021 The XLS Authors
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

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "libs/json11/json11.hpp"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11_abseil/statusor_caster.h"
#include "xls/common/status/import_status_module.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/interp_value_helpers.h"
#include "xls/fuzzer/sample.h"

namespace py = pybind11;

namespace xls::dslx {

PYBIND11_MODULE(cpp_sample, m) {
  ImportStatusModule();

  py::enum_<TopType>(m, "TopType")
      .value("function", TopType::kFunction)
      .value("proc", TopType::kProc);

  py::class_<SampleOptions>(m, "SampleOptions")
      .def(
          py::init([](std::optional<bool> input_is_dslx,
                      std::optional<std::vector<std::string>> ir_converter_args,
                      std::optional<bool> convert_to_ir,
                      std::optional<bool> optimize_ir,
                      std::optional<bool> use_jit, std::optional<bool> codegen,
                      std::optional<std::vector<std::string>> codegen_args,
                      std::optional<bool> simulate,
                      std::optional<std::string> simulator,
                      std::optional<bool> use_system_verilog,
                      std::optional<int64_t> timeout_seconds,
                      std::optional<int64_t> calls_per_sample,
                      std::optional<int64_t> proc_ticks,
                      std::optional<TopType> top_type) {
            std::map<std::string, json11::Json> json;
            if (input_is_dslx) {
              json["input_is_dslx"] = *input_is_dslx;
            }
            if (ir_converter_args) {
              json["ir_converter_args"] = *ir_converter_args;
            }
            if (convert_to_ir) {
              json["convert_to_ir"] = *convert_to_ir;
            }
            if (optimize_ir) {
              json["optimize_ir"] = *optimize_ir;
            }
            if (use_jit) {
              json["use_jit"] = *use_jit;
            }
            if (codegen) {
              json["codegen"] = *codegen;
            }
            if (codegen_args) {
              json["codegen_args"] = *codegen_args;
            }
            if (simulate) {
              json["simulate"] = *simulate;
            }
            if (simulator) {
              json["simulator"] = *simulator;
            }
            if (use_system_verilog) {
              json["use_system_verilog"] = *use_system_verilog;
            }
            if (timeout_seconds) {
              json["timeout_seconds"] = static_cast<int>(*timeout_seconds);
            }
            if (calls_per_sample) {
              json["calls_per_sample"] = static_cast<int>(*calls_per_sample);
            }
            if (proc_ticks) {
              json["proc_ticks"] = static_cast<int>(*proc_ticks);
            }
            if (top_type) {
              json["top_type"] = static_cast<int>(*top_type);
            }
            return SampleOptions::FromJson(json11::Json(json).dump()).value();
          }),
          py::arg("input_is_dslx") = absl::nullopt,
          py::arg("ir_converter_args") = absl::nullopt,
          py::arg("convert_to_ir") = absl::nullopt,
          py::arg("optimize_ir") = absl::nullopt,
          py::arg("use_jit") = absl::nullopt,
          py::arg("codegen") = absl::nullopt,
          py::arg("codegen_args") = absl::nullopt,
          py::arg("simulate") = absl::nullopt,
          py::arg("simulator") = absl::nullopt,
          py::arg("use_system_verilog") = absl::nullopt,
          py::arg("timeout_seconds") = absl::nullopt,
          py::arg("calls_per_sample") = absl::nullopt,
          py::arg("proc_ticks") = absl::nullopt,
          py::arg("top_type") = absl::nullopt)
      .def("__eq__", &SampleOptions::operator==)
      .def("__ne__", &SampleOptions::operator!=)
      .def_static("from_json", &SampleOptions::FromJson)
      .def("to_json", &SampleOptions::ToJsonText)
      .def_property_readonly("input_is_dslx", &SampleOptions::input_is_dslx)
      .def_property_readonly("ir_converter_args",
                             &SampleOptions::ir_converter_args)
      .def_property_readonly("convert_to_ir", &SampleOptions::convert_to_ir)
      .def_property_readonly("simulate", &SampleOptions::simulate)
      .def_property_readonly("simulator", &SampleOptions::simulator)
      .def_property_readonly("optimize_ir", &SampleOptions::optimize_ir)
      .def_property_readonly("use_jit", &SampleOptions::use_jit)
      .def_property_readonly("codegen", &SampleOptions::codegen)
      .def_property_readonly("codegen_args", &SampleOptions::codegen_args)
      .def_property_readonly("use_system_verilog",
                             &SampleOptions::use_system_verilog)
      .def_property_readonly("timeout_seconds", &SampleOptions::timeout_seconds)
      .def_property_readonly("calls_per_sample",
                             &SampleOptions::calls_per_sample)
      .def_property_readonly("proc_ticks", &SampleOptions::proc_ticks)
      .def_property_readonly("top_type", &SampleOptions::top_type)
      .def(
          "replace",
          [](const SampleOptions& self, std::optional<bool> input_is_dslx,
             std::optional<std::vector<std::string>> codegen_args) {
            SampleOptions updated = self;
            if (input_is_dslx) {
              updated.set_input_is_dslx(*input_is_dslx);
            }
            if (codegen_args) {
              updated.set_codegen_args(*codegen_args);
            }
            return updated;
          },
          py::arg("input_is_dslx") = absl::nullopt,
          py::arg("codegen_args") = absl::nullopt)
      // Pickling is required by the multiprocess fuzzer which pickles options
      // to send to the separate worker process.
      .def(py::pickle(
          [](const SampleOptions& options) {
            return py::make_tuple(options.ToJsonText());
          },
          [](py::tuple t) {
            return SampleOptions::FromJson(t[0].cast<std::string>()).value();
          }));

  py::class_<Sample>(m, "Sample")
      .def(py::init(
               [](std::string input_text, SampleOptions options,
                  std::optional<std::vector<std::vector<dslx::InterpValue>>>
                      args_batch,
                  std::optional<std::vector<std::string>> ir_channel_names) {
                 std::vector<std::vector<dslx::InterpValue>> args_batch_vec;
                 if (args_batch.has_value()) {
                   args_batch_vec = std::move(*args_batch);
                 }
                 return Sample(std::move(input_text), std::move(options),
                               std::move(args_batch_vec),
                               std::move(ir_channel_names));
               }),
           py::arg("input_text"), py::arg("options"),
           py::arg("args_batch") = absl::nullopt,
           py::arg("ir_channel_names") = absl::nullopt)
      .def(py::pickle(
          [](const Sample& self) { return py::make_tuple(self.Serialize()); },
          [](const py::tuple& t) {
            std::string s = t[0].cast<std::string>();
            return Sample::Deserialize(s).value();
          }))
      .def("__eq__", &Sample::operator==)
      .def("__ne__", &Sample::operator!=)
      .def(
          "to_crasher",
          [](const Sample& self, std::string_view error_message) {
            return self.ToCrasher(error_message);
          },
          py::arg("error_message") = absl::nullopt)
      .def("serialize", &Sample::Serialize)
      .def_static("deserialize", &Sample::Deserialize)
      .def_property_readonly("options", &Sample::options)
      .def_property_readonly("input_text", &Sample::input_text)
      .def_property_readonly("args_batch", &Sample::args_batch)
      .def_property_readonly("ir_channel_names", &Sample::ir_channel_names);

  m.def("parse_args",
        [](std::string_view args_text) -> absl::StatusOr<py::tuple> {
          XLS_ASSIGN_OR_RETURN(std::vector<dslx::InterpValue> args,
                               ParseArgs(args_text));
          py::tuple t(args.size());
          for (int64_t i = 0; i < args.size(); ++i) {
            t[i] = args[i];
          }
          return t;
        });
  m.def("parse_args_batch", &ParseArgsBatch);
  m.def("args_batch_to_text", &ArgsBatchToText);
  m.def("parse_ir_channel_names", &ParseIrChannelNames);
  m.def("ir_channel_names_to_text", &IrChannelNamesToText);
}

}  // namespace xls::dslx
