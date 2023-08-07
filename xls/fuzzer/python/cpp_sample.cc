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

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11_abseil/statusor_caster.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/import_status_module.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/interp_value_helpers.h"
#include "xls/fuzzer/sample.h"
#include "xls/fuzzer/sample.pb.h"

namespace py = pybind11;

namespace xls::dslx {

PYBIND11_MODULE(cpp_sample, m) {
  ImportStatusModule();

  py::enum_<fuzzer::SampleType>(m, "SampleType")
      .value("function", fuzzer::SAMPLE_TYPE_FUNCTION)
      .value("proc", fuzzer::SAMPLE_TYPE_PROC);

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
                      std::optional<fuzzer::SampleType> sample_type) {
            fuzzer::SampleOptionsProto proto =
                SampleOptions::DefaultOptionsProto();
            if (input_is_dslx) {
              proto.set_input_is_dslx(*input_is_dslx);
            }
            if (ir_converter_args) {
              for (const std::string& arg : ir_converter_args.value()) {
                proto.add_ir_converter_args(arg);
              }
            }
            if (convert_to_ir) {
              proto.set_convert_to_ir(*convert_to_ir);
            }
            if (optimize_ir) {
              proto.set_optimize_ir(*optimize_ir);
            }
            if (use_jit) {
              proto.set_use_jit(*use_jit);
            }
            if (codegen) {
              proto.set_codegen(*codegen);
            }
            if (codegen_args) {
              for (const std::string& arg : codegen_args.value()) {
                proto.add_codegen_args(arg);
              }
            }
            if (simulate) {
              proto.set_simulate(*simulate);
            }
            if (simulator) {
              proto.set_simulator(*simulator);
            }
            if (use_system_verilog) {
              proto.set_use_system_verilog(*use_system_verilog);
            }
            if (timeout_seconds) {
              proto.set_timeout_seconds(*timeout_seconds);
            }
            if (calls_per_sample) {
              proto.set_calls_per_sample(*calls_per_sample);
            }
            if (proc_ticks) {
              proto.set_proc_ticks(*proc_ticks);
            }
            if (sample_type) {
              proto.set_sample_type(*sample_type);
            }
            return SampleOptions::FromProto(proto).value();
          }),
          py::arg("input_is_dslx") = std::nullopt,
          py::arg("ir_converter_args") = std::nullopt,
          py::arg("convert_to_ir") = std::nullopt,
          py::arg("optimize_ir") = std::nullopt,
          py::arg("use_jit") = std::nullopt, py::arg("codegen") = std::nullopt,
          py::arg("codegen_args") = std::nullopt,
          py::arg("simulate") = std::nullopt,
          py::arg("simulator") = std::nullopt,
          py::arg("use_system_verilog") = std::nullopt,
          py::arg("timeout_seconds") = std::nullopt,
          py::arg("calls_per_sample") = std::nullopt,
          py::arg("proc_ticks") = std::nullopt,
          py::arg("top_type") = std::nullopt)
      .def("__eq__", &SampleOptions::operator==)
      .def("__ne__", &SampleOptions::operator!=)
      .def_static("from_pbtxt", &SampleOptions::FromPbtxt)
      .def("to_pbtxt", &SampleOptions::ToPbtxt)
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
      .def_property_readonly("top_type", &SampleOptions::sample_type)
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
          py::arg("input_is_dslx") = std::nullopt,
          py::arg("codegen_args") = std::nullopt)
      // Pickling is required by the multiprocess fuzzer which pickles options
      // to send to the separate worker process.
      .def(py::pickle(
          [](const SampleOptions& options) {
            return py::make_tuple(options.proto().DebugString());
          },
          [](py::tuple t) {
            fuzzer::SampleOptionsProto proto;
            XLS_CHECK_OK(ParseTextProto(t[0].cast<std::string>(),
                                        /*file_name=*/"", &proto));
            return SampleOptions::FromProto(proto).value();
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
                 std::vector<std::string> ir_channel_names_vec;
                 if (ir_channel_names.has_value()) {
                   ir_channel_names_vec = ir_channel_names.value();
                 }
                 return Sample(std::move(input_text), std::move(options),
                               std::move(args_batch_vec),
                               std::move(ir_channel_names_vec));
               }),
           py::arg("input_text"), py::arg("options"),
           py::arg("args_batch") = std::nullopt,
           py::arg("ir_channel_names") = std::nullopt)
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
          py::arg("error_message") = std::nullopt)
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
