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

// Exposes AST generation capability (as is needed for fuzzing) to Python code
// (which currently drives the sampling / running process).

#include <memory>

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor_pybind_caster.h"
#include "xls/dslx/ast.h"
#include "xls/fuzzer/ast_generator.h"
#include "xls/fuzzer/sample_generator.h"

namespace py = pybind11;

namespace xls::dslx {

PYBIND11_MODULE(cpp_ast_generator, m) {
  ImportStatusModule();
  py::module::import("xls.dslx.python.interp_value");

  py::class_<RngState>(m, "RngState")
      .def(py::init([](int64_t seed) { return RngState{std::mt19937(seed)}; }))
      .def("random", &RngState::RandomDouble)
      .def("randrange", &RngState::RandRange)
      .def("randrange_biased_towards_zero",
           &RngState::RandRangeBiasedTowardsZero);

  py::class_<AstGeneratorOptions>(m, "AstGeneratorOptions")
      .def(py::init([](std::optional<bool> disallow_divide,
                       std::optional<bool> emit_loops,
                       std::optional<bool> short_samples,
                       std::optional<int64_t> max_width_bits_types,
                       std::optional<int64_t> max_width_aggregate_types,
                       std::optional<std::vector<BinopKind>> binop_allowlist,
                       std::optional<bool> generate_empty_tuples,
                       std::optional<bool> emit_gate,
                       std::optional<bool> generate_proc) {
             AstGeneratorOptions options;
             if (disallow_divide.has_value()) {
               options.disallow_divide = disallow_divide.value();
             }
             if (short_samples.has_value()) {
               options.short_samples = short_samples.value();
             }
             if (max_width_bits_types.has_value()) {
               options.max_width_bits_types = max_width_bits_types.value();
             }
             if (max_width_aggregate_types.has_value()) {
               options.max_width_aggregate_types =
                   max_width_aggregate_types.value();
             }
             if (binop_allowlist.has_value()) {
               options.binop_allowlist = absl::btree_set<BinopKind>(
                   binop_allowlist->begin(), binop_allowlist->end());
             }
             if (generate_empty_tuples.has_value()) {
               options.generate_empty_tuples = generate_empty_tuples.value();
             }
             if (emit_gate.has_value()) {
               options.emit_gate = emit_gate.value();
             }
             if (generate_proc.has_value()) {
               options.generate_proc = generate_proc.value();
             }
             return options;
           }),
           py::arg("disallow_divide") = absl::nullopt,
           py::arg("emit_loops") = absl::nullopt,
           py::arg("short_samples") = absl::nullopt,
           py::arg("max_width_bits_types") = absl::nullopt,
           py::arg("max_width_aggregate_types") = absl::nullopt,
           py::arg("binop_allowlist") = absl::nullopt,
           py::arg("generate_empty_tuples") = absl::nullopt,
           py::arg("emit_gate") = absl::nullopt,
           py::arg("generate_proc") = absl::nullopt);

  m.def("generate",
        [](const AstGeneratorOptions& options,
           RngState& state) -> absl::StatusOr<std::string> {
          AstGenerator g(options, &state.rng());
          XLS_ASSIGN_OR_RETURN(std::unique_ptr<Module> module,
                               g.Generate("main", "test"));
          return module->ToString();
        });

  m.def("choose_bit_pattern", [](int64_t bit_count, RngState& state) -> Bits {
    AstGenerator g(AstGeneratorOptions(), &state.rng());
    return g.ChooseBitPattern(bit_count);
  });

  m.def("generate_arguments",
        [](const std::vector<const ConcreteType*>& arg_types,
           RngState* rng) -> absl::StatusOr<py::tuple> {
          XLS_ASSIGN_OR_RETURN(auto arguments,
                               GenerateArguments(arg_types, rng));
          py::tuple result(arguments.size());
          for (int64_t i = 0; i < arguments.size(); ++i) {
            result[i] = arguments[i];
          }
          return result;
        });
  m.def("generate_sample", GenerateSample, py::arg("options"),
        py::arg("calls_per_sample"), py::arg("default_options"),
        py::arg("rng"));
}

}  // namespace xls::dslx
