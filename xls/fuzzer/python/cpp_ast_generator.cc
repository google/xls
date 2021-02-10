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

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor_pybind_caster.h"
#include "xls/fuzzer/ast_generator.h"

namespace py = pybind11;

namespace xls::dslx {

// Holds our RNG state and is wrapped so our Python code can use the same
// underlying engine for its sequences.
struct RngState {
  std::mt19937 rng;
};

PYBIND11_MODULE(cpp_ast_generator, m) {
  py::class_<RngState>(m, "RngState")
      .def(py::init([](int64 seed) { return RngState{std::mt19937(seed)}; }))
      .def("random",
           [](RngState& self) -> double {
             std::uniform_real_distribution<double> d(0.0, 1.0);
             return d(self.rng);
           })
      .def("randrange",
           [](RngState& self, int64 limit) -> int64 {
             std::uniform_int_distribution<int64> d(0, limit - 1);
             return d(self.rng);
           })
      .def("randrange_biased_towards_zero",
           [](RngState& self, int64 limit) -> absl::StatusOr<int64> {
             XLS_RET_CHECK_GT(limit, 0);
             if (limit == 1) {  // Only one possible value.
               return 0;
             }
             std::array<double, 3> i = {{0, 0, limit - 1.0}};
             std::array<double, 3> w = {{0, 1, 0}};
             std::piecewise_linear_distribution<double> d(i.begin(), i.end(),
                                                          w.begin());
             double triangular = d(self.rng);
             int64 result = static_cast<int64>(std::ceil(triangular)) - 1;
             XLS_CHECK_GE(result, 0);
             XLS_CHECK_LT(result, limit);
             return result;
           });

  py::class_<AstGeneratorOptions>(m, "AstGeneratorOptions")
      .def(py::init([](absl::optional<bool> disallow_divide,
                       absl::optional<bool> emit_loops,
                       absl::optional<bool> short_samples,
                       absl::optional<int64> max_width_bits_types,
                       absl::optional<int64> max_width_aggregate_types,
                       absl::optional<std::vector<BinopKind>> binop_allowlist) {
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
             return options;
           }),
           py::arg("disallow_divide") = absl::nullopt,
           py::arg("emit_loops") = absl::nullopt,
           py::arg("short_samples") = absl::nullopt,
           py::arg("max_width_bits_types") = absl::nullopt,
           py::arg("max_width_aggregate_types") = absl::nullopt,
           py::arg("binop_allowlist") = absl::nullopt);

  m.def("generate",
        [](const AstGeneratorOptions& options,
           RngState& state) -> absl::StatusOr<std::string> {
          AstGenerator g(options, &state.rng);
          XLS_ASSIGN_OR_RETURN(auto pair,
                               g.GenerateFunctionInModule("main", "test"));
          return pair.second->ToString();
        });

  m.def("choose_bit_pattern", [](int64 bit_count, RngState& state) -> Bits {
    AstGenerator g(AstGeneratorOptions(), &state.rng);
    return g.ChooseBitPattern(bit_count);
  });
}

}  // namespace xls::dslx
