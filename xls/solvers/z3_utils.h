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

// Helper routines for commonly-encountered patterns when working with Z3.
#ifndef XLS_SOLVERS_Z3_UTILS_H_
#define XLS_SOLVERS_Z3_UTILS_H_

#include <string>

#include "absl/strings/string_view.h"
#include "xls/ir/type.h"
#include "../z3/src/api/z3.h"

namespace xls {
namespace solvers {
namespace z3 {

// Helper class for establishing an error callback / turning it into a status
// via RAII.
class ScopedErrorHandler {
 public:
  explicit ScopedErrorHandler(Z3_context ctx);

  ~ScopedErrorHandler();

  absl::Status status() const { return status_; }

 private:
  static void Handler(Z3_context c, Z3_error_code e);
  Z3_context ctx_;
  ScopedErrorHandler* prev_handler_;
  absl::Status status_;
};

// Creates a Z3 solver that will use the specified number of threads.
// This is a refcounted object and will need to be unref'ed once no longer
// needed.
Z3_solver CreateSolver(Z3_context ctx, int num_threads);

// Printing / output functions ------------------------------------------------
// Prints the solver's result, and, if satisfiable, prints a model demonstrating
// such a case.
// If "hexify" is true, then all output values will be converted from boolean or
// decimal to hex.
std::string SolverResultToString(Z3_context ctx, Z3_solver solver,
                                 Z3_lbool satisfiable, bool hexify = true);

// Returns a string representation of the given node interpreted under the given
// model.
// If "hexify" is true, then all output values will be converted from boolean or
// decimal to hex.
std::string QueryNode(Z3_context ctx, Z3_model model, Z3_ast node,
                      bool hexify = true);

// Converts Z3 boolean (#b[01]+) output values to hex. Any range matching the
// pattern "#b[01]+" will be converted to "#x[0-9a-f]+", where the values are
// converted accordingly.
std::string HexifyOutput(const std::string& input);

// Returns a string containing a binary-formatted version of the given bits
// interpreted under "model".
std::string BitVectorToString(Z3_context ctx,
                              const std::vector<Z3_ast>& z3_bits,
                              Z3_model model);

// Converts a XLS IR Type to the corresponding Z3 sort.
Z3_sort TypeToSort(Z3_context ctx, const Type& type);

// Common Z3 unsigned multiplication (between DSLX and IR level).
Z3_ast DoUnsignedMul(Z3_context ctx, Z3_ast lhs, Z3_ast rhs, int result_size);

Z3_ast BitVectorToBoolean(Z3_context c, Z3_ast bit_vector);

}  // namespace z3
}  // namespace solvers
}  // namespace xls

#endif  // XLS_SOLVERS_Z3_UTILS_H_
