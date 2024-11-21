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

// This program proves or disproves that the XLS 2x32-bit floating-point adder
// produces results less than a given (absolute) error bound when compared to
// a reference (in this case the Z3 floating-point type), using the Z3 SMT
// solver.
//
// With default flags, it proves that results are _exactly_ identical when
// subnormals are flushed to zero.

#include <cstdint>
#include <filesystem>  // NOLINT
#include <iostream>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "absl/base/casts.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/solvers/z3_ir_translator.h"
#include "xls/solvers/z3_utils.h"
#include "z3/src/api/z3_api.h"
#include "z3/src/api/z3_fpa.h"

ABSL_FLAG(absl::Duration, timeout, absl::InfiniteDuration(),
          "How long to wait for the proof to complete.");
ABSL_FLAG(bool, flush_subnormals, true,
          "Flush input and output subnormals to 0. If this flag is false, "
          "the proof (and this test) will fail, as it expects ZERO (0.0f) "
          "error between the calculations.\n"
          "This option exists to demonstrate validity of result.");
ABSL_FLAG(bool, reference_use_opt_ir, true,
          "Whether or not to use optimized IR or not.");
ABSL_FLAG(
    uint32_t, error_bound, 0,
    "The error bound to prove. Z3 will aim to prove that the maximum "
    "error between its FP impl and ours - for all inputs - will be "
    "less than this value. This is an absolute, not relative, value. "
    "This is specified as a uint32_t to enable, e.g., subnormal values to "
    "be specified.");

namespace xls {

constexpr const char kIrPath[] = "xls/dslx/stdlib/float32_add.ir";
constexpr const char kOptIrPath[] = "xls/dslx/stdlib/float32_add.opt.ir";
constexpr const char kFunctionName[] = "__float32__add";

using ::xls::solvers::z3::IrTranslator;

// Adds an error comparison to the translated XLS function. To do so:
//  - We convert the input arguments into Z3 floating-point types.
//  - We flush any subnormals to 0 (as is done in the XLS function).
//  - Perform Z3-internal FP addition.
//  - Again flush subnormals to 0.
//  - Take the absolute value of the difference between the two results.
// "Proper" FP error calculation would take the size of the arguments into
// effect, but it suffices for now to be draconian - we've been allowing 0 or
// <smallest sum of subnormals> + iota error.
//
// Returns "actual" vs. "expected" nodes (via reference) to query (via
// QueryNode) on failure.
static absl::StatusOr<Z3_ast> CreateReferenceComparisonFunction(
    Function* function, IrTranslator* translator, bool flush_subnormals,
    Z3_ast* expected, Z3_ast* actual) {
  // Get the translated XLS function, and create its return value.
  Z3_context ctx = translator->ctx();
  Z3_ast result = translator->GetTranslation(function->return_value());

  // The params to and result from float32:add are FP32s, which are tuples of:
  //  - u1: sign
  //  - u8: exponent
  //  - u23: fractional part
  // Which are trivially converted to Z3 floating-point types.
  XLS_ASSIGN_OR_RETURN(Z3_ast xls_result, translator->ToFloat32(result));
  *actual = xls_result;

  // Create Z3 floating-point elements:
  std::vector<Z3_ast> z3_params;
  z3_params.reserve(function->params().size());
  for (const auto& param : function->params()) {
    Z3_ast translation = translator->GetTranslation(param);
    XLS_ASSIGN_OR_RETURN(Z3_ast fp, translator->ToFloat32(translation));
    if (flush_subnormals) {
      XLS_ASSIGN_OR_RETURN(fp, translator->FloatFlushSubnormal(fp));
    }
    z3_params.push_back(fp);
  }

  // Construct the Z3 FP add:
  Z3_ast rounding_mode = Z3_mk_fpa_round_nearest_ties_to_even(ctx);
  Z3_ast z3_result =
      Z3_mk_fpa_add(ctx, rounding_mode, z3_params[0], z3_params[1]);
  if (flush_subnormals) {
    XLS_ASSIGN_OR_RETURN(z3_result, translator->FloatFlushSubnormal(z3_result));
  }
  *expected = z3_result;

  // Format NaNs like we expect (with 0x400000 in the fraction).
  Z3_ast is_nan = Z3_mk_fpa_is_nan(ctx, z3_result);
  Z3_ast positive_nan = Z3_mk_fpa_numeral_int_uint(ctx, false, 0xFF, 0x400000,
                                                   Z3_mk_fpa_sort_32(ctx));
  Z3_ast negative_nan = Z3_mk_fpa_numeral_int_uint(ctx, true, 0xFF, 0x400000,
                                                   Z3_mk_fpa_sort_32(ctx));
  Z3_ast signed_nan = Z3_mk_ite(ctx, Z3_mk_fpa_is_negative(ctx, z3_result),
                                negative_nan, positive_nan);
  z3_result = Z3_mk_ite(ctx, is_nan, signed_nan, z3_result);

  // Compare the two results.
  Z3_ast error = Z3_mk_fpa_abs(
      ctx, Z3_mk_fpa_sub(ctx, rounding_mode, z3_result, xls_result));

  return error;
}

static absl::StatusOr<std::unique_ptr<Package>> GetIr(bool opt_ir) {
  XLS_ASSIGN_OR_RETURN(std::filesystem::path ir_path,
                       GetXlsRunfilePath(opt_ir ? kOptIrPath : kIrPath));
  XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(ir_path));
  return Parser::ParsePackage(ir_text);
}

static absl::Status CompareToReference(bool use_opt_ir, uint32_t error_bound,
                                       bool flush_subnormals,
                                       absl::Duration timeout) {
  XLS_ASSIGN_OR_RETURN(auto package, GetIr(use_opt_ir));
  XLS_ASSIGN_OR_RETURN(auto function, package->GetFunction(kFunctionName));

  // Translate our IR into a matching Z3 AST.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<IrTranslator> translator,
                       IrTranslator::CreateAndTranslate(function));
  // "Wrap" that computation with another (also in Z3) to independently compute
  // the sum and calculate the difference between the two results (the error).
  Z3_ast expected;
  Z3_ast actual;
  XLS_ASSIGN_OR_RETURN(Z3_ast error, CreateReferenceComparisonFunction(
                                         function, translator.get(),
                                         flush_subnormals, &expected, &actual));

  // Define the maximum allowable error for the proof to succeed.
  Z3_context ctx = translator->ctx();
  Z3_ast bounds = Z3_mk_fpa_numeral_float(
      ctx, absl::bit_cast<float>(error_bound), Z3_mk_fpa_sort_32(ctx));

  // Push all that work into z3, and have the solver do its work.
  translator->SetTimeout(timeout);

  Z3_solver solver =
      solvers::z3::CreateSolver(ctx, std::thread::hardware_concurrency());
  Z3_ast objective = Z3_mk_fpa_gt(ctx, error, bounds);
  Z3_solver_assert(ctx, solver, objective);

  // Finally, print the output to the terminal in gorgeous two-color ASCII.
  Z3_lbool satisfiable = Z3_solver_check(ctx, solver);
  std::cout << solvers::z3::SolverResultToString(ctx, solver, satisfiable)
            << '\n'
            << std::flush;

  Z3_solver_dec_ref(ctx, solver);
  return absl::OkStatus();
}

}  // namespace xls

int main(int argc, char** argv) {
  xls::InitXls(argv[0], argc, argv);

  return xls::ExitStatus(xls::CompareToReference(
      absl::GetFlag(FLAGS_reference_use_opt_ir),
      absl::GetFlag(FLAGS_error_bound), absl::GetFlag(FLAGS_flush_subnormals),
      absl::GetFlag(FLAGS_timeout)));
}
