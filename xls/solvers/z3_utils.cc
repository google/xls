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
#include "xls/solvers/z3_utils.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/common/logging/logging.h"
#include "xls/common/source_location.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/type.h"
#include "../z3/src/api/z3_api.h"
#include "re2/re2.h"

namespace xls {
namespace solvers {
namespace z3 {

// Since the callback from z3 does not pass a void* as context we rely on RAII
// to establish this thread local value for retrieval from the static error
// handler.
thread_local ScopedErrorHandler* g_handler = nullptr;

ScopedErrorHandler::ScopedErrorHandler(Z3_context ctx,
                                       xabsl::SourceLocation source_location)
    : ctx_(ctx), source_location_(source_location) {
  Z3_set_error_handler(ctx_, Handler);
  prev_handler_ = g_handler;
  g_handler = this;
}

ScopedErrorHandler::~ScopedErrorHandler() {
  Z3_set_error_handler(ctx_, nullptr);
  CHECK_EQ(g_handler, this);
  g_handler = prev_handler_;
}

/* static */ void ScopedErrorHandler::Handler(Z3_context c, Z3_error_code e) {
  g_handler->status_ = absl::InternalError(absl::StrFormat(
      "Z3 error @ %s:%d: %s", g_handler->source_location_.file_name(),
      g_handler->source_location_.line(), Z3_get_error_msg(c, e)));
  XLS_LOG(ERROR) << g_handler->status_;
}

namespace {

// Helper for TypeToSort below.
Z3_sort CreateTupleSort(Z3_context ctx, const Type& type) {
  const TupleType* tuple_type = type.AsTupleOrDie();
  Z3_func_decl mk_tuple_decl;
  std::string tuple_type_str = tuple_type->ToString();
  Z3_symbol tuple_sort_name = Z3_mk_string_symbol(ctx, tuple_type_str.c_str());

  absl::Span<Type* const> element_types = tuple_type->element_types();
  int64_t num_elements = element_types.size();
  std::vector<Z3_symbol> field_names;
  std::vector<Z3_sort> field_sorts;
  field_names.reserve(num_elements);
  field_sorts.reserve(num_elements);

  for (int i = 0; i < num_elements; i++) {
    field_names.push_back(
        Z3_mk_string_symbol(ctx, absl::StrCat(tuple_type_str, "_", i).c_str()));
    field_sorts.push_back(TypeToSort(ctx, *element_types[i]));
  }

  // Populated in Z3_mk_tuple_sort.
  std::vector<Z3_func_decl> proj_decls;
  proj_decls.resize(num_elements);
  return Z3_mk_tuple_sort(ctx, tuple_sort_name, num_elements,
                          field_names.data(), field_sorts.data(),
                          &mk_tuple_decl, proj_decls.data());
}

}  // namespace

Z3_solver CreateSolver(Z3_context ctx, int num_threads) {
  Z3_params params = Z3_mk_params(ctx);
  Z3_params_inc_ref(ctx, params);
  Z3_params_set_uint(ctx, params, Z3_mk_string_symbol(ctx, "sat.threads"),
                     num_threads);
  Z3_params_set_uint(ctx, params, Z3_mk_string_symbol(ctx, "threads"),
                     num_threads);

  Z3_solver solver = Z3_mk_solver(ctx);
  Z3_solver_inc_ref(ctx, solver);
  Z3_solver_set_params(ctx, solver, params);

  Z3_params_dec_ref(ctx, params);
  return solver;
}

std::string SolverResultToString(Z3_context ctx, Z3_solver solver,
                                 Z3_lbool satisfiable, bool hexify) {
  std::string result_str;
  switch (satisfiable) {
    case Z3_L_TRUE:
      result_str = "true";
      break;
    case Z3_L_FALSE:
      result_str = "false";
      break;
    case Z3_L_UNDEF:
      result_str = "undef";
      break;
    default:
      result_str = "invalid";
  }

  std::string output =
      absl::StrFormat("Solver result; satisfiable: %s\n", result_str);
  if (satisfiable == Z3_L_TRUE) {
    Z3_model model = Z3_solver_get_model(ctx, solver);
    absl::StrAppend(&output, "\n  Model:\n```", Z3_model_to_string(ctx, model),
                    "```");
  }

  if (hexify) {
    output = HexifyOutput(output);
  }
  return output;
}

std::string QueryNode(Z3_context ctx, Z3_model model, Z3_ast node,
                      bool hexify) {
  Z3_ast node_eval;
  Z3_model_eval(ctx, model, node, true, &node_eval);
  std::string output = Z3_ast_to_string(ctx, node_eval);
  if (hexify) {
    output = HexifyOutput(output);
  }
  return output;
}

std::string HexifyOutput(const std::string& input) {
  std::string text = input;
  std::string match;
  // If this was ever used to match a wall of text, it'd be faster to chop up
  // the input, rather than searching over the whole string over and over...but
  // that's not the expected use case.
  while (RE2::PartialMatch(text, "(#b[01]+)", &match)) {
    BitsRope rope(match.size() - 2);
    for (int i = match.size() - 1; i >= 2; i--) {
      rope.push_back(match[i] == '1');
    }

    std::string new_text = BitsToString(rope.Build(), FormatPreference::kHex);
    new_text[0] = '#';
    CHECK(RE2::Replace(&text, "#b[01]+", new_text));
  }

  return text;
}

std::string BitVectorToString(Z3_context ctx,
                              const std::vector<Z3_ast>& z3_bits,
                              Z3_model model) {
  constexpr const char kZ3One[] = "#x1";
  BitsRope rope(z3_bits.size());

  for (auto z3_bit : z3_bits) {
    rope.push_back(QueryNode(ctx, model, z3_bit) == kZ3One);
  }
  Bits bits = bits_ops::Reverse(rope.Build());
  return absl::StrCat("0b", BitsToRawDigits(bits, FormatPreference::kBinary,
                                            /*emit_leading_zeros=*/true));
}

Z3_sort TypeToSort(Z3_context ctx, const Type& type) {
  switch (type.kind()) {
    case TypeKind::kBits:
      return Z3_mk_bv_sort(ctx, type.GetFlatBitCount());
    case TypeKind::kTuple:
      return CreateTupleSort(ctx, type);
    case TypeKind::kArray: {
      const ArrayType* array_type = type.AsArrayOrDie();
      Z3_sort element_sort = TypeToSort(ctx, *array_type->element_type());
      Z3_sort index_sort =
          Z3_mk_bv_sort(ctx, Bits::MinBitCountUnsigned(array_type->size()));
      return Z3_mk_array_sort(ctx, index_sort, element_sort);
    }
    case TypeKind::kToken: {
      // Token types don't contain any data. A 0-field tuple is a convenient
      // way to let (most of) the rest of the z3 infrastructure treat
      // token like a normal data-type.
      return CreateTupleSort(ctx, TupleType(/*members=*/{}));
    }
    default:
      XLS_LOG(FATAL) << "Unsupported type kind: "
                     << TypeKindToString(type.kind());
  }
}

Z3_ast DoUnsignedMul(Z3_context ctx, Z3_ast lhs, Z3_ast rhs, int result_size) {
  int lhs_size = Z3_get_bv_sort_size(ctx, Z3_get_sort(ctx, lhs));
  int rhs_size = Z3_get_bv_sort_size(ctx, Z3_get_sort(ctx, rhs));

  int operation_size = std::max(result_size, std::max(lhs_size, rhs_size));
  // Add an extra 0 MSb to make sure Z3 knows that we are doing unsigned
  // multiplication.
  operation_size += 1;
  lhs = Z3_mk_zero_ext(ctx, operation_size - lhs_size, lhs);
  rhs = Z3_mk_zero_ext(ctx, operation_size - rhs_size, rhs);

  Z3_ast result = Z3_mk_bvmul(ctx, lhs, rhs);
  return Z3_mk_extract(ctx, result_size - 1, 0, result);
}

Z3_ast BitVectorToBoolean(Z3_context c, Z3_ast bit_vector) {
  std::unique_ptr<bool[]> bits(new bool[1]);
  bits[0] = true;
  return Z3_mk_eq(c, bit_vector, Z3_mk_bv_numeral(c, 1, &bits[0]));
}

}  // namespace z3
}  // namespace solvers
}  // namespace xls
