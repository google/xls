// Copyright 2023 The XLS Authors
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

#include "xls/solvers/z3_op_translator.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/macros.h"
#include "absl/log/check.h"
#include "external/z3/src/api/z3.h"  // IWYU pragma: keep
#include "external/z3/src/api/z3_api.h"

namespace xls::solvers::z3 {

Z3_ast Z3OpTranslator::OrBool(Z3_ast lhs, Z3_ast rhs) {
  Z3_ast args[] = {lhs, rhs};
  return Z3_mk_or(z3_ctx_, /*num_args=*/ABSL_ARRAYSIZE(args), args);
}

Z3_ast Z3OpTranslator::AndBool(Z3_ast lhs, Z3_ast rhs) {
  Z3_ast args[] = {lhs, rhs};
  return Z3_mk_and(z3_ctx_, /*num_args=*/ABSL_ARRAYSIZE(args), args);
}

std::vector<Z3_ast> Z3OpTranslator::ExplodeBits(Z3_ast arg) {
  std::vector<Z3_ast> bits;
  int64_t bit_count = GetBvBitCount(arg);
  bits.reserve(bit_count);
  for (int64_t i = 0; i < bit_count; ++i) {
    bits.push_back(Extract(arg, i));
  }
  return bits;
}

Z3_ast Z3OpTranslator::SignExt(Z3_ast arg, int64_t new_bit_count) {
  int64_t input_bit_count = GetBvBitCount(arg);
  CHECK_GE(new_bit_count, input_bit_count);
  CHECK_GE(input_bit_count, 0);
  CHECK_GE(new_bit_count, 0);
  return Z3_mk_sign_ext(
      z3_ctx_, static_cast<unsigned int>(new_bit_count - input_bit_count), arg);
}

Z3_ast Z3OpTranslator::Extract(Z3_ast arg, int64_t bitno) {
  unsigned int unsigned_bitno = static_cast<unsigned int>(bitno);
  CHECK_EQ(unsigned_bitno, bitno);
  return Z3_mk_extract(z3_ctx_, unsigned_bitno, unsigned_bitno, arg);
}

Z3_ast Z3OpTranslator::Fill(bool value, int64_t bit_count) {
  unsigned int ubit_count = static_cast<unsigned int>(bit_count);
  CHECK_EQ(bit_count, ubit_count);
  std::unique_ptr<bool[]> bits(new bool[ubit_count]);
  for (int64_t i = 0; i < bit_count; ++i) {
    bits[i] = value;
  }
  return Z3_mk_bv_numeral(z3_ctx_, ubit_count, &bits[0]);
}

Z3_ast Z3OpTranslator::MakeBvParam(int64_t bit_count, std::string_view name) {
  unsigned int ubit_count = static_cast<unsigned int>(bit_count);
  Z3_sort type = Z3_mk_bv_sort(z3_ctx_, ubit_count);
  return Z3_mk_const(
      z3_ctx_, Z3_mk_string_symbol(z3_ctx_, std::string(name).c_str()), type);
}

}  // namespace xls::solvers::z3
