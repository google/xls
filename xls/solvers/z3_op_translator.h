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

#ifndef XLS_SOLVERS_Z3_OP_TRANSLATOR_H_
#define XLS_SOLVERS_Z3_OP_TRANSLATOR_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/types/span.h"
#include "xls/ir/bits.h"
#include "external/z3/src/api/z3.h"  // IWYU pragma: keep
#include "external/z3/src/api/z3_api.h"

namespace xls::solvers::z3 {

// Helpers for Z3 translation -- these wrap a Z3 context and give slightly
// easier / safer / more effectively enumerated methods for adding operations
// that correspond similarly to XLS IR operations.
//
// See also solvers::z3::IrTranslator which orchestrates translation using this
// helper layer.
class Z3OpTranslator {
 public:
  explicit Z3OpTranslator(Z3_context z3_ctx) : z3_ctx_(z3_ctx) {}

  // Helpers for building bit-vector operations, which are generally what we
  // use.
  Z3_ast Sub(Z3_ast lhs, Z3_ast rhs) { return Z3_mk_bvsub(z3_ctx_, lhs, rhs); }
  Z3_ast And(Z3_ast lhs, Z3_ast rhs) { return Z3_mk_bvand(z3_ctx_, lhs, rhs); }
  Z3_ast Or(Z3_ast lhs, Z3_ast rhs) { return Z3_mk_bvor(z3_ctx_, lhs, rhs); }
  Z3_ast Xor(Z3_ast lhs, Z3_ast rhs) { return Z3_mk_bvxor(z3_ctx_, lhs, rhs); }
  Z3_ast Not(Z3_ast arg) { return Z3_mk_bvnot(z3_ctx_, arg); }
  Z3_ast ReduceOr(Z3_ast arg) { return Z3_mk_bvredor(z3_ctx_, arg); }
  Z3_ast EqZero(Z3_ast arg) { return Not(Z3_mk_bvredor(z3_ctx_, arg)); }
  Z3_ast Eq(Z3_ast lhs, Z3_ast rhs) { return EqZero(Xor(lhs, rhs)); }
  Z3_ast If(Z3_ast cond, Z3_ast consequent, Z3_ast alternate) {
    return Cond(NeZeroBool(cond), consequent, alternate);
  }

  // Returns a boolean-kinded result that says whether lhs == rhs.
  Z3_ast EqBool(Z3_ast lhs, Z3_ast rhs) { return Z3_mk_eq(z3_ctx_, lhs, rhs); }

  // Takes boolean kinds as arguments and produces a boolean kind result.
  Z3_ast OrBool(Z3_ast lhs, Z3_ast rhs);

  // Takes boolean kinds as arguments and produces a boolean kind result.
  Z3_ast AndBool(Z3_ast lhs, Z3_ast rhs);

  Z3_ast True() { return Z3_mk_true(z3_ctx_); }
  Z3_ast False() { return Z3_mk_false(z3_ctx_); }

  Z3_ast ZextBy1b(Z3_ast arg) { return Z3_mk_zero_ext(z3_ctx_, 1, arg); }
  Z3_ast SextBy1b(Z3_ast arg) { return Z3_mk_sign_ext(z3_ctx_, 1, arg); }

  // Extracts bit "bitno" from the given argument, returns a single-bit
  // bitvector result.
  //
  // Note that the most significant bit (as visible via arithmetic operations
  // and similar) is at `bitno = bit_count - 1`.
  Z3_ast Extract(Z3_ast arg, int64_t bitno);

  // Returns the kind of the sort of the AST node, e.g. Z3_BOOL_SORT,
  // Z3_BV_SORT, etc.
  Z3_sort_kind GetSortKind(Z3_ast a) {
    return Z3_get_sort_kind(z3_ctx_, Z3_get_sort(z3_ctx_, a));
  }

  std::string GetSortName(Z3_ast a) {
    return Z3_get_symbol_string(
        z3_ctx_, Z3_get_sort_name(z3_ctx_, Z3_get_sort(z3_ctx_, a)));
  }

  Z3_ast Cond(Z3_ast cond, Z3_ast match, Z3_ast nomatch) {
    return Z3_mk_ite(z3_ctx_, cond, match, nomatch);
  }

  Z3_ast UDiv(Z3_ast lhs, Z3_ast rhs) {
    // Z3's bvudiv matches XLS's udiv semantics on divide by zero. This does not
    // seem to be consistently documented, but
    // https://cs.nyu.edu/pipermail/smt-lib/2017/001206.html (and other places)
    // say that bvudiv with rhs=0 is defined as yielding all-ones.
    return Z3_mk_bvudiv(z3_ctx_, lhs, rhs);
  }

  Z3_ast SDiv(Z3_ast lhs, Z3_ast rhs) {
    // Z3's bvsdiv seems to differ from XLS's sdiv semantics on divide by zero.
    // The Z3 sdiv is undefined for rhs=0; the XLS behavior is
    // (rhs == 0 ? (lhs < 0 ? MIN_INT : MAX_INT) : lhs / rhs
    // Implement that using a conditional.
    const int64_t result_bits = GetBvBitCount(lhs);
    Z3_ast max_signed_int =  // MAX_INT for this bit-width
        (result_bits > 1)
            ? ConcatN({Fill(false, 1), Fill(true, result_bits - 1)})
            : Fill(false, 1);
    Z3_ast min_signed_int =  // MIN_INT for this bit width
        (result_bits > 1)
            ? ConcatN({Fill(true, 1), Fill(false, result_bits - 1)})
            : Fill(true, 1);

    return Cond(EqZeroBool(rhs),
                Cond(NeZeroBool(Msb(lhs)), min_signed_int, max_signed_int),
                Z3_mk_bvsdiv(z3_ctx_, lhs, rhs));
  }

  Z3_ast UMod(Z3_ast lhs, Z3_ast rhs) {
    // XLS behavior for mod-by-zero: (rhs == 0) ? 0 : lhs % rhs
    return Cond(EqZeroBool(rhs), rhs, Z3_mk_bvurem(z3_ctx_, lhs, rhs));
  }

  Z3_ast SMod(Z3_ast lhs, Z3_ast rhs) {
    // XLS behavior for mod-by-zero: (rhs == 0) ? 0 : lhs % rhs
    return Cond(EqZeroBool(rhs), rhs, Z3_mk_bvsrem(z3_ctx_, lhs, rhs));
  }

  int64_t GetBvBitCount(Z3_ast arg) {
    Z3_sort sort = Z3_get_sort(z3_ctx_, arg);
    return Z3_get_bv_sort_size(z3_ctx_, sort);
  }

  // Explodes bits in the bit-vector Z3 value "arg" such that the LSb is in
  // index 0 of the return value.
  std::vector<Z3_ast> ExplodeBits(Z3_ast arg);

  Z3_ast Msb(Z3_ast arg) {
    int64_t bit_count = GetBvBitCount(arg);
    return Extract(arg, bit_count - 1);
  }

  Z3_ast SignExt(Z3_ast arg, int64_t new_bit_count);

  // Concatenates args such that arg[0]'s most significant bit is the most
  // significant bit of the result, and arg[args.size()-1]'s least significant
  // bit is the least significant bit of the result.
  Z3_ast ConcatN(absl::Span<const Z3_ast> args) {
    Z3_ast accum = args[0];
    for (int64_t i = 1; i < args.size(); ++i) {
      accum = Z3_mk_concat(z3_ctx_, accum, args[i]);
    }
    return accum;
  }

  // Returns whether lhs < rhs -- this is determined by zero-extending the
  // values and testing whether lhs - rhs < 0
  Z3_ast ULt(Z3_ast lhs, Z3_ast rhs) {
    return Msb(Sub(ZextBy1b(lhs), ZextBy1b(rhs)));
  }
  Z3_ast ULe(Z3_ast lhs, Z3_ast rhs) { return Or(ULt(lhs, rhs), Eq(lhs, rhs)); }
  Z3_ast UGt(Z3_ast lhs, Z3_ast rhs) { return Not(ULe(lhs, rhs)); }
  Z3_ast UGe(Z3_ast lhs, Z3_ast rhs) { return Not(ULt(lhs, rhs)); }

  Z3_ast ULtBool(Z3_ast lhs, Z3_ast rhs) {
    return Z3_mk_bvult(z3_ctx_, lhs, rhs);
  }
  Z3_ast ULeBool(Z3_ast lhs, Z3_ast rhs) {
    return Z3_mk_bvule(z3_ctx_, lhs, rhs);
  }
  Z3_ast UGtBool(Z3_ast lhs, Z3_ast rhs) {
    return Z3_mk_bvugt(z3_ctx_, lhs, rhs);
  }
  Z3_ast UGeBool(Z3_ast lhs, Z3_ast rhs) {
    return Z3_mk_bvuge(z3_ctx_, lhs, rhs);
  }

  // Returns whether lhs < rhs -- this is determined by sign-extending the
  // values and testing whether lhs - rhs < 0
  Z3_ast SLt(Z3_ast lhs, Z3_ast rhs) {
    return Msb(Sub(SextBy1b(lhs), SextBy1b(rhs)));
  }

  Z3_ast Min(Z3_ast lhs, Z3_ast rhs) {
    Z3_ast lt = Z3_mk_bvult(z3_ctx_, lhs, rhs);
    return Z3_mk_ite(z3_ctx_, lt, lhs, rhs);
  }

  // Returns a bit vector filled with "bit_count" digits of "value".
  Z3_ast Fill(bool value, int64_t bit_count);

  // For use in solver assertions, we have to use the "mk_eq" form that creates
  // a bool (in lieu of a bit vector). We put the "Bool" suffix on these helper
  // routines.
  Z3_ast EqZeroBool(Z3_ast arg) {
    int64_t bits = GetBvBitCount(arg);
    return Z3_mk_eq(z3_ctx_, arg, Fill(false, bits));
  }

  Z3_ast NeZeroBool(Z3_ast arg) { return Z3_mk_not(z3_ctx_, EqZeroBool(arg)); }

  Z3_ast NeBool(Z3_ast lhs, Z3_ast rhs) {
    return Z3_mk_not(z3_ctx_, Z3_mk_eq(z3_ctx_, lhs, rhs));
  }

  // Makes a bit-count-sized bitvector parameter with the given name.
  Z3_ast MakeBvParam(int64_t bit_count, std::string_view name);

  Z3_context z3_ctx_;
};

}  // namespace xls::solvers::z3

#endif  // XLS_SOLVERS_Z3_OP_TRANSLATOR_H_
