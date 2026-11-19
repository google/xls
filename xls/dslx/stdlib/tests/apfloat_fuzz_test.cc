// Copyright 2026 The XLS Authors
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

// Differential FuzzTest and exhaustive oracle harness for DSLX apfloat
// operations. Compares AOT-compiled (coverage-instrumented) DSLX wrappers
// against llvm::APFloat (IEEE-754 with each operation's documented DAZ/FTZ and
// canonical +qNaN contract) and Eigen::bfloat16.

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/base/casts.h"
#include "absl/base/no_destructor.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "Eigen/Core"
#include "llvm/include/llvm/ADT/APFloat.h"
#include "llvm/include/llvm/ADT/APInt.h"
#include "llvm/include/llvm/ADT/APSInt.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_bf16_add_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_bf16_add_lza_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_bf16_ceil_daz_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_bf16_ceil_with_denorms_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_bf16_cmp_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_bf16_floor_daz_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_bf16_floor_with_denorms_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_bf16_fma_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_bf16_from_s32_rne_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_bf16_from_s32_rz_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_bf16_ldexp_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_bf16_mul_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_bf16_round_rna_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_bf16_round_rne_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_bf16_sub_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_bf16_to_f32_upcast_daz_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_bf16_to_f32_upcast_denorms_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_bf16_to_s32_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_bf16_to_u32_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_bf16_trunc_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_e4m3_add_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_e4m3_fma_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_e4m3_from_s64_rne_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_e4m3_from_s64_rz_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_e4m3_mul_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_f32_add_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_f32_add_lza_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_f32_cast_to_fixed_s32_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_f32_fast_rsqrt_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_f32_fma_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_f32_from_int32_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_f32_from_s64_rne_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_f32_from_s64_rz_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_f32_from_uint32_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_f32_ldexp_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_f32_mul_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_f32_to_bf16_downcast_rna_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_f32_to_bf16_downcast_rne_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_f32_to_hf16_downcast_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_f32_to_hf16_downcast_rne_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_f32_to_s32_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_f32_to_u32_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_f64_add_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_f64_fma_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_f64_mul_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_hf16_add_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_hf16_fma_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_hf16_from_s32_rne_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_hf16_from_s32_rz_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_hf16_mul_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/apfloat_fuzz_hf16_to_f32_upcast_denorms_jit_wrapper.h"

namespace xls::dslx::apfloat_fuzz {
namespace {

using ::llvm::APFloat;
using ::llvm::APFloatBase;
using ::llvm::APInt;
using ::llvm::APSInt;

template <typename UintT>
struct FormatSpec {
  const llvm::fltSemantics* semantics;
  int bit_width;
  int exp_bits;
  int frac_bits;
  UintT canonical_qnan;
};

const FormatSpec<uint8_t>& E4M3Spec() {
  static const FormatSpec<uint8_t> kSpec = {
      .semantics = &APFloatBase::Float8E4M3(),
      .bit_width = 8,
      .exp_bits = 4,
      .frac_bits = 3,
      .canonical_qnan = 0x7c,
  };
  return kSpec;
}

const FormatSpec<uint16_t>& Hf16Spec() {
  static const FormatSpec<uint16_t> kSpec = {
      .semantics = &APFloatBase::IEEEhalf(),
      .bit_width = 16,
      .exp_bits = 5,
      .frac_bits = 10,
      .canonical_qnan = 0x7e00,
  };
  return kSpec;
}

const FormatSpec<uint16_t>& Bf16Spec() {
  static const FormatSpec<uint16_t> kSpec = {
      .semantics = &APFloatBase::BFloat(),
      .bit_width = 16,
      .exp_bits = 8,
      .frac_bits = 7,
      .canonical_qnan = 0x7fc0,
  };
  return kSpec;
}

const FormatSpec<uint32_t>& F32Spec() {
  static const FormatSpec<uint32_t> kSpec = {
      .semantics = &APFloatBase::IEEEsingle(),
      .bit_width = 32,
      .exp_bits = 8,
      .frac_bits = 23,
      .canonical_qnan = 0x7fc00000u,
  };
  return kSpec;
}

const FormatSpec<uint64_t>& F64Spec() {
  static const FormatSpec<uint64_t> kSpec = {
      .semantics = &APFloatBase::IEEEdouble(),
      .bit_width = 64,
      .exp_bits = 11,
      .frac_bits = 52,
      .canonical_qnan = 0x7ff8000000000000ull,
  };
  return kSpec;
}

template <typename WrapperT>
WrapperT& GetWrapper() {
  static absl::NoDestructor<std::unique_ptr<WrapperT>> wrapper([]() {
    auto created = WrapperT::Create();
    CHECK_OK(created);
    return std::move(*created);
  }());
  return **wrapper;
}

template <typename UintT>
UintT PackFields(const FormatSpec<UintT>& spec, uint64_t sign, uint64_t bexp,
                 uint64_t frac) {
  const uint64_t exp_mask = (uint64_t{1} << spec.exp_bits) - 1;
  const uint64_t frac_mask = (uint64_t{1} << spec.frac_bits) - 1;
  return static_cast<UintT>(((sign & 1) << (spec.exp_bits + spec.frac_bits)) |
                            ((bexp & exp_mask) << spec.frac_bits) |
                            (frac & frac_mask));
}

template <typename UintT>
APFloat DecodeFloat(const FormatSpec<UintT>& spec, UintT bits, bool daz) {
  APFloat value(*spec.semantics, APInt(spec.bit_width, bits));
  if (daz && value.isDenormal()) {
    return APFloat::getZero(*spec.semantics, value.isNegative());
  }
  return value;
}

template <typename UintT>
UintT EncodeExpected(const FormatSpec<UintT>& spec, APFloat value, bool ftz) {
  if (value.isNaN()) {
    return spec.canonical_qnan;
  }
  if (ftz && value.isDenormal()) {
    value = APFloat::getZero(*spec.semantics, value.isNegative());
  }
  return static_cast<UintT>(value.bitcastToAPInt().getZExtValue());
}

template <typename UintT>
std::string HexBits(const FormatSpec<UintT>& spec, UintT bits) {
  int hex_digits = spec.bit_width / 4;
  return absl::StrFormat("0x%0*x", hex_digits, static_cast<uint64_t>(bits));
}

uint16_t FlushBf16Subnormal(uint16_t bits) {
  if ((bits & 0x7f80u) == 0 && (bits & 0x007fu) != 0) {
    return static_cast<uint16_t>(bits & 0x8000u);
  }
  return bits;
}

uint32_t FlushF32Subnormal(uint32_t bits) {
  if ((bits & 0x7f800000u) == 0 && (bits & 0x007fffffu) != 0) {
    return bits & 0x80000000u;
  }
  return bits;
}

template <typename UintT>
UintT OracleAdd(const FormatSpec<UintT>& spec, UintT a_bits, UintT b_bits) {
  APFloat result = DecodeFloat(spec, a_bits, /*daz=*/true);
  result.add(DecodeFloat(spec, b_bits, /*daz=*/true),
             APFloat::rmNearestTiesToEven);
  return EncodeExpected(spec, result, /*ftz=*/true);
}

template <typename UintT>
UintT OracleSub(const FormatSpec<UintT>& spec, UintT a_bits, UintT b_bits) {
  APFloat result = DecodeFloat(spec, a_bits, /*daz=*/true);
  result.subtract(DecodeFloat(spec, b_bits, /*daz=*/true),
                  APFloat::rmNearestTiesToEven);
  return EncodeExpected(spec, result, /*ftz=*/true);
}

template <typename UintT>
UintT OracleMul(const FormatSpec<UintT>& spec, UintT a_bits, UintT b_bits) {
  APFloat result = DecodeFloat(spec, a_bits, /*daz=*/true);
  result.multiply(DecodeFloat(spec, b_bits, /*daz=*/true),
                  APFloat::rmNearestTiesToEven);
  return EncodeExpected(spec, result, /*ftz=*/true);
}

template <typename UintT>
UintT OracleFma(const FormatSpec<UintT>& spec, UintT a_bits, UintT b_bits,
                UintT c_bits) {
  APFloat result = DecodeFloat(spec, a_bits, /*daz=*/true);
  result.fusedMultiplyAdd(DecodeFloat(spec, b_bits, /*daz=*/true),
                          DecodeFloat(spec, c_bits, /*daz=*/true),
                          APFloat::rmNearestTiesToEven);
  return EncodeExpected(spec, result, /*ftz=*/true);
}

template <typename UintT>
UintT OracleLdexp(const FormatSpec<UintT>& spec, UintT a_bits, int32_t exp) {
  APFloat input = DecodeFloat(spec, a_bits, /*daz=*/true);
  APFloat result = llvm::scalbn(input, exp, APFloat::rmNearestTiesToEven);
  return EncodeExpected(spec, result, /*ftz=*/true);
}

template <typename UintT, typename IntT>
UintT OracleFromSignedInt(const FormatSpec<UintT>& spec, IntT value,
                          APFloat::roundingMode rounding_mode) {
  APFloat result(*spec.semantics);
  APInt integer_bits(sizeof(IntT) * 8, static_cast<uint64_t>(value),
                     /*isSigned=*/true);
  result.convertFromAPInt(integer_bits, /*IsSigned=*/true, rounding_mode);
  return EncodeExpected(spec, result, /*ftz=*/false);
}

template <typename UintT, typename UnsignedIntT>
UintT OracleFromUnsignedInt(const FormatSpec<UintT>& spec, UnsignedIntT value,
                            APFloat::roundingMode rounding_mode) {
  APFloat result(*spec.semantics);
  APInt integer_bits(sizeof(UnsignedIntT) * 8, static_cast<uint64_t>(value),
                     /*isSigned=*/false);
  result.convertFromAPInt(integer_bits, /*IsSigned=*/false, rounding_mode);
  return EncodeExpected(spec, result, /*ftz=*/false);
}

template <typename IntT, typename UintT>
IntT OracleToInt(const FormatSpec<UintT>& spec, UintT bits, bool is_unsigned) {
  APFloat input = DecodeFloat(spec, bits, /*daz=*/false);
  APSInt out(sizeof(IntT) * 8, is_unsigned);
  bool is_exact = false;
  input.convertToInteger(out, APFloat::rmTowardZero, &is_exact);
  return static_cast<IntT>(is_unsigned ? out.getZExtValue()
                                       : out.getSExtValue());
}

int32_t OracleCastToFixedS32(uint32_t bits) {
  APFloat input = DecodeFloat(F32Spec(), bits, /*daz=*/false);
  if (input.isNaN() || input.isInfinity()) {
    return std::numeric_limits<int32_t>::min();
  }
  APSInt out(32, /*isUnsigned=*/false);
  bool is_exact = false;
  APFloat::opStatus status =
      input.convertToInteger(out, APFloat::rmTowardZero, &is_exact);
  if (status & APFloat::opInvalidOp) {
    return std::numeric_limits<int32_t>::min();
  }
  return static_cast<int32_t>(out.getSExtValue());
}

template <typename ToUintT, typename FromUintT>
ToUintT OracleConvert(const FormatSpec<FromUintT>& from_spec,
                      const FormatSpec<ToUintT>& to_spec, FromUintT bits,
                      APFloat::roundingMode rounding_mode, bool daz, bool ftz) {
  APFloat value = DecodeFloat(from_spec, bits, daz);
  bool loses_info = false;
  value.convert(*to_spec.semantics, rounding_mode, &loses_info);
  return EncodeExpected(to_spec, value, ftz);
}

template <typename UintT>
UintT OracleRoundToIntegral(const FormatSpec<UintT>& spec, UintT bits,
                            APFloat::roundingMode rounding_mode, bool daz) {
  APFloat value = DecodeFloat(spec, bits, daz);
  value.roundToIntegral(rounding_mode);
  return EncodeExpected(spec, value, /*ftz=*/false);
}

uint8_t OracleBf16CmpMask(uint16_t a_bits, uint16_t b_bits) {
  APFloat a = DecodeFloat(Bf16Spec(), a_bits, /*daz=*/true);
  APFloat b = DecodeFloat(Bf16Spec(), b_bits, /*daz=*/true);
  APFloat::cmpResult cmp = a.compare(b);
  uint8_t eq = cmp == APFloat::cmpEqual ? 1 : 0;
  uint8_t lt = cmp == APFloat::cmpLessThan ? 1 : 0;
  uint8_t lte = (eq | lt);
  uint8_t gt = cmp == APFloat::cmpGreaterThan ? 1 : 0;
  uint8_t gte = (eq | gt);
  return static_cast<uint8_t>((gte << 4) | (gt << 3) | (lte << 2) | (lt << 1) |
                              eq);
}

// Exhaustive tests for 8-bit binary ops and 16-bit unary ops.

TEST(ApfloatExhaustiveTest, E4M3AddAndMulMatchIeeeDazFtz) {
  const FormatSpec<uint8_t>& spec = E4M3Spec();
  auto& add_wrapper = GetWrapper<E4M3Add>();
  auto& mul_wrapper = GetWrapper<E4M3Mul>();
  for (uint32_t a = 0; a < 256; ++a) {
    for (uint32_t b = 0; b < 256; ++b) {
      uint8_t a8 = static_cast<uint8_t>(a);
      uint8_t b8 = static_cast<uint8_t>(b);
      XLS_ASSERT_OK_AND_ASSIGN(uint8_t actual_add, add_wrapper.Run(a8, b8));
      uint8_t expected_add = OracleAdd(spec, a8, b8);
      ASSERT_EQ(actual_add, expected_add)
          << "e4m3_add(" << HexBits(spec, a8) << ", " << HexBits(spec, b8)
          << "): actual=" << HexBits(spec, actual_add)
          << " expected=" << HexBits(spec, expected_add);

      XLS_ASSERT_OK_AND_ASSIGN(uint8_t actual_mul, mul_wrapper.Run(a8, b8));
      uint8_t expected_mul = OracleMul(spec, a8, b8);
      ASSERT_EQ(actual_mul, expected_mul)
          << "e4m3_mul(" << HexBits(spec, a8) << ", " << HexBits(spec, b8)
          << "): actual=" << HexBits(spec, actual_mul)
          << " expected=" << HexBits(spec, expected_mul);
    }
  }
}

TEST(ApfloatExhaustiveTest, Bf16UnaryOpsMatchIeee) {
  const FormatSpec<uint16_t>& bf16 = Bf16Spec();
  const FormatSpec<uint32_t>& f32 = F32Spec();
  auto& to_s32 = GetWrapper<Bf16ToS32>();
  auto& to_u32 = GetWrapper<Bf16ToU32>();
  auto& ceil_denorms = GetWrapper<Bf16CeilWithDenorms>();
  auto& ceil_daz = GetWrapper<Bf16CeilDaz>();
  auto& floor_denorms = GetWrapper<Bf16FloorWithDenorms>();
  auto& floor_daz = GetWrapper<Bf16FloorDaz>();
  auto& trunc_op = GetWrapper<Bf16Trunc>();
  auto& round_rne = GetWrapper<Bf16RoundRne>();
  auto& round_rna = GetWrapper<Bf16RoundRna>();
  auto& upcast_denorms = GetWrapper<Bf16ToF32UpcastDenorms>();
  auto& upcast_daz = GetWrapper<Bf16ToF32UpcastDaz>();

  for (uint32_t i = 0; i < 65536; ++i) {
    uint16_t bits = static_cast<uint16_t>(i);

    XLS_ASSERT_OK_AND_ASSIGN(uint32_t actual_s32_raw, to_s32.Run(bits));
    ASSERT_EQ(static_cast<int32_t>(actual_s32_raw),
              OracleToInt<int32_t>(bf16, bits, /*is_unsigned=*/false))
        << "bf16_to_s32(" << HexBits(bf16, bits) << ")";

    XLS_ASSERT_OK_AND_ASSIGN(uint32_t actual_u32, to_u32.Run(bits));
    ASSERT_EQ(actual_u32,
              OracleToInt<uint32_t>(bf16, bits, /*is_unsigned=*/true))
        << "bf16_to_u32(" << HexBits(bf16, bits) << ")";

    XLS_ASSERT_OK_AND_ASSIGN(uint16_t actual_ceil, ceil_denorms.Run(bits));
    ASSERT_EQ(actual_ceil,
              OracleRoundToIntegral(bf16, bits, APFloat::rmTowardPositive,
                                    /*daz=*/false))
        << "bf16_ceil_with_denorms(" << HexBits(bf16, bits) << ")";

    XLS_ASSERT_OK_AND_ASSIGN(uint16_t actual_ceil_daz, ceil_daz.Run(bits));
    ASSERT_EQ(actual_ceil_daz,
              OracleRoundToIntegral(bf16, bits, APFloat::rmTowardPositive,
                                    /*daz=*/true))
        << "bf16_ceil_daz(" << HexBits(bf16, bits) << ")";

    XLS_ASSERT_OK_AND_ASSIGN(uint16_t actual_floor, floor_denorms.Run(bits));
    ASSERT_EQ(actual_floor,
              OracleRoundToIntegral(bf16, bits, APFloat::rmTowardNegative,
                                    /*daz=*/false))
        << "bf16_floor_with_denorms(" << HexBits(bf16, bits) << ")";

    XLS_ASSERT_OK_AND_ASSIGN(uint16_t actual_floor_daz, floor_daz.Run(bits));
    ASSERT_EQ(actual_floor_daz,
              OracleRoundToIntegral(bf16, bits, APFloat::rmTowardNegative,
                                    /*daz=*/true))
        << "bf16_floor_daz(" << HexBits(bf16, bits) << ")";

    XLS_ASSERT_OK_AND_ASSIGN(uint16_t actual_trunc, trunc_op.Run(bits));
    ASSERT_EQ(actual_trunc,
              OracleRoundToIntegral(bf16, bits, APFloat::rmTowardZero,
                                    /*daz=*/false))
        << "bf16_trunc(" << HexBits(bf16, bits) << ")";

    XLS_ASSERT_OK_AND_ASSIGN(uint16_t actual_rne, round_rne.Run(bits));
    ASSERT_EQ(actual_rne,
              OracleRoundToIntegral(bf16, bits, APFloat::rmNearestTiesToEven,
                                    /*daz=*/false))
        << "bf16_round_rne(" << HexBits(bf16, bits) << ")";

    XLS_ASSERT_OK_AND_ASSIGN(uint16_t actual_rna, round_rna.Run(bits));
    ASSERT_EQ(actual_rna,
              OracleRoundToIntegral(bf16, bits, APFloat::rmNearestTiesToAway,
                                    /*daz=*/false))
        << "bf16_round_rna(" << HexBits(bf16, bits) << ")";

    XLS_ASSERT_OK_AND_ASSIGN(uint32_t actual_upcast, upcast_denorms.Run(bits));
    ASSERT_EQ(actual_upcast,
              OracleConvert(bf16, f32, bits, APFloat::rmNearestTiesToEven,
                            /*daz=*/false, /*ftz=*/false))
        << "bf16_to_f32_upcast_denorms(" << HexBits(bf16, bits) << ")";

    XLS_ASSERT_OK_AND_ASSIGN(uint32_t actual_upcast_daz, upcast_daz.Run(bits));
    ASSERT_EQ(actual_upcast_daz,
              OracleConvert(bf16, f32, bits, APFloat::rmNearestTiesToEven,
                            /*daz=*/true, /*ftz=*/false))
        << "bf16_to_f32_upcast_daz(" << HexBits(bf16, bits) << ")";
  }
}

TEST(ApfloatExhaustiveTest, Hf16UpcastDenormsMatchesIeee) {
  const FormatSpec<uint16_t>& hf16 = Hf16Spec();
  const FormatSpec<uint32_t>& f32 = F32Spec();
  auto& upcast = GetWrapper<Hf16ToF32UpcastDenorms>();
  for (uint32_t i = 0; i < 65536; ++i) {
    uint16_t bits = static_cast<uint16_t>(i);
    XLS_ASSERT_OK_AND_ASSIGN(uint32_t actual, upcast.Run(bits));
    uint32_t expected =
        OracleConvert(hf16, f32, bits, APFloat::rmNearestTiesToEven,
                      /*daz=*/false, /*ftz=*/false);
    ASSERT_EQ(actual, expected)
        << "hf16_to_f32_upcast_denorms(" << HexBits(hf16, bits) << ")";
  }
}

// FuzzTest properties for binary, ternary, cast, and wide-format operations.

void E4M3FmaMatchesIeeeDazFtz(uint8_t a, uint8_t b, uint8_t c) {
  const FormatSpec<uint8_t>& spec = E4M3Spec();
  XLS_ASSERT_OK_AND_ASSIGN(uint8_t actual, GetWrapper<E4M3Fma>().Run(a, b, c));
  uint8_t expected = OracleFma(spec, a, b, c);
  EXPECT_EQ(actual, expected)
      << "e4m3_fma(" << HexBits(spec, a) << ", " << HexBits(spec, b) << ", "
      << HexBits(spec, c) << "): actual=" << HexBits(spec, actual)
      << " expected=" << HexBits(spec, expected);
}
FUZZ_TEST(ApfloatFuzzTest, E4M3FmaMatchesIeeeDazFtz);

void Bf16AddSubMulMatchIeeeAndEigen(uint16_t a, uint16_t b) {
  const FormatSpec<uint16_t>& spec = Bf16Spec();
  Eigen::bfloat16 eigen_a =
      Eigen::numext::bit_cast<Eigen::bfloat16>(FlushBf16Subnormal(a));
  Eigen::bfloat16 eigen_b =
      Eigen::numext::bit_cast<Eigen::bfloat16>(FlushBf16Subnormal(b));

  auto check_eigen = [&](uint16_t actual, Eigen::bfloat16 eigen_result,
                         const char* op_name) {
    uint16_t eigen_bits =
        FlushBf16Subnormal(Eigen::numext::bit_cast<uint16_t>(eigen_result));
    if (Eigen::numext::isnan(eigen_result)) {
      EXPECT_EQ(actual, spec.canonical_qnan)
          << op_name << "(" << HexBits(spec, a) << ", " << HexBits(spec, b)
          << ") vs Eigen NaN";
    } else {
      EXPECT_EQ(actual, eigen_bits)
          << op_name << "(" << HexBits(spec, a) << ", " << HexBits(spec, b)
          << ") vs Eigen: actual=" << HexBits(spec, actual)
          << " eigen=" << HexBits(spec, eigen_bits);
    }
  };

  XLS_ASSERT_OK_AND_ASSIGN(uint16_t actual_add,
                           GetWrapper<Bf16Add>().Run(a, b));
  uint16_t expected_add = OracleAdd(spec, a, b);
  EXPECT_EQ(actual_add, expected_add)
      << "bf16_add(" << HexBits(spec, a) << ", " << HexBits(spec, b) << ")";
  check_eigen(actual_add, eigen_a + eigen_b, "bf16_add");

  XLS_ASSERT_OK_AND_ASSIGN(uint16_t actual_add_lza,
                           GetWrapper<Bf16AddLza>().Run(a, b));
  EXPECT_EQ(actual_add_lza, expected_add)
      << "bf16_add_lza(" << HexBits(spec, a) << ", " << HexBits(spec, b) << ")";

  XLS_ASSERT_OK_AND_ASSIGN(uint16_t actual_sub,
                           GetWrapper<Bf16Sub>().Run(a, b));
  EXPECT_EQ(actual_sub, OracleSub(spec, a, b))
      << "bf16_sub(" << HexBits(spec, a) << ", " << HexBits(spec, b) << ")";
  check_eigen(actual_sub, eigen_a - eigen_b, "bf16_sub");

  XLS_ASSERT_OK_AND_ASSIGN(uint16_t actual_mul,
                           GetWrapper<Bf16Mul>().Run(a, b));
  EXPECT_EQ(actual_mul, OracleMul(spec, a, b))
      << "bf16_mul(" << HexBits(spec, a) << ", " << HexBits(spec, b) << ")";
  check_eigen(actual_mul, eigen_a * eigen_b, "bf16_mul");
}
FUZZ_TEST(ApfloatFuzzTest, Bf16AddSubMulMatchIeeeAndEigen);

void Bf16FmaMatchesIeeeDazFtz(uint16_t a, uint16_t b, uint16_t c) {
  const FormatSpec<uint16_t>& spec = Bf16Spec();
  XLS_ASSERT_OK_AND_ASSIGN(uint16_t actual, GetWrapper<Bf16Fma>().Run(a, b, c));
  uint16_t expected = OracleFma(spec, a, b, c);
  EXPECT_EQ(actual, expected)
      << "bf16_fma(" << HexBits(spec, a) << ", " << HexBits(spec, b) << ", "
      << HexBits(spec, c) << "): actual=" << HexBits(spec, actual)
      << " expected=" << HexBits(spec, expected);
}
FUZZ_TEST(ApfloatFuzzTest, Bf16FmaMatchesIeeeDazFtz);

void Bf16LdexpMatchesIeeeDazFtz(uint16_t a, int32_t exp) {
  const FormatSpec<uint16_t>& spec = Bf16Spec();
  XLS_ASSERT_OK_AND_ASSIGN(uint16_t actual, GetWrapper<Bf16Ldexp>().Run(
                                                a, static_cast<uint32_t>(exp)));
  uint16_t expected = OracleLdexp(spec, a, exp);
  EXPECT_EQ(actual, expected) << "bf16_ldexp(" << HexBits(spec, a) << ", "
                              << exp << "): actual=" << HexBits(spec, actual)
                              << " expected=" << HexBits(spec, expected);
}
FUZZ_TEST(ApfloatFuzzTest, Bf16LdexpMatchesIeeeDazFtz);

void Bf16ComparisonsMatchIeeeDaz(uint16_t a, uint16_t b) {
  const FormatSpec<uint16_t>& spec = Bf16Spec();
  XLS_ASSERT_OK_AND_ASSIGN(uint8_t actual, GetWrapper<Bf16Cmp>().Run(a, b));
  EXPECT_EQ(actual, OracleBf16CmpMask(a, b))
      << "bf16_cmp(" << HexBits(spec, a) << ", " << HexBits(spec, b) << ")";
}
FUZZ_TEST(ApfloatFuzzTest, Bf16ComparisonsMatchIeeeDaz);

void Hf16AddAndMulMatchIeeeDazFtz(uint16_t a, uint16_t b) {
  const FormatSpec<uint16_t>& spec = Hf16Spec();
  XLS_ASSERT_OK_AND_ASSIGN(uint16_t actual_add,
                           GetWrapper<Hf16Add>().Run(a, b));
  EXPECT_EQ(actual_add, OracleAdd(spec, a, b))
      << "hf16_add(" << HexBits(spec, a) << ", " << HexBits(spec, b) << ")";

  XLS_ASSERT_OK_AND_ASSIGN(uint16_t actual_mul,
                           GetWrapper<Hf16Mul>().Run(a, b));
  EXPECT_EQ(actual_mul, OracleMul(spec, a, b))
      << "hf16_mul(" << HexBits(spec, a) << ", " << HexBits(spec, b) << ")";
}
FUZZ_TEST(ApfloatFuzzTest, Hf16AddAndMulMatchIeeeDazFtz);

void Hf16FmaMatchesIeeeDazFtz(uint16_t a, uint16_t b, uint16_t c) {
  const FormatSpec<uint16_t>& spec = Hf16Spec();
  XLS_ASSERT_OK_AND_ASSIGN(uint16_t actual, GetWrapper<Hf16Fma>().Run(a, b, c));
  uint16_t expected = OracleFma(spec, a, b, c);
  EXPECT_EQ(actual, expected)
      << "hf16_fma(" << HexBits(spec, a) << ", " << HexBits(spec, b) << ", "
      << HexBits(spec, c) << "): actual=" << HexBits(spec, actual)
      << " expected=" << HexBits(spec, expected);
}
FUZZ_TEST(ApfloatFuzzTest, Hf16FmaMatchesIeeeDazFtz);

void F32AddAndMulMatchIeeeDazFtz(uint32_t a, uint32_t b) {
  const FormatSpec<uint32_t>& spec = F32Spec();
  uint32_t expected_add = OracleAdd(spec, a, b);
  XLS_ASSERT_OK_AND_ASSIGN(uint32_t actual_add, GetWrapper<F32Add>().Run(a, b));
  EXPECT_EQ(actual_add, expected_add)
      << "f32_add(" << HexBits(spec, a) << ", " << HexBits(spec, b) << ")";

  XLS_ASSERT_OK_AND_ASSIGN(uint32_t actual_add_lza,
                           GetWrapper<F32AddLza>().Run(a, b));
  EXPECT_EQ(actual_add_lza, expected_add)
      << "f32_add_lza(" << HexBits(spec, a) << ", " << HexBits(spec, b) << ")";

  XLS_ASSERT_OK_AND_ASSIGN(uint32_t actual_mul, GetWrapper<F32Mul>().Run(a, b));
  EXPECT_EQ(actual_mul, OracleMul(spec, a, b))
      << "f32_mul(" << HexBits(spec, a) << ", " << HexBits(spec, b) << ")";
}
FUZZ_TEST(ApfloatFuzzTest, F32AddAndMulMatchIeeeDazFtz);

void F32FmaMatchesIeeeDazFtz(uint32_t a, uint32_t b, uint32_t c) {
  const FormatSpec<uint32_t>& spec = F32Spec();
  XLS_ASSERT_OK_AND_ASSIGN(uint32_t actual, GetWrapper<F32Fma>().Run(a, b, c));
  uint32_t expected = OracleFma(spec, a, b, c);
  EXPECT_EQ(actual, expected)
      << "f32_fma(" << HexBits(spec, a) << ", " << HexBits(spec, b) << ", "
      << HexBits(spec, c) << "): actual=" << HexBits(spec, actual)
      << " expected=" << HexBits(spec, expected);
}
FUZZ_TEST(ApfloatFuzzTest, F32FmaMatchesIeeeDazFtz);

void F32LdexpMatchesIeeeDazFtz(uint32_t a, int32_t exp) {
  const FormatSpec<uint32_t>& spec = F32Spec();
  XLS_ASSERT_OK_AND_ASSIGN(uint32_t actual, GetWrapper<F32Ldexp>().Run(
                                                a, static_cast<uint32_t>(exp)));
  uint32_t expected = OracleLdexp(spec, a, exp);
  EXPECT_EQ(actual, expected) << "f32_ldexp(" << HexBits(spec, a) << ", " << exp
                              << "): actual=" << HexBits(spec, actual)
                              << " expected=" << HexBits(spec, expected);
}
FUZZ_TEST(ApfloatFuzzTest, F32LdexpMatchesIeeeDazFtz);

void F32ToIntAndUintMatchIeeeSaturating(uint32_t a) {
  const FormatSpec<uint32_t>& spec = F32Spec();
  XLS_ASSERT_OK_AND_ASSIGN(uint32_t actual_s32_raw,
                           GetWrapper<F32ToS32>().Run(a));
  EXPECT_EQ(static_cast<int32_t>(actual_s32_raw),
            OracleToInt<int32_t>(spec, a, /*is_unsigned=*/false))
      << "f32_to_s32(" << HexBits(spec, a) << ")";

  XLS_ASSERT_OK_AND_ASSIGN(uint32_t actual_u32, GetWrapper<F32ToU32>().Run(a));
  EXPECT_EQ(actual_u32, OracleToInt<uint32_t>(spec, a, /*is_unsigned=*/true))
      << "f32_to_u32(" << HexBits(spec, a) << ")";

  XLS_ASSERT_OK_AND_ASSIGN(uint32_t actual_fixed_raw,
                           GetWrapper<F32CastToFixedS32>().Run(a));
  EXPECT_EQ(static_cast<int32_t>(actual_fixed_raw), OracleCastToFixedS32(a))
      << "f32_cast_to_fixed_s32(" << HexBits(spec, a) << ")";
}
FUZZ_TEST(ApfloatFuzzTest, F32ToIntAndUintMatchIeeeSaturating);

void F64AddAndMulMatchIeeeDazFtz(uint64_t a, uint64_t b) {
  const FormatSpec<uint64_t>& spec = F64Spec();
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t actual_add, GetWrapper<F64Add>().Run(a, b));
  EXPECT_EQ(actual_add, OracleAdd(spec, a, b))
      << "f64_add(" << HexBits(spec, a) << ", " << HexBits(spec, b) << ")";

  XLS_ASSERT_OK_AND_ASSIGN(uint64_t actual_mul, GetWrapper<F64Mul>().Run(a, b));
  EXPECT_EQ(actual_mul, OracleMul(spec, a, b))
      << "f64_mul(" << HexBits(spec, a) << ", " << HexBits(spec, b) << ")";
}
FUZZ_TEST(ApfloatFuzzTest, F64AddAndMulMatchIeeeDazFtz);

void F64FmaMatchesIeeeDazFtz(uint64_t a, uint64_t b, uint64_t c) {
  const FormatSpec<uint64_t>& spec = F64Spec();
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t actual, GetWrapper<F64Fma>().Run(a, b, c));
  uint64_t expected = OracleFma(spec, a, b, c);
  EXPECT_EQ(actual, expected)
      << "f64_fma(" << HexBits(spec, a) << ", " << HexBits(spec, b) << ", "
      << HexBits(spec, c) << "): actual=" << HexBits(spec, actual)
      << " expected=" << HexBits(spec, expected);
}
FUZZ_TEST(ApfloatFuzzTest, F64FmaMatchesIeeeDazFtz);

void CastFromFixedRneMatchesIeee(int32_t value_32, int64_t value_64) {
  XLS_ASSERT_OK_AND_ASSIGN(
      uint8_t actual_e4m3,
      GetWrapper<E4M3FromS64Rne>().Run(static_cast<uint64_t>(value_64)));
  EXPECT_EQ(actual_e4m3, OracleFromSignedInt(E4M3Spec(), value_64,
                                             APFloat::rmNearestTiesToEven))
      << "e4m3_from_s64_rne(" << value_64 << ")";

  XLS_ASSERT_OK_AND_ASSIGN(
      uint16_t actual_hf16,
      GetWrapper<Hf16FromS32Rne>().Run(static_cast<uint32_t>(value_32)));
  EXPECT_EQ(actual_hf16, OracleFromSignedInt(Hf16Spec(), value_32,
                                             APFloat::rmNearestTiesToEven))
      << "hf16_from_s32_rne(" << value_32 << ")";

  XLS_ASSERT_OK_AND_ASSIGN(
      uint16_t actual_bf16,
      GetWrapper<Bf16FromS32Rne>().Run(static_cast<uint32_t>(value_32)));
  EXPECT_EQ(actual_bf16, OracleFromSignedInt(Bf16Spec(), value_32,
                                             APFloat::rmNearestTiesToEven))
      << "bf16_from_s32_rne(" << value_32 << ")";

  XLS_ASSERT_OK_AND_ASSIGN(
      uint32_t actual_f32,
      GetWrapper<F32FromS64Rne>().Run(static_cast<uint64_t>(value_64)));
  EXPECT_EQ(actual_f32, OracleFromSignedInt(F32Spec(), value_64,
                                            APFloat::rmNearestTiesToEven))
      << "f32_from_s64_rne(" << value_64 << ")";
}
FUZZ_TEST(ApfloatFuzzTest, CastFromFixedRneMatchesIeee);

void CastFromFixedRzMatchesIeee(int32_t value_32, int64_t value_64) {
  XLS_ASSERT_OK_AND_ASSIGN(
      uint8_t actual_e4m3,
      GetWrapper<E4M3FromS64Rz>().Run(static_cast<uint64_t>(value_64)));
  EXPECT_EQ(actual_e4m3,
            OracleFromSignedInt(E4M3Spec(), value_64, APFloat::rmTowardZero))
      << "e4m3_from_s64_rz(" << value_64 << ")";

  XLS_ASSERT_OK_AND_ASSIGN(
      uint16_t actual_hf16,
      GetWrapper<Hf16FromS32Rz>().Run(static_cast<uint32_t>(value_32)));
  EXPECT_EQ(actual_hf16,
            OracleFromSignedInt(Hf16Spec(), value_32, APFloat::rmTowardZero))
      << "hf16_from_s32_rz(" << value_32 << ")";

  XLS_ASSERT_OK_AND_ASSIGN(
      uint16_t actual_bf16,
      GetWrapper<Bf16FromS32Rz>().Run(static_cast<uint32_t>(value_32)));
  EXPECT_EQ(actual_bf16,
            OracleFromSignedInt(Bf16Spec(), value_32, APFloat::rmTowardZero))
      << "bf16_from_s32_rz(" << value_32 << ")";

  XLS_ASSERT_OK_AND_ASSIGN(
      uint32_t actual_f32,
      GetWrapper<F32FromS64Rz>().Run(static_cast<uint64_t>(value_64)));
  EXPECT_EQ(actual_f32,
            OracleFromSignedInt(F32Spec(), value_64, APFloat::rmTowardZero))
      << "f32_from_s64_rz(" << value_64 << ")";
}
FUZZ_TEST(ApfloatFuzzTest, CastFromFixedRzMatchesIeee);

void F32DowncastToBf16AndHf16MatchesIeeeAndEigen(uint32_t a) {
  const FormatSpec<uint32_t>& f32 = F32Spec();
  const FormatSpec<uint16_t>& bf16 = Bf16Spec();
  const FormatSpec<uint16_t>& hf16 = Hf16Spec();

  XLS_ASSERT_OK_AND_ASSIGN(uint16_t actual_bf16_rne,
                           GetWrapper<F32ToBf16DowncastRne>().Run(a));
  EXPECT_EQ(actual_bf16_rne,
            OracleConvert(f32, bf16, a, APFloat::rmNearestTiesToEven,
                          /*daz=*/true, /*ftz=*/true))
      << "f32_to_bf16_downcast_rne(" << HexBits(f32, a) << ")";

  // Cross-check against Eigen::bfloat16 on DAZ-flushed input.
  float flushed_f32 = absl::bit_cast<float>(FlushF32Subnormal(a));
  Eigen::bfloat16 eigen_bf16(flushed_f32);
  if (std::isnan(flushed_f32)) {
    EXPECT_EQ(actual_bf16_rne, bf16.canonical_qnan);
  } else {
    EXPECT_EQ(actual_bf16_rne,
              FlushBf16Subnormal(Eigen::numext::bit_cast<uint16_t>(eigen_bf16)))
        << "f32_to_bf16_downcast_rne vs Eigen(" << HexBits(f32, a) << ")";
  }

  XLS_ASSERT_OK_AND_ASSIGN(uint16_t actual_bf16_rna,
                           GetWrapper<F32ToBf16DowncastRna>().Run(a));
  EXPECT_EQ(actual_bf16_rna,
            OracleConvert(f32, bf16, a, APFloat::rmNearestTiesToAway,
                          /*daz=*/true, /*ftz=*/false))
      << "f32_to_bf16_downcast_rna(" << HexBits(f32, a) << ")";

  XLS_ASSERT_OK_AND_ASSIGN(uint16_t actual_hf16_rne,
                           GetWrapper<F32ToHf16DowncastRne>().Run(a));
  EXPECT_EQ(actual_hf16_rne,
            OracleConvert(f32, hf16, a, APFloat::rmNearestTiesToEven,
                          /*daz=*/true, /*ftz=*/true))
      << "f32_to_hf16_downcast_rne(" << HexBits(f32, a) << ")";

  XLS_ASSERT_OK_AND_ASSIGN(uint16_t actual_hf16_denorm,
                           GetWrapper<F32ToHf16Downcast>().Run(a));
  EXPECT_EQ(actual_hf16_denorm,
            OracleConvert(f32, hf16, a, APFloat::rmNearestTiesToEven,
                          /*daz=*/true, /*ftz=*/false))
      << "f32_to_hf16_downcast(" << HexBits(f32, a) << ")";
}
FUZZ_TEST(ApfloatFuzzTest, F32DowncastToBf16AndHf16MatchesIeeeAndEigen);

void Float32IntConversionsMatchIeee(int32_t signed_val, uint32_t unsigned_val) {
  const FormatSpec<uint32_t>& f32 = F32Spec();
  XLS_ASSERT_OK_AND_ASSIGN(
      uint32_t actual_s32,
      GetWrapper<F32FromInt32>().Run(static_cast<uint32_t>(signed_val)));
  EXPECT_EQ(actual_s32,
            OracleFromSignedInt(f32, signed_val, APFloat::rmNearestTiesToEven))
      << "f32_from_int32(" << signed_val << ")";

  XLS_ASSERT_OK_AND_ASSIGN(uint32_t actual_u32,
                           GetWrapper<F32FromUint32>().Run(unsigned_val));
  EXPECT_EQ(actual_u32, OracleFromUnsignedInt(f32, unsigned_val,
                                              APFloat::rmNearestTiesToEven))
      << "f32_from_uint32(" << unsigned_val << ")";
}
FUZZ_TEST(ApfloatFuzzTest, Float32IntConversionsMatchIeee);

void Float32FastRsqrtMatchesReference(uint32_t a) {
  const FormatSpec<uint32_t>& spec = F32Spec();
  XLS_ASSERT_OK_AND_ASSIGN(uint32_t actual, GetWrapper<F32FastRsqrt>().Run(a));
  const uint32_t sign = a >> 31;
  const uint32_t bexp = (a >> 23) & 0xffU;
  const uint32_t frac = a & 0x7fffffU;

  if (bexp == 0) {
    EXPECT_EQ(actual, PackFields(spec, sign, 0xff, 0))
        << "f32_fast_rsqrt(" << HexBits(spec, a) << ")";
    return;
  }
  if ((bexp == 0xff && frac != 0) || sign == 1) {
    EXPECT_EQ(actual, spec.canonical_qnan)
        << "f32_fast_rsqrt(" << HexBits(spec, a) << ")";
    return;
  }
  if (bexp == 0xff && frac == 0) {
    EXPECT_EQ(actual, uint32_t{0})
        << "f32_fast_rsqrt(" << HexBits(spec, a) << ")";
    return;
  }

  double x = static_cast<double>(absl::bit_cast<float>(a));
  double expected = 1.0 / std::sqrt(x);
  float actual_f = absl::bit_cast<float>(actual);
  ASSERT_TRUE(std::isfinite(actual_f))
      << "f32_fast_rsqrt(" << HexBits(spec, a) << ") returned non-finite "
      << HexBits(spec, actual);
  EXPECT_GT(actual_f, 0.0f)
      << "f32_fast_rsqrt(" << HexBits(spec, a) << ")=" << HexBits(spec, actual);
  double rel_err =
      std::abs(static_cast<double>(actual_f) - expected) / expected;
  EXPECT_LE(rel_err, 0.01) << "f32_fast_rsqrt(" << HexBits(spec, a)
                           << "): actual=" << actual_f
                           << " expected=" << expected
                           << " rel_err=" << rel_err;
}
FUZZ_TEST(ApfloatFuzzTest, Float32FastRsqrtMatchesReference);

}  // namespace
}  // namespace xls::dslx::apfloat_fuzz
