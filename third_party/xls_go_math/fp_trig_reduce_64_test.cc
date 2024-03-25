// Copyright 2009 The Go Authors. All rights reserved.
// Copyright 2021 The XLS Authors
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Random-sampling test for the DSLX 2x64
// floating-point trig reduction unit.
#include <cmath>
#include <tuple>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/numeric/int128.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/value_utils.h"
#include "xls/ir/value_view_utils.h"
#include "xls/tools/testbench_builder.h"
#include "third_party/xls_go_math/fp_trig_reduce_jit_wrapper.h"

ABSL_FLAG(bool, use_opt_ir, true, "Use optimized IR.");
ABSL_FLAG(int, num_threads, 4,
          "Number of threads to use. Set to 0 to use all.");
ABSL_FLAG(int64_t, num_samples, 1024 * 1024,
          "Number of random samples to test.");

namespace xls {

// Returned bits of fp_trig_reduce.
typedef struct {
  // Fields are reversed relative to xls
  // because earlier fields in an xls tuple
  // occupy higher bits.
  double fraction;
  uint8_t j;
} ResultT;

using PackedFloat64 =
    PackedTupleView<PackedBitsView<1>, PackedBitsView<11>, PackedBitsView<52>>;
using PackedU3Float64 = PackedTupleView<PackedBitsView<3>, PackedFloat64>;

double FlushSubnormals(double value) {
  if (std::fpclassify(value) == FP_SUBNORMAL) {
    return 0;
  }
  return value;
}

bool ZeroOrSubnormal(double value) {
  return value == 0 || std::fpclassify(value) == FP_SUBNORMAL;
}

// Cast from uint64_t to double.
double unflatten(uint64_t in) { return absl::bit_cast<double>(in); }

// Cast from double to uint64_t
uint64_t flatten(double in) { return absl::bit_cast<uint64_t>(in); }

// Return the number of leading zeros in num;
uint64_t clz(uint64_t num) {
  uint64_t leading_zeros = 0;
  for (uint64_t mask = static_cast<uint64_t>(1) << 63; !(mask & num);
       mask = mask >> 1) {
    leading_zeros++;
  }
  return leading_zeros;
}

// Generates double with reasonably unformly random bit patterns.
double IndexToInput(uint64_t index) {
  thread_local absl::BitGen bitgen;
  double a = absl::Uniform(bitgen, 0u, std::numeric_limits<uint64_t>::max());
  return a;
}

ResultT ComputeExpected(fp::FpTrigReduce* jit_wrapper, double x) {
  // Bits of 4/Pi.
  constexpr uint64_t MPI4[] = {
      0x0000000000000001, 0x45f306dc9c882a53, 0xf84eafa3ea69bb81,
      0xb6c52b3278872083, 0xfca2c757bd778ac3, 0x6e48dc74849ba5c0,
      0x0c925dd413a32439, 0xfc3bd63962534e7d, 0xd1046bea5d768909,
      0xd338e04d68befc82, 0x7323ac7306a673e9, 0x3908bf177bf25076,
      0x3ff12fffbc0b301f, 0xde5e2316b414da3e, 0xda6cfd9e4f96136e,
      0x9e8c7ecd3cbfd45a, 0xea4f758fd7cbe2f6, 0x7a0e73ef14a525d4,
      0xd7f6bf623f1aba10, 0xac06608df8f6d757,
  };

  constexpr uint64_t FP64_EXP_SHIFT = 52;
  constexpr uint64_t FP64_EXP_BIAS = 1023;
  constexpr uint64_t FP64_EXP_MASK = 0x7ff;

  if (x < M_PI_4) {
    return ResultT{.fraction = x, .j = static_cast<char>(0)};
  }

  // Extract out the integer and exponent such that,
  // x = ix * 2 ** exp.
  uint64_t ix = flatten(x);
  int exp = static_cast<int>((ix >> FP64_EXP_SHIFT) & FP64_EXP_MASK) -
            FP64_EXP_BIAS - FP64_EXP_SHIFT;
  ix = ~(FP64_EXP_MASK << FP64_EXP_SHIFT) & ix;
  ix |= uint64_t(1) << FP64_EXP_SHIFT;

  // Use the exponent to extract the 3 appropriate uint64 digits from MPI4,
  uint64_t digit = uint64_t(exp + 61) / 64;
  uint64_t bitshift = uint64_t(exp + 61) % 64;

  uint64_t z0 = (MPI4[digit] << bitshift) |
                // A shift of size >= type size (e.g. 64)
                // is undefined behavior in C.
                (bitshift == 0 ? 0 : (MPI4[digit + 1] >> (64 - bitshift)));
  uint64_t z1 = (MPI4[digit + 1] << bitshift) |
                (bitshift == 0 ? 0 : (MPI4[digit + 2] >> (64 - bitshift)));
  uint64_t z2 = (MPI4[digit + 2] << bitshift) |
                (bitshift == 0 ? 0 : (MPI4[digit + 3] >> (64 - bitshift)));

  // Multiply mantissa by the digits and extract the upper two digits (hi, lo).
  absl::uint128 z2_prod =
      static_cast<absl::uint128>(z2) * static_cast<absl::uint128>(ix);
  uint64_t z2hi = absl::Uint128High64(z2_prod);
  absl::uint128 z1_prod =
      static_cast<absl::uint128>(z1) * static_cast<absl::uint128>(ix);
  uint64_t z1hi = absl::Uint128High64(z1_prod);
  uint64_t z1lo = absl::Uint128Low64(z1_prod);
  uint64_t z0lo = z0 * ix;

  // We probably don't have to break up z1 to do this, but since this
  // code is just for testing, we will follow the source as closely as possible
  absl::uint128 z_sum = static_cast<absl::uint128>(z1lo) +
                        static_cast<absl::uint128>(z2hi) +
                        (static_cast<absl::uint128>(z1hi) << 64) +
                        (static_cast<absl::uint128>(z0lo) << 64);
  uint64_t hi = absl::Uint128High64(z_sum);
  uint64_t lo = absl::Uint128Low64(z_sum);

  // The top 3 bits are j.
  uint8_t j = hi >> 61;
  // Extract the fraction and find its magnitude.
  hi = hi << 3 | lo >> 61;
  // Note: This shift is not present in the go implementation, but
  // this appears to be an error. Without this shift, we duplicate
  // the 3 MSBs of lo in the fraction (assuming there is enough shifting
  // for this error to be exposed). Further, any bits visible after this
  // would also be incorrect because they have been shifted 3 places to
  // the right.
  lo = lo << 3;

  uint64_t lz = clz(hi);
  uint64_t new_exp = FP64_EXP_BIAS - (lz + 1);

  // Clear implicit mantissa bit and shift into place.
  hi = (hi << (lz + 1)) | (lo >> (64 - (lz + 1)));

  hi = hi >> (64 - FP64_EXP_SHIFT);

  // Include the exponent and convert to a float.
  hi |= new_exp << FP64_EXP_SHIFT;

  double z = unflatten(hi);
  if ((j & 1) == 1) {
    j++;
    j &= 7;
    z--;
  }

  // Multiply the fractional part by pi/4.
  z = z * M_PI_4;
  return ResultT{.fraction = z, .j = j};
}

// Computes FP addition via DSLX & the JIT.
ResultT ComputeActual(fp::FpTrigReduce* jit_wrapper, double x) {
  PackedFloat64 packed_x(reinterpret_cast<uint8_t*>(&x), 5);
  ResultT result;
  result.j = 0x0;
  result.fraction = 0.0;
  PackedU3Float64 packed_result(reinterpret_cast<uint8_t*>(&result), 0);
  CHECK_OK(jit_wrapper->Run(packed_x, packed_result));
  result.j = result.j & 7;
  return result;
}

// Compares expected vs. actual results.
bool CompareResults(ResultT a, ResultT b) {
  return ((a.j & 7) == (b.j & 7)) && (a.fraction == b.fraction);
}

absl::Status RealMain(bool use_opt_ir, uint64_t num_samples, int num_threads) {
  TestbenchBuilder<double, ResultT, fp::FpTrigReduce> builder(
      ComputeExpected, ComputeActual,
      []() { return fp::FpTrigReduce::Create().value(); });
  builder.SetIndexToInputFn(IndexToInput).SetCompareResultsFn(CompareResults);
  if (num_threads != 0) {
    builder.SetNumThreads(num_threads);
  }
  if (num_samples != 0) {
    builder.SetNumSamples(num_samples);
  }
  return builder.Build().Run();
}

}  // namespace xls

int main(int argc, char** argv) {
  xls::InitXls(argv[0], argc, argv);
  QCHECK_OK(xls::RealMain(absl::GetFlag(FLAGS_use_opt_ir),
                          absl::GetFlag(FLAGS_num_samples),
                          absl::GetFlag(FLAGS_num_threads)));
  return 0;
}
