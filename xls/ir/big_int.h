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

#ifndef XLS_IR_BIG_INT_H_
#define XLS_IR_BIG_INT_H_

#include <cstdint>
#include <ostream>
#include <string>
#include <tuple>

#include "absl/status/statusor.h"
#include "openssl/bn.h"
#include "xls/ir/bits.h"

namespace xls {

// Class which wraps OpenSSL's bignum library to provide support for arbitrary
// width integer arithmetic operations.
class BigInt {
 public:
  // Make (un)signed BigInt from the given bits object. MakeSigned assumes a
  // twos-complement representation.
  static BigInt MakeSigned(const Bits& bits);
  static BigInt MakeUnsigned(const Bits& bits);

  // Returns 0 (as a BigInt).
  static BigInt Zero();

  // Returns 1 (as a BigInt).
  static BigInt One();

  BigInt();
  BigInt(const BigInt& other);
  BigInt(BigInt&& other);
  ~BigInt();
  BigInt& operator=(const BigInt& other);
  BigInt& operator=(BigInt&& other);

  bool operator==(const BigInt& other) const;
  bool operator!=(const BigInt& other) const { return !(*this == other); }

  std::string ToDecimalString() const;
  std::string ToHexString() const;

  // Returns the BigInt value as a (un)signed Bits object. The Bits object
  // returned from ToSignedBits is in twos-complement representation. If a width
  // is unspecified, then the Bits object has the minimum number of bits
  // required to hold the value.
  Bits ToSignedBits() const;
  Bits ToUnsignedBits() const;

  // Returns the BigInt value as a Bits object of the specified width. Returns
  // an error if the value doesn't fit in the specified bit count.
  absl::StatusOr<Bits> ToSignedBitsWithBitCount(int64_t bit_count) const;
  absl::StatusOr<Bits> ToUnsignedBitsWithBitCount(int64_t bit_count) const;

  // Returns the minimum number of bits required to hold this BigInt value. For
  // SignedBitCount this is the number of bits required to hold the value in
  // twos-complement representation.
  int64_t SignedBitCount() const;
  int64_t UnsignedBitCount() const;

  // Various arithmetic and comparison operations.
  static BigInt Add(const BigInt& lhs, const BigInt& rhs);
  static BigInt Sub(const BigInt& lhs, const BigInt& rhs);
  static BigInt Negate(const BigInt& input);

  // Returns absolute value.
  static BigInt Absolute(const BigInt& input);

  static BigInt Mul(const BigInt& lhs, const BigInt& rhs);
  static BigInt Div(const BigInt& lhs, const BigInt& rhs);
  static BigInt Mod(const BigInt& lhs, const BigInt& rhs);

  static bool LessThan(const BigInt& lhs, const BigInt& rhs);
  static bool GreaterThan(const BigInt& lhs, const BigInt& rhs);

  // Returns true when input is even.
  static bool IsEven(const BigInt& input);

  // Returns true when input is a power of two. Otherwise returns false,
  // including when input is non-positive.
  static bool IsPowerOfTwo(const BigInt& input);

  // Returns ceiling(logbase2(input)). Input must be non-negative.
  // CeilingLog2(0) returns most-negative int64_t.
  static int64_t CeilingLog2(const BigInt& input);

  // Returns (odd, y) such that input = odd * 2^y. odd has the same sign as
  // input.
  //
  // That is, factorizes input into an odd number and a power of two. Returns
  // the odd number and the exponent.
  //
  // Special case: FactorizePowerOfTwo(0) = 0 * 2^0
  static std::tuple<BigInt, int64_t> FactorizePowerOfTwo(const BigInt& input);

  // Operator overloads
  BigInt operator+(const BigInt& rhs) const { return Add(*this, rhs); }
  BigInt operator-(const BigInt& rhs) const { return Sub(*this, rhs); }
  BigInt operator*(const BigInt& rhs) const { return Mul(*this, rhs); }
  BigInt operator/(const BigInt& rhs) const { return Div(*this, rhs); }
  bool operator<(const BigInt& rhs) const { return LessThan(*this, rhs); }
  bool operator<=(const BigInt& rhs) const;
  bool operator>(const BigInt& rhs) const { return GreaterThan(*this, rhs); }
  bool operator>=(const BigInt& rhs) const;

  // Returns 2^e
  static BigInt Exp2(int64_t e);

 private:
  BIGNUM bn_{};
};

std::ostream& operator<<(std::ostream& os, const BigInt& big_int);

}  // namespace xls

#endif  // XLS_IR_BIG_INT_H_
