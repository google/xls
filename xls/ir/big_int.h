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

  BigInt();
  BigInt(const BigInt& other);
  BigInt(BigInt&& other);
  ~BigInt();
  BigInt& operator=(const BigInt& other);
  BigInt& operator=(BigInt&& other);

  bool operator==(const BigInt& other) const;
  bool operator!=(const BigInt& other) const { return !(*this == other); }

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
  static BigInt Mul(const BigInt& lhs, const BigInt& rhs);
  static BigInt Div(const BigInt& lhs, const BigInt& rhs);
  static BigInt Mod(const BigInt& lhs, const BigInt& rhs);
  static BigInt Negate(const BigInt& input);
  static bool LessThan(const BigInt& lhs, const BigInt& rhs);

 private:
  BIGNUM bn_;
};

std::ostream& operator<<(std::ostream& os, const BigInt& big_int);

}  // namespace xls

#endif  // XLS_IR_BIG_INT_H_
