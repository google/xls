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

#include "xls/ir/big_int.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "openssl/base.h"
#include "openssl/bn.h"
#include "openssl/mem.h"
#include "xls/common/bits_util.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/bits.h"

namespace xls {

BigInt::BigInt() { BN_init(&bn_); }

BigInt::BigInt(const BigInt& other) {
  BN_init(&bn_);
  CHECK(BN_copy(&bn_, &other.bn_));
}

BigInt::BigInt(BigInt&& other) {
  memcpy(&bn_, &other.bn_, sizeof(bn_));
  BN_init(&other.bn_);
}

BigInt::~BigInt() { BN_free(&bn_); }

BigInt& BigInt::operator=(const BigInt& other) {
  BN_copy(&bn_, &other.bn_);
  return *this;
}

BigInt& BigInt::operator=(BigInt&& other) {
  CHECK(BN_copy(&bn_, &other.bn_));
  return *this;
}

bool BigInt::operator==(const BigInt& other) const {
  return BN_cmp(&bn_, &other.bn_) == 0;
}

/* static */ BigInt BigInt::MakeUnsigned(const Bits& bits) {
  BigInt value;
  std::vector<uint8_t> byte_vector = bits.ToBytes();
  // ToBytes returns in little-endian. BN expects big-endian.
  std::reverse(byte_vector.begin(), byte_vector.end());
  CHECK(BN_bin2bn(byte_vector.data(), byte_vector.size(), &value.bn_));
  return value;
}

/* static */ BigInt BigInt::MakeSigned(const Bits& bits) {
  if (bits.bit_count() == 0 || !bits.msb()) {
    return MakeUnsigned(bits);
  }
  BigInt value;
  // 'bits' is a twos-complement negative number, invert the bits and add one to
  // get the magnitude. Then negate it to produce the correct value in the
  // BigInt.
  std::vector<uint8_t> byte_vector = bits.ToBytes();
  // ToBytes returns in little-endian. BN expects big-endian.
  std::reverse(byte_vector.begin(), byte_vector.end());
  for (auto& byte : byte_vector) {
    byte = ~byte;
  }
  // ToBytes pads the most significant bits with zeroes to fit the return value
  // in whole bytes. Set them back to zero after the bit inversion above.
  int bits_in_msb = bits.bit_count() % 8;
  if (bits_in_msb) {  // There are pad bits.
    byte_vector[0] &= Mask(bits_in_msb);
  }

  CHECK(BN_bin2bn(byte_vector.data(), byte_vector.size(), &value.bn_));
  CHECK(BN_add_word(&value.bn_, 1));
  BN_set_negative(&value.bn_, 1);
  return value;
}

/* static */ BigInt BigInt::Zero() {
  BigInt value;
  BN_zero(&value.bn_);
  return value;
}

/* static */ BigInt BigInt::One() {
  BigInt value;
  CHECK(BN_one(&value.bn_));
  return value;
}

Bits BigInt::ToSignedBits() const {
  if (BN_is_zero(&bn_)) {
    return Bits();
  }
  bool is_negative = BN_is_negative(&bn_);

  // In twos-complement, negative values are stored as their positive
  // counterpart - 1, bit inverted. First compute the positive counterpart - 1.
  BigInt decremented_if_negative = *this;
  BN_set_negative(&decremented_if_negative.bn_, 0);
  if (is_negative) {
    CHECK(BN_sub_word(&decremented_if_negative.bn_, 1));
  }

  std::vector<uint8_t> byte_vector;
  byte_vector.resize(BN_num_bytes(&decremented_if_negative.bn_));
  CHECK(BN_bn2bin_padded(byte_vector.data(), byte_vector.size(),
                         &decremented_if_negative.bn_));

  if (is_negative) {
    for (uint8_t& byte : byte_vector) {
      byte = ~byte;
    }
  }

  int64_t result_bit_count = SignedBitCount();
  BitsRope rope(result_bit_count);
  // FromBytes expects bytes in little endian order.
  std::reverse(byte_vector.begin(), byte_vector.end());
  rope.push_back(Bits::FromBytes(byte_vector, result_bit_count - 1));
  rope.push_back(is_negative ? Bits::AllOnes(1) : Bits(1));
  return rope.Build();
}

Bits BigInt::ToUnsignedBits() const {
  CHECK(!BN_is_negative(&bn_));
  int64_t bit_count = BN_num_bits(&bn_);
  std::vector<uint8_t> byte_vector;
  byte_vector.resize(BN_num_bytes(&bn_));

  CHECK(BN_bn2bin_padded(byte_vector.data(), byte_vector.size(), &bn_));
  // FromBytes expects bytes in little endian order.
  std::reverse(byte_vector.begin(), byte_vector.end());
  return Bits::FromBytes(byte_vector, bit_count);
}

std::string BigInt::ToDecimalString() const {
  char* s = BN_bn2dec(&bn_);
  CHECK(s != nullptr) << "BigNum allocation failure";
  std::string str(s);
  OPENSSL_free(s);
  return str;
}

std::string BigInt::ToHexString() const {
  char* s = BN_bn2hex(&bn_);
  CHECK(s != nullptr) << "BigNum allocation failure";
  std::string str(s);
  OPENSSL_free(s);
  return str;
}

absl::StatusOr<Bits> BigInt::ToSignedBitsWithBitCount(int64_t bit_count) const {
  int64_t min_bit_count = SignedBitCount();
  if (bit_count < min_bit_count) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Specified bit count (%d) is less than minimum required (%d)!",
        bit_count, min_bit_count));
  }

  // Can't use bits_ops due to circular dependency.
  BitsRope rope(bit_count);
  Bits bits = ToSignedBits();
  rope.push_back(bits);
  if (BN_is_negative(&bn_)) {
    rope.push_back(Bits::AllOnes(bit_count - min_bit_count));
  } else {
    rope.push_back(Bits(bit_count - min_bit_count));
  }
  return rope.Build();
}

absl::StatusOr<Bits> BigInt::ToUnsignedBitsWithBitCount(
    int64_t bit_count) const {
  int64_t min_bit_count = UnsignedBitCount();
  if (bit_count < min_bit_count) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Specified bit count (%d) is less than minimum required (%d)!",
        bit_count, min_bit_count));
  }

  // Can't use bits_ops due to circular dependency.
  BitsRope rope(bit_count);
  Bits bits = ToUnsignedBits();
  rope.push_back(bits);
  rope.push_back(Bits(bit_count - min_bit_count));
  return rope.Build();
}

int64_t BigInt::SignedBitCount() const {
  if (BN_is_zero(&bn_)) {
    return 0;
  }
  if (BN_is_negative(&bn_)) {
    // Value is negative. If it only has a single bit set (eg, 0x8000) then the
    // minimum number of bits to represent the value in twos-complement is the
    // same as the minimum number of bits for the unsigned representation of
    // the magnitude. Otherwise, it requires one extra bit.
    BigInt magnitude = Negate(*this);
    CHECK(BN_sub_word(&magnitude.bn_, 1));
    return BN_num_bits(&magnitude.bn_) + 1;
  }

  // Value is positive. Twos complement requires a leading zero.
  return BN_num_bits(&bn_) + 1;
}

int64_t BigInt::UnsignedBitCount() const {
  CHECK_EQ(BN_is_negative(&bn_), 0) << "Value must be non-negative.";
  if (BN_is_negative(&bn_)) {
    return 0;
  }
  return BN_num_bits(&bn_);
}

/* static */ BigInt BigInt::Add(const BigInt& lhs, const BigInt& rhs) {
  BigInt value;
  CHECK(BN_add(&value.bn_, &lhs.bn_, &rhs.bn_));
  return value;
}

/* static */ BigInt BigInt::Sub(const BigInt& lhs, const BigInt& rhs) {
  BigInt value;
  CHECK(BN_sub(&value.bn_, &lhs.bn_, &rhs.bn_));
  return value;
}

/* static */ BigInt BigInt::Mul(const BigInt& lhs, const BigInt& rhs) {
  BigInt value;
  // Note: The documentation about BN_CTX in bn.h indicates that it's possible
  // to pass null to public methods that take a BN_CTX*, but that's not true.
  BN_CTX* ctx = BN_CTX_new();
  CHECK(ctx);
  CHECK(BN_mul(&value.bn_, &lhs.bn_, &rhs.bn_, ctx));
  BN_CTX_free(ctx);
  return value;
}

/* static */ BigInt BigInt::Div(const BigInt& lhs, const BigInt& rhs) {
  BigInt value;
  // Note: The documentation about BN_CTX in bn.h indicates that it's possible
  // to pass null to public methods that take a BN_CTX*, but that's not true.
  BN_CTX* ctx = BN_CTX_new();
  CHECK(ctx);
  CHECK(BN_div(&value.bn_, /*rem=*/nullptr, &lhs.bn_, &rhs.bn_, ctx));
  BN_CTX_free(ctx);
  return value;
}

/* static */ BigInt BigInt::Mod(const BigInt& lhs, const BigInt& rhs) {
  BigInt value;
  // Note: The documentation about BN_CTX in bn.h indicates that it's possible
  // to pass null to public methods that take a BN_CTX*, but that's not true.
  BN_CTX* ctx = BN_CTX_new();
  CHECK(ctx);
  CHECK(BN_div(/*quotient=*/nullptr, /*rem=*/&value.bn_, &lhs.bn_, &rhs.bn_,
               ctx));
  BN_CTX_free(ctx);
  return value;
}

/* static */ BigInt BigInt::Exp2(int64_t e) {
  CHECK_GE(e, 0);
  BigInt one = BigInt::One();
  if (e == 0) {
    return one;
  }

  CHECK_LE(e, int64_t{std::numeric_limits<int>::max()});

  BigInt value;
  BN_lshift(&value.bn_, &one.bn_, static_cast<int>(e));
  return value;
}

/* static */ BigInt BigInt::Negate(const BigInt& input) {
  BigInt value = input;
  BN_set_negative(&value.bn_, !BN_is_negative(&value.bn_));
  return value;
}

/* static */ BigInt BigInt::Absolute(const BigInt& input) {
  BigInt r = input;
  BN_set_negative(&r.bn_, 0);
  return r;
}

/* static */ bool BigInt::LessThan(const BigInt& lhs, const BigInt& rhs) {
  return BN_cmp(&lhs.bn_, &rhs.bn_) < 0;
}

/* static */ bool BigInt::GreaterThan(const BigInt& lhs, const BigInt& rhs) {
  return BN_cmp(&lhs.bn_, &rhs.bn_) > 0;
}

bool BigInt::operator<=(const BigInt& rhs) const {
  return BN_cmp(&this->bn_, &rhs.bn_) <= 0;
}

bool BigInt::operator>=(const BigInt& rhs) const {
  return BN_cmp(&this->bn_, &rhs.bn_) >= 0;
}

/* static */ bool BigInt::IsEven(const BigInt& input) {
  return BN_is_odd(&input.bn_) == 0;
}

/* static */ bool BigInt::IsPowerOfTwo(const BigInt& input) {
  if (input < BigInt::One()) {
    return false;
  }

  const int64_t num_significant_bits = BN_num_bits(&input.bn_);
  return BigInt::Exp2(num_significant_bits - 1) == input;
}

/* static */ int64_t BigInt::CeilingLog2(const BigInt& input) {
  CHECK(!BN_is_negative(&input.bn_));

  if (input == BigInt::Zero()) {
    return std::numeric_limits<int64_t>::min();
  }

  if (input == BigInt::One()) {
    return 0;
  }

  const int64_t num_significant_bits = BN_num_bits(&input.bn_);
  if (BigInt::Exp2(num_significant_bits - 1) == input) {
    return num_significant_bits - 1;
  }

  return num_significant_bits;
}

/* static */ std::tuple<BigInt, int64_t> BigInt::FactorizePowerOfTwo(
    const BigInt& input) {
  const BigInt zero = BigInt::Zero();
  const BigInt two = BigInt::Exp2(1);
  int64_t power = 0;
  BigInt x = input;
  while (IsEven(x) && x != zero) {
    x = x / two;
    power = power + 1;
  }
  return std::make_tuple(x, power);
}

std::ostream& operator<<(std::ostream& os, const BigInt& big_int) {
  if (BigInt::LessThan(big_int, BigInt())) {
    os << "-" << BigInt::Negate(big_int).ToUnsignedBits().ToDebugString();
  } else {
    os << big_int.ToUnsignedBits().ToDebugString();
  }
  return os;
}

}  // namespace xls
