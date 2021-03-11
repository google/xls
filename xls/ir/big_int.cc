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

#include "absl/status/statusor.h"
#include "openssl/bn.h"
#include "xls/common/logging/logging.h"

namespace xls {

BigInt::BigInt() { BN_init(&bn_); }

BigInt::BigInt(const BigInt& other) {
  BN_init(&bn_);
  BN_copy(&bn_, &other.bn_);
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
  memcpy(&bn_, &other.bn_, sizeof(bn_));
  BN_init(&other.bn_);
  return *this;
}

bool BigInt::operator==(const BigInt& other) const {
  return BN_cmp(&bn_, &other.bn_) == 0;
}

/* static */
BigInt BigInt::MakeUnsigned(const Bits& bits) {
  BigInt value;
  std::vector<uint8_t> byte_vector = bits.ToBytes();
  BN_bin2bn(byte_vector.data(), byte_vector.size(), &value.bn_);
  return value;
}

/* static */
BigInt BigInt::MakeSigned(const Bits& bits) {
  if (bits.bit_count() == 0 || !bits.msb()) {
    return MakeUnsigned(bits);
  }
  BigInt value;
  // 'bits' is a twos-complement negative number, invert the bits and add one to
  // get the magnitude. Then negate it to produce the correct value in the
  // BigInt.
  std::vector<uint8_t> byte_vector = bits.ToBytes();
  for (auto& byte : byte_vector) {
    byte = ~byte;
  }
  // ToBytes pads the most significant bits with zeroes to fit the return value
  // in whole bytes. Set them back to zero after the bit inversion above.
  int bits_in_msb = bits.bit_count() % 8;
  if (bits_in_msb) {  // There are pad bits.
    byte_vector[0] &= Mask(bits_in_msb);
  }

  BN_bin2bn(byte_vector.data(), byte_vector.size(), &value.bn_);
  BN_add_word(&value.bn_, 1);
  BN_set_negative(&value.bn_, 1);
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
    BN_sub_word(&decremented_if_negative.bn_, 1);
  }

  std::vector<uint8_t> byte_vector;
  byte_vector.resize(BN_num_bytes(&decremented_if_negative.bn_));
  XLS_CHECK(BN_bn2bin_padded(byte_vector.data(), byte_vector.size(),
                             &decremented_if_negative.bn_));

  if (is_negative) {
    for (uint8_t& byte : byte_vector) {
      byte = ~byte;
    }
  }

  int64_t result_bit_count = SignedBitCount();
  BitsRope rope(result_bit_count);
  rope.push_back(Bits::FromBytes(byte_vector, result_bit_count - 1));
  rope.push_back(is_negative ? Bits::AllOnes(1) : Bits(1));
  return rope.Build();
}

Bits BigInt::ToUnsignedBits() const {
  XLS_CHECK(!BN_is_negative(&bn_));
  int64_t bit_count = BN_num_bits(&bn_);
  std::vector<uint8_t> byte_vector;
  byte_vector.resize(BN_num_bytes(&bn_));

  XLS_CHECK(BN_bn2bin_padded(byte_vector.data(), byte_vector.size(), &bn_));
  return Bits::FromBytes(byte_vector, bit_count);
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
    BN_sub_word(&magnitude.bn_, 1);
    return BN_num_bits(&magnitude.bn_) + 1;
  }

  // Value is positive. Twos complement requires a leading zero.
  return BN_num_bits(&bn_) + 1;
}

int64_t BigInt::UnsignedBitCount() const {
  XLS_CHECK_EQ(BN_is_negative(&bn_), 0) << "Value must be non-negative.";
  if (BN_is_negative(&bn_)) {
    return 0;
  }
  return BN_num_bits(&bn_);
}

/* static */ BigInt BigInt::Add(const BigInt& lhs, const BigInt& rhs) {
  BigInt value;
  BN_add(&value.bn_, &lhs.bn_, &rhs.bn_);
  return value;
}

/* static */ BigInt BigInt::Sub(const BigInt& lhs, const BigInt& rhs) {
  BigInt value;
  BN_sub(&value.bn_, &lhs.bn_, &rhs.bn_);
  return value;
}

/* static */ BigInt BigInt::Mul(const BigInt& lhs, const BigInt& rhs) {
  BigInt value;
  // Note: The documentation about BN_CTX in bn.h indicates that it's possible
  // to pass null to public methods that take a BN_CTX*, but that's not true.
  BN_CTX* ctx = BN_CTX_new();
  BN_mul(&value.bn_, &lhs.bn_, &rhs.bn_, ctx);
  BN_CTX_free(ctx);
  return value;
}

/* static */ BigInt BigInt::Div(const BigInt& lhs, const BigInt& rhs) {
  BigInt value;
  // Note: The documentation about BN_CTX in bn.h indicates that it's possible
  // to pass null to public methods that take a BN_CTX*, but that's not true.
  BN_CTX* ctx = BN_CTX_new();
  BN_div(&value.bn_, /*rem=*/nullptr, &lhs.bn_, &rhs.bn_, ctx);
  BN_CTX_free(ctx);
  return value;
}

/* static */ BigInt BigInt::Mod(const BigInt& lhs, const BigInt& rhs) {
  BigInt value;
  // Note: The documentation about BN_CTX in bn.h indicates that it's possible
  // to pass null to public methods that take a BN_CTX*, but that's not true.
  BN_CTX* ctx = BN_CTX_new();
  BN_div(/*quotient=*/nullptr, /*rem=*/&value.bn_, &lhs.bn_, &rhs.bn_, ctx);
  BN_CTX_free(ctx);
  return value;
}

/* static */ BigInt BigInt::Negate(const BigInt& input) {
  BigInt value = input;
  BN_set_negative(&value.bn_, !BN_is_negative(&value.bn_));
  return value;
}

/* static */ bool BigInt::LessThan(const BigInt& lhs, const BigInt& rhs) {
  return BN_cmp(&lhs.bn_, &rhs.bn_) < 0;
}

std::ostream& operator<<(std::ostream& os, const BigInt& big_int) {
  if (BigInt::LessThan(big_int, BigInt())) {
    os << "-"
       << BigInt::Negate(big_int).ToUnsignedBits().ToString(
              FormatPreference::kHex, /*include_bit_count=*/true);
  } else {
    os << big_int.ToUnsignedBits().ToString(FormatPreference::kHex,
                                            /*include_bit_count=*/true);
  }
  return os;
}

}  // namespace xls
