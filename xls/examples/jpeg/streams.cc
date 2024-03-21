// Copyright 2021 The XLS Authors
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

#include "xls/examples/jpeg/streams.h"

#include <climits>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/examples/jpeg/constants.h"

namespace xls::jpeg {

absl::Status EofError(std::string s) {
  return absl::InvalidArgumentError(absl::StrCat("JPEG EofError: ", s));
}
absl::Status MalformedInputError(std::string s) {
  return absl::InvalidArgumentError(
      absl::StrCat("JPEG MalformedInputError: ", s));
}

// -- SpanPopper

absl::StatusOr<std::optional<uint8_t>> SpanPopper::operator()() {
  if (i_ >= bytes_.size()) {
    return std::nullopt;
  }
  return bytes_[i_++];
}

// -- ByteStream

ByteStream::ByteStream(
    std::function<absl::StatusOr<std::optional<uint8_t>>()> pop)
    : pop_(std::move(pop)) {}

absl::Status ByteStream::DropExpectedMulti(absl::Span<const uint8_t> want,
                                           std::string_view message) {
  for (int i = 0; i < want.size(); ++i) {
    XLS_RETURN_IF_ERROR(DropExpected(want[i], message));
  }
  return absl::OkStatus();
}

absl::StatusOr<uint16_t> ByteStream::PopHiLo() {
  XLS_ASSIGN_OR_RETURN(uint8_t hi, Pop());
  XLS_ASSIGN_OR_RETURN(uint8_t lo, Pop());
  return static_cast<uint16_t>(hi) << 8 | static_cast<uint16_t>(lo);
}

absl::StatusOr<bool> ByteStream::AtEof() {
  XLS_ASSIGN_OR_RETURN(std::optional<uint8_t> peek_eof, PeekEof());
  return !peek_eof.has_value();
}

absl::StatusOr<uint8_t> ByteStream::Peek() {
  XLS_ASSIGN_OR_RETURN(std::optional<uint8_t> peek_eof, PeekEof());
  if (peek_eof.has_value()) {
    return peek_eof.value();
  }
  return EofError("Unexpected end of JPEG byte stream");
}

absl::StatusOr<std::vector<uint8_t>> ByteStream::PopN(int n) {
  std::vector<uint8_t> popped;
  popped.reserve(n);
  for (int i = 0; i < n; ++i) {
    XLS_ASSIGN_OR_RETURN(uint8_t x, Pop());
    popped.push_back(x);
  }
  return popped;
}

absl::StatusOr<std::pair<uint8_t, uint8_t>> ByteStream::Pop2xU4() {
  XLS_ASSIGN_OR_RETURN(uint8_t b, Pop());
  return std::make_pair(b >> 4, b & 0xf);
}

absl::Status ByteStream::DropExpected(uint8_t want, std::string_view message) {
  XLS_ASSIGN_OR_RETURN(uint8_t got, Peek());
  if (got == want) {
    XLS_RETURN_IF_ERROR(Drop());
    return absl::OkStatus();
  }
  return MalformedInputError(
      absl::StrFormat("Expected %#x (%s) got %#x (%c) at byte index %#x", want,
                      message, got, got, popped_index_));
}

absl::StatusOr<uint8_t> ByteStream::Pop() {
  XLS_ASSIGN_OR_RETURN(uint8_t b, Peek());
  CHECK_OK(Drop());
  return b;
}

absl::StatusOr<std::optional<uint8_t>> ByteStream::PeekEof() {
  if (lookahead_.has_value()) {
    return lookahead_;
  }
  XLS_ASSIGN_OR_RETURN(std::optional<uint8_t> b, pop_());
  if (b.has_value()) {
    // Should never "see the end" then "not the end".
    XLS_RET_CHECK(!saw_end_);
    lookahead_ = b;
    return b;
  }

  saw_end_ = true;
  return b;
}

absl::Status ByteStream::Drop() {
  XLS_RETURN_IF_ERROR(Peek().status());
  XLS_RET_CHECK(lookahead_.has_value());
  lookahead_ = std::nullopt;
  popped_index_ += 1;
  return absl::OkStatus();
}

// -- BitStream

absl::StatusOr<bool> BitStream::AtEof() {
  XLS_ASSIGN_OR_RETURN(bool bytes_at_eof, byte_stream_->AtEof());
  return bytes_at_eof && lookahead_bits_ == 0;
}

absl::StatusOr<uint16_t> BitStream::PeekN(uint8_t n) {
  XLS_RET_CHECK_LE(n, 16);
  while (lookahead_bits_ < n) {
    XLS_ASSIGN_OR_RETURN(bool at_eof, AtEof());
    if (at_eof) {
      // We fill with 0xff bytes at the end of the stream.
      lookahead_ <<= 8;
      lookahead_ |= 0xff;
      lookahead_bits_ += 8;
    } else {
      XLS_RETURN_IF_ERROR(LookaheadOneByte());
    }
  }
  return static_cast<uint16_t>(lookahead_ >> (lookahead_bits_ - n));
}

absl::Status BitStream::LookaheadOneByte() {
  XLS_RET_CHECK_LE(lookahead_bits_ + 8, sizeof(lookahead_) * CHAR_BIT);
  XLS_ASSIGN_OR_RETURN(uint8_t byte, byte_stream_->Pop());
  lookahead_ <<= 8;
  lookahead_ |= byte;
  lookahead_bits_ += 8;
  if (byte == 0xff) {
    XLS_ASSIGN_OR_RETURN(uint8_t marker, byte_stream_->Pop());
    VLOG(3) << absl::StreamFormat("Bit stream encountered marker: %#02x",
                                  marker);
    // Not clear whether we ever have to do anything with this marker or we can
    // assume it's always effectively EOI.
    if (marker != kEoiMarker) {
      return MalformedInputError(absl::StrFormat(
          "Unexpected marker encountered in bit scan: %#x", marker));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<int16_t> BitStream::PopCoeff(uint8_t n) {
  XLS_RET_CHECK_LT(n, 16);
  if (n == 0) {
    return 0;
  }
  XLS_ASSIGN_OR_RETURN(uint16_t bits, PopN(n));
  bool msb_set = (bits >> (n - 1)) != 0;
  // Given the table in the header to project from a bit stream value (e.g. 0b00
  // to -3 in the two bit space, within the 16 bit value used as storage) we
  // have to set all the high bits above the bit pattern (which in the example
  // takes us to -4) and then add one. (You can see that two's complement
  // negation clearly doesn't do what we want here, since negating zero doesn't
  // get us to -4!)
  uint16_t adjusted =
      msb_set ? bits : (bits | (static_cast<uint16_t>(-1) << n)) + 1;
  return static_cast<int16_t>(adjusted);
}

absl::StatusOr<uint16_t> BitStream::PopN(uint8_t n) {
  if (n == 0) {
    return 0;
  }
  XLS_RET_CHECK_LE(n, 16);
  XLS_RETURN_IF_ERROR(PeekN(n).status());
  XLS_RET_CHECK_LE(n, lookahead_bits_);
  uint32_t result = lookahead_ >> (lookahead_bits_ - n);
  lookahead_bits_ -= n;
  lookahead_ = lookahead_ & ((1 << lookahead_bits_) - 1);
  XLS_RET_CHECK_EQ(result & 0xffff, result);
  return static_cast<uint16_t>(result);
}

}  // namespace xls::jpeg
