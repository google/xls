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

// In the decoding of JPEGs there is fundamentally a stream of bytes from the
// start of the file to the end. Metadata comes at the start of the file in the
// form of segments delimited by markers, generally encoded in 1x8-bit, 2x4-bit,
// or 1x16-bit value wise pieces arranged in sequence according to the grammar
// of JPEG files.
//
// Once the metadata has been established the decoding enters the "scan data"
// segment -- this is typically the trailing segment and contains Huffman-coded
// variable-length pieces of data that must be unfurled (sequentially due to
// variable length prefixes) and turned into their fixed forms (YCbCr color
// space "minimum coded unit" 8x8 blocks of pixel data).
//
// Since the scan data comes in chunks that are a variable number of bits, the
// ByteStream is adapted with a BitStream layer, to more easily process the
// variable-bit-length scan data out of the byte-based underlying stream.
//
// Some helpful references for more context:
//
// * JPEG Still Image Compression Standard (1993) (which is a giant pink book
//   which also contains a draft of the JPEG spec)
// * "the spec" CCITT T.81 (1992): https://www.w3.org/Graphics/JPEG/itu-t81.pdf
// * JFIF spec (which augments / refines the JPEG standard document):
//   https://www.w3.org/Graphics/JPEG/jfif3.pdf
// * Helpful walkthrough content such as "Everything You Need to Know About
//   JPEG":
//   https://www.youtube.com/watch?v=Sls8zdGU4cQ&list=PLpsTn9TA_Q8VMDyOPrDKmSJYt1DLgDZU4

#ifndef XLS_EXAMPLES_JPEG_STREAMS_H_
#define XLS_EXAMPLES_JPEG_STREAMS_H_

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"

namespace xls::jpeg {

// Returns a status that indicates an unexpected End of File was encountered in
// the input stream, with the given message "s".
absl::Status EofError(std::string s);

// Returns a status that indicates some aspect of the contents of the file being
// scanned was malformed according to what we accept, which is based on the
// definitions in the JPEG specification; e.g. since we only accept JFIF
// standard JPEG files, a file that was lacking the appropriate JFIF indicator
// at the start would be considered malformed.
absl::Status MalformedInputError(std::string s);

// Pops bytes out of a span and presents a function interface, e.g. for use in
// test cases.
class SpanPopper {
 public:
  explicit SpanPopper(absl::Span<const uint8_t> bytes) : bytes_(bytes) {}

  // Pops the next byte in the underlying span; when the limit is reached,
  // returns nullopt. Does not return a Status, but StatusOr is the return type
  // to conform to the popper interface used by ByteStream below.
  absl::StatusOr<std::optional<uint8_t>> operator()();

 private:
  absl::Span<const uint8_t> bytes_;
  int64_t i_ = 0;
};

class ByteStream {
 public:
  explicit ByteStream(
      std::function<absl::StatusOr<std::optional<uint8_t>>()> pop);

  // Drops the target "want" value from the head of the underlying byte stream,
  // or returns a MalformedInputError status that includes "message".
  absl::Status DropExpected(uint8_t want, std::string_view message);

  // Convenience wrapper that calls `DropExpected()` above for each of the byte
  // values in "want" in sequence.
  absl::Status DropExpectedMulti(absl::Span<const uint8_t> want,
                                 std::string_view message);

  // Pops a byte from the head of the underlying byte stream.
  //
  // Returns an error status if the end of the underlying byte stream has been
  // reached.
  absl::StatusOr<uint8_t> Pop();

  // Pops a compile-time-constant number of bytes "N" into a std::array of bytes
  // sized N.
  template <int N>
  absl::StatusOr<std::array<uint8_t, N>> PopN() {
    std::array<uint8_t, N> popped;
    for (int i = 0; i < N; ++i) {
      XLS_ASSIGN_OR_RETURN(uint8_t x, Pop());
      popped[i] = x;
    }
    return popped;
  }

  // Convenience wrapper for popping the next N bytes via `Pop()`.
  absl::StatusOr<std::vector<uint8_t>> PopN(int n);

  // Convenience wrapper for popping a byte and breaking it into two four bit
  // values -- the pair is given as (most significant nibble, least significant
  // nibble).
  absl::StatusOr<std::pair<uint8_t, uint8_t>> Pop2xU4();

  // Performs a peek operation at a position that may be the end of file --
  // returns an optional value to indicate whether end of file was reached.
  absl::StatusOr<std::optional<uint8_t>> PeekEof();

  // Peek at a byte in the stream.
  absl::StatusOr<uint8_t> Peek();

  // Pops a 16-bit value from the stream (with the most significant byte
  // appearing first in the stream, as is the convention for JPEG files).
  absl::StatusOr<uint16_t> PopHiLo();

  absl::StatusOr<bool> AtEof();

  // Returns the index of the last byte that was popped from the stream.
  int32_t popped_index() const { return popped_index_; }

 private:
  // Drops the byte at the head of the byte stream.
  absl::Status Drop();

  // Lambda that pops an underlying byte stream.
  std::function<absl::StatusOr<std::optional<uint8_t>>()> pop_;

  std::optional<uint8_t> lookahead_;
  int32_t popped_index_ = 0;  // i.e. index of lookahead, if it is present.
  bool saw_end_ = false;
};

// See file level comment for context -- the BitStream is a layer on top of an
// underlying ByteStream that assists with processing variable-bit-length
// Huffman-encoded data.
class BitStream {
 public:
  explicit BitStream(ByteStream* byte_stream) : byte_stream_(byte_stream) {}

  // Returns whether the bit stream has encountered the end of the underlying
  // file.
  absl::StatusOr<bool> AtEof();

  // Peeks at the "n" bits at the head of the bit stream.
  //
  // Precondition: n <= 16
  absl::StatusOr<uint16_t> PeekN(uint8_t n);

  // Pops the "n" bits at the head of the bit stream into a 16-bit integer for
  // storage. The "n" bits that are popped are placed in the least significant
  // bits of the result value.
  //
  // Precondition: n <= 16
  absl::StatusOr<uint16_t> PopN(uint8_t n);

  // Pops a coefficient value from the "n" bits at the head of the bit stream.
  //
  // Coefficients are encoded per the helpful visualization in Table 11-4 in
  // chapter 11 of "JPEG: Still Comage Data Compression Standard" (1993) --
  // values are "spread" in exclusive ranges determined via bit counts around 0
  // -- with the binary pattern added as the rightmost column:
  //
  //  bit count              values                    binary
  //  ----------   ----------------------  ----------------------------------
  //      1      |           -1, 1        |             0, 1
  //      2      |       -3, -2, 2, 3     |        00, 01, 10, 11
  //      3      |  -7, ..., -4, 4, ... 7 | 000, ..., 011, 100, 101, 110, 111
  //      ..
  //      15
  //
  // Note that negative values correspond to those bit patterns which have 0 as
  // their most significant bit, and positive values have 1 as their most
  // significant bit.
  //
  // Precondition: n <= 15
  absl::StatusOr<int16_t> PopCoeff(uint8_t n);

 private:
  // "Pumps" one byte from the underlying byte stream into our lookahead data.
  //
  // Implementation note: the byte becomes the least significant bits of
  // lookahead_.
  //
  // Precondition: lookahead_bits_ + 8 <= 32 (we cannot overpopulate our
  // limited-size lookahead data integer used as storage).
  absl::Status LookaheadOneByte();

  ByteStream* byte_stream_;
  uint32_t lookahead_ = 0;
  uint8_t lookahead_bits_ = 0;
};

}  // namespace xls::jpeg

#endif  // XLS_EXAMPLES_JPEG_STREAMS_H_
