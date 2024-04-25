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

// This is the "golden reference model" for our JPEG decoding functionality --
// it's not optimized for speed, just attempts to give implementation clarity
// and a basis for comparison for our DSL version as that comes online, and
// helps us establish unit tests for cross checking functionality.
//
// See streams.h file-level comment for references on the JPEG decoding process.

#ifndef XLS_EXAMPLES_JPEG_JPEG_GRM_H_
#define XLS_EXAMPLES_JPEG_JPEG_GRM_H_

#include <array>
#include <cstdint>
#include <ostream>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/examples/jpeg/constants.h"
#include "xls/examples/jpeg/streams.h"

namespace xls::jpeg {

struct Rgb {
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

// Helpers (equality, stream formatting) for testing with Rgb values.
bool operator==(const Rgb& lhs, const Rgb& rhs);
std::ostream& operator<<(std::ostream& os, const Rgb& rgb);

// Represents the result of decoding a JPEG.
struct DecodedJpeg {
  uint16_t y;
  uint16_t x;
  // Note: this is a 2D array as an invariant (i.e. it should not be ragged, all
  // lines should have the same `.size()`, which is the "number of samples per
  // line");
  std::vector<std::vector<Rgb>> lines;

  // Returns the number of pixels in the image by inspecting the metadata about
  // the image.
  int64_t GetPixelCount() const { return y * x; }

  // Returns the total number of pixels in the image by inspecting the value
  // storage. Once the JPEG is decoded this should be the same as
  // `GetPixelCount()` as an invariant.
  int64_t GetPixelCountViaLineData() const {
    if (lines.empty()) {
      return 0;
    }
    return lines.size() * lines[0].size();
  }
};

// Decodes a JPEG from the given byte strema or returns an error.
absl::StatusOr<DecodedJpeg> DecodeJpeg(ByteStream& byte_stream);

// Internal namespace is exposed in the header for unit testing purposes.
namespace internal {

// Note: conversion is from JFIF specification
// https://www.w3.org/Graphics/JPEG/jfif3.pdf page 3 -- notably Y has 256 levels
// [0, 255] while the chrominance channels (Cb and Cr) are offset by 128, so the
// [0, 255] values correspond to [-.5, .5] in the real number line. This implies
// that greyscale colors lacking in chrominance are given as (X, 128, 128).
Rgb ToRgb(uint8_t y, uint8_t cb, uint8_t cr);

// Notes whether the Huffman table applies to the DC information (no frequency
// variation within an MCU) or AC information (the components varying by
// frequency within the MCU).
enum class HuffmanTableClass {
  kDc,
  kAc,
};

// An "expanded" value from the Huffman metadata encoded in the JPEG stream --
// the order in which entries are placed in the stream help us expand the 16-bit
// "code" value -- if the lookahead in the bit stream matches the "code" value,
// this entry will be the (unique) match in the Huffman table.
struct PrefixEntry {
  // The number of effective bits in "code". Must be in range [1, 16].
  uint8_t bits;

  // The prefix code to match on -- the effective bits in the code will be
  // placed in the LSbs of this 16-bit storage.
  uint16_t code;

  // The value given in the metadata, which concatenates two 4-bit values of
  // "leading zeros" and "bits to pop", referred to as RRRR and SSSS in the
  // spec.
  uint8_t value;

  // Returns the upper nibble of "value" which indicates the number of leading
  // zeros to skip in coefficients before decoding the value indicated by
  // `GetBitsToPop()`.
  uint8_t GetLeadingZeros() const { return value >> 4; }

  // Returns the lower nibble of "value" which indicates the number of bits to
  // pop from the bit stream.
  uint8_t GetBitsToPop() const { return value & 0xf; }
};

// Stores data for a Huffman decoding table, as extracted from the byte stream.
//
// It is postprocessed via HuffmanTableExpandEntries once "entries" has been
// populated by the JPEG metadata to create more easily matched-on PrefixEntry
// records.
struct HuffmanTable {
  HuffmanTableClass table_class;
  uint8_t table_index;
  std::array<std::vector<uint8_t>, kHuffmanCodeSizeLimit> entries;
  std::vector<PrefixEntry> expanded;
};

// Expands the (compact) Huffman tables received from the byte stream into
// entries with the prefix bit patterns that we can more easily prefix match on.
absl::Status HuffmanTableExpandEntries(HuffmanTable* table);

// Matches the 16 bits of provided lookahead from the scan stream against the
// Huffman table, and returns a pointer to a matching prefix entry, or nullptr
// if no entry matches.
//
// Note that the lookahead bits that are currently being matched start with the
// MSb of "lookahead". e.g. if the table has entries for:
//
//    #1: 0
//    #2: 10
// And the lookahead is:
//
//    bits[16]:0b1000_0000_0000_0000
//
// Then entry #2 will be matched.
const PrefixEntry* MatchLookahead(const HuffmanTable& table,
                                  uint16_t lookahead);

}  // namespace internal
}  // namespace xls::jpeg

#endif  // XLS_EXAMPLES_JPEG_JPEG_GRM_H_
