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

#ifndef XLS_EXAMPLES_JPEG_CONSTANTS_H_
#define XLS_EXAMPLES_JPEG_CONSTANTS_H_

#include <array>
#include <cstdint>

namespace xls::jpeg {

// An "MCU" is a "Minimum Coded Unit" which is an 8x8 grid of values.
inline constexpr int kMcuHeight = 8;
inline constexpr int kMcuWidth = 8;
inline constexpr int kCoeffPerMcu = kMcuHeight * kMcuWidth;

// The JPEG byte stream has "markers" that delimit the sections of the JPEG
// (first metadata, then scan data). This all-bits-set byte is an escape that
// notes that a marker identifier bytes (indicating what kind of section we're
// decoding, e.g. kSof0Marker perhaps) will come next.
inline constexpr uint8_t kMarkerStart = 0xff;

// JPEG markers defined in the specification, see
// https://www.w3.org/Graphics/JPEG/ -- for more details see the spec document.
inline constexpr uint8_t kSof0Marker = 0xc0;  // Start of Frame (Baseline DCT)
inline constexpr uint8_t kDhtMarker = 0xc4;   // Define Huffman Table(s)
inline constexpr uint8_t kSoiMarker = 0xd8;   // Start of Image
inline constexpr uint8_t kEoiMarker = 0xd9;   // End of Image
inline constexpr uint8_t kSosMarker = 0xda;   // Start of Scan
inline constexpr uint8_t kDqtMarker = 0xdb;   // Define Quantization Table(s)
inline constexpr uint8_t kApp0Marker = 0xe0;  // "Application 0" (JFIF metadata)
inline constexpr uint8_t kComMarker = 0xfe;   // Comment

inline constexpr uint8_t kChrominanceZero = 128;  // See internal::ToRgb().

// The prefix codes in the stream can be at most 16 bits long.
inline constexpr uint8_t kHuffmanCodeSizeLimit = 16;

// The number of bits that we need to pop after a given prefix code does not
// exceed 11 bits. Technically there's encoding room in the DHT data for up to
// 15 bits, but the values encoded in the stream can't exceed 1024, so 11 bits
// is the max we'll observe.
inline constexpr uint8_t kBitsToPopLimit = 11;

// Currently we only support 3 color comonents, since we assume YCbCr.
inline constexpr uint8_t kColorLimit = 3;

// When the image uses three color components (without downsampling) this is the
// order in which MCU components are interleaved in the stream.
inline constexpr uint8_t kYIndex = 0;
inline constexpr uint8_t kCbIndex = 1;
inline constexpr uint8_t kCrIndex = 2;

// Coefficients are encoded in the scan data in an order that should maximize
// compression in frequency space; that is, the highest frequency components
// should come at the end of the block so that we can easily squash them to zero
// and say "end of block" as early as possible.
//
// This "zig zag map" indicates where coefficient values (decoded from the
// stream) scatter before performing an IDCT. That is, as we scan coefficients
// in the stream:
//
//    frequency_data[kZigZagMap[coeffno]] = value;
//
// Note that this is the inverse of the encoder-side zigzag map; e.g. the
// encoder's map looks like:
//
//  0,  1,  5, ...
//  2,  4, ...
//  3, ...
//
// For example, you can see the value at index 2 in the map below is "8", and 8
// is the index of the value "2" in the map above.
inline constexpr std::array<uint8_t, kCoeffPerMcu> kZigZagMap = {
    0,  1,  8,  16, 9,  2,  3,  10,  //
    17, 24, 32, 25, 18, 11, 4,  5,   //
    12, 19, 26, 33, 40, 48, 41, 34,  //
    27, 20, 13, 6,  7,  14, 21, 28,  //
    35, 42, 49, 56, 57, 50, 43, 36,  //
    29, 22, 15, 23, 30, 37, 44, 51,  //
    58, 59, 52, 45, 38, 31, 39, 46,  //
    53, 60, 61, 54, 47, 55, 62, 63,  //
};

}  // namespace xls::jpeg

#endif  // XLS_EXAMPLES_JPEG_CONSTANTS_H_
