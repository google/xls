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

#include "xls/examples/jpeg/jpeg_grm.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/examples/jpeg/constants.h"
#include "xls/examples/jpeg/idct_chen_jit_wrapper.h"

namespace xls::jpeg {
namespace internal {

// "Start of Frame 0" (the segment used in the baseline JPEG decoding) metadata.
struct Sof0Data {
  uint16_t y;
  uint16_t x;
  uint8_t nf;
  std::array<uint8_t, kColorLimit> component_ids;
  std::array<uint8_t, kColorLimit> component_qtabs;

  std::string ToString() const;
};

// Describes the data collected for a given color component within the "Start of
// Scan" section.
struct SosComponentData {
  uint8_t component;  // Component ID.
  uint8_t dc_table;   // DC Huffman table selector.
  uint8_t ac_table;   // AC Huffman table selector.

  std::string ToString() const;
};

// Notes the precision of values in the quantization table -- right now we only
// support 8-bit precision.
enum class Precision {
  k8b,
  k16b,
};

// Quantization scaling factors -- these are used on the data that results from
// the popping of a `PrefixEntry::GetBitsToPop()` quantity, using the index
// corresponding to the currently decoded coefficient.
struct QuantizationTable {
  Precision precision;
  uint8_t identifier;

  // Quantization coefficient data (one per coefficient in the MCU).
  std::array<uint8_t, kCoeffPerMcu> data;
};

// Overall context of metadata gathered for JPEG decoding of a given image.
struct DecodeCtx {
  std::optional<Sof0Data> sof0_data;
  std::vector<SosComponentData> component_data;
  std::vector<HuffmanTable> huffman_tables;
  std::vector<QuantizationTable> quantization_tables;
};

absl::Status HuffmanTableExpandEntries(HuffmanTable* table) {
  XLS_RET_CHECK(table->expanded.empty());
  uint16_t current = 0;
  for (int i = 0; i < table->entries.size(); ++i) {
    // Note: index 0 contains all entries of Huffman length 1, index 1
    // contains all Huffman entries of length 2, and so on.
    uint8_t bit_length = i + 1;
    for (uint8_t value : table->entries[i]) {
      PrefixEntry entry{.bits = bit_length, .code = current, .value = value};
      // In the DC table there should be no leading zero coefficients.
      if (table->table_class == HuffmanTableClass::kDc &&
          entry.GetLeadingZeros() != 0) {
        return MalformedInputError(
            absl::StrFormat("In the DC table there should be no leading zero "
                            "coefficients; got %d leading zeros",
                            entry.GetLeadingZeros()));
      }
      if (entry.GetBitsToPop() > kBitsToPopLimit) {
        return MalformedInputError(
            absl::StrFormat("Expect the 'bit count to pop' for Huffman symbol "
                            "should be <= %d bits; got: %d",
                            entry.GetBitsToPop(), kBitsToPopLimit));
      }
      table->expanded.push_back(std::move(entry));
      current += 1;
    }
    current <<= 1;
  }
  return absl::OkStatus();
}

static uint8_t ClampToU8(int32_t x) {
  if (x < 0) {
    return 0;
  }
  if (x > 0xff) {
    return 0xff;
  }
  return x;
}

static std::string ToString(const std::array<int32_t, kCoeffPerMcu>& data) {
  return absl::StrCat("{", absl::StrJoin(data, ", "), "}");
}

static uint8_t GetQuantizationTableNumber(const Sof0Data& sof0_data,
                                          uint8_t componentno) {
  return sof0_data.component_qtabs[componentno];
}

// Performs an IDCT transform on the input values (cosine frequency space to
// spatial) and then translates the result values into the uint8_t range (by
// adding 128 to translate values in the domain [-128, 127] to ones in the [0,
// 255] range).
static absl::StatusOr<std::array<uint8_t, kCoeffPerMcu>> IdctToU8(
    const std::array<int32_t, kCoeffPerMcu>& data) {
  // Implementation note: we use a singleton here to avoid reloading each time.
  static std::unique_ptr<IdctChen> idct = IdctChen::Create().value();
  std::array<int32_t, kCoeffPerMcu> spatial;
  VLOG(3) << "pre-IDCT: " << ToString(data);
  using PackedArrayViewT = PackedArrayView<PackedBitsView<32>, kCoeffPerMcu>;
  XLS_RETURN_IF_ERROR(idct->Run(
      PackedArrayViewT(&const_cast<std::array<int32_t, kCoeffPerMcu>&>(data)),
      PackedArrayViewT(&spatial)));
  VLOG(3) << "post-IDCT: " << ToString(spatial);
  // "Clamp" translated values (centered around 0) into the 0..255 range
  std::array<uint8_t, kCoeffPerMcu> clamped;
  for (int i = 0; i < spatial.size(); ++i) {
    int32_t x = spatial[i];
    clamped[i] = ClampToU8(x + 128);
  }
  return clamped;
}

const PrefixEntry* MatchLookahead(const HuffmanTable& table,
                                  uint16_t lookahead) {
  for (const PrefixEntry& entry : table.expanded) {
    if (lookahead >> (sizeof(lookahead) * CHAR_BIT - entry.bits) ==
        entry.code) {
      return &entry;
    }
  }
  return nullptr;
}

static const HuffmanTable* FindHuffmanTable(
    absl::Span<const HuffmanTable> huffman_tables,
    HuffmanTableClass table_class, uint8_t dc_tableno) {
  for (const HuffmanTable& ht : huffman_tables) {
    if (ht.table_class == table_class && ht.table_index == dc_tableno) {
      return &ht;
    }
  }
  return nullptr;
}

// Decodes an MCU for a single component (i.e. color channel) and returns it.
//
// Args:
//  c: Component index (0-based within the number of components in a frame which
//    the spec calls "nf")
//  ctx: Holds the metadata needed for decoding an MCU.
//  bit_stream: The bit stream being popped from (for the decoding of scan
//    data).
//  dc_carry: The DC component value which carries from the previous MCU
//    decodings -- these are additively encoded, so when we decode MCU0 in
//    component 0 it updates `dc_carry[0] += my_decoded_dc_value` for subsequent
//    use in MCU1; i.e. it is "loop carried" across MCU decodings for a
//    component.
static absl::StatusOr<std::array<uint8_t, kCoeffPerMcu>> DecodeComponentMcu(
    uint8_t c, const DecodeCtx& ctx, BitStream* bit_stream,
    std::array<int32_t, kColorLimit>* dc_carry) {
  VLOG(3) << absl::StreamFormat("Decoding MCU for component at index: %d", c);
  std::array<int32_t, kCoeffPerMcu> data = {0};
  XLS_ASSIGN_OR_RETURN(bool at_eof, bit_stream->AtEof());
  if (at_eof) {
    VLOG(3) << "At end of bit stream, trivial MCU";
    return IdctToU8(data);
  }

  uint8_t qtabno = GetQuantizationTableNumber(ctx.sof0_data.value(), c);
  if (qtabno >= ctx.quantization_tables.size()) {
    return MalformedInputError(
        absl::StrFormat("Component %d referred to quantization table %d which "
                        "was not given; have %d",
                        c, qtabno, ctx.quantization_tables.size()));
  }
  const QuantizationTable& qtable = ctx.quantization_tables.at(qtabno);

  {
    uint8_t dc_tableno = ctx.component_data[c].dc_table;
    const HuffmanTable* dc_table = FindHuffmanTable(
        ctx.huffman_tables, HuffmanTableClass::kDc, dc_tableno);
    XLS_ASSIGN_OR_RETURN(uint16_t lookahead,
                         bit_stream->PeekN(kHuffmanCodeSizeLimit));
    const PrefixEntry* matched = MatchLookahead(*dc_table, lookahead);
    if (matched == nullptr) {
      return MalformedInputError(absl::StrFormat(
          "Lookahead scan data %#04x did not match any Huffman prefix entry.",
          lookahead));
    }
    XLS_RETURN_IF_ERROR(bit_stream->PopN(matched->bits).status());
    XLS_RET_CHECK_EQ(matched->GetLeadingZeros(), 0)
        << "no leading zeros for DC entry";
    uint8_t to_pop = matched->GetBitsToPop();
    XLS_RET_CHECK_LT(matched->bits + to_pop,
                     kHuffmanCodeSizeLimit + kBitsToPopLimit);
    XLS_ASSIGN_OR_RETURN(int16_t symbol, bit_stream->PopCoeff(to_pop));
    VLOG(3) << "Popped DC symbol: " << symbol;
    (*dc_carry)[c] += static_cast<int32_t>(symbol);
    int32_t quantization_coeff = qtable.data[0];
    int32_t value = (*dc_carry)[c] * quantization_coeff;
    data[0] = value;
    VLOG(3) << "Dequantized DC value (index 0): " << data[0]
            << " quantization coeff: " << quantization_coeff;
  }

  uint8_t ac_tableno = ctx.component_data[c].ac_table;
  const HuffmanTable* ac_table =
      FindHuffmanTable(ctx.huffman_tables, HuffmanTableClass::kAc, ac_tableno);
  if (ac_table == nullptr) {
    return MalformedInputError(
        absl::StrFormat("Invalid reference to AC table: %d", ac_tableno));
  }

  int coeff = 1;
  while (coeff < kCoeffPerMcu) {
    // If we're at the EOF we can't look ahead.
    XLS_ASSIGN_OR_RETURN(bool at_eof, bit_stream->AtEof());
    if (at_eof) {
      break;
    }
    XLS_ASSIGN_OR_RETURN(uint16_t lookahead,
                         bit_stream->PeekN(kHuffmanCodeSizeLimit));
    const PrefixEntry* matched = MatchLookahead(*ac_table, lookahead);
    XLS_RETURN_IF_ERROR(bit_stream->PopN(matched->bits).status());
    if (matched->value == 0) {
      // AC coefficient indicates End of Block.
      VLOG(3) << "Popped AC coefficient indicating End of Block";
      break;
    }

    uint8_t leading_zeros = matched->GetLeadingZeros();
    coeff += leading_zeros;
    VLOG(3) << absl::StreamFormat(
        "Saw %d leading zeros, now at coefficient %d.", leading_zeros, coeff);
    if (coeff >= kCoeffPerMcu) {
      return MalformedInputError(absl::StrFormat(
          "Attempted to address out-of-range MCU coefficient %d", coeff));
    }

    uint8_t to_pop = matched->GetBitsToPop();
    XLS_ASSIGN_OR_RETURN(int16_t symbol, bit_stream->PopCoeff(to_pop));
    VLOG(3) << "Popped AC symbol: " << symbol;
    uint8_t index = kZigZagMap[coeff];
    int32_t value =
        static_cast<int32_t>(symbol) * static_cast<int32_t>(qtable.data[coeff]);
    VLOG(3) << absl::StreamFormat("Dequantized AC value (index %d): %d", index,
                                  value);
    data[index] = value;
    coeff += 1;
  }

  return IdctToU8(data);
}

Rgb ToRgb(uint8_t y, uint8_t cb, uint8_t cr) {
  int32_t crz = int32_t{cr} - kChrominanceZero;
  int32_t cbz = int32_t{cb} - kChrominanceZero;
  // clang-format off
  double r = y                 + 1.402   * crz;
  double g = y - 0.34414 * cbz - 0.71414 * crz;
  double b = y + 1.772   * cbz;
  // clang-format on
  VLOG(3) << absl::StreamFormat("YCbCr (%d, %d, %d) => RGB (%f, %f, %f)", y, cb,
                                cr, r, g, b);
  return Rgb{.r = ClampToU8(r), .g = ClampToU8(g), .b = ClampToU8(b)};
}

// Decodes the MCUs (8x8 blocks) present in the given byte stream as "scan
// data".
//
// Args:
//  nf: Number of color channels.
//  mcu_height: Number of 8x8 blocks in height.
//  mcu_width: Number of 8x8 blocks in width.
//  ctx: Decode information context.
//  byte_stream: The byte stream containing the scan data.
//
// Returns:
//  A decoded JPEG (with RGB pixel values, y height, and x width).
static absl::StatusOr<DecodedJpeg> DecodeMcus(uint8_t nf, uint16_t mcu_height,
                                              uint16_t mcu_width,
                                              const DecodeCtx& ctx,
                                              ByteStream* byte_stream) {
  VLOG(3) << "Quantization table count at start of MCU decoding: "
          << ctx.quantization_tables.size();
  if (nf != 3) {
    return absl::UnimplementedError(
        "Only 3 color channels are currently handled.");
  }
  BitStream bit_stream(byte_stream);
  std::vector<std::vector<Rgb>> lines(ctx.sof0_data->y);
  for (auto& line : lines) {
    line.resize(ctx.sof0_data->x);
  }
  std::array<int32_t, 3> dc_carry = {0, 0, 0};
  for (int32_t mcu_y = 0; mcu_y < mcu_height; ++mcu_y) {
    VLOG(3) << "Decoding MCU row: " << mcu_y;
    for (int32_t mcu_x = 0; mcu_x < mcu_height; ++mcu_x) {
      VLOG(3) << "Decoding MCU column: " << mcu_x;
      std::array<std::array<uint8_t, kCoeffPerMcu>, kColorLimit> color_data;
      for (uint8_t c = 0; c < nf; ++c) {
        XLS_ASSIGN_OR_RETURN(
            color_data[c], DecodeComponentMcu(c, ctx, &bit_stream, &dc_carry));
      }
      for (int i = 0; i < kCoeffPerMcu; ++i) {
        uint8_t y = color_data[kYIndex][i];
        uint8_t cb = color_data[kCbIndex][i];
        uint8_t cr = color_data[kCrIndex][i];
        // Note that MCUs are discrete 8x8 blocks, so if the image is, say, 1
        // pixel wide, there will be extra columns that we don't want to write
        // into the pixel buffer.
        int row = mcu_y * kMcuWidth + i / kMcuWidth;
        int col = mcu_x * kMcuHeight + i % kMcuHeight;
        if (row < ctx.sof0_data->y && col < ctx.sof0_data->x) {
          Rgb rgb = ToRgb(y, cb, cr);
          lines[row][col] = rgb;
          VLOG(3) << absl::StreamFormat(
                         "Color converted row/col (%d, %d) YCbCr{%u %u %u} => ",
                         row, col, y, cb, cr)
                  << rgb;
        }
      }
    }
  }

  VLOG(3) << absl::StreamFormat(
      "Completed decoding MCUs; height: %d width: %d => lines: %d", mcu_height,
      mcu_width, lines.size());
  return DecodedJpeg{ctx.sof0_data->y, ctx.sof0_data->x, std::move(lines)};
}

static absl::Status DecodeApp0Segment(ByteStream& byte_stream, DecodeCtx& ctx) {
  XLS_ASSIGN_OR_RETURN(uint16_t length, byte_stream.PopHiLo());
  // Note: the length is not generally necessary to be used in the APP0 segment,
  // since sizes of entities to pop are all known.
  VLOG(3) << "APP0 segment length: " << length;
  // "JFIF\0" indicator.
  XLS_RETURN_IF_ERROR(byte_stream.DropExpectedMulti(
      {0x4a, 0x46, 0x49, 0x46, 0x00}, "JFIF indicator"));
  XLS_ASSIGN_OR_RETURN(uint8_t version_major, byte_stream.Pop());
  XLS_ASSIGN_OR_RETURN(uint8_t version_minor, byte_stream.Pop());
  VLOG(5) << absl::StreamFormat("APP0 version: %u.%u", version_major,
                                version_minor);
  // 0: no units, X and Y specify pixel aspect ratio.
  // 1: X and Y are dots per inch.
  // 2: X and Y are dots per cm.
  XLS_ASSIGN_OR_RETURN(uint8_t units, byte_stream.Pop());
  if (units != 0) {
    return absl::UnimplementedError(
        "Only pixel-based aspect ratios are supported.");
  }
  XLS_ASSIGN_OR_RETURN(uint16_t x_density, byte_stream.PopHiLo());
  XLS_ASSIGN_OR_RETURN(uint16_t y_density, byte_stream.PopHiLo());
  VLOG(5) << absl::StreamFormat("density x: %u y: %u", x_density, y_density);
  XLS_ASSIGN_OR_RETURN(uint8_t x_thumbnail, byte_stream.Pop());
  XLS_ASSIGN_OR_RETURN(uint8_t y_thumbnail, byte_stream.Pop());
  XLS_ASSIGN_OR_RETURN(std::vector<uint8_t> thumbnail_rgb,
                       byte_stream.PopN(static_cast<int32_t>(x_thumbnail) *
                                        static_cast<int32_t>(y_thumbnail)));
  return absl::OkStatus();
}

static absl::Status DecodeSof0Segment(ByteStream& byte_stream, DecodeCtx& ctx) {
  XLS_ASSIGN_OR_RETURN(uint16_t length_u16, byte_stream.PopHiLo());
  // Note: the length is not generally necessary to be used in the SOF0 segment,
  // since sizes of entities to pop are all known.
  VLOG(3) << "SOF0 segment length: " << length_u16;
  // p: sample precision (precision in bits for samples of the components
  // in the frame).
  XLS_ASSIGN_OR_RETURN(uint8_t p, byte_stream.Pop());
  XLS_RET_CHECK_EQ(p, 8);
  // y: number of lines
  XLS_ASSIGN_OR_RETURN(uint16_t y, byte_stream.PopHiLo());
  // x: number of samples per line
  XLS_ASSIGN_OR_RETURN(uint16_t x, byte_stream.PopHiLo());
  // nf: number of image component in frame
  XLS_ASSIGN_OR_RETURN(uint8_t nf, byte_stream.Pop());
  XLS_RET_CHECK_EQ(nf, 3);
  std::array<uint8_t, kColorLimit> component_qtabs = {0, 0, 0};
  // TODO(leary): 2021-06-18 We need a test example that really exercises the
  // component id remapping capability specified in the standard. We need to
  // scan them out of the stream, but right now it's more decorative than
  // anything.
  std::array<uint8_t, kColorLimit> component_ids = {0, 0, 0};
  for (int i = 0; i < nf; ++i) {
    // "component identifier"
    XLS_ASSIGN_OR_RETURN(uint8_t ci, byte_stream.Pop());
    component_ids[i] = ci;
    // {horizontal, vertical} sampling factor.
    XLS_ASSIGN_OR_RETURN(auto hi_vi, byte_stream.Pop2xU4());
    auto [hi, vi] = hi_vi;
    XLS_RET_CHECK_EQ(hi, 1) << "horizontal sampling factor is not 1";
    XLS_RET_CHECK_EQ(vi, 1) << "vertical sampling factor is not 1";
    // "quantization table destination selector"
    XLS_ASSIGN_OR_RETURN(uint8_t tqi, byte_stream.Pop());
    component_qtabs[i] = tqi;
  }
  ctx.sof0_data = Sof0Data{.y = y,
                           .x = x,
                           .nf = nf,
                           .component_ids = component_ids,
                           .component_qtabs = component_qtabs};
  VLOG(3) << "SOF0: " << ctx.sof0_data->ToString();
  return absl::OkStatus();
}

static absl::Status DecodeDhtSegment(ByteStream& byte_stream, DecodeCtx& ctx) {
  XLS_ASSIGN_OR_RETURN(uint16_t length_u16, byte_stream.PopHiLo());
  auto length = static_cast<int32_t>(length_u16);
  length -= 2;  // The two length bytes themselves are included.
  while (length > 0) {
    // tc: table class; 0: DC or lossless; 1: AC
    // th: Huffman table destination identifier; one of four possible
    // destinations
    XLS_ASSIGN_OR_RETURN(auto tc_th, byte_stream.Pop2xU4());
    auto [tc, th] = tc_th;
    XLS_ASSIGN_OR_RETURN(auto l_i_lens,
                         byte_stream.PopN<kHuffmanCodeSizeLimit>());
    length -= kHuffmanCodeSizeLimit + 1;
    std::array<std::vector<uint8_t>, kHuffmanCodeSizeLimit> data;
    for (int64_t i = 0; i < kHuffmanCodeSizeLimit; ++i) {
      int32_t l_i_len = static_cast<int32_t>(l_i_lens[i]);
      length -= l_i_len;
      XLS_ASSIGN_OR_RETURN(std::vector<uint8_t> l_i_data,
                           byte_stream.PopN(l_i_len));
      data[i] = std::move(l_i_data);
    }
    ctx.huffman_tables.push_back(
        HuffmanTable{.table_class = static_cast<HuffmanTableClass>(tc),
                     .table_index = th,
                     .entries = std::move(data)});
    XLS_RETURN_IF_ERROR(HuffmanTableExpandEntries(&ctx.huffman_tables.back()));
  }
  return absl::OkStatus();
}

static absl::Status DecodeDqtSegment(ByteStream& byte_stream, DecodeCtx& ctx) {
  XLS_ASSIGN_OR_RETURN(uint16_t length_u16, byte_stream.PopHiLo());
  auto length = static_cast<int32_t>(length_u16);
  length -= 2;  // The two length bytes themselves are included.
  VLOG(3) << "DQT marker; length: " << length;
  while (length > 0) {
    // p_q: quantization table element precision; 0: 8-bit 1: 16-bit
    // t_q: quantization table destination identifier
    XLS_ASSIGN_OR_RETURN(auto pq_tq, byte_stream.Pop2xU4());
    auto [p_q, t_q] = pq_tq;
    XLS_RET_CHECK_LT(t_q, 4)
        << "qtable destination identifier out of range: " << t_q;
    // q_k: Quantization table elements (in zig-zag scan order).
    XLS_ASSIGN_OR_RETURN(auto q_k, byte_stream.PopN<kCoeffPerMcu>());
    VLOG(3) << absl::StreamFormat("qtable at index %d has identifier %d",
                                  ctx.quantization_tables.size(), t_q);
    ctx.quantization_tables.push_back(
        QuantizationTable{.precision = static_cast<Precision>(p_q),
                          .identifier = t_q,
                          .data = q_k});
    VLOG(3) << "Quantization table count now: "
            << ctx.quantization_tables.size();
    length -= static_cast<int32_t>(kCoeffPerMcu) + 1;
  }
  XLS_RET_CHECK_EQ(length, 0);
  return absl::OkStatus();
}

static absl::StatusOr<DecodedJpeg> DecodeSosSegment(ByteStream& byte_stream,
                                                    DecodeCtx& ctx) {
  XLS_ASSIGN_OR_RETURN(uint16_t ls, byte_stream.PopHiLo());
  VLOG(3) << "SOS segment length: " << ls;
  XLS_ASSIGN_OR_RETURN(uint8_t ns, byte_stream.Pop());
  for (int64_t i = 0; i < ns; ++i) {
    XLS_ASSIGN_OR_RETURN(uint8_t csj, byte_stream.Pop());
    XLS_ASSIGN_OR_RETURN(auto tdj_taj, byte_stream.Pop2xU4());
    auto [tdj, taj] = tdj_taj;
    XLS_RET_CHECK_LT(tdj, 4);
    XLS_RET_CHECK_LT(taj, 4);
    ctx.component_data.push_back(
        SosComponentData{.component = csj, .dc_table = tdj, .ac_table = taj});
    VLOG(3) << absl::StreamFormat("SOS component %d: %s", i,
                                  ctx.component_data.back().ToString());
  }
  // ss: Start of spectral or predictor selection -- we need this to be
  // zero.
  XLS_ASSIGN_OR_RETURN(uint8_t ss, byte_stream.Pop());
  XLS_RET_CHECK_EQ(ss, 0);
  // se: End of spectral section.
  XLS_ASSIGN_OR_RETURN(uint8_t se, byte_stream.Pop());
  XLS_RET_CHECK_EQ(se, 63);
  // ah: Successive approximation bit position high.
  // al: Successive approximation bit position low.
  XLS_ASSIGN_OR_RETURN(auto ah_al, byte_stream.Pop2xU4());
  auto [ah, al] = ah_al;
  XLS_RET_CHECK_EQ(ah, 0);
  XLS_RET_CHECK_EQ(al, 0);

  if (!ctx.sof0_data.has_value()) {
    return MalformedInputError(
        "Start of Frame segment was not encountered before Start of Scan "
        "segment.");
  }

  const Sof0Data& sof0 = ctx.sof0_data.value();
  uint16_t mcu_height = CeilOfRatio(sof0.y, uint16_t{8});
  uint16_t mcu_width = CeilOfRatio(sof0.x, uint16_t{8});

  XLS_ASSIGN_OR_RETURN(
      DecodedJpeg result,
      DecodeMcus(sof0.nf, mcu_height, mcu_width, ctx, &byte_stream));
  XLS_RET_CHECK_EQ(result.GetPixelCount(), result.GetPixelCountViaLineData());
  return result;
}

std::string SosComponentData::ToString() const {
  return absl::StrFormat(
      "SosData{.component = %u, dc_table = %u, ac_table = %u}", component,
      dc_table, ac_table);
}

std::string Sof0Data::ToString() const {
  std::string component_ids_str = absl::StrJoin(component_ids, ", ");
  std::string component_qtabs_str = absl::StrJoin(component_qtabs, ", ");
  return absl::StrFormat(
      "Sof0Data{.y = %u, .x = %u, .nf = %u, .component_ids = {%s}, "
      ".component_qtabs = {%s}}",
      y, x, nf, component_ids_str, component_qtabs_str);
}

}  // namespace internal

bool operator==(const Rgb& lhs, const Rgb& rhs) {
  return lhs.r == rhs.r && lhs.g == rhs.g && lhs.b == rhs.b;
}

std::ostream& operator<<(std::ostream& os, const Rgb& rgb) {
  os << absl::StreamFormat("Rgb{.r=%d, .g=%d, .b=%d}", rgb.r, rgb.g, rgb.b);
  return os;
}

absl::StatusOr<DecodedJpeg> DecodeJpeg(ByteStream& byte_stream) {
  internal::DecodeCtx ctx;

  while (true) {
    XLS_RETURN_IF_ERROR(
        byte_stream.DropExpected(kMarkerStart, "marker start byte"));
    XLS_ASSIGN_OR_RETURN(uint8_t marker_byte, byte_stream.Pop());
    switch (marker_byte) {
      case kSoiMarker:
        break;  // No payload.
      case kApp0Marker:
        XLS_RETURN_IF_ERROR(DecodeApp0Segment(byte_stream, ctx));
        break;
      case kSof0Marker:  // Start of Frame (baseline decode)
        XLS_RETURN_IF_ERROR(DecodeSof0Segment(byte_stream, ctx));
        break;
      case kSosMarker:  // Start of Scan
        return DecodeSosSegment(byte_stream, ctx);
      case kDqtMarker:  // Define Quantization Table(s)
        XLS_RETURN_IF_ERROR(DecodeDqtSegment(byte_stream, ctx));
        break;
      case kDhtMarker:  // Define Huffman Table(s)
        XLS_RETURN_IF_ERROR(DecodeDhtSegment(byte_stream, ctx));
        break;
      default:  // Unknown / unhandled marker.
        return MalformedInputError(
            absl::StrFormat("Unhandled marker: %#x byte index: %#x",
                            marker_byte, byte_stream.popped_index()));
    }
  }
}

}  // namespace xls::jpeg
