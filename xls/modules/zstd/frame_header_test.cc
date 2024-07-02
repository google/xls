// Copyright 2024 The XLS Authors
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

#define ZSTD_STATIC_LINKING_ONLY 1
#include <zstd.h>

#include <bitset>
#include <climits>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>

#include "absl/container/flat_hash_map.h"
#include "fuzztest/fuzztest.h"
#include "fuzztest/googletest_fixture_adapter.h"
#include "gtest/gtest.h"
#include "lib/common/zstd_errors.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/ir_convert/ir_converter.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/ir/events.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/modules/zstd/data_generator.h"

namespace xls {
namespace {

// Must be in sync with FrameHeaderStatus from
// xls/modules/zstd/frame_header.x
enum FrameHeaderStatus {
  OK,
  CORRUPTED,
  NO_ENOUGH_DATA,
  UNSUPPORTED_WINDOW_SIZE
};

class FrameHeaderTest : public xls::IrTestBase {
 public:
  // Prepare simulation environment
  void SetUp() {
    XLS_ASSERT_OK_AND_ASSIGN(std::filesystem::path path,
                             xls::GetXlsRunfilePath(this->file));
    XLS_ASSERT_OK_AND_ASSIGN(std::string module_text,
                             xls::GetFileContents(path));

    auto import_data = xls::dslx::CreateImportDataForTest();
    XLS_ASSERT_OK_AND_ASSIGN(
        xls::dslx::TypecheckedModule checked_module,
        xls::dslx::ParseAndTypecheck(module_text, this->file_name,
                                     this->module_name, &import_data));

    auto options = xls::dslx::ConvertOptions{};
    /* FIXME: The following code should work with a parametrized version of
     * the `parse_frame_header` function. However, it seems that
     * the symbolic_bindings are not correctly propagated inside
     * ConvertOneFunction. To leverage the problem, a simple specialization
     * of the function is used (`parse_frame_header_128`).
     * Once the problem is solved, we can restore the code below.
     */
    // auto symbolic_bindings = xls::dslx::ParametricEnv(
    //     absl::flat_hash_map<std::string, xls::dslx::InterpValue>{
    //         {"CAPACITY", xls::dslx::InterpValue::MakeUBits(/*bit_count=*/32,
    //                                                     /*value=*/32)}});
    dslx::ParametricEnv* symbolic_bindings = nullptr;
    XLS_ASSERT_OK_AND_ASSIGN(
        this->converted, xls::dslx::ConvertOneFunction(
                             checked_module.module, function_name, &import_data,
                             symbolic_bindings, options));
  }

  // Parse Buffer contents with ZSTD library, prepare inputs for DSLX simulation
  // based on the buffer contents, form the expected output from the simulation,
  // run the simulation of frame header parser and compare the results against
  // expected values.
  void ParseAndCompareWithZstd(absl::Span<const uint8_t> buffer) {
    absl::Span<const uint8_t> input_buffer;
    ZSTD_frameHeader zstd_fh;
    size_t result;
    std::vector<uint8_t> buffer_extended(dslx_buffer_size_bytes, 0);

    // Extend buffer contents to 128 bits if necessary
    if (buffer.size() < dslx_buffer_size_bytes) {
      std::copy(buffer.begin(), buffer.end(), buffer_extended.begin());
      input_buffer = absl::MakeSpan(buffer_extended);
    } else {
      input_buffer = buffer;
    }

    // Parse input buffer with libzstd and write it as ZSTD_frameHeader
    ASSERT_TRUE(!buffer.empty() && buffer.data() != nullptr);
    result = ZSTD_getFrameHeader_advanced(
        &zstd_fh, buffer.data(), buffer.size(), ZSTD_f_zstd1_magicless);

    // Decide on the expected status
    FrameHeaderStatus expected_status = FrameHeaderStatus::OK;
    if (result != 0) {
      if (ZSTD_isError(result)) {
        switch (ZSTD_getErrorCode(result)) {
          case ZSTD_error_frameParameter_windowTooLarge:
            expected_status = FrameHeaderStatus::UNSUPPORTED_WINDOW_SIZE;
            break;
          case ZSTD_error_frameParameter_unsupported:
            // Occurs when reserved_bit == 1, should result in CORRUPTED state
          default:
            // Provided data is corrupted. Unable to correctly parse ZSTD frame.
            expected_status = FrameHeaderStatus::CORRUPTED;
            break;
        }
      } else {
        // Provided data is to small to correctly parse ZSTD frame, should
        // have `result` bytes, got `buffer.size()` bytes.
        expected_status = FrameHeaderStatus::NO_ENOUGH_DATA;
      }
    }

    auto input = CreateDSLXSimulationInput(buffer.size(), input_buffer);
    absl::flat_hash_map<std::string, Value> hashed_input = {{"buffer", input}};

    auto expected_frame_header_result = CreateExpectedFrameHeaderResult(
        &zstd_fh, input, buffer, expected_status);

    RunAndExpectEq(hashed_input, expected_frame_header_result, this->converted,
                   true, true);
  }

  const char* file = "xls/modules/zstd/frame_header_test.x";
  const char* module_name = "frame_header_test";
  const char* file_name = "frame_header_test.x";
  const char* function_name = "parse_frame_header_128";
  std::string converted;

 private:
  const uint32_t dslx_buffer_size = 128;
  const uint32_t dslx_buffer_size_bytes =
      (dslx_buffer_size + CHAR_BIT - 1) / CHAR_BIT;

  void PrintZSTDFrameHeader(ZSTD_frameHeader* fh) {
    std::cout << std::hex;
    std::cout << "zstd_fh->frameContentSize: 0x" << fh->frameContentSize
              << std::endl;
    std::cout << "zstd_fh->windowSize: 0x" << fh->windowSize << std::endl;
    std::cout << "zstd_fh->blockSizeMax: 0x" << fh->blockSizeMax << std::endl;
    std::cout << "zstd_fh->frameType: 0x" << fh->frameType << std::endl;
    std::cout << "zstd_fh->headerSize: 0x" << fh->headerSize << std::endl;
    std::cout << "zstd_fh->dictID: 0x" << fh->dictID << std::endl;
    std::cout << "zstd_fh->checksumFlag: 0x" << fh->checksumFlag << std::endl;
  }

  // Form DSLX Value representing ZSTD Frame header based on data parsed with
  // ZSTD library. Represents DSLX struct `FrameHeader`.
  Value CreateExpectedFrameHeader(ZSTD_frameHeader* fh) {
    return Value::Tuple({
        /*window_size:*/ Value(UBits(fh->windowSize, 64)),
        /*frame_content_size:*/ Value(UBits(fh->frameContentSize, 64)),
        /*dictionary_id:*/ Value(UBits(fh->dictID, 32)),
        /*content_checksum_flag: */ Value(UBits(fh->checksumFlag, 1)),
    });
  }

  // Create DSLX Value representing Buffer contents after parsing frame header
  // in simulation. Represents DSLX struct `Buffer`.
  Value CreateExpectedBuffer(Value dslx_simulation_input,
                             absl::Span<const uint8_t> input_buffer,
                             uint32_t consumed_bytes_count,
                             FrameHeaderStatus expected_status) {
    // Return original buffer contents
    if (expected_status == FrameHeaderStatus::NO_ENOUGH_DATA)
      return dslx_simulation_input;
    // Critical failure - return empty buffer
    if (expected_status == FrameHeaderStatus::CORRUPTED)
      return Value::Tuple({/*contents:*/ Value(UBits(0, dslx_buffer_size)),
                           /*length:*/ Value(UBits(0, 32))});

    // Frame Header parsing succeeded. Expect output buffer contents with
    // removed first `consumed_bytes_count` bytes and extended to
    // dslx_buffer_size if necessary
    int bytes_to_extend =
        dslx_buffer_size_bytes - (input_buffer.size() - consumed_bytes_count);
    std::vector output_buffer(input_buffer.begin() + consumed_bytes_count,
                              input_buffer.end());
    for (int i = 0; i < bytes_to_extend; i++) {
      output_buffer.push_back(0);
    }

    auto expected_buffer_contents =
        Value(Bits::FromBytes(output_buffer, dslx_buffer_size));
    auto output_buffer_size_bits =
        (input_buffer.size() - consumed_bytes_count) * CHAR_BIT;
    uint32_t expected_buffer_size = output_buffer_size_bits > dslx_buffer_size
                                        ? dslx_buffer_size
                                        : output_buffer_size_bits;

    return Value::Tuple({/*contents:*/ expected_buffer_contents,
                         /*length:*/ Value(UBits(expected_buffer_size, 32))});
  }

  // Prepare DSLX Value representing Full Result of frame header parsing
  // simulation. It consists of expected status, parsing result and buffer
  // contents after parsing. Represents DSLX struct `FrameHeaderResult`.
  Value CreateExpectedFrameHeaderResult(ZSTD_frameHeader* fh,
                                        Value dslx_simulation_input,
                                        absl::Span<const uint8_t> input_buffer,
                                        FrameHeaderStatus expected_status) {
    auto expected_buffer = CreateExpectedBuffer(
        dslx_simulation_input, input_buffer, fh->headerSize, expected_status);
    auto expected_frame_header = CreateExpectedFrameHeader(fh);
    return Value::Tuple({/*status:*/ Value(UBits(expected_status, 2)),
                         /*header:*/ expected_frame_header,
                         /*buffer:*/ expected_buffer});
  }

  // Return DSLX Value used as input argument for running frame header parsing
  // simulation. Represents DSLX struct `Buffer`.
  Value CreateDSLXSimulationInput(size_t buffer_size,
                                  absl::Span<const uint8_t> input_buffer) {
    size_t size = buffer_size;

    // ignore buffer contents that won't fit into specialized buffer
    if (buffer_size > dslx_buffer_size_bytes) size = dslx_buffer_size_bytes;

    return Value::Tuple(
        {/*contents:*/ Value(Bits::FromBytes(input_buffer, dslx_buffer_size)),
         /*length:*/ Value(UBits(size * CHAR_BIT, 32))});
  }
};

/* TESTS */

TEST(ZstdLib, Version) { ASSERT_EQ(ZSTD_VERSION_STRING, "1.4.7"); }

TEST_F(FrameHeaderTest, ParseFrameHeaderSuccess) {
  std::vector<uint8_t> buffer{0xC2, 0x09, 0xFE, 0xCA, 0xEF, 0xCD,
                              0xAB, 0x90, 0x78, 0x56, 0x34, 0x12};
  this->ParseAndCompareWithZstd(buffer);
}

TEST_F(FrameHeaderTest, ParseFrameHeaderFailCorruptedReservedBit) {
  std::vector<uint8_t> buffer{0xEA, 0xFE, 0xCA, 0xEF, 0xCD, 0xAB,
                              0x90, 0x78, 0x56, 0x34, 0x12};
  this->ParseAndCompareWithZstd(buffer);
}

TEST_F(FrameHeaderTest, ParseFrameHeaderFailUnsupportedWindowSizeTooBig) {
  std::vector<uint8_t> buffer{0x10, 0xD3};
  this->ParseAndCompareWithZstd(buffer);
}

TEST_F(FrameHeaderTest, ParseFrameHeaderFailNoEnoughData) {
  std::vector<uint8_t> buffer{0xD3, 0xED};
  this->ParseAndCompareWithZstd(buffer);
}

// NO_ENOUGH_DATA has priority over CORRUPTED from reserved bit
TEST_F(FrameHeaderTest, ParseFrameHeaderFailNoEnoughDataReservedBit) {
  std::vector<uint8_t> buffer{0xED, 0xD3};
  this->ParseAndCompareWithZstd(buffer);
}

class FrameHeaderSeededTest : public FrameHeaderTest,
                              public ::testing::WithParamInterface<uint32_t> {
 public:
  static const uint32_t random_headers_count = 50;
};

// Test `random_headers_count` instances of randomly generated valid
// frame headers, generated with `decodecorpus` tool.
TEST_P(FrameHeaderSeededTest, ParseMultipleFrameHeaders) {
  auto seed = GetParam();
  auto frame_header = zstd::GenerateFrameHeader(seed, false);
  ASSERT_TRUE(frame_header.ok());
  this->ParseAndCompareWithZstd(frame_header.value());
}

INSTANTIATE_TEST_SUITE_P(
    FrameHeaderSeededTest, FrameHeaderSeededTest,
    ::testing::Range<uint32_t>(0, FrameHeaderSeededTest::random_headers_count));

class FrameHeaderFuzzTest
    : public fuzztest::PerFuzzTestFixtureAdapter<FrameHeaderTest> {
 public:
  void ParseMultipleRandomFrameHeaders(std::vector<uint8_t> frame_header) {
    this->ParseAndCompareWithZstd(frame_header);
  }
};

// Perform UNDETERMINISTIC FuzzTests with input vectors of variable length and
// contents. Frame Headers generated by FuzzTests can be invalid.
// This test checks if negative cases are handled correctly.
FUZZ_TEST_F(FrameHeaderFuzzTest, ParseMultipleRandomFrameHeaders)
    .WithDomains(fuzztest::Arbitrary<std::vector<uint8_t>>()
                     .WithMinSize(1)
                     .WithMaxSize(16));

}  // namespace
}  // namespace xls
