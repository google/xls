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

#include <cstdint>
#include <iostream>
#include <string>

#include "absl/base/casts.h"
#include "absl/flags/flag.h"
#include "absl/types/span.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"

ABSL_FLAG(std::string, input, "", "Input data to CRC.");
ABSL_FLAG(uint32_t, polynomial, 0xEDB88320, "CRC polynomial value to use.");

namespace {

uint32_t Crc32Reference(absl::Span<const uint8_t> input, uint32_t polynomial) {
  uint32_t crc = -1U;
  for (uint8_t byte : input) {
    crc = crc ^ byte;
    for (int i = 0; i < 8; ++i) {
      uint32_t mask = -(crc & 1U);
      crc = (crc >> 1U) ^ (polynomial & mask);
    }
  }
  crc = ~crc;
  return crc;
}

}  // namespace

int main(int argc, char** argv) {
  xls::InitXls(argv[0], argc, argv);
  std::string input = absl::GetFlag(FLAGS_input);
  absl::Span<const uint8_t> data(absl::bit_cast<uint8_t*>(input.data()),
                                 input.size());
  XLS_LOG(INFO) << "Performing CRC on " << data.size()
                << " byte(s) of input data.";
  uint32_t result = Crc32Reference(data, absl::GetFlag(FLAGS_polynomial));
  std::cout << std::hex << result << '\n';
  return 0;
}
