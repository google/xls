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

#include <iostream>

#include "absl/base/casts.h"
#include "absl/flags/flag.h"
#include "absl/types/span.h"
#include "xls/common/init_xls.h"
#include "xls/common/integral_types.h"
#include "xls/common/logging/logging.h"

ABSL_FLAG(std::string, input, "", "Input data to CRC.");
ABSL_FLAG(uint32, polynomial, 0xEDB88320, "CRC polynomial value to use.");

namespace {

uint32 Crc32Reference(absl::Span<const uint8> input, uint32 polynomial) {
  uint32 crc = -1U;
  for (uint8 byte : input) {
    crc = crc ^ byte;
    for (int i = 0; i < 8; ++i) {
      uint32 mask = -(crc & 1U);
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
  absl::Span<const uint8> data(absl::bit_cast<uint8*>(input.data()),
                               input.size());
  XLS_LOG(INFO) << "Performing CRC on " << data.size()
                << " byte(s) of input data.";
  uint32 result = Crc32Reference(data, absl::GetFlag(FLAGS_polynomial));
  std::cout << std::hex << result << std::endl;
  return 0;
}
