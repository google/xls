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

ABSL_FLAG(std::string, input, "", "Input data (numeric) to Adler32.");

namespace {

const uint32_t MOD_ADLER = 65521;

uint32_t Adler32Sequential(absl::Span<const uint8_t> buf, size_t len) {
  uint32_t a = 1;
  uint32_t b = 0;
  for (int i = 0; i < len; ++i) {
    a = (a + static_cast<uint32_t>(buf[i])) % MOD_ADLER;
    b = (b + a) % MOD_ADLER;
  }
  return (b << 16) | a;
}

}  // namespace

int main(int argc, char** argv) {
  xls::InitXls(argv[0], argc, argv);
  std::string input = absl::GetFlag(FLAGS_input);
  absl::Span<const uint8_t> data(absl::bit_cast<uint8_t*>(input.data()),
                                 input.size());
  XLS_LOG(INFO) << "Performing Adler32 on " << data.size()
                << " byte(s) of input data (string).";
  uint32_t result = Adler32Sequential(data, data.size());
  std::cout << std::hex << result << '\n';
  return 0;
}
