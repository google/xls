// MachSuite (BSD-3) License
//
// Copyright (c) 2014-2015, the President and Fellows of Harvard College.
// Copyright 2021 The XLS Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of Harvard University nor the names of its contributors
// may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cstdio>
#include <filesystem>
#include <iostream>

#include "absl/base/casts.h"
#include "absl/flags/flag.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"

#define FFT_SIZE 1024

ABSL_FLAG(bool, hex, false, "Display float data in hex format.");

constexpr const char kRealInPath[] =
    "third_party/xls_machsuite/fft/test_data/real_in";
constexpr const char kImgInPath[] =
    "third_party/xls_machsuite/fft/test_data/img_in";
constexpr const char kRealTwidInPath[] =
    "third_party/xls_machsuite/fft/test_data/real_twid_in";
constexpr const char kImgTwidInPath[] =
    "third_party/xls_machsuite/fft/test_data/img_twid_in";

namespace xls {
namespace {

using xls::GetFileContents;
using xls::GetXlsRunfilePath;

// Note: This reference uses full IEEE-compliant floating-
// point operations, whereas the XLS version uses floating-
// point operatoins that do not support subnormal numbers.
// This version could be modified to flush subnormal numbers
// to guaruntee equivalence for all inputs. However, this is
// not necessary for the input tested.

void fft(float real[FFT_SIZE], float img[FFT_SIZE],
         float real_twid[FFT_SIZE / 2], float img_twid[FFT_SIZE / 2]) {
  int even, odd, span, log, rootindex;
  float temp;
  log = 0;

  for (span = FFT_SIZE >> 1; span; span >>= 1, log++) {
    for (odd = span; odd < FFT_SIZE; odd++) {
      odd |= span;
      even = odd ^ span;

      temp = real[even] + real[odd];
      real[odd] = real[even] - real[odd];
      real[even] = temp;

      temp = img[even] + img[odd];
      img[odd] = img[even] - img[odd];
      img[even] = temp;

      rootindex = (even << log) & (FFT_SIZE - 1);
      if (rootindex) {
        temp =
            real_twid[rootindex] * real[odd] - img_twid[rootindex] * img[odd];
        img[odd] =
            real_twid[rootindex] * img[odd] + img_twid[rootindex] * real[odd];
        real[odd] = temp;
      }
    }
  }
}

// Load 'data' with the floats contained the file pointed at
// location 'path'. Expects exactly one float per line.
absl::Status ReadInputData(const char* path, float data[]) {
  XLS_ASSIGN_OR_RETURN(std::filesystem::path run_path, GetXlsRunfilePath(path));
  XLS_ASSIGN_OR_RETURN(std::string data_text, GetFileContents(run_path));

  int64_t data_idx = 0;
  for (auto line : absl::StrSplit(data_text, '\n')) {
    if (!line.empty()) {
      XLS_CHECK(absl::SimpleAtof(line, &data[data_idx]));
      data_idx++;
    }
  }

  return absl::OkStatus();
}

union FloatUintUnion {
  float f32;
  uint32_t u32;
};

// Display the contents of 'data' with name 'name' and
// 'size' number of entries.
void DisplayData(float data[], std::string name, int64_t size) {
  bool display_hex = absl::GetFlag(FLAGS_hex);
  std::cout << name << " = [" << std::endl;

  for (int64_t idx = 0; idx < size; ++idx) {
    if (display_hex) {
      std::cout << absl::StreamFormat("0x%a,\n",
                                      absl::bit_cast<uint32_t>(data[idx]));
    } else {
      std::cout << data[idx] << "," << std::endl;
    }
  }

  std::cout << "]" << std::endl;
}

absl::Status RealMain() {
  // Get input data.
  float real[FFT_SIZE];
  float img[FFT_SIZE];
  float real_twid[FFT_SIZE / 2];
  float img_twid[FFT_SIZE / 2];
  XLS_RETURN_IF_ERROR((ReadInputData(kRealInPath, real)));
  XLS_RETURN_IF_ERROR((ReadInputData(kImgInPath, img)));
  XLS_RETURN_IF_ERROR((ReadInputData(kRealTwidInPath, real_twid)));
  XLS_RETURN_IF_ERROR((ReadInputData(kImgTwidInPath, img_twid)));

  fft(real, img, real_twid, img_twid);
  DisplayData(real, "real_out", FFT_SIZE);
  DisplayData(img, "img_out", FFT_SIZE);

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  xls::InitXls(argv[0], argc, argv);
  QCHECK_OK(xls::RealMain());
  return 0;
}
