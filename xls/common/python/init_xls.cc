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

#include "xls/common/init_xls.h"

#include "absl/base/call_once.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "absl/types/span.h"
#include "pybind11/pybind11.h"
#include "pybind11_abseil/absl_casters.h"

namespace py = pybind11;

namespace xls {
namespace {

// Creates a vector of char* from a std::string span.
std::vector<char*> MakeCharStarVector(absl::Span<const std::string> str_vec) {
  std::vector<char*> char_vec;
  for (const std::string& s : str_vec) {
    char_vec.push_back(const_cast<char*>(s.data()));
  }
  return char_vec;
}

void InitXlsWrapper(absl::Span<const std::string> argv) {
  // Make this function idempotent by wrapping the guts in call_once.
  static absl::once_flag once;
  absl::call_once(once, [&argv] {
    // Pass the python argv to InitXls for flag parsing on the C++
    // side. Unfortunately, any flag which is not defined on the C++ side will
    // result in program termination with an error so specify every flag in argv
    // to --undefok so the flag is ignored if it is not defined.
    std::vector<std::string> argv_flags;
    for (const std::string& arg : argv) {
      if (absl::StartsWith(arg, "-")) {
        std::vector<std::string> split_arg = absl::StrSplit(arg, '=');
        std::string flag_with_dashes = split_arg.front();
        // Strip off one or two dashes at the start of the string.
        argv_flags.push_back(std::string(
            absl::StripPrefix(absl::StripPrefix(flag_with_dashes, "-"), "-")));
      }
    }
    std::vector<std::string> argv_vec(argv.begin(), argv.end());
    argv_vec.push_back("--undefok=" + absl::StrJoin(argv_flags, ","));
    std::vector<char*> char_vec = MakeCharStarVector(argv_vec);
    InitXls("", char_vec.size(), char_vec.data());
  });
}

}  // namespace

PYBIND11_MODULE(init_xls, m) {
  m.def("init_xls", &InitXlsWrapper, py::arg("argv"));
}

}  // namespace xls
