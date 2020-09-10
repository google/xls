// Copyright 2020 Google LLC
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

#include "xls/dslx/cpp_pos.h"

#include "re2/re2.h"

namespace xls::dslx {

/* static */ xabsl::StatusOr<Pos> Pos::FromString(absl::string_view s) {
  std::string filename;
  int64 lineno, colno;
  if (RE2::FullMatch(s, "(.*):(\\d+):(\\d+)", &filename, &lineno, &colno)) {
    return Pos(filename, lineno, colno);
  }

  return absl::InvalidArgumentError(
      absl::StrFormat("Cannot convert string to position: \"%s\"", s));
}

}  // namespace xls::dslx
