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

#include "xls/dslx/pos.h"

#include "absl/status/statusor.h"
#include "re2/re2.h"

namespace xls::dslx {

/* static */ absl::StatusOr<Span> Span::FromString(std::string_view s) {
  std::string filename;
  int64_t start_lineno, start_colno, limit_lineno, limit_colno;
  if (RE2::FullMatch(s, R"((.*):(\d+):(\d+)-(\d+):(\d+))", &filename,
                     &start_lineno, &start_colno, &limit_lineno,
                     &limit_colno)) {
    // The values used for display are 1-based, whereas the backing storage is
    // zero-based.
    return Span(Pos(filename, start_lineno - 1, start_colno - 1),
                Pos(filename, limit_lineno - 1, limit_colno - 1));
  }

  return absl::InvalidArgumentError(
      absl::StrFormat("Cannot convert string to span: \"%s\"", s));
}

/* static */ absl::StatusOr<Pos> Pos::FromString(std::string_view s) {
  std::string filename;
  int64_t lineno, colno;
  if (RE2::FullMatch(s, "(.*):(\\d+):(\\d+)", &filename, &lineno, &colno)) {
    return Pos(filename, lineno - 1, colno - 1);
  }

  return absl::InvalidArgumentError(
      absl::StrFormat("Cannot convert string to position: \"%s\"", s));
}

Span FakeSpan() {
  Pos fake_pos("<fake>", 0, 0);
  return Span(fake_pos, fake_pos);
}

}  // namespace xls::dslx
