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

#include "xls/dslx/frontend/pos.h"

#include <cstdint>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "re2/re2.h"

namespace xls::dslx {

/* static */ absl::StatusOr<Span> Span::FromString(std::string_view s,
                                                   FileTable& file_table) {
  static const LazyRE2 kFileLocationSpanRe = {
      .pattern_ = R"((.*):(\d+):(\d+)-(\d+):(\d+))"};
  std::string_view filename;
  int64_t start_lineno, start_colno, limit_lineno, limit_colno;
  if (RE2::FullMatch(s, *kFileLocationSpanRe, &filename, &start_lineno,
                     &start_colno, &limit_lineno, &limit_colno)) {
    // The values used for display are 1-based, whereas the backing storage is
    // zero-based.
    const Fileno fileno = file_table.GetOrCreate(filename);
    return Span(Pos(fileno, start_lineno - 1, start_colno - 1),
                Pos(fileno, limit_lineno - 1, limit_colno - 1));
  }

  return absl::InvalidArgumentError(
      absl::StrFormat("Cannot convert string to span: \"%s\"", s));
}

/* static */ absl::StatusOr<Pos> Pos::FromString(std::string_view s,
                                                 FileTable& file_table) {
  static const LazyRE2 kFileLocationRe = {.pattern_ = R"((.*):(\d+):(\d+))"};
  std::string_view filename;
  int64_t lineno, colno;
  if (RE2::FullMatch(s, *kFileLocationRe, &filename, &lineno, &colno)) {
    const Fileno fileno = file_table.GetOrCreate(filename);
    return Pos(fileno, lineno - 1, colno - 1);
  }

  return absl::InvalidArgumentError(
      absl::StrFormat("Cannot convert string to position: \"%s\"", s));
}

Span FakeSpan() {
  Pos fake_pos(Fileno(0), 0, 0);
  return Span(fake_pos, fake_pos);
}

}  // namespace xls::dslx
