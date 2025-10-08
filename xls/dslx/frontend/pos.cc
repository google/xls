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
#include <string>
#include <string_view>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "xls/dslx/status_payload.pb.h"
#include "re2/re2.h"

namespace xls::dslx {

Fileno FileTable::GetOrCreate(std::string_view path) {
  CHECK(!absl::StartsWith(path, "file://"))
      << "FileTable does not support URIs as paths: " << path;
  VLOG(5) << absl::StreamFormat("FileTable::GetOrCreate: %s", path);
  auto it = path_to_number_.find(path);
  if (it == path_to_number_.end()) {
    Fileno this_fileno = next_fileno_++;
    path_to_number_.emplace_hint(it, std::string(path), this_fileno);
    number_to_path_.emplace(this_fileno, std::string(path));
    VLOG(5) << "Added to filetable";
    return this_fileno;
  }
  VLOG(5) << "Already in filetable";
  return it->second;
}

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

PosProto ToProto(const Pos& pos, const FileTable& file_table) {
  PosProto proto;
  proto.set_filename(pos.GetFilename(file_table));
  proto.set_lineno(static_cast<int32_t>(pos.lineno()));
  proto.set_colno(static_cast<int32_t>(pos.colno()));
  return proto;
}

SpanProto ToProto(const Span& span, const FileTable& file_table) {
  SpanProto proto;
  *proto.mutable_start() = ToProto(span.start(), file_table);
  *proto.mutable_limit() = ToProto(span.limit(), file_table);
  return proto;
}

std::string ToHumanString(const SpanProto& proto, bool v2) {
  if (v2) {
    return absl::StrFormat(
        "%s:%d:%d-%d:%d", proto.start().filename(), proto.start().lineno() + 1,
        proto.start().colno() + 1, proto.limit().lineno() + 1,
        proto.limit().colno() + 1);
  } else {
    return absl::StrFormat("%d:%d-%d:%d", proto.start().lineno(),
                           proto.start().colno(), proto.limit().lineno(),
                           proto.limit().colno());
  }
}

}  // namespace xls::dslx
