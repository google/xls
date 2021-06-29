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

#ifndef XLS_DSLX_POS_H_
#define XLS_DSLX_POS_H_

#include <cstdint>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/logging/logging.h"

namespace xls::dslx {

// Represents a position in the text (file, line, column).
class Pos {
 public:
  static absl::StatusOr<Pos> FromString(absl::string_view s);

  Pos() : filename_(""), lineno_(0), colno_(0) {}
  Pos(std::string filename, int64_t lineno, int64_t colno)
      : filename_(std::move(filename)), lineno_(lineno), colno_(colno) {}

  std::string ToString() const {
    return absl::StrFormat("%s:%d:%d", filename_, lineno_ + 1, colno_ + 1);
  }
  std::string ToStringNoFile() const {
    return absl::StrFormat("%d:%d", lineno_ + 1, colno_ + 1);
  }

  std::string ToRepr() const {
    return absl::StrFormat("Pos(\"%s\", %d, %d)", filename_, lineno_, colno_);
  }

  bool operator<(const Pos& other) const {
    XLS_CHECK_EQ(filename_, other.filename_);
    if (lineno_ < other.lineno_) {
      return true;
    }
    if (lineno_ > other.lineno_) {
      return false;
    }
    if (colno_ < other.colno_) {
      return true;
    }
    return false;
  }
  bool operator==(const Pos& other) const {
    XLS_CHECK_EQ(filename_, other.filename_);
    return lineno_ == other.lineno_ && colno_ == other.colno_;
  }
  bool operator!=(const Pos& other) const { return !(*this == other); }
  bool operator<=(const Pos& other) const {
    return (*this) < other || (*this) == other;
  }
  bool operator>=(const Pos& other) const { return !((*this) < other); }

  const std::string& filename() const { return filename_; }
  int64_t lineno() const { return lineno_; }
  int64_t colno() const { return colno_; }

  Pos BumpCol() const { return Pos(filename_, lineno_, colno_ + 1); }

 private:
  std::string filename_;
  int64_t lineno_;
  int64_t colno_;
};

inline std::ostream& operator<<(std::ostream& os, const Pos& pos) {
  os << pos.ToString();
  return os;
}

// Represents a positional span in the text.
class Span {
 public:
  static absl::StatusOr<Span> FromString(absl::string_view s);
  static Span Fake() { return Span(Pos(), Pos()); }

  Span(Pos start, Pos limit)
      : start_(std::move(start)), limit_(std::move(limit)) {
    XLS_CHECK_EQ(start_.filename(), limit_.filename());
    XLS_CHECK_LE(start_, limit_);
  }
  Span() = default;

  const std::string& filename() const { return start().filename(); }
  const Pos& start() const { return start_; }
  const Pos& limit() const { return limit_; }

  bool operator==(const Span& other) const {
    return start_ == other.start_ && limit_ == other.limit_;
  }
  bool operator!=(const Span& other) const { return !(*this == other); }

  std::string ToString() const {
    return absl::StrFormat("%s-%d:%d", start().ToString(), limit().lineno() + 1,
                           limit().colno() + 1);
  }

  std::string ToRepr() const {
    return absl::StrFormat("Span(%s, %s)", start().ToRepr(), limit().ToRepr());
  }

  Span CloneWithLimit(Pos limit) const { return Span(start_, limit); }

 private:
  Pos start_;
  Pos limit_;
};

inline std::ostream& operator<<(std::ostream& os, const Span& span) {
  os << span.ToString();
  return os;
}

// Helper that allows conversion of optional Spans to strings.
inline std::string SpanToString(const absl::optional<Span>& span) {
  if (!span.has_value()) {
    return "<no span>";
  }
  return span->ToString();
}

}  // namespace xls::dslx

#endif  // XLS_DSLX_POS_H_
