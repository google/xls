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

#ifndef XLS_DSLX_FRONTEND_POS_H_
#define XLS_DSLX_FRONTEND_POS_H_

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/strong_int.h"

namespace xls::dslx {

// Note that fileno 0 is generally reserved for "no known file", as in the
// default constructor of a pos.
XLS_DEFINE_STRONG_INT_TYPE(Fileno, uint32_t);

// Holds file paths in an "interned" arena that can be indexed by a file
// number. This prevents string copies for "flyweight" style objects like
// `Pos`.
class FileTable {
 public:
  FileTable() {
    number_to_path_.emplace(Fileno(0), "<no-file>");
    path_to_number_.emplace("<no-file>", Fileno(0));
  }

  Fileno GetOrCreate(std::string_view path) {
    auto it = path_to_number_.find(path);
    if (it == path_to_number_.end()) {
      Fileno this_fileno = next_fileno_++;
      path_to_number_.emplace_hint(it, std::string(path), this_fileno);
      number_to_path_.emplace(this_fileno, std::string(path));
      return this_fileno;
    }
    return it->second;
  }

  std::string_view Get(Fileno fileno) const {
    return number_to_path_.at(fileno);
  }

 private:
  Fileno next_fileno_ = Fileno(1);
  absl::flat_hash_map<Fileno, std::string> number_to_path_;
  absl::flat_hash_map<std::string, Fileno> path_to_number_;
};

// Represents a position in the text (file, line, column).
class Pos {
 public:
  static absl::StatusOr<Pos> FromString(std::string_view s,
                                        FileTable& file_table);

  Pos() : fileno_(Fileno(0)), lineno_(0), colno_(0) {}

  Pos(Fileno fileno, int64_t lineno, int64_t colno)
      : fileno_(fileno), lineno_(lineno), colno_(colno) {}

  std::string ToString(const FileTable& table) const {
    return absl::StrFormat("%s:%d:%d", table.Get(fileno_), lineno_ + 1,
                           colno_ + 1);
  }
  std::string ToStringNoFile() const {
    return absl::StrFormat("%d:%d", lineno_ + 1, colno_ + 1);
  }

  std::string ToRepr(const FileTable& table) const {
    return absl::StrFormat("Pos(\"%s\", %d, %d)", GetFilename(table), lineno_,
                           colno_);
  }

  bool operator<(const Pos& other) const {
    CHECK_EQ(fileno_, other.fileno_);
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
    CHECK_EQ(fileno_, other.fileno_);
    return lineno_ == other.lineno_ && colno_ == other.colno_;
  }
  bool operator!=(const Pos& other) const { return !(*this == other); }
  bool operator<=(const Pos& other) const {
    return (*this) < other || (*this) == other;
  }
  bool operator>(const Pos& other) const { return !(*this <= other); }
  bool operator>=(const Pos& other) const { return !((*this) < other); }

  std::string_view GetFilename(const FileTable& table) const {
    return table.Get(fileno_);
  }

  Fileno fileno() const { return fileno_; }

  // Note: these lineno/colno values are zero-based.
  int64_t lineno() const { return lineno_; }
  int64_t colno() const { return colno_; }

  // Returns a line number as we'd display it to a human (i.e. with one-based
  // indexing, the first line of the file is called "line 1" instead of "line
  // 0").
  int64_t GetHumanLineno() const { return lineno_ + 1; }

  Pos BumpCol() const { return Pos(fileno_, lineno_, colno_ + 1); }

 private:
  Fileno fileno_;
  int64_t lineno_;
  int64_t colno_;
};

inline std::ostream& operator<<(std::ostream& os, const Pos& pos) {
  os << absl::StreamFormat("Fileno(%d):%d:%d", pos.fileno().value(),
                           pos.lineno() + 1, pos.colno() + 1);
  return os;
}

// Represents a positional span in the text.
class Span {
 public:
  static absl::StatusOr<Span> FromString(std::string_view s,
                                         FileTable& file_table);
  static Span Fake() { return Span(Pos(), Pos()); }

  Span(Pos start, Pos limit) : start_(start), limit_(limit) {
    CHECK_EQ(start_.fileno(), limit_.fileno());
    CHECK(start_ <= limit_);
  }
  Span() = default;

  std::ostream& operator<<(std::ostream& os) const {
    os << start() << "-" << limit().lineno() + 1 << ":" << limit().colno() + 1;
    return os;
  }

  std::string_view GetFilename(const FileTable& table) const {
    return start().GetFilename(table);
  }
  const Pos& start() const { return start_; }
  const Pos& limit() const { return limit_; }
  Fileno fileno() const { return start().fileno(); }

  bool operator==(const Span& other) const {
    return start_ == other.start_ && limit_ == other.limit_;
  }
  bool operator!=(const Span& other) const { return !(*this == other); }

  std::string ToString(const FileTable& table) const {
    return absl::StrFormat("%s-%d:%d", start().ToString(table),
                           limit().lineno() + 1, limit().colno() + 1);
  }

  std::string ToRepr(const FileTable& table) const {
    return absl::StrFormat("Span(%s, %s)", start().ToRepr(table),
                           limit().ToRepr(table));
  }

  bool empty() const { return start_ == limit_; }

  // Returns true iff the given "pos" is contained within this span.
  //
  // (Note that spans have an exclusive limit position, so the limit position is
  // not considered to be contained in the span.)
  bool Contains(const Pos& pos) const { return start_ <= pos && pos < limit_; }

  // Returns whether "span" is contained within this span.
  //
  // Note: a span is considered to be contained inside of itself.
  bool Contains(const Span& span) const {
    return start() <= span.start() && span.limit() <= limit();
  }

  bool operator<(const Span& other) const {
    return start_ < other.start_ ||
           (start_ == other.start_ && limit_ < other.limit_);
  }

 private:
  Pos start_;
  Pos limit_;
};

// Helper that allows conversion of optional Spans to strings.
inline std::string SpanToString(const std::optional<Span>& span,
                                const FileTable& table) {
  if (!span.has_value()) {
    return "<no span>";
  }
  return span->ToString(table);
}

// Returns a "fake" span suitable for use in creating testing ASTs.
Span FakeSpan();

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_POS_H_
