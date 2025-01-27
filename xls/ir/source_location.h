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

#ifndef XLS_IR_SOURCE_LOCATION_H_
#define XLS_IR_SOURCE_LOCATION_H_

#include <compare>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/ir/fileno.h"

namespace xls {

// Stores a tuple of unique identifiers specifying a file id and line number.
// SourceLocation objects are added to XLS IR nodes and used for debug tracing.
class SourceLocation {
 public:
  SourceLocation() : SourceLocation(Fileno(0), Lineno(0), Colno(0)) {}
  SourceLocation(Fileno fileno, Lineno lineno, Colno colno)
      : fileno_(fileno), lineno_(lineno), colno_(colno) {}
  SourceLocation(const SourceLocation& other) = default;
  SourceLocation& operator=(const SourceLocation& other) = default;

  Fileno fileno() const { return fileno_; }
  Lineno lineno() const { return lineno_; }
  Colno colno() const { return colno_; }

  std::string ToString() const {
    return absl::StrFormat("%d,%d,%d", fileno_.value(), lineno_.value(),
                           colno_.value());
  }

  std::strong_ordering operator<=>(const SourceLocation& other) const {
    if (fileno_ != other.fileno_) {
      return fileno_.value() <=> other.fileno_.value();
    }
    if (lineno_ != other.lineno_) {
      return lineno_.value() <=> other.lineno_.value();
    }
    return colno_.value() <=> other.colno_.value();
  }

 private:
  Fileno fileno_;
  Lineno lineno_;
  Colno colno_;
};

struct SourceInfo {
  std::vector<SourceLocation> locations;

  SourceInfo() = default;
  explicit SourceInfo(const SourceLocation& loc) : locations({loc}) {}
  explicit SourceInfo(std::vector<SourceLocation>&& locs)
      : locations(std::move(locs)) {}
  explicit SourceInfo(absl::Span<const SourceLocation> locs)
      : locations(locs.begin(), locs.end()) {}

  bool Empty() const { return locations.empty(); }

  std::string ToString() const {
    std::vector<std::string> strings;
    strings.reserve(locations.size());
    for (const SourceLocation& location : locations) {
      strings.push_back(absl::StrFormat("(%s)", location.ToString()));
    }
    return absl::StrFormat("[%s]", absl::StrJoin(strings, ", "));
  }
};

}  // namespace xls

#endif  // XLS_IR_SOURCE_LOCATION_H_
