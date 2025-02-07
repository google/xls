// Copyright 2022 The XLS Authors
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

#ifndef XLS_DSLX_WARNING_COLLECTOR_H_
#define XLS_DSLX_WARNING_COLLECTOR_H_

#include <string>
#include <utility>
#include <vector>

#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/warning_kind.h"

namespace xls::dslx {

// Type-safe wrapper around a vector for accumulating warnings during the
// frontend processing of DSL files.
//
// The WarningKindSet given on construction determines what errors are enabled
// -- all non-enabled errors are dropped when they're attempted to be added to
// the collector.
class WarningCollector {
 public:
  struct Entry {
    Span span;
    WarningKind kind;
    std::string message;
  };

  explicit WarningCollector(WarningKindSet enabled) : enabled_(enabled) {}

  void Add(Span span, WarningKind kind, std::string message) {
    if (WarningIsEnabled(enabled_, kind)) {
      warnings_.push_back(Entry{std::move(span), kind, std::move(message)});
    }
  }

  const std::vector<Entry>& warnings() const { return warnings_; }

  bool empty() const { return warnings_.empty(); }

 private:
  const WarningKindSet enabled_;
  std::vector<Entry> warnings_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_WARNING_COLLECTOR_H_
