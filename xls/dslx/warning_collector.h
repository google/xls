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
#include <vector>

#include "xls/dslx/pos.h"

namespace xls::dslx {

// Type-safe wrapper around a vector for accumulating warnings during the
// frontend processing of DSL files.
//
// Implementation note: this may grow filtering mechanisms a la `-Wno-X` which
// is why we spring for a type safe wrapper around of the gate.
class WarningCollector {
 public:
  struct Entry {
    Span span;
    std::string message;
  };

  void Add(Span span, std::string message) {
    warnings_.push_back(Entry{std::move(span), std::move(message)});
  }

  const std::vector<Entry>& warnings() const { return warnings_; }

 private:
  std::vector<Entry> warnings_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_WARNING_COLLECTOR_H_
