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

#ifndef XLS_IR_NAME_UNIQUER_H_
#define XLS_IR_NAME_UNIQUER_H_

#include <cstdint>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/types/span.h"

namespace xls {

// Class for generating unique names. Keeps track of names that have currently
// been seen/generated. The names returned by GetUniqueName are guaranteed to
// be distinct for this instance of the class.  The names will be
// sanitized to match regexp "[a-zA-Z_][a-zA-Z0-9_]*".
class NameUniquer {
 public:
  explicit NameUniquer(std::string_view separator,
                       absl::Span<const std::string> reserved_names = {})
      : separator_(separator),
        reserved_names_(reserved_names.begin(), reserved_names.end()) {}

  NameUniquer(const NameUniquer&) = delete;
  NameUniquer operator=(const NameUniquer&) = delete;

  // Return a sanitized unique name which starts with the given (sanitized)
  // prefix. Names are uniqued by adding a numeric suffix if necessary separated
  // from the given prefix by "separator_". For example,
  // GetSanitizedUniqueName("foo") might return "foo__1" if "foo" is not
  // available.
  std::string GetSanitizedUniqueName(std::string_view prefix);

  // Returns true if the given str is a valid Verilog, and thus XLS, identifier.
  static bool IsValidIdentifier(std::string_view str);

 private:
  // Used to track and generate new identifiers for the same instruction name
  // root.
  class SequentialIdGenerator {
   public:
    SequentialIdGenerator() = default;

    // Tries to register the given ID. If the ID is not registered already,
    // register the ID and return it. Otherwise if the ID is registered return
    // the next available ID.
    int64_t RegisterId(int64_t id) {
      int64_t result;
      if (used_.insert(id).second) {
        // ID has not been used before.
        result = id;
      } else {
        // ID has been used before, find the next one
        CHECK(used_.insert(next_).second);
        result = next_;
      }

      // Advance next_ to the first unregistered value.
      while (used_.contains(next_)) {
        ++next_;
      }

      return result;
    }

    // Returns the next available unique ID.
    int64_t NextId() { return RegisterId(next_); }

   private:
    // The next identifier to be tried.
    int64_t next_ = 1;

    // Set of all the identifiers which has been used.
    absl::flat_hash_set<int64_t> used_;
  };

  // The string to use to separate the prefix of the name from the uniquing
  // integer value.
  std::string separator_;

  // Set of reserved names. Names in this set will be sanitized by adding a '_'
  // prefix.
  absl::flat_hash_set<std::string> reserved_names_;

  // Map from name prefix to the generator data structure which tracks used
  // identifiers and generates new ones.
  struct PrefixTracker {
    // Whether the bare prefix (no numeric suffix) is taken as a name.
    bool bare_prefix_taken = false;
    // Numeric suffix generator for guaranteeing uniqueness.
    SequentialIdGenerator generator;
  };
  absl::flat_hash_map<std::string, PrefixTracker> generated_names_;
};

}  // namespace xls

#endif  // XLS_IR_NAME_UNIQUER_H_
