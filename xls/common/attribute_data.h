// Copyright 2026 The XLS Authors
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

#ifndef XLS_COMMON_ATTRIBUTE_DATA_H_
#define XLS_COMMON_ATTRIBUTE_DATA_H_

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace xls {

enum class AttributeKind : uint8_t {
  kCfg,
  kDerive,
  kDslxFormatDisable,
  kExternVerilog,
  kSvType,
  kTest,
  kTestProc,
  kQuickcheck,
  kChannelStrictness,
  kFuzzTest,
};

// Converts an AttributeKind to a string, e.g., "fuzz_test"
std::string AttributeKindToString(AttributeKind kind);

class AttributeData {
 public:
  // Represents a key (first)-value (second) pair to an attribute where both
  // key and value are strings. The value string is either back-ticked
  // or quoted, indicated by the is_backticked field.
  struct StringKeyValueArgument {
    std::string first;
    std::string second;
    bool is_backticked;
  };

  using IntKeyValueArgument = std::pair<std::string, int64_t>;

  // Represents a quoted string argument as opposed to a bare identifier-like
  // string.
  struct StringLiteralArgument {
    std::string text;
  };

  // Note that a std::string argument is the simplest kind and looks like an
  // unquoted identifier, like `test` in `#[cfg(test)]`.
  using Argument = std::variant<std::string, StringLiteralArgument,
                                StringKeyValueArgument, IntKeyValueArgument>;

  AttributeData(AttributeKind kind, std::vector<Argument> args)
      : kind_(kind), args_(std::move(args)) {}

  AttributeData(const AttributeData& other) = default;

  AttributeKind kind() const { return kind_; }
  const std::vector<Argument>& args() const { return args_; }
  void AddArgument(Argument arg) { args_.push_back(std::move(arg)); }

 private:
  const AttributeKind kind_;
  std::vector<Argument> args_;
};

}  // namespace xls

#endif  // XLS_COMMON_ATTRIBUTE_DATA_H_
