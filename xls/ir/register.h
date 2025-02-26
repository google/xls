// Copyright 2021 The XLS Authors
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

#ifndef XLS_IR_REGISTER_H_
#define XLS_IR_REGISTER_H_

#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"

namespace xls {

struct ResetBehavior {
  bool asynchronous;
  bool active_low;

  bool operator==(const ResetBehavior& other) const {
    return asynchronous == other.asynchronous && active_low == other.active_low;
  }
  std::string ToString() const;
};

inline std::ostream& operator<<(std::ostream& os, const ResetBehavior& rb) {
  os << rb.ToString();
  return os;
}

class RegisterWrite;

// Data structure representing a RTL-level register. These constructs are
// contained in and owned by Blocks and lower to registers in Verilog.
class Register {
 public:
  Register(std::string_view name, Type* type,
           std::optional<Value> reset_value = std::nullopt)
      : name_(name), type_(type), reset_value_(std::move(reset_value)) {}

  const std::string& name() const { return name_; }
  Type* type() const { return type_; }
  const std::optional<Value>& reset_value() const { return reset_value_; }

  std::string ToString() const;
  absl::Status SetResetValue(std::optional<Value> reset_value) {
    if (reset_value.has_value() &&
        !ValueConformsToType(reset_value.value(), type())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid reset value `%s` is not the same type as the register: %s",
          reset_value->ToString(), type()->ToString()));
    }
    reset_value = reset_value_;
    return absl::OkStatus();
  }

 private:
  std::string name_;
  Type* type_;
  std::optional<Value> reset_value_;
};

}  // namespace xls

#endif  // XLS_IR_REGISTER_H_
