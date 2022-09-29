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

#include "absl/strings/string_view.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {

// Data structure describing the reset behavior of a register.
struct Reset {
  Value reset_value;
  bool asynchronous;
  bool active_low;
};

class RegisterWrite;

// Data structure representing a RTL-level register. These constructs are
// contained in and owned by Blocks and lower to registers in Verilog.
class Register {
 public:
  Register(std::string_view name, Type* type, std::optional<Reset> reset)
      : name_(name), type_(type), reset_(std::move(reset)) {}

  const std::string& name() const { return name_; }
  Type* type() const { return type_; }
  const std::optional<Reset>& reset() const { return reset_; }

  std::string ToString() const;

 private:
  // RegisterWrite can access Register so that it can update
  // reset information.
  friend RegisterWrite;
  void UpdateReset(Reset reset_info) { reset_ = reset_info; }

  std::string name_;
  Type* type_;
  std::optional<Reset> reset_;
};

}  // namespace xls

#endif  // XLS_IR_REGISTER_H_
