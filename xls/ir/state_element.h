// Copyright 2024 The XLS Authors
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

#ifndef XLS_IR_STATE_ELEMENT_H_
#define XLS_IR_STATE_ELEMENT_H_

#include <string>
#include <string_view>
#include <utility>

#include "absl/log/check.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"

namespace xls {

// Data structure representing a proc state element. These constructs are
// contained in and owned by Procs.
class StateElement {
 public:
  StateElement(std::string_view name, Type* type, Value initial_value)
      : name_(name), type_(type), initial_value_(initial_value) {
    CHECK(ValueConformsToType(initial_value, type));
  }

  const std::string& name() const { return name_; }
  Type* type() const { return type_; }
  const Value& initial_value() const { return initial_value_; }

  void SetName(std::string_view name) { name_ = name; }
  void SetName(std::string&& name) { name_ = std::move(name); }

  std::string ToString() const;

 private:
  std::string name_;
  Type* type_;
  Value initial_value_;
};

}  // namespace xls

#endif  // XLS_IR_STATE_ELEMENT_H_
