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

#include "xls/codegen/codegen_options.h"

#include "xls/common/proto_adaptor_utils.h"

namespace xls::verilog {

CodegenOptions& CodegenOptions::entry(absl::string_view name) {
  entry_ = name;
  return *this;
}

CodegenOptions& CodegenOptions::module_name(absl::string_view name) {
  module_name_ = name;
  return *this;
}

CodegenOptions& CodegenOptions::reset(absl::string_view name, bool asynchronous,
                                      bool active_low) {
  reset_proto_ = ResetProto();
  reset_proto_->set_name(ToProtoString(name));
  reset_proto_->set_asynchronous(asynchronous);
  reset_proto_->set_active_low(active_low);
  return *this;
}

CodegenOptions& CodegenOptions::clock_name(absl::string_view clock_name) {
  clock_name_ = std::move(clock_name);
  return *this;
}

CodegenOptions& CodegenOptions::use_system_verilog(bool value) {
  use_system_verilog_ = value;
  return *this;
}

CodegenOptions& CodegenOptions::flop_inputs(bool value) {
  flop_inputs_ = value;
  return *this;
}

CodegenOptions& CodegenOptions::flop_outputs(bool value) {
  flop_outputs_ = value;
  return *this;
}

CodegenOptions& CodegenOptions::assert_format(absl::string_view value) {
  assert_format_ = std::string{value};
  return *this;
}

}  // namespace xls::verilog
