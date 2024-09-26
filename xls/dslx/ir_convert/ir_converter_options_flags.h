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

#ifndef XLS_DSLX_IR_CONVERT_IR_CONVERTER_OPTIONS_FLAGS_H_
#define XLS_DSLX_IR_CONVERT_IR_CONVERTER_OPTIONS_FLAGS_H_

#include <optional>

#include "absl/flags/declare.h"
#include "absl/status/statusor.h"
#include "xls/dslx/ir_convert/ir_converter_options_flags.pb.h"

ABSL_DECLARE_FLAG(std::optional<std::string>,
                  ir_converter_options_used_textproto_file);

namespace xls {

absl::StatusOr<IrConverterOptionsFlagsProto> GetIrConverterOptionsFlagsProto();

}  // namespace xls

#endif  // XLS_DSLX_IR_CONVERT_IR_CONVERTER_OPTIONS_FLAGS_H_
