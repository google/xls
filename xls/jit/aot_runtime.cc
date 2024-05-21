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
#include "xls/jit/aot_runtime.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "google/protobuf/text_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/package.h"
#include "xls/jit/type_layout.h"
#include "xls/jit/type_layout.pb.h"

namespace xls::aot_compile {

/* static */ absl::StatusOr<std::unique_ptr<FunctionTypeLayout>>
FunctionTypeLayout::Create(std::string_view serialized_arg_layouts,
                           std::string_view serialized_result_layout) {
  auto dummy_package = std::make_unique<Package>("__aot_compiler");

  TypeLayoutsProto arg_layouts_proto;
  if (!google::protobuf::TextFormat::ParseFromString(
          std::string(serialized_arg_layouts), &arg_layouts_proto)) {
    return absl::InvalidArgumentError(
        "Unable to parse TypeLayoutsProto for arguments");
  }
  std::vector<TypeLayout> arg_layouts;
  for (const TypeLayoutProto& layout_proto : arg_layouts_proto.layouts()) {
    XLS_ASSIGN_OR_RETURN(
        TypeLayout arg_layout,
        TypeLayout::FromProto(layout_proto, dummy_package.get()));
    arg_layouts.push_back(std::move(arg_layout));
  }
  TypeLayoutProto result_layout_proto;
  if (!google::protobuf::TextFormat::ParseFromString(
          std::string(serialized_result_layout), &result_layout_proto)) {
    return absl::InvalidArgumentError(
        "Unable to parse TypeLayoutProto for result");
  }
  XLS_ASSIGN_OR_RETURN(
      TypeLayout result_layout,
      TypeLayout::FromProto(result_layout_proto, dummy_package.get()));
  return absl::WrapUnique(new FunctionTypeLayout(std::move(dummy_package),
                                                 std::move(arg_layouts),
                                                 std::move(result_layout)));
}

}  // namespace xls::aot_compile
