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

#include "xls/tools/codegen_flags_handler_registry.h"

#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "xls/codegen/codegen_options.h"
#include "xls/common/status/status_macros.h"
#include "xls/tools/codegen_flags.pb.h"

namespace xls {
namespace {

struct RegisteredHandler {
  std::string_view name;
  CodegenFlagsHandler handler;
  CodegenFlagsParser parser;
};

std::vector<RegisteredHandler>& GetRegistry() {
  static auto* registry = new std::vector<RegisteredHandler>();
  return *registry;
}

}  // namespace

void CodegenFlagsHandlerRegistry::Register(std::string_view name,
                                           CodegenFlagsHandler handler,
                                           CodegenFlagsParser parser) {
  GetRegistry().push_back({name, std::move(handler), std::move(parser)});
}

absl::Status CodegenFlagsHandlerRegistry::ParseFlags(CodegenFlagsProto& proto) {
  for (const auto& registered : GetRegistry()) {
    if (registered.parser != nullptr) {
      XLS_RETURN_IF_ERROR(registered.parser(proto));
    }
  }
  return absl::OkStatus();
}

absl::Status CodegenFlagsHandlerRegistry::Process(
    const CodegenFlagsProto& proto, verilog::CodegenOptions& options) {
  for (const auto& registered : GetRegistry()) {
    XLS_RETURN_IF_ERROR(registered.handler(proto, options));
  }
  return absl::OkStatus();
}

}  // namespace xls
