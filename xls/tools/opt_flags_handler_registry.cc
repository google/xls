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

#include "xls/tools/opt_flags_handler_registry.h"

#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "xls/common/status/status_macros.h"
#include "xls/passes/optimization_pass.h"
#include "xls/tools/opt_flags.pb.h"

namespace xls {
namespace {

class RegistryList {
 public:
  static RegistryList& Get();
  absl::Status ParseFlags(OptFlagsProto& proto) {
    for (const auto& registered : registry_) {
      if (registered.parser != nullptr) {
        XLS_RETURN_IF_ERROR(registered.parser(proto));
      }
    }
    return absl::OkStatus();
  }
  absl::Status Process(const OptFlagsProto& proto,
                       OptimizationPassOptions& options) {
    for (const auto& registered : registry_) {
      if (registered.handler != nullptr) {
        XLS_RETURN_IF_ERROR(registered.handler(proto, options));
      }
    }
    return absl::OkStatus();
  }
  void Register(std::string_view name, OptFlagsHandler handler,
                OptFlagsParser parser) {
    registry_.push_back({name, std::move(handler), std::move(parser)});
  }

  struct RegisteredHandler {
    std::string_view name;
    OptFlagsHandler handler;
    OptFlagsParser parser;
  };

 private:
  std::vector<RegisteredHandler> registry_;
};

RegistryList& RegistryList::Get() {
  static RegistryList registry;
  return registry;
}

}  // namespace

void OptFlagsHandlerRegistry::Register(std::string_view name,
                                       OptFlagsHandler handler,
                                       OptFlagsParser parser) {
  RegistryList::Get().Register(name, std::move(handler), std::move(parser));
}

absl::Status OptFlagsHandlerRegistry::ParseFlags(OptFlagsProto& proto) {
  return RegistryList::Get().ParseFlags(proto);
}

absl::Status OptFlagsHandlerRegistry::Process(
    const OptFlagsProto& proto, OptimizationPassOptions& options) {
  return RegistryList::Get().Process(proto, options);
}

}  // namespace xls
