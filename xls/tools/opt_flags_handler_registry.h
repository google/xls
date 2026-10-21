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

#ifndef XLS_TOOLS_OPT_FLAGS_HANDLER_REGISTRY_H_
#define XLS_TOOLS_OPT_FLAGS_HANDLER_REGISTRY_H_

#include <functional>
#include <string_view>

#include "absl/status/status.h"
#include "xls/common/module_initializer.h"
#include "xls/passes/optimization_pass.h"
#include "xls/tools/opt_flags.pb.h"

namespace xls {

using OptFlagsHandler =
    std::function<absl::Status(const OptFlagsProto&, OptimizationPassOptions&)>;
using OptFlagsParser = std::function<absl::Status(OptFlagsProto&)>;

// Registry to manage custom opt_main option extensions.
// Enables downstream pass modules to register flag-parsing hooks and options
// handler callbacks. At runtime, the registry parses command-line flags into
// The OptFlagsProto (keeping them trackable/serializable) and then
// configures the C++ OptimizationPassOptions container.
//
// Expected usage:
// 1. Declare proto extension fields on OptFlagsProto in a local proto file.
// 2. Define custom properties in a OptOptionExtension C++ struct.
// 3. Register your pass handlers using the registration macro helper.
//
// Example registration:
//   namespace {
//   ABSL_FLAG(std::string, my_pass_config, "none", "My pass setting");
//   class MyOptions : public OptimizationPassOptionExtension {
//    public:
//     static constexpr std::string_view kExtensionName = "my_options";
//     MyOptions(int x) : x_(x) {}
//     std::string_view extension_name() const override {
//       return kExtensionName;
//     }
//     int x() const { return x_; }
//    private:
//     int x_;
//     ...
//   };
//
//   absl::Status MyHandler(const OptFlagsProto& proto,
//                          OptimizationPassOptions& options) {
//     if (proto.HasExtension(my_extension_id)) {
//        XLS_RETURN_IF_ERROR(options.AddExtension(std::make_unique<MyOptions>(...));
//     }
//     return absl::OkStatus();
//   }
//
//   absl::Status MyParser(OptFlagsProto& proto) {
//     std::string val = absl::GetFlag(FLAGS_my_pass_config);
//     if (val != "none") {
//       proto.SetExtension(my_extension_id, val);
//     }
//     return absl::OkStatus();
//   }
//
//   XLS_REGISTER_OPT_FLAGS_HANDLER(my_pass, MyHandler, MyParser);
//   }  // namespace
class OptFlagsHandlerRegistry {
 public:
  // Register the given handler.
  //
  // NB Not thread safe.
  static void Register(std::string_view name, OptFlagsHandler handler,
                       OptFlagsParser parser);
  static absl::Status ParseFlags(OptFlagsProto& proto);
  static absl::Status Process(const OptFlagsProto& proto,
                              OptimizationPassOptions& options);
};

#define XLS_REGISTER_OPT_FLAGS_HANDLER(name, handler, parser)         \
  XLS_REGISTER_MODULE_INITIALIZER(name, {                             \
    ::xls::OptFlagsHandlerRegistry::Register(#name, handler, parser); \
  });

}  // namespace xls

#endif  // XLS_TOOLS_OPT_FLAGS_HANDLER_REGISTRY_H_
