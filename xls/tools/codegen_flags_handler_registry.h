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

#ifndef XLS_TOOLS_CODEGEN_FLAGS_HANDLER_REGISTRY_H_
#define XLS_TOOLS_CODEGEN_FLAGS_HANDLER_REGISTRY_H_

#include <functional>
#include <string_view>

#include "absl/status/status.h"
#include "xls/codegen/codegen_options.h"
#include "xls/tools/codegen_flags.pb.h"

namespace xls {

using CodegenFlagsHandler = std::function<absl::Status(
    const CodegenFlagsProto&, verilog::CodegenOptions&)>;
using CodegenFlagsParser = std::function<absl::Status(CodegenFlagsProto&)>;

// Registry to manage custom codegen option extensions.
// Enables downstream pass modules to register flag-parsing hooks and options
// handler callbacks. At runtime, the registry parses command-line flags into
// The CodegenFlagsProto (keeping them trackable/serializable) and then
// configures the C++ verilog::CodegenOptions container.
//
// Expected usage:
// 1. Declare proto extension fields on CodegenFlagsProto in a local proto file.
// 2. Define custom properties in a verilog::CodegenOptionExtension C++ struct.
// 3. Register your pass handlers using the registration macro helper.
//
// Example registration:
//   ABSL_FLAG(std::string, my_pass_config, "none", "My pass setting");
//
//   absl::Status MyHandler(const CodegenFlagsProto& proto,
//                          verilog::CodegenOptions& options) {
//     if (proto.HasExtension(my_extension_id)) {
//        options.add_extension(std::make_unique<MyOptions>(...));
//     }
//     return absl::OkStatus();
//   }
//
//   absl::Status MyParser(CodegenFlagsProto& proto) {
//     std::string val = absl::GetFlag(FLAGS_my_pass_config);
//     if (val != "none") {
//       proto.SetExtension(my_extension_id, val);
//     }
//     return absl::OkStatus();
//   }
//
//   XLS_REGISTER_CODEGEN_FLAGS_HANDLER("my_pass", MyHandler, MyParser);
class CodegenFlagsHandlerRegistry {
 public:
  static void Register(std::string_view name, CodegenFlagsHandler handler,
                       CodegenFlagsParser parser);
  static absl::Status ParseFlags(CodegenFlagsProto& proto);
  static absl::Status Process(const CodegenFlagsProto& proto,
                              verilog::CodegenOptions& options);
};

#define XLS_REGISTER_CODEGEN_FLAGS_HANDLER(name, handler, parser) \
  XLS_REGISTER_CODEGEN_FLAGS_HANDLER_UNIQHelper(__LINE__, name, handler, parser)

#define XLS_REGISTER_CODEGEN_FLAGS_HANDLER_UNIQHelper(line, name, handler, \
                                                      parser)              \
  XLS_REGISTER_CODEGEN_FLAGS_HANDLER_UNIQ(line, name, handler, parser)

#define XLS_REGISTER_CODEGEN_FLAGS_HANDLER_UNIQ(line, name, handler, parser) \
  static bool const registrator_##line = []() {                              \
    ::xls::CodegenFlagsHandlerRegistry::Register(name, handler, parser);     \
    return true;                                                             \
  }();

}  // namespace xls

#endif  // XLS_TOOLS_CODEGEN_FLAGS_HANDLER_REGISTRY_H_
