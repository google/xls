// Copyright 2020 The XLS Authors
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

#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/codegen/module_signature.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/tools/io_strategy.h"
#include "xls/tools/io_strategy_factory.h"
#include "xls/tools/wrap_io.h"

ABSL_FLAG(std::string, wrapped_module_name, "",
          "name of 'device function' module being wrapped");
ABSL_FLAG(std::string, instance_name, "device_function",
          "instance name to use");
ABSL_FLAG(std::string, signature_proto_path, "",
          "path to function signature proto");
ABSL_FLAG(std::string, target_device, "", "target device kind (e.g. ice40)");
ABSL_FLAG(std::string, include, "",
          "path to include with 'device function' module");

namespace xls {
namespace tools {
namespace {

absl::Status RealMain() {
  std::string signature_proto_path = absl::GetFlag(FLAGS_signature_proto_path);
  QCHECK_NE(signature_proto_path, "") << "Must provide -signature_proto_path";

  std::string target_device = absl::GetFlag(FLAGS_target_device);
  QCHECK_NE(target_device, "") << "Must provide -target_device";

  std::string instance_name = absl::GetFlag(FLAGS_instance_name);
  XLS_CHECK_NE(instance_name, "") << "Must provide -instance_name";

  std::string wrapped_module_name = absl::GetFlag(FLAGS_wrapped_module_name);
  XLS_CHECK_NE(wrapped_module_name, "") << "Must provide -wrapped_module_name";

  std::string include = absl::GetFlag(FLAGS_include);
  XLS_CHECK_NE(include, "") << "Must provide -include";

  verilog::ModuleSignatureProto signature_proto;
  QCHECK_OK(ParseTextProtoFile(signature_proto_path, &signature_proto));
  auto signature_status = verilog::ModuleSignature::FromProto(signature_proto);
  QCHECK_OK(signature_status.status());
  verilog::ModuleSignature signature = signature_status.value();

  verilog::VerilogFile f(verilog::FileType::kVerilog);
  f.AddInclude(include, SourceInfo());

  absl::StatusOr<std::unique_ptr<verilog::IOStrategy>> io_strategy_status =
      verilog::IOStrategyFactory::CreateForDevice(target_device, &f);
  QCHECK_OK(io_strategy_status.status());
  auto io_strategy = std::move(io_strategy_status).value();
  absl::StatusOr<verilog::Module*> module_status = verilog::WrapIO(
      wrapped_module_name, instance_name, signature, io_strategy.get(), &f);
  if (module_status.ok()) {
    std::cout << f.Emit() << '\n';
  }
  return module_status.status();
}

}  // namespace
}  // namespace tools
}  // namespace xls

int main(int argc, char** argv) {
  xls::InitXls(argv[0], argc, argv);
  return xls::ExitStatus(xls::tools::RealMain());
}
