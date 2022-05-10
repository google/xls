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

#include "google/protobuf/text_format.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/include/llvm/IR/DataLayout.h"
#include "llvm/include/llvm/IR/LLVMContext.h"
#include "llvm/include/llvm/Support/TargetSelect.h"

namespace xls::aot_compile {

std::unique_ptr<GlobalData> InitGlobalData(
    absl::string_view fn_type_textproto) {
  llvm::InitializeNativeTarget();
  auto ctx = std::make_unique<llvm::LLVMContext>();
  auto error_or_target_builder =
      llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!error_or_target_builder) {
    std::cerr << absl::StrCat(
        "Unable to detect host: ",
        llvm::toString(error_or_target_builder.takeError()));
    exit(1);
  }

  auto error_or_target_machine = error_or_target_builder->createTargetMachine();
  if (!error_or_target_machine) {
    std::cerr << absl::StrCat(
        "Unable to create target machine: ",
        llvm::toString(error_or_target_machine.takeError()));
    exit(1);
  }

  std::unique_ptr<llvm::TargetMachine> target_machine =
      std::move(error_or_target_machine.get());
  llvm::DataLayout data_layout = target_machine->createDataLayout();
  auto type_converter =
      std::make_unique<::xls::LlvmTypeConverter>(ctx.get(), data_layout);
  auto global_data = absl::WrapUnique(
      new GlobalData{std::move(ctx), data_layout, std::move(type_converter),
                     ::xls::Package("dummy")});
  ::xls::FunctionTypeProto fn_type_proto;
  google::protobuf::TextFormat::ParseFromString(std::string(fn_type_textproto),
                                      &fn_type_proto);
  global_data->fn_type =
      global_data->package.GetFunctionTypeFromProto(fn_type_proto).value();

  std::vector<::xls::Type*> param_types;
  for (const auto& param_type : global_data->fn_type->parameters()) {
    global_data->borrowed_param_types.push_back(param_type);
  }

  return global_data;
}

}  // namespace xls::aot_compile
