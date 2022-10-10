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
#include "xls/jit/orc_jit.h"

namespace xls::aot_compile {

std::unique_ptr<GlobalData> InitGlobalData(
    std::string_view fn_type_textproto) {
  auto package = std::make_unique<Package>("dummy");
  auto jit_runtime =
      std::make_unique<JitRuntime>(OrcJit::CreateDataLayout().value());
  ::xls::FunctionTypeProto fn_type_proto;
  google::protobuf::TextFormat::ParseFromString(std::string(fn_type_textproto),
                                      &fn_type_proto);
  FunctionType* fn_type =
      package->GetFunctionTypeFromProto(fn_type_proto).value();
  std::vector<::xls::Type*> param_types(fn_type->parameters().begin(),
                                        fn_type->parameters().end());
  return std::make_unique<GlobalData>(
      GlobalData{.jit_runtime = std::move(jit_runtime),
                 .package = std::move(package),
                 .fn_type = fn_type,
                 .param_types = std::move(param_types),
                 .return_type = fn_type->return_type()});
}

}  // namespace xls::aot_compile
