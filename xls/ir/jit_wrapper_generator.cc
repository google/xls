// Copyright 2020 Google LLC
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
#include "xls/ir/jit_wrapper_generator.h"

#include "absl/strings/substitute.h"

namespace xls {

std::string GenerateWrapperHeader(const Function& function,
                                  absl::string_view class_name) {
  // $0 : Class name
  // $1 : Function params
  // $2 : Function name
  constexpr const char header_template[] =
      R"(// Automatically-generated file! DO NOT EDIT!
#include <memory>

#include "xls/common/status/statusor.h"
#include "xls/ir/llvm_ir_jit.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls {

// JIT execution wrapper for the $2 XLS IR module.
class $0 {
 public:
  static xabsl::StatusOr<$0> Create();

  xabsl::StatusOr<Value> Run($1);

 private:
  $0(std::unique_ptr<Package> package, std::unique_ptr<LlvmIrJit> jit);

  std::unique_ptr<Package> package_;
  std::unique_ptr<LlvmIrJit> jit_;
};

}  // namespace xls
)";

  std::vector<std::string> params;
  for (const Param* param : function.params()) {
    params.push_back(absl::StrCat("Value ", param->name()));
  }

  return absl::Substitute(header_template, class_name,
                          absl::StrJoin(params, ","), function.name());
}

std::string GenerateWrapperSource(const Function& function,
                                  absl::string_view class_name,
                                  const std::filesystem::path& header_path) {
  // Use an extra '-' delimiter so we can embed a traditional-looking raw string
  // in the source.
  //  $0 : Class name
  //  $1 : IR text
  //  $2 : Param list
  //  $3 : Arg list
  //  $4 : Arg list size
  //  $5 : Header path
  //  $6 : Function name (not camelized)
  constexpr const char source_template[] =
      R"-(// Automatically-generated file! DO NOT EDIT!
#include "$5"
#include "xls/ir/ir_parser.h"

namespace xls {

constexpr const char ir_text[] = R"($1
)";

xabsl::StatusOr<$0> $0::Create() {
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSIGN_OR_RETURN(Function* function, package->GetFunction("$6"));
  XLS_ASSIGN_OR_RETURN(auto jit, LlvmIrJit::Create(function));
  return $0(std::move(package), std::move(jit));
}

$0::$0(std::unique_ptr<Package> package, std::unique_ptr<LlvmIrJit> jit)
    : package_(std::move(package)), jit_(std::move(jit)) { }

xabsl::StatusOr<Value> $0::Run($2) {
  Value args[$4] = { $3 };
  // Special form to handle zero-argument spans.
  return jit_->Run(absl::MakeSpan(args, $4));
}

}  // namespace xls
)-";
  std::vector<std::string> param_list;
  for (const Param* param : function.params()) {
    param_list.push_back(absl::StrCat("Value ", param->name()));
  }
  std::string params = absl::StrJoin(param_list, ", ");

  std::vector<std::string> arg_list;
  for (const Param* param : function.params()) {
    arg_list.push_back(param->name());
  }
  std::string args = absl::StrJoin(arg_list, ", ");

  return absl::Substitute(
      source_template, class_name, function.package()->DumpIr(), params, args,
      arg_list.size(), header_path.string(), function.name());
}

GeneratedJitWrapper GenerateJitWrapper(
    const Function& function, const std::string& class_name,
    const std::filesystem::path& header_path) {
  GeneratedJitWrapper wrapper;
  wrapper.header = GenerateWrapperHeader(function, class_name);
  wrapper.source = GenerateWrapperSource(function, class_name, header_path);
  return wrapper;
}

}  // namespace xls
