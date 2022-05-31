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

#include "xls/flows/ir_wrapper.h"

#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/ir_converter.h"
#include "xls/dslx/mangle.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls {
using dslx::Module;
using dslx::TypecheckedModule;

absl::StatusOr<IrWrapper> IrWrapper::Create(
    absl::string_view ir_package_name, std::unique_ptr<Module> top_module,
    absl::string_view top_module_path, std::unique_ptr<Module> other_module,
    absl::string_view other_module_path) {
  std::vector<std::unique_ptr<Module>> other_module_vec;
  other_module_vec.push_back(std::move(other_module));

  std::vector<absl::string_view> other_module_path_vec;
  other_module_path_vec.push_back(other_module_path);

  return Create(ir_package_name, std::move(top_module), top_module_path,
                absl::MakeSpan(other_module_vec),
                absl::MakeSpan(other_module_path_vec));
}

absl::StatusOr<IrWrapper> IrWrapper::Create(
    absl::string_view ir_package_name, std::unique_ptr<Module> top_module,
    absl::string_view top_module_path,
    absl::Span<std::unique_ptr<Module>> other_modules,
    absl::Span<absl::string_view> other_modules_path) {
  IrWrapper ir_wrapper(ir_package_name);

  // Compile DSLX
  XLS_RET_CHECK(other_modules.size() == other_modules_path.size());
  for (int64_t i = 0; i < other_modules.size(); ++i) {
    XLS_RET_CHECK(other_modules[i] != nullptr);

    XLS_ASSIGN_OR_RETURN(
        TypecheckedModule module_typechecked,
        TypecheckModule(std::move(other_modules[i]), other_modules_path[i],
                        &ir_wrapper.import_data_));
    XLS_VLOG_LINES(3, module_typechecked.module->ToString());
  }

  XLS_RET_CHECK(top_module != nullptr);
  XLS_ASSIGN_OR_RETURN(TypecheckedModule top_typechecked,
                       TypecheckModule(std::move(top_module), top_module_path,
                                       &ir_wrapper.import_data_));
  ir_wrapper.top_module_ = top_typechecked.module;
  XLS_VLOG_LINES(3, ir_wrapper.top_module_->ToString());

  // Convert into IR
  const dslx::ConvertOptions convert_options = {
      .emit_positions = true, .emit_fail_as_assert = true, .verify_ir = true};

  XLS_RET_CHECK_OK(dslx::ConvertModuleIntoPackage(
      ir_wrapper.top_module_, &ir_wrapper.import_data_, convert_options,
      /*traverse_tests=*/false, ir_wrapper.package_.get()));

  // TODO(tedhong): 2022-05-17 Optimize IR

  return std::move(ir_wrapper);
}

absl::StatusOr<Function*> IrWrapper::GetIrFunction(
    absl::string_view name) const {
  XLS_RET_CHECK(top_module_ != nullptr);

  XLS_ASSIGN_OR_RETURN(std::string mangled_name,
                       dslx::MangleDslxName(top_module_->name(), name,
                                            dslx::CallingConvention::kTypical));

  return package_->GetFunction(mangled_name);
}

}  // namespace xls
