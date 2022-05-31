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

#ifndef XLS_FLOWS_IR_WRAPPER_H_
#define XLS_FLOWS_IR_WRAPPER_H_

#include <string>
#include <utility>
#include <variant>

#include "absl/status/statusor.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/default_dslx_stdlib_path.h"
#include "xls/dslx/import_data.h"
#include "xls/ir/function.h"
#include "xls/ir/proc.h"
#include "xls/jit/function_jit.h"

namespace xls {

// This class owns and is responsible for the flow to take ownership of a set
// of DSLX modules, compile/typecheck them, and convert them into an
// IR package.
//
// Additional convenience functions are available.
class IrWrapper {
 public:
  // Retrieve a specific dslx module.
  absl::StatusOr<dslx::Module*> GetDslxModule(absl::string_view name) const;

  // Retrieve a specific top-level function from the compiled BOP IR.
  //
  // name is the unmangled name.
  absl::StatusOr<Function*> GetIrFunction(absl::string_view name) const;

  // Retrieve top level package.
  absl::StatusOr<Package*> GetIrPackage() const;

  // Retrieve and create (if needed) the JIT for the given function name.
  absl::StatusOr<FunctionJit*> GetAndMaybeCreateFunctionJit(
      absl::string_view name);

  // Takes ownership of a set of DSLX modules, converts to IR and creates
  // an IrWrapper object.
  static absl::StatusOr<IrWrapper> Create(
      absl::string_view ir_package_name,
      std::unique_ptr<dslx::Module> top_module,
      absl::string_view top_module_path,
      std::unique_ptr<dslx::Module> other_module,
      absl::string_view other_module_path);

  static absl::StatusOr<IrWrapper> Create(
      absl::string_view ir_package_name,
      std::unique_ptr<dslx::Module> top_module,
      absl::string_view top_module_path,
      absl::Span<std::unique_ptr<dslx::Module>> other_modules,
      absl::Span<absl::string_view> other_modules_path);

 private:
  // Construct this object with a default ImportData.
  explicit IrWrapper(absl::string_view package_name)
      : import_data_(dslx::CreateImportData(xls::kDefaultDslxStdlibPath,
                                            /*additional_search_paths=*/{})),
        package_(std::make_unique<Package>(package_name)) {}

  // Pointers to the each of the DSLX modules explicitly given to this wrapper.
  //
  // Ownership of this and all other DSLX modules is with import_data_;
  dslx::Module* top_module_;
  std::vector<dslx::Module*> other_modules_;

  // Holds typechecked DSLX modules.
  dslx::ImportData import_data_;

  // IR Package.
  std::unique_ptr<Package> package_;

  // Holds pre-compiled IR Jit.
  absl::flat_hash_map<Function*, std::unique_ptr<FunctionJit>>
      pre_compiled_function_jit_;
};

}  // namespace xls

#endif  // XLS_FLOWS_IR_WRAPPER_H_
