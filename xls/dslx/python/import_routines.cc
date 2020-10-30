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

#include "xls/dslx/import_routines.h"

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor_pybind_caster.h"
#include "xls/dslx/python/cpp_ast.h"
#include "xls/dslx/type_info.h"

namespace py = pybind11;

namespace xls::dslx {

using PyTypecheckFn =
    std::function<std::shared_ptr<TypeInfo>(ModuleHolder module)>;

PYBIND11_MODULE(import_routines, m) {
  ImportStatusModule();

  py::class_<ModuleInfo>(m, "ModuleInfo")
      .def("__getitem__",
           [](const ModuleInfo& self, int64 index)
               -> absl::variant<ModuleHolder, std::shared_ptr<TypeInfo>> {
             switch (index) {
               case 0:
                 return ModuleHolder(self.module.get(), self.module);
               case 1:
                 return self.type_info;
               default:
                 throw py::index_error("Index out of bounds");
             }
           });

  py::class_<ImportTokens>(m, "ImportTokens")
      .def(py::init([](std::vector<std::string> pieces) {
        return absl::make_unique<ImportTokens>(std::move(pieces));
      }));

  py::class_<ImportCache>(m, "ImportCache")
      .def(py::init())
      .def("clear", &ImportCache::Clear);

  m.def(
      "do_import",
      [](PyTypecheckFn py_ftypecheck, const ImportTokens& subject,
         ImportCache* cache) -> absl::StatusOr<ModuleInfo> {
        // The typecheck callback we get from Python uses the "ModuleHolder"
        // type -- make a conversion lambda that turns a std::shared_ptr<Module>
        // into its "Holder" form so we can invoke the Python-provided
        // typechecking callback.
        auto ftypecheck = [&](std::shared_ptr<Module> module)
            -> absl::StatusOr<std::shared_ptr<TypeInfo>> {
          ModuleHolder holder(module.get(), module);
          try {
            return py_ftypecheck(holder);
          } catch (std::exception& e) {
            // Note: normally we would throw a Python positional error, but
            // since this is a somewhat rare condition and everything is being
            // ported to C++ we don't do the super-nice thing and instead throw
            // a status error if you have a typecheck-failure-under-import.
            return absl::InternalError(e.what());
          }
        };
        // With the appropriately typed callback we can now call DoImport().
        XLS_ASSIGN_OR_RETURN(const ModuleInfo* info,
                             DoImport(ftypecheck, subject, cache));
        return *info;
      },
      py::arg("ftypecheck"), py::arg("subject"), py::arg("cache"));
}

}  // namespace xls::dslx
