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

#include "absl/status/statusor.h"
#include "pybind11_abseil/import_status_module.h"

namespace xls {

inline void ImportStatusModule() {
  // Copied from ImportStatusModule from pybind11_abseil. This is a work around
  // for the missing symbol error: _ZN8pybind116google18ImportStatusModuleEb
  // TODO(meheff): 2022/12/01 Remove this library when
  // pybind11::google::ImportStatusModule can be called directly.
  if (!PyGILState_Check()) {
    pybind11::pybind11_fail("ImportStatusModule() PyGILState_Check() failure.");
  }
  pybind11::module_::import(
      PYBIND11_TOSTRING(PYBIND11_ABSEIL_STATUS_MODULE_PATH));
}

}  // namespace xls
