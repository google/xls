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

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor_pybind_caster.h"
#include "xls/dslx/concrete_type.h"
#include "xls/dslx/deduce.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/interp_bindings.h"
#include "xls/dslx/python/cpp_ast.h"
#include "xls/dslx/python/errors.h"

namespace py = pybind11;

namespace xls::dslx {

PYBIND11_MODULE(cpp_deduce, m) {
  ImportStatusModule();
  py::module::import("xls.dslx.python.cpp_concrete_type");

  static py::exception<TypeInferenceError> type_inference_exc(
      m, "TypeInferenceError");

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const TypeInferenceError& e) {
      type_inference_exc.attr("span") = e.span();
      if (const ConcreteType* type = e.type()) {
        type_inference_exc.attr("type_") = type->CloneToUnique();
      } else {
        type_inference_exc.attr("type_") = py::none();
      }
      type_inference_exc.attr("message") = e.message();
      type_inference_exc(e.what());
    }
  });

  static py::exception<XlsTypeError> xls_type_exc(m, "XlsTypeError");
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const XlsTypeError& e) {
      XLS_VLOG(5) << "Translating XlsTypeError: " << e.what();
      xls_type_exc.attr("span") = e.span();
      if (e.lhs() == nullptr) {
        pybind11::setattr(xls_type_exc, "lhs_type", nullptr);
      } else {
        pybind11::setattr(xls_type_exc, "lhs_type",
                          pybind11::cast(e.lhs()->CloneToUnique()));
      }
      if (e.rhs() == nullptr) {
        pybind11::setattr(xls_type_exc, "rhs_type", nullptr);
      } else {
        pybind11::setattr(xls_type_exc, "rhs_type",
                          pybind11::cast(e.rhs()->CloneToUnique()));
      }
      xls_type_exc.attr("message") = e.message();
      xls_type_exc(e.what());
    }
  });
}

}  // namespace xls::dslx
