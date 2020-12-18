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
#include "xls/dslx/dslx_builtins.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/interp_bindings.h"
#include "xls/dslx/python/callback_converters.h"
#include "xls/dslx/python/cpp_ast.h"
#include "xls/dslx/python/errors.h"

namespace py = pybind11;

namespace xls::dslx {
namespace {

using PyTypeAndBindings =
    std::pair<std::unique_ptr<ConcreteType>, SymbolicBindings>;

using PySignatureFn = std::function<absl::StatusOr<PyTypeAndBindings>(
    const std::vector<const ConcreteType*>& arg_types, absl::string_view name,
    const Span& span, DeduceCtx* ctx,
    absl::optional<std::vector<ParametricBindingHolder>>
        py_parametric_bindings)>;

// Wraps up fsignature with an exception throwing layer (for throwing exceptions
// to Python land).
PySignatureFn ToPy(SignatureFn fsignature) {
  return [fsignature](
             const std::vector<const ConcreteType*>& arg_types,
             absl::string_view name, const Span& span, DeduceCtx* ctx,
             absl::optional<std::vector<ParametricBindingHolder>>
                 py_parametric_bindings) -> absl::StatusOr<PyTypeAndBindings> {
    absl::optional<std::vector<ParametricBinding*>> parametric_bindings;
    if (py_parametric_bindings.has_value()) {
      parametric_bindings.emplace();
      for (ParametricBindingHolder& pbh : py_parametric_bindings.value()) {
        parametric_bindings.value().push_back(&pbh.deref());
      }
    }
    auto statusor = fsignature(arg_types, name, span, ctx, parametric_bindings);
    TryThrowArgCountMismatchError(statusor.status());
    TryThrowXlsTypeError(statusor.status());
    TryThrowTypeInferenceError(statusor.status());
    XLS_RETURN_IF_ERROR(statusor.status());
    TypeAndBindings& tab = statusor.value();
    return std::make_pair(std::move(tab.type),
                          std::move(tab.symbolic_bindings));
  };
}

}  // namespace

PYBIND11_MODULE(cpp_dslx_builtins, m) {
  ImportStatusModule();

  m.def("get_fsignature",
        [](absl::string_view builtin_name) -> absl::StatusOr<PySignatureFn> {
          XLS_ASSIGN_OR_RETURN(SignatureFn fsignature,
                               GetParametricBuiltinSignature(builtin_name));
          return ToPy(std::move(fsignature));
        });

  {
    std::set<std::string> parametric_builtin_names;
    for (const auto& item : GetParametricBuiltins()) {
      parametric_builtin_names.insert(item.first);
    }
    m.attr("PARAMETRIC_BUILTIN_NAMES") = std::move(parametric_builtin_names);
  }
  {
    std::unordered_set<std::string> value;
    for (const std::string& item : GetUnaryParametricBuiltinNames()) {
      value.insert(item);
    }
    m.attr("UNARY_PARAMETRIC_BUILTIN_NAMES") = std::move(value);
  }
}

}  // namespace xls::dslx
