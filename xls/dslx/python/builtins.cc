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

#include "xls/dslx/builtins.h"

#include "absl/base/casts.h"
#include "absl/strings/match.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor_pybind_caster.h"
#include "xls/dslx/cpp_bindings.h"
#include "xls/dslx/python/cpp_ast.h"
#include "xls/dslx/python/errors.h"

namespace py = pybind11;

namespace xls::dslx {

// Signature for a DSLX interpreter builtin function defined in C++.
using BuiltinFn = absl::StatusOr<InterpValue> (*)(absl::Span<const InterpValue>,
                                                  const Span&, Invocation*,
                                                  SymbolicBindings*);

// Python type used for representing symbolic bindings.
using PySymbolicBindings = std::vector<std::pair<std::string, int64>>;

// Wraps the C++ defined DSLX builtin function f so it conforms to the
// pybind11-desired signature, and does standard things like throw FailureError
// statuses as exceptions.
std::function<absl::StatusOr<InterpValue>(const std::vector<InterpValue>&,
                                          const Span&, InvocationHolder,
                                          absl::optional<PySymbolicBindings>)>
WrapBuiltin(BuiltinFn f) {
  return [f](const std::vector<InterpValue>& args, const Span& span,
             InvocationHolder invocation,
             absl::optional<PySymbolicBindings> py_sym_bindings) {
    absl::optional<SymbolicBindings> sym_bindings;
    SymbolicBindings* psym_bindings;
    if (py_sym_bindings.has_value()) {
      sym_bindings.emplace(*py_sym_bindings);
      psym_bindings = &sym_bindings.value();
    } else {
      psym_bindings = nullptr;
    }
    absl::StatusOr<InterpValue> v =
        f(args, span, &invocation.deref(), psym_bindings);
    TryThrowFailureError(v.status());
    return v;
  };
}

PYBIND11_MODULE(builtins, m) {
  ImportStatusModule();

  static py::exception<FailureError> failure_exc(m, "FailureError");

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const FailureError& e) {
      py::object& e_type = failure_exc;
      py::object instance = e_type();
      instance.attr("message") = e.what();
      instance.attr("span") = e.span();
      PyErr_SetObject(failure_exc.ptr(), instance.ptr());
    }
  });

  m.def("throw_fail_error",
        [](Span span, const std::string& s) { throw FailureError(s, span); });

  m.def("add_with_carry", WrapBuiltin(BuiltinAddWithCarry));
  m.def("and_reduce", WrapBuiltin(BuiltinAndReduce));
  m.def("assert_eq", WrapBuiltin(BuiltinAssertEq));
  m.def("assert_lt", WrapBuiltin(BuiltinAssertLt));
  m.def("bit_slice", WrapBuiltin(BuiltinBitSlice));
  m.def("clz", WrapBuiltin(BuiltinClz));
  m.def("ctz", WrapBuiltin(BuiltinCtz));
  m.def("enumerate", WrapBuiltin(BuiltinEnumerate));
  m.def("one_hot_sel", WrapBuiltin(BuiltinOneHotSel));
  m.def("one_hot", WrapBuiltin(BuiltinOneHot));
  m.def("or_reduce", WrapBuiltin(BuiltinOrReduce));
  m.def("range", WrapBuiltin(BuiltinRange));
  m.def("rev", WrapBuiltin(BuiltinRev));
  m.def("signex", WrapBuiltin(BuiltinSignex));
  m.def("slice", WrapBuiltin(BuiltinSlice));
  m.def("xor_reduce", WrapBuiltin(BuiltinXorReduce));
}

}  // namespace xls::dslx
