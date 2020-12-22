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
#include "xls/dslx/python/callback_converters.h"
#include "xls/dslx/python/cpp_ast.h"
#include "xls/dslx/python/errors.h"

namespace py = pybind11;

namespace xls::dslx {

PYBIND11_MODULE(cpp_deduce, m) {
  ImportStatusModule();
  py::module::import("xls.dslx.python.cpp_concrete_type");

  py::class_<FnStackEntry>(m, "FnStackEntry")
      .def("__repr__", &FnStackEntry::ToReprString)
      .def_property_readonly(
          "name", [](const FnStackEntry& entry) { return entry.name; })
      .def_property_readonly(
          "symbolic_bindings",
          [](const FnStackEntry& entry) { return entry.symbolic_bindings; });

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

  m.def("type_inference_error",
        [](const Span& span, ConcreteType* type,
           const std::string& suffix) -> absl::Status {
          std::unique_ptr<ConcreteType> t;
          if (type != nullptr) {
            t = type->CloneToUnique();
          }
          throw TypeInferenceError(span, std::move(t), suffix);
        });

  py::class_<DeduceCtx, std::shared_ptr<DeduceCtx>>(m, "DeduceCtx")
      .def(py::init([](const std::shared_ptr<TypeInfo>& type_info,
                       ModuleHolder module, PyDeduceFn deduce_function,
                       PyTypecheckFunctionFn typecheck_function,
                       PyTypecheckFn typecheck_module,
                       const std::vector<std::string>& additional_search_paths,
                       absl::optional<ImportCache*> import_cache) {
        ImportCache* pimport_cache = import_cache ? *import_cache : nullptr;
        return std::make_shared<DeduceCtx>(
            type_info, module.deref().shared_from_this(),
            ToCppDeduce(deduce_function),
            ToCppTypecheckFunction(typecheck_function),
            ToCppTypecheck(typecheck_module), additional_search_paths,
            pimport_cache);
      }))
      .def("add_fn_stack_entry",
           [](DeduceCtx& self, std::string name,
              const SymbolicBindings& sym_bindings) {
             self.AddFnStackEntry(FnStackEntry{name, sym_bindings});
           })
      .def("pop_fn_stack_entry",
           [](DeduceCtx& self) { return self.PopFnStackEntry(); })
      .def("peek_fn_stack",
           [](DeduceCtx& self) {
             absl::optional<FnStackEntry> result;
             if (!self.fn_stack().empty()) {
               result = self.fn_stack().back();
             }
             return result;
           })
      .def("add_derived_type_info",
           [](DeduceCtx& self) { self.AddDerivedTypeInfo(); })
      .def("pop_derived_type_info",
           [](DeduceCtx& self) { return self.PopDerivedTypeInfo(); })
      .def_property_readonly(
          "import_cache",
          [](const DeduceCtx& ctx) -> absl::optional<ImportCache*> {
            ImportCache* cache = ctx.import_cache();
            if (cache == nullptr) {
              return absl::nullopt;
            }
            return cache;
          })
      .def_property_readonly(
          "type_info", [](const DeduceCtx& ctx) { return ctx.type_info(); })
      .def_property_readonly("module",
                             [](const DeduceCtx& ctx) {
                               return ModuleHolder(ctx.module().get(),
                                                   ctx.module());
                             })
      .def_property_readonly("typecheck_module",
                             [](const DeduceCtx& ctx) {
                               return ToPyTypecheck(ctx.typecheck_module());
                             })
      .def("typecheck_function",
           [](const DeduceCtx& self, FunctionHolder function, DeduceCtx* ctx) {
             return self.typecheck_function()(&function.deref(), ctx);
           })
      .def("make_ctx", [](const DeduceCtx& self,
                          const std::shared_ptr<TypeInfo>& new_type_info,
                          ModuleHolder new_module) {
        return self.MakeCtx(new_type_info, new_module.module());
      });

  m.def("resolve", &Resolve);

  m.def("deduce", [](AstNodeHolder node, DeduceCtx* ctx) {
    auto statusor = Deduce(&node.deref(), ctx);
    TryThrowTypeInferenceError(statusor.status());
    TryThrowXlsTypeError(statusor.status());
    TryThrowKeyError(statusor.status());
    TryThrowTypeMissingError(statusor.status());
    TryThrowArgCountMismatchError(statusor.status());
    return statusor;
  });
}

}  // namespace xls::dslx
