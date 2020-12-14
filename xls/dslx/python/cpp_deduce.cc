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

  py::class_<FnStackEntry>(m, "FnStackEntry")
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
      if (e.type() == nullptr) {
        type_inference_exc.attr("type_") = nullptr;
      } else {
        type_inference_exc.attr("type_") = e.type()->CloneToUnique();
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
      xls_type_exc.attr("span") = e.span();
      if (e.lhs() == nullptr) {
        xls_type_exc.attr("lhs_type") = nullptr;
      } else {
        xls_type_exc.attr("lhs_type") = e.lhs()->CloneToUnique();
      }
      if (e.rhs() == nullptr) {
        xls_type_exc.attr("rhs_type") = nullptr;
      } else {
        xls_type_exc.attr("rhs_type") = e.rhs()->CloneToUnique();
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
  m.def(
      "xls_type_error",
      [](const Span& span, ConcreteType* plhs, ConcreteType* prhs,
         const std::string& suffix) {
        std::unique_ptr<ConcreteType> lhs;
        std::unique_ptr<ConcreteType> rhs;
        if (plhs != nullptr) {
          lhs = plhs->CloneToUnique();
        }
        if (prhs != nullptr) {
          rhs = prhs->CloneToUnique();
        }
        throw XlsTypeError(span, std::move(lhs), std::move(rhs), suffix);
      },
      py::arg("span"), py::arg("lhs_type"), py::arg("rhs_type"),
      py::arg("suffix"));

  m.def("type_missing_error_set_node",
        [](py::object self, AstNodeHolder node) { self.attr("node") = node; });
  m.def("type_missing_error_set_span",
        [](py::object self, const Span& span) { self.attr("span") = span; });
  m.def("type_missing_error_set_user",
        [](py::object self, AstNodeHolder node) { self.attr("user") = node; });

  py::class_<DeduceCtx, std::shared_ptr<DeduceCtx>>(m, "DeduceCtx")
      .def(py::init([](const std::shared_ptr<TypeInfo>& type_info,
                       ModuleHolder module, PyDeduceFn deduce_function,
                       PyTypecheckFunctionFn typecheck_function,
                       PyTypecheckFn typecheck_module,
                       absl::optional<ImportCache*> import_cache) {
        ImportCache* pimport_cache = import_cache ? *import_cache : nullptr;
        return std::make_shared<DeduceCtx>(
            type_info, module.deref().shared_from_this(),
            ToCppDeduce(deduce_function),
            ToCppTypecheckFunction(typecheck_function),
            ToCppTypecheck(typecheck_module), pimport_cache);
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

  m.def("check_bitwidth", [](NumberHolder number, const ConcreteType& type) {
    auto status = CheckBitwidth(number.deref(), type);
    TryThrowTypeInferenceError(status);
    return status;
  });
  m.def("resolve", &Resolve);

#define DELEGATE_DEDUCE(__type)                                      \
  m.def("deduce_" #__type, [](__type##Holder node, DeduceCtx* ctx) { \
    auto statusor = Deduce##__type(&node.deref(), ctx);              \
    TryThrowTypeInferenceError(statusor.status());                   \
    TryThrowXlsTypeError(statusor.status());                         \
    TryThrowKeyError(statusor.status());                             \
    TryThrowTypeMissingError(statusor.status());                     \
    return statusor;                                                 \
  })

  DELEGATE_DEDUCE(Array);
  DELEGATE_DEDUCE(Attr);
  DELEGATE_DEDUCE(Binop);
  DELEGATE_DEDUCE(Cast);
  DELEGATE_DEDUCE(ColonRef);
  DELEGATE_DEDUCE(ConstantArray);
  DELEGATE_DEDUCE(ConstantDef);
  DELEGATE_DEDUCE(EnumDef);
  DELEGATE_DEDUCE(For);
  DELEGATE_DEDUCE(Index);
  DELEGATE_DEDUCE(Let);
  DELEGATE_DEDUCE(Match);
  DELEGATE_DEDUCE(Number);
  DELEGATE_DEDUCE(Param);
  DELEGATE_DEDUCE(SplatStructInstance);
  DELEGATE_DEDUCE(StructDef);
  DELEGATE_DEDUCE(StructInstance);
  DELEGATE_DEDUCE(Ternary);
  DELEGATE_DEDUCE(TypeDef);
  DELEGATE_DEDUCE(TypeRef);
  DELEGATE_DEDUCE(Unop);
  DELEGATE_DEDUCE(XlsTuple);

  DELEGATE_DEDUCE(BuiltinTypeAnnotation);
  DELEGATE_DEDUCE(ArrayTypeAnnotation);
  DELEGATE_DEDUCE(TupleTypeAnnotation);

#undef DELEGATE_DEDUCE
}

}  // namespace xls::dslx
