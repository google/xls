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

#include "xls/dslx/cpp_ast.h"

#include "absl/base/casts.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor_pybind_caster.h"
#include "xls/dslx/python/cpp_ast.h"

namespace py = pybind11;

namespace xls::dslx {

template <typename UnwrappedT, typename IterableHolderT>
std::vector<UnwrappedT> Unwrap(IterableHolderT& held) {
  std::vector<UnwrappedT> results;
  for (auto& item : held) {
    results.push_back(&item.deref());
  }
  return results;
}

template <typename UnwrappedT0, typename UnwrappedT1, typename HolderPairT>
std::vector<std::pair<UnwrappedT0, UnwrappedT1>> UnwrapPair(
    const std::vector<HolderPairT>& held) {
  std::vector<std::pair<UnwrappedT0, UnwrappedT1>> results;
  for (const HolderPairT& item : held) {
    results.push_back({&item.first.deref(), &item.second.deref()});
  }
  return results;
}

template <typename OrigT>
py::tuple Wrap(absl::Span<OrigT const> xs,
               const std::shared_ptr<Module>& module) {
  py::tuple results(xs.size());
  // TODO(leary): 2020-09-08 Figure out how to enable -Wno-signed-compare,
  // Normally we ignore comparison warnings but this one gets replicated lots of
  // times in instantiations below, so it's worth squashing to avoid too much
  // noise in the build logs.
  for (int64 i = 0; i < static_cast<int64>(xs.size()); ++i) {
    results[i] = AstNodeHolder(ToAstNode(xs[i]), module);
  }
  return results;
}

template <typename FirstHolderT, typename SecondHolderT, typename FirstT,
          typename SecondT>
std::vector<std::pair<FirstHolderT, SecondHolderT>> WrapPair(
    absl::Span<std::pair<FirstT, SecondT> const> xs,
    const std::shared_ptr<Module>& module) {
  std::vector<std::pair<FirstHolderT, SecondHolderT>> results;
  results.reserve(xs.size());
  for (auto& x : xs) {
    results.push_back(
        {FirstHolderT(x.first, module), SecondHolderT(x.second, module)});
  }
  return results;
}

template <typename SecondHolderT, typename FirstT, typename SecondT>
std::vector<std::pair<std::string, SecondHolderT>> WrapStringPair(
    absl::Span<std::pair<FirstT, SecondT> const> xs,
    const std::shared_ptr<Module>& module) {
  std::vector<std::pair<std::string, SecondHolderT>> results;
  results.reserve(xs.size());
  for (auto& x : xs) {
    results.push_back({x.first, SecondHolderT(ToAstNode(x.second), module)});
  }
  return results;
}

std::vector<ExprHolder> WrapExprs(absl::Span<Expr* const> xs,
                                  const std::shared_ptr<Module>& module) {
  std::vector<ExprHolder> results;
  results.reserve(xs.size());
  for (Expr* x : xs) {
    results.push_back(ExprHolder(x, module));
  }
  return results;
}

PYBIND11_MODULE(cpp_ast, m) {
  py::enum_<BuiltinType>(m, "BuiltinType")
#define VALUE(__enum, __pyattr, ...) .value(#__pyattr, BuiltinType::__enum)
      XLS_DSLX_BUILTIN_TYPE_EACH(VALUE)
#undef VALUE
          .export_values();

  m.def("is_constant", [](AstNodeHolder a) { return IsConstant(&a.deref()); });

  // class EnumMember
  py::class_<EnumMember>(m, "EnumMember")
      .def(py::init([](NameDefHolder name, NumberHolder value) {
        return EnumMember{&name.deref(), &value.deref()};
      }))
      .def(py::init([](NameDefHolder name, NameDefHolder value) {
        return EnumMember{&name.deref(), &value.deref()};
      }))
      .def("get_name_value",
           [](EnumMember& self, EnumHolder enum_) {
             return std::make_pair(
                 NameDefHolder(self.name, enum_.module()),
                 AstNodeHolder(ToAstNode(self.value), enum_.module()));
           })
      .def_property_readonly("identifier", [](const EnumMember& self) {
        return self.name->identifier();
      });

  // class FreeVariables
  py::class_<FreeVariables>(m, "FreeVariables")
      // TODO(leary): 2020-08-28 Create/find pybind conversions for absl
      // containers.
      .def("get_name_def_tups",
           [](FreeVariables& self, ModuleHolder module) {
             return WrapStringPair<AstNodeHolder>(
                 absl::Span<std::pair<std::string, AnyNameDef> const>(
                     self.GetNameDefTuples()),
                 module.module());
           })
      .def("drop_builtin_defs",
           [](const FreeVariables& self) { return self.DropBuiltinDefs(); })
      .def("get_name_defs",
           [](const FreeVariables& self, ModuleHolder module) {
             return Wrap<AnyNameDef>(self.GetNameDefs(), module.module());
           })
      .def("keys", [](FreeVariables& self) {
        auto keys = self.Keys();
        return std::unordered_set<std::string>(keys.begin(), keys.end());
      });

  // class AstNode (abstract base class)
  py::class_<AstNodeHolder>(m, "AstNode")
      .def("__hash__",
           [](AstNodeHolder self) {
             return absl::bit_cast<uint64>(&self.deref());
           })
      .def("__eq__",
           [](AstNodeHolder self, absl::optional<AstNodeHolder> other) {
             if (!other.has_value()) {
               return false;
             }
             return &self.deref() == &other->deref();
           })
      .def("__str__",
           [](AstNodeHolder self) { return self.deref().ToString(); })
      .def_property_readonly(
          "children",
          [](AstNodeHolder self) {
            return Wrap<AstNode*>(self.deref().GetChildren(/*want_types=*/true),
                                  self.module());
          })
      .def("get_free_variables", [](AstNodeHolder self, Pos start_pos) {
        return self.deref().GetFreeVariables(start_pos);
      });

  // class Expr
  py::class_<ExprHolder, AstNodeHolder>(m, "Expr").def_property_readonly(
      "span", [](ExprHolder self) { return self.deref().span(); });

  // class Module
  py::object ast_module =
      py::class_<ModuleHolder>(m, "Module")
          .def(py::init([](std::string name) {
            auto module = std::make_shared<Module>(std::move(name));
            return ModuleHolder(module.get(), module);
          }))
          .def("__str__",
               [](ModuleHolder module) { return module.deref().ToString(); })
          .def_property_readonly(
              "name", [](ModuleHolder module) { return module.deref().name(); })
          .def("get_function",
               [](ModuleHolder module, absl::string_view target_name)
                   -> xabsl::StatusOr<FunctionHolder> {
                 xabsl::StatusOr<Function*> f =
                     module.deref().GetFunction(target_name);
                 if (!f.status().ok() &&
                     f.status().code() == absl::StatusCode::kNotFound) {
                   throw py::key_error(std::string(f.status().message()));
                 }
                 return FunctionHolder(f.value(), module.module());
               })
          .def("get_typedef_by_name",
               [](ModuleHolder module) {
                 auto map = module.deref().GetTypeDefinitionByName();
                 std::unordered_map<std::string, AstNodeHolder> m;
                 for (auto& item : map) {
                   m.insert({item.first, AstNodeHolder(ToAstNode(item.second),
                                                       module.module())});
                 }
                 return m;
               })
          .def("get_typedefs",
               [](ModuleHolder module) {
                 auto map = module.deref().GetTypeDefinitionByName();
                 std::vector<AstNodeHolder> results;
                 for (auto& item : map) {
                   results.push_back(
                       AstNodeHolder(ToAstNode(item.second), module.module()));
                 }
                 return results;
               })
          .def("get_constant_by_name",
               [](ModuleHolder module) {
                 auto map = module.deref().GetConstantByName();
                 std::unordered_map<std::string, ConstantDefHolder> m;
                 for (auto& item : map) {
                   m.insert({item.first,
                             ConstantDefHolder(item.second, module.module())});
                 }
                 return m;
               })
          .def("get_function_by_name",
               [](ModuleHolder module) {
                 auto map = module.deref().GetFunctionByName();
                 std::unordered_map<std::string, FunctionHolder> m;
                 for (auto& item : map) {
                   m.insert({item.first,
                             FunctionHolder(item.second, module.module())});
                 }
                 return m;
               })
          .def("get_quickchecks",
               [](ModuleHolder module) {
                 return Wrap<QuickCheck*>(module.deref().GetQuickChecks(),
                                          module.module());
               })
          .def("get_structs",
               [](ModuleHolder module) {
                 return Wrap<Struct*>(module.deref().GetStructs(),
                                      module.module());
               })
          .def("get_constants",
               [](ModuleHolder module) {
                 return Wrap<ConstantDef*>(module.deref().GetConstantDefs(),
                                           module.module());
               })
          .def("get_functions",
               [](ModuleHolder module) {
                 return Wrap<Function*>(module.deref().GetFunctions(),
                                        module.module());
               })
          .def("get_tests",
               [](ModuleHolder module) {
                 return Wrap<Test*>(module.deref().GetTests(), module.module());
               })
          .def(
              "get_test_names",
              [](ModuleHolder module) { return module.deref().GetTestNames(); })
          .def("get_test",
               [](ModuleHolder module,
                  absl::string_view name) -> xabsl::StatusOr<TestHolder> {
                 XLS_ASSIGN_OR_RETURN(Test * test,
                                      module.deref().GetTest(name));
                 return TestHolder(test, module.module());
               })
          .def("add_top",
               [](ModuleHolder module, AstNodeHolder node) -> absl::Status {
                 XLS_ASSIGN_OR_RETURN(ModuleMember member,
                                      AsModuleMember(&node.deref()));
                 module.deref().AddTop(member);
                 return absl::OkStatus();
               })
          .def_property_readonly("top", [](ModuleHolder self) {
            return Wrap(self.deref().top(), self.module());
          });

  m.attr("AstNodeOwner") = ast_module;

  // class Enum
  py::class_<EnumHolder, AstNodeHolder>(m, "Enum")
      .def(py::init([](ModuleHolder module, Span span, NameDefHolder name_def,
                       TypeAnnotationHolder type,
                       std::vector<EnumMember> values, bool is_public) {
        auto* self = module.deref().Make<Enum>(std::move(span),
                                               &name_def.deref(), &type.deref(),
                                               std::move(values), is_public);
        return EnumHolder(self, module.module());
      }))
      .def("set_signedness",
           [](EnumHolder self, bool is_signed) {
             self.deref().set_signedness(is_signed);
           })
      .def("get_signedness",
           [](EnumHolder self) { return self.deref().signedness(); })
      .def("has_value",
           [](EnumHolder self, absl::string_view name) {
             return self.deref().HasValue(name);
           })
      .def("get_value",
           [](EnumHolder self,
              absl::string_view name) -> xabsl::StatusOr<AstNodeHolder> {
             XLS_ASSIGN_OR_RETURN(auto value, self.deref().GetValue(name));
             return AstNodeHolder(ToAstNode(value), self.module());
           })
      .def_property_readonly(
          "public", [](EnumHolder self) { return self.deref().is_public(); })
      .def_property_readonly(
          "identifier",
          [](EnumHolder self) { return self.deref().identifier(); })
      .def_property_readonly("name",
                             [](EnumHolder self) -> NameDefHolder {
                               return NameDefHolder(self.deref().name_def(),
                                                    self.module());
                             })
      .def_property_readonly("type_",
                             [](EnumHolder self) -> TypeAnnotationHolder {
                               return TypeAnnotationHolder(self.deref().type(),
                                                           self.module());
                             })
      .def_property_readonly(
          "values", [](EnumHolder self) -> const std::vector<EnumMember>& {
            return self.deref().values();
          });

  // class TypeDef
  py::class_<TypeDefHolder, AstNodeHolder>(m, "TypeDef")
      .def(py::init([](ModuleHolder module, NameDefHolder name_def,
                       TypeAnnotationHolder type, bool is_public) {
             auto* self = module.deref().Make<TypeDef>(
                 &name_def.deref(), &type.deref(), is_public);
             return TypeDefHolder(self, module.module());
           }),
           py::arg("module"), py::arg("name_def"), py::arg("type"),
           py::arg("public"))
      .def_property_readonly("type_",
                             [](TypeDefHolder self) {
                               return TypeAnnotationHolder(self.deref().type(),
                                                           self.module());
                             })
      .def_property_readonly(
          "identifier",
          [](TypeDefHolder self) { return self.deref().identifier(); })
      .def_property_readonly(
          "public", [](TypeDefHolder self) { return self.deref().is_public(); })
      // TODO(leary): 2020-08-27 Rename to name_def.
      .def_property_readonly("name", [](TypeDefHolder self) {
        return NameDefHolder(self.deref().name_def(), self.module());
      });

  // class Slice
  py::class_<SliceHolder, AstNodeHolder>(m, "Slice")
      .def(py::init([](ModuleHolder module, Span span,
                       absl::optional<NumberHolder> start,
                       absl::optional<NumberHolder> limit) {
             Number* start_ptr = start ? &start->deref() : nullptr;
             Number* limit_ptr = limit ? &limit->deref() : nullptr;
             auto* self = module.deref().Make<Slice>(std::move(span), start_ptr,
                                                     limit_ptr);
             return SliceHolder(self, module.module());
           }),
           py::arg("module"), py::arg("span"), py::arg("start"),
           py::arg("limit"))
      .def_property_readonly(
          "start",
          [](SliceHolder self) -> absl::optional<NumberHolder> {
            if (self.deref().start() == nullptr) {
              return absl::nullopt;
            }
            return NumberHolder(self.deref().start(), self.module());
          })
      .def_property_readonly(
          "limit", [](SliceHolder self) -> absl::optional<NumberHolder> {
            if (self.deref().limit() == nullptr) {
              return absl::nullopt;
            }
            return NumberHolder(self.deref().limit(), self.module());
          });

  // class Struct
  py::class_<StructHolder, AstNodeHolder>(m, "Struct")
      .def(py::init(
          [](ModuleHolder module, NameDefHolder name_def,
             std::vector<ParametricBindingHolder> parametric_bindings,
             std::vector<std::pair<NameDefHolder, TypeAnnotationHolder>>
                 members,
             bool is_public) {
            auto* self = module.deref().Make<Struct>(
                &name_def.deref(),
                Unwrap<ParametricBinding*>(parametric_bindings),
                UnwrapPair<NameDef*, TypeAnnotation*>(members), is_public);
            return StructHolder(self, module.module());
          }))
      // TODO(leary): 2020-08-27 Rename to name_def.
      .def_property_readonly("name",
                             [](StructHolder self) {
                               return NameDefHolder(self.deref().name_def(),
                                                    self.module());
                             })
      .def_property_readonly("identifier",
                             [](StructHolder self) {
                               return self.deref().name_def()->identifier();
                             })
      .def("is_parametric",
           [](StructHolder self) {
             return !self.deref().parametric_bindings().empty();
           })
      .def_property_readonly(
          "public", [](StructHolder self) { return self.deref().is_public(); })
      .def_property_readonly(
          "member_names",
          [](StructHolder self) { return self.deref().GetMemberNames(); })
      .def_property_readonly(
          "members",
          [](StructHolder self) {
            return WrapPair<NameDefHolder, TypeAnnotationHolder>(
                absl::MakeSpan(self.deref().members()), self.module());
          })
      .def_property_readonly("parametric_bindings", [](StructHolder self) {
        return Wrap<ParametricBinding*>(self.deref().parametric_bindings(),
                                        self.module());
      });

  // class Import
  py::class_<ImportHolder, AstNodeHolder>(m, "Import")
      .def(py::init([](ModuleHolder module, Span span,
                       std::vector<std::string> name, NameDefHolder name_def,
                       absl::optional<std::string> alias) {
        auto* self =
            module.deref().Make<Import>(std::move(span), std::move(name),
                                        &name_def.deref(), std::move(alias));
        return ImportHolder(self, module.module());
      }))
      .def_property_readonly(
          "span", [](ImportHolder self) { return self.deref().span(); })
      .def_property_readonly(
          "identifier",
          [](ImportHolder self) { return self.deref().identifier(); })
      .def_property_readonly("name", [](ImportHolder self) {
        const std::vector<std::string>& name = self.deref().name();
        py::tuple t(name.size());
        for (int64 i = 0; i < name.size(); ++i) {
          t[i] = name[i];
        }
        return t;
      });

  // class ModRef
  py::class_<ModRefHolder, ExprHolder>(m, "ModRef")
      .def(py::init([](ModuleHolder module, Span span, ImportHolder mod,
                       std::string attr) {
        auto* self = module.deref().Make<ModRef>(std::move(span), &mod.deref(),
                                                 std::move(attr));
        return ModRefHolder(self, module.module());
      }))
      // TODO(leary): 2020-09-04 Rename to import_.
      .def_property_readonly("mod",
                             [](ModRefHolder self) {
                               return ImportHolder(self.deref().import(),
                                                   self.module());
                             })
      // TODO(leary): 2020-09-04 Rename to attr.
      .def_property_readonly(
          "value", [](ModRefHolder self) { return self.deref().attr(); });

  // class BuiltinNameDef
  py::class_<BuiltinNameDefHolder, AstNodeHolder>(m, "BuiltinNameDef")
      .def(py::init([](ModuleHolder module, std::string identifier) {
        auto* self = module.deref().Make<BuiltinNameDef>(identifier);
        return BuiltinNameDefHolder(self, module.module());
      }))
      .def_property_readonly("identifier", [](BuiltinNameDefHolder self) {
        return self.deref().identifier();
      });

  // class TypeAnnotation
  py::class_<TypeAnnotationHolder, AstNodeHolder>(m, "TypeAnnotation")
      .def_property_readonly("span", [](TypeAnnotationHolder self) {
        return self.deref().span();
      });

  // class BuiltinTypeAnnotation
  py::class_<BuiltinTypeAnnotationHolder, TypeAnnotationHolder>(
      m, "BuiltinTypeAnnotation")
      .def(py::init(
          [](ModuleHolder module, Span span, BuiltinType builtin_type) {
            auto* self = module.deref().Make<BuiltinTypeAnnotation>(
                std::move(span), builtin_type);
            return BuiltinTypeAnnotationHolder(self, module.module());
          }))
      .def_property_readonly("signedness_and_bits",
                             [](BuiltinTypeAnnotationHolder self) {
                               return std::make_tuple(
                                   self.deref().GetSignedness(),
                                   self.deref().GetBitCount());
                             })
      .def_property_readonly("bits",
                             [](BuiltinTypeAnnotationHolder self) {
                               return self.deref().GetBitCount();
                             })
      .def_property_readonly("signedness",
                             [](BuiltinTypeAnnotationHolder self) {
                               return self.deref().GetSignedness();
                             });

  // class ArrayTypeAnnotation
  py::class_<ArrayTypeAnnotationHolder, TypeAnnotationHolder>(
      m, "ArrayTypeAnnotation")
      .def(py::init([](ModuleHolder module, Span span,
                       TypeAnnotationHolder element_type, ExprHolder dim) {
        auto* self = module.deref().Make<ArrayTypeAnnotation>(
            std::move(span), &element_type.deref(), &dim.deref());
        return ArrayTypeAnnotationHolder(self, module.module());
      }))
      .def_property_readonly("element_type",
                             [](ArrayTypeAnnotationHolder self) {
                               return TypeAnnotationHolder(
                                   self.deref().element_type(), self.module());
                             })
      .def_property_readonly("dim", [](ArrayTypeAnnotationHolder self) {
        return ExprHolder(self.deref().dim(), self.module());
      });

  // class TupleTypeAnnotation
  py::class_<TupleTypeAnnotationHolder, TypeAnnotationHolder>(
      m, "TupleTypeAnnotation")
      .def(py::init([](ModuleHolder module, Span span,
                       std::vector<TypeAnnotationHolder> members) {
        auto* self = module.deref().Make<TupleTypeAnnotation>(
            std::move(span), Unwrap<TypeAnnotation*>(members));
        return TupleTypeAnnotationHolder(self, module.module());
      }))
      .def_property_readonly("members", [](TupleTypeAnnotationHolder self) {
        return Wrap<TypeAnnotation*>(self.deref().members(), self.module());
      });

  // class TypeRefTypeAnnotation
  py::class_<TypeRefTypeAnnotationHolder, TypeAnnotationHolder>(
      m, "TypeRefTypeAnnotation")
      .def(py::init([](ModuleHolder module, Span span, TypeRefHolder type_ref,
                       absl::optional<std::vector<ExprHolder>> parametrics) {
        std::vector<Expr*> parametric_ptrs;
        if (parametrics) {
          parametric_ptrs = Unwrap<Expr*>(*parametrics);
        }
        auto* self = module.deref().Make<TypeRefTypeAnnotation>(
            std::move(span), &type_ref.deref(), std::move(parametric_ptrs));
        return TypeRefTypeAnnotationHolder(self, module.module());
      }))
      .def_property_readonly("parametrics",
                             [](TypeRefTypeAnnotationHolder self) {
                               return Wrap<Expr*>(self.deref().parametrics(),
                                                  self.module());
                             })
      .def_property_readonly("type_ref", [](TypeRefTypeAnnotationHolder self) {
        return TypeRefHolder(self.deref().type_ref(), self.module());
      });

  // class TypeRef
  py::class_<TypeRefHolder, AstNodeHolder>(m, "TypeRef")
      .def(py::init([](ModuleHolder module, Span span, std::string text,
                       TypeDefHolder type_def) {
        auto* self = module.deref().Make<TypeRef>(std::move(span), text,
                                                  &type_def.deref());
        return TypeRefHolder(self, module.module());
      }))
      .def(py::init([](ModuleHolder module, Span span, std::string text,
                       StructHolder struct_) {
        auto* self = module.deref().Make<TypeRef>(std::move(span), text,
                                                  &struct_.deref());
        return TypeRefHolder(self, module.module());
      }))
      .def(py::init([](ModuleHolder module, Span span, std::string text,
                       EnumHolder enum_) {
        auto* self =
            module.deref().Make<TypeRef>(std::move(span), text, &enum_.deref());
        return TypeRefHolder(self, module.module());
      }))
      .def(py::init([](ModuleHolder module, Span span, std::string text,
                       ModRefHolder mod_ref) {
        auto* self = module.deref().Make<TypeRef>(std::move(span), text,
                                                  &mod_ref.deref());
        return TypeRefHolder(self, module.module());
      }))
      .def_property_readonly("type_def",
                             [](TypeRefHolder self) {
                               return AstNodeHolder(
                                   ToAstNode(self.deref().type_definition()),
                                   self.module());
                             })
      .def_property_readonly(
          "text", [](TypeRefHolder self) { return self.deref().text(); })
      .def_property_readonly(
          "span", [](TypeRefHolder self) { return self.deref().span(); });

  py::class_<WildcardPatternHolder, AstNodeHolder>(m, "WildcardPattern")
      .def(py::init([](ModuleHolder module, Span span) {
        auto* self = module.deref().Make<WildcardPattern>(std::move(span));
        return WildcardPatternHolder(self, module.module());
      }))
      .def_property_readonly("span", [](WildcardPatternHolder self) {
        return self.deref().span();
      });

  // class NameDefTree
#define INIT_OVERLOAD(__type)                                                  \
  .def(py::init([](ModuleHolder module, Span span, __type##Holder leaf) {      \
         auto* self =                                                          \
             module.deref().Make<NameDefTree>(std::move(span), &leaf.deref()); \
         return NameDefTreeHolder(self, module.module());                      \
       }),                                                                     \
       py::arg("module"), py::arg("span"), py::arg("tree"))

  py::class_<NameDefTreeHolder, AstNodeHolder>(m, "NameDefTree")
      INIT_OVERLOAD(NameDef)          //
      INIT_OVERLOAD(NameRef)          //
      INIT_OVERLOAD(EnumRef)          //
      INIT_OVERLOAD(ModRef)           //
      INIT_OVERLOAD(WildcardPattern)  //
      INIT_OVERLOAD(Number)
          .def(py::init([](ModuleHolder module, Span span,
                           std::vector<NameDefTreeHolder> nodes) {
                 auto* self = module.deref().Make<NameDefTree>(
                     std::move(span), Unwrap<NameDefTree*>(nodes));
                 return NameDefTreeHolder(self, module.module());
               }),
               py::arg("module"), py::arg("span"), py::arg("tree"))
          .def("is_irrefutable",
               [](NameDefTreeHolder self) {
                 return self.deref().IsIrrefutable();
               })
          .def("is_leaf",
               [](NameDefTreeHolder self) { return self.deref().is_leaf(); })
          .def("get_leaf",
               [](NameDefTreeHolder self) -> xabsl::StatusOr<AstNodeHolder> {
                 if (!self.deref().is_leaf()) {
                   return absl::InvalidArgumentError(
                       "NameDefTree AST node is not a leaf.");
                 }
                 return AstNodeHolder(ToAstNode(self.deref().leaf()),
                                      self.module());
               })
          .def("flatten1",
               [](NameDefTreeHolder self) {
                 std::vector<AstNodeHolder> result;
                 for (absl::variant<NameDefTree::Leaf, NameDefTree*> item :
                      self.deref().Flatten1()) {
                   if (absl::holds_alternative<NameDefTree*>(item)) {
                     result.push_back(AstNodeHolder(
                         absl::get<NameDefTree*>(item), self.module()));
                   } else {
                     result.push_back(AstNodeHolder(
                         ToAstNode(absl::get<NameDefTree::Leaf>(item)),
                         self.module()));
                   }
                 }
                 return result;
               })
          .def_property_readonly(
              "span",
              [](NameDefTreeHolder self) { return self.deref().span(); })
          .def_property_readonly(
              "tree",
              [](NameDefTreeHolder self)
                  -> absl::variant<AstNodeHolder, py::tuple> {
                if (self.deref().is_leaf()) {
                  return AstNodeHolder(ToAstNode(self.deref().leaf()),
                                       self.module());
                }
                return Wrap<NameDefTree*>(absl::MakeSpan(self.deref().nodes()),
                                          self.module());
              });

#undef INIT_OVERLOAD

  // class Param
  py::class_<ParamHolder, AstNodeHolder>(m, "Param")
      .def(py::init([](ModuleHolder module, NameDefHolder name_def,
                       TypeAnnotationHolder type) {
        auto* self =
            module.deref().Make<Param>(&name_def.deref(), &type.deref());
        return ParamHolder(self, module.module());
      }))
      .def_property_readonly("type_",
                             [](ParamHolder self) {
                               return TypeAnnotationHolder(self.deref().type(),
                                                           self.module());
                             })
      .def_property_readonly(
          "span", [](ParamHolder self) { return self.deref().span(); })
      // TODO(leary): 2020-08-27 Rename to name_def.
      .def_property_readonly("name", [](ParamHolder self) {
        return NameDefHolder(self.deref().name_def(), self.module());
      });

  // class ParametricBinding
  py::class_<ParametricBindingHolder, AstNodeHolder>(m, "ParametricBinding")
      .def(py::init([](ModuleHolder module, NameDefHolder name_def,
                       TypeAnnotationHolder type,
                       absl::optional<ExprHolder> expr) {
        Expr* expr_ptr = nullptr;
        if (expr) {
          expr_ptr = &expr->deref();
        }
        auto* self = module.deref().Make<ParametricBinding>(
            &name_def.deref(), &type.deref(), expr_ptr);
        return ParametricBindingHolder(self, module.module());
      }))
      // TODO(leary): 2020-08-28 Switch to name_def.
      .def_property_readonly("name",
                             [](ParametricBindingHolder self) {
                               return NameDefHolder(self.deref().name_def(),
                                                    self.module());
                             })
      .def_property_readonly("type_",
                             [](ParametricBindingHolder self) {
                               return TypeAnnotationHolder(self.deref().type(),
                                                           self.module());
                             })
      .def_property_readonly(
          "span",
          [](ParametricBindingHolder self) { return self.deref().span(); })
      .def_property_readonly(
          "expr",
          [](ParametricBindingHolder self) -> absl::optional<ExprHolder> {
            if (self.deref().expr() == nullptr) {
              return absl::nullopt;
            }
            return ExprHolder(self.deref().expr(), self.module());
          });

  py::class_<ProcHolder, AstNodeHolder>(m, "Proc")
      .def(py::init([](ModuleHolder module, Span span, NameDefHolder name_def,
                       std::vector<ParamHolder> proc_params,
                       std::vector<ParamHolder> iter_params, ExprHolder body,
                       bool is_public) {
        auto* self = module.deref().Make<Proc>(
            std::move(span), &name_def.deref(), Unwrap<Param*>(proc_params),
            Unwrap<Param*>(iter_params), &body.deref(), is_public);
        return ProcHolder(self, module.module());
      }))
      .def_property_readonly("name_def", [](ProcHolder self) {
        return NameDefHolder(self.deref().name_def(), self.module());
      });

  py::class_<WidthSliceHolder, AstNodeHolder>(m, "WidthSlice")
      .def(py::init([](ModuleHolder module, Span span, ExprHolder start,
                       TypeAnnotationHolder width) {
        auto* self = module.deref().Make<WidthSlice>(
            std::move(span), &start.deref(), &width.deref());
        return WidthSliceHolder(self, module.module());
      }))
      .def_property_readonly("start",
                             [](WidthSliceHolder self) {
                               return ExprHolder(self.deref().start(),
                                                 self.module());
                             })
      .def_property_readonly("width", [](WidthSliceHolder self) {
        return TypeAnnotationHolder(self.deref().width(), self.module());
      });

  // class Function
  py::class_<FunctionHolder, AstNodeHolder>(m, "Function")
      .def(py::init([](ModuleHolder module, Span span, NameDefHolder name_def,
                       const std::vector<ParametricBindingHolder>&
                           parametric_bindings,
                       const std::vector<ParamHolder>& params,
                       absl::optional<TypeAnnotationHolder> return_type,
                       ExprHolder body, bool is_public) {
             TypeAnnotation* return_type_ptr = nullptr;
             if (return_type) {
               return_type_ptr = &return_type->deref();
             }
             auto* self = module.deref().Make<Function>(
                 std::move(span), &name_def.deref(),
                 Unwrap<ParametricBinding*>(parametric_bindings),
                 Unwrap<Param*>(params), return_type_ptr, &body.deref(),
                 is_public);
             return FunctionHolder(self, module.module());
           }),
           py::arg("module"), py::arg("span"), py::arg("name_def"),
           py::arg("parametric_bindings"), py::arg("params"),
           py::arg("return_type"), py::arg("body"), py::arg("public"))
      .def("get_free_parametric_keys",
           [](FunctionHolder self) {
             auto keys = self.deref().GetFreeParametricKeys();
             return std::unordered_set<std::string>(keys.begin(), keys.end());
           })
      .def("is_parametric",
           [](FunctionHolder self) { return self.deref().is_parametric(); })
      .def("is_public",
           [](FunctionHolder self) { return self.deref().is_public(); })
      .def_property_readonly(
          "identifier",
          [](FunctionHolder self) { return self.deref().identifier(); })
      .def_property_readonly(
          "span", [](FunctionHolder self) { return self.deref().span(); })
      .def_property_readonly(
          "public",
          [](FunctionHolder self) { return self.deref().is_public(); })
      .def_property_readonly("parametric_bindings",
                             [](FunctionHolder self) {
                               return Wrap<ParametricBinding*>(
                                   self.deref().parametric_bindings(),
                                   self.module());
                             })
      .def_property_readonly("params",
                             [](FunctionHolder self) {
                               return Wrap<Param*>(
                                   absl::MakeSpan(self.deref().params()),
                                   self.module());
                             })
      .def_property_readonly("body",
                             [](FunctionHolder self) {
                               return ExprHolder(self.deref().body(),
                                                 self.module());
                             })
      .def_property_readonly(
          "return_type",
          [](FunctionHolder self) -> absl::optional<TypeAnnotationHolder> {
            TypeAnnotation* return_type = self.deref().return_type();
            if (return_type == nullptr) {
              return absl::nullopt;
            }
            return TypeAnnotationHolder(return_type, self.module());
          })
      // TODO(leary): 2020-08-27 Rename to name_def.
      .def_property_readonly("name", [](FunctionHolder self) {
        return NameDefHolder(self.deref().name_def(), self.module());
      });

  // class Test
  py::class_<TestHolder, AstNodeHolder>(m, "Test")
      .def(py::init(
          [](ModuleHolder module, NameDefHolder name_def, ExprHolder body) {
            auto* self =
                module.deref().Make<Test>(&name_def.deref(), &body.deref());
            return TestHolder(self, module.module());
          }))
      // TODO(leary): 2020-09-04 Rename to name_def.
      .def_property_readonly("name",
                             [](TestHolder self) {
                               return NameDefHolder(self.deref().name_def(),
                                                    self.module());
                             })
      .def_property_readonly("body", [](TestHolder self) {
        return ExprHolder(self.deref().body(), self.module());
      });

  // class TestFunction
  py::class_<TestFunctionHolder, TestHolder>(m, "TestFunction")
      .def(py::init([](ModuleHolder module, FunctionHolder f) {
        auto* self = module.deref().Make<TestFunction>(&f.deref());
        return TestFunctionHolder(self, module.module());
      }));

  // class QuickCheck
  py::class_<QuickCheckHolder, AstNodeHolder>(m, "QuickCheck")
      .def(py::init([](ModuleHolder module, Span span, FunctionHolder f,
                       absl::optional<int64> test_count) {
        auto* self = module.deref().Make<QuickCheck>(std::move(span),
                                                     &f.deref(), test_count);
        return QuickCheckHolder(self, module.module());
      }))
      .def_property_readonly(
          "test_count",
          [](QuickCheckHolder self) { return self.deref().test_count(); })
      .def_property_readonly("f", [](QuickCheckHolder self) {
        return FunctionHolder(self.deref().f(), self.module());
      });

  // class NameDef
  py::class_<NameDefHolder, AstNodeHolder>(m, "NameDef")
      .def(py::init([](ModuleHolder module, Span span, std::string identifier) {
        auto* self = module.deref().Make<NameDef>(span, identifier);
        return NameDefHolder(self, module.module());
      }))
      .def_property_readonly(
          "identifier",
          [](NameDefHolder self) { return self.deref().identifier(); })
      .def_property_readonly(
          "span", [](NameDefHolder self) { return self.deref().span(); });

  // class Next
  py::class_<NextHolder, ExprHolder>(m, "Next").def(
      py::init([](ModuleHolder module, Span span) {
        auto* self = module.deref().Make<Next>(std::move(span));
        return NextHolder(self, module.module());
      }));

  // class Carry
  py::class_<CarryHolder, ExprHolder>(m, "Carry")
      .def(py::init([](ModuleHolder module, Span span, WhileHolder loop) {
        auto* self = module.deref().Make<Carry>(std::move(span), &loop.deref());
        return CarryHolder(self, module.module());
      }))
      .def_property_readonly("loop", [](CarryHolder self) {
        return WhileHolder(self.deref().loop(), self.module());
      });

  // class StructInstance
  py::class_<StructInstanceHolder, ExprHolder>(m, "StructInstance")
      .def(
          py::init([](ModuleHolder module, Span span, StructHolder struct_,
                      std::vector<std::pair<std::string, ExprHolder>> members) {
            std::vector<std::pair<std::string, Expr*>> member_ptrs;
            for (auto& item : members) {
              member_ptrs.push_back({item.first, &item.second.deref()});
            }

            auto* self = module.deref().Make<StructInstance>(
                std::move(span), &struct_.deref(), std::move(member_ptrs));
            return StructInstanceHolder(self, module.module());
          }))
      .def(
          py::init([](ModuleHolder module, Span span, ModRefHolder struct_,
                      std::vector<std::pair<std::string, ExprHolder>> members) {
            std::vector<std::pair<std::string, Expr*>> member_ptrs;
            for (auto& item : members) {
              member_ptrs.push_back({item.first, &item.second.deref()});
            }

            auto* self = module.deref().Make<StructInstance>(
                std::move(span), &struct_.deref(), std::move(member_ptrs));
            return StructInstanceHolder(self, module.module());
          }))
      .def_property_readonly(
          "struct_text",
          [](StructInstanceHolder self) {
            return StructDefinitionToText(self.deref().struct_def());
          })
      .def_property_readonly("unordered_members",
                             [](StructInstanceHolder self) {
                               return WrapStringPair<ExprHolder>(
                                   self.deref().GetUnorderedMembers(),
                                   self.module());
                             })
      .def("get_ordered_members",
           [](StructInstanceHolder self, StructHolder struct_def) {
             auto ordered = self.deref().GetOrderedMembers(&struct_def.deref());
             return WrapStringPair<ExprHolder>(
                 absl::Span<const std::pair<std::string, Expr*>>(ordered),
                 self.module());
           })
      .def_property_readonly("struct", [](StructInstanceHolder self) {
        return AstNodeHolder(ToAstNode(self.deref().struct_def()),
                             self.module());
      });

  // class SplatStructInstance
  py::class_<SplatStructInstanceHolder, ExprHolder>(m, "SplatStructInstance")
      .def(py::init([](ModuleHolder module, Span span, StructHolder struct_,
                       std::vector<std::pair<std::string, ExprHolder>> members,
                       ExprHolder splatted) {
        std::vector<std::pair<std::string, Expr*>> member_ptrs;
        for (auto& item : members) {
          member_ptrs.push_back({item.first, &item.second.deref()});
        }

        auto* self = module.deref().Make<SplatStructInstance>(
            std::move(span), &struct_.deref(), std::move(member_ptrs),
            &splatted.deref());
        return SplatStructInstanceHolder(self, module.module());
      }))
      .def_property_readonly(
          "struct_text",
          [](SplatStructInstanceHolder self) {
            return StructDefinitionToText(self.deref().struct_def());
          })
      .def_property_readonly("members",
                             [](SplatStructInstanceHolder self) {
                               return WrapStringPair<ExprHolder>(
                                   absl::MakeSpan(self.deref().members()),
                                   self.module());
                             })
      .def_property_readonly("struct",
                             [](SplatStructInstanceHolder self) {
                               return AstNodeHolder(
                                   ToAstNode(self.deref().struct_def()),
                                   self.module());
                             })
      .def_property_readonly("splatted", [](SplatStructInstanceHolder self) {
        return ExprHolder(self.deref().splatted(), self.module());
      });

  // class NameRef
  py::class_<NameRefHolder, ExprHolder>(m, "NameRef")
      .def(py::init([](ModuleHolder module, Span span, std::string identifier,
                       NameDefHolder name_def) {
        auto* self = module.deref().Make<NameRef>(
            std::move(span), std::move(identifier), &name_def.deref());
        return NameRefHolder(self, module.module());
      }))
      .def(py::init([](ModuleHolder module, Span span, std::string identifier,
                       BuiltinNameDefHolder name_def) {
        auto* self = module.deref().Make<NameRef>(
            std::move(span), std::move(identifier), &name_def.deref());
        return NameRefHolder(self, module.module());
      }))
      .def_property_readonly("name_def",
                             [](NameRefHolder self) {
                               return AstNodeHolder(
                                   ToAstNode(self.deref().name_def()),
                                   self.module());
                             })
      .def_property_readonly("identifier", [](NameRefHolder self) {
        return self.deref().identifier();
      });

  // ConstRef
  py::class_<ConstRefHolder, NameRefHolder>(m, "ConstRef")
      .def(py::init([](ModuleHolder module, Span span, std::string identifier,
                       NameDefHolder name_def) {
        auto* self = module.deref().Make<ConstRef>(
            std::move(span), std::move(identifier), &name_def.deref());
        return ConstRefHolder(self, module.module());
      }))
      .def(py::init([](ModuleHolder module, Span span, std::string identifier,
                       BuiltinNameDefHolder name_def) {
        auto* self = module.deref().Make<ConstRef>(
            std::move(span), std::move(identifier), &name_def.deref());
        return ConstRefHolder(self, module.module());
      }));

  // class Invocation
  py::class_<InvocationHolder, ExprHolder>(m, "Invocation")
      .def(py::init([](ModuleHolder module, Span span, ExprHolder callee,
                       std::vector<ExprHolder> args) {
             auto* self = module.deref().Make<Invocation>(
                 std::move(span), &callee.deref(), Unwrap<Expr*>(args));
             return InvocationHolder(self, module.module());
           }),
           py::arg("module"), py::arg("span"), py::arg("callee"),
           py::arg("args"))
      .def("format_args",
           [](InvocationHolder self) { return self.deref().FormatArgs(); })
      .def_property_readonly("callee",
                             [](InvocationHolder self) {
                               return ExprHolder(self.deref().callee(),
                                                 self.module());
                             })
      .def_property_readonly("symbolic_bindings",
                             [](InvocationHolder self) {
                               return self.deref().symbolic_bindings();
                             })
      .def_property_readonly("args", [](InvocationHolder self) {
        return WrapExprs(self.deref().args(), self.module());
      });

  // class Cast
  py::class_<CastHolder, ExprHolder>(m, "Cast")
      .def(py::init([](ModuleHolder module, Span span,
                       TypeAnnotationHolder type, ExprHolder expr) {
        auto* self = module.deref().Make<Cast>(std::move(span), &expr.deref(),
                                               &type.deref());
        return CastHolder(self, module.module());
      }))
      .def_property_readonly("expr",
                             [](CastHolder self) {
                               return ExprHolder(self.deref().expr(),
                                                 self.module());
                             })
      .def_property_readonly("type_", [](CastHolder self) {
        return TypeAnnotationHolder(self.deref().type(), self.module());
      });

  // class EnumRef
  py::class_<EnumRefHolder, ExprHolder>(m, "EnumRef")
      .def(py::init([](ModuleHolder module, Span span, EnumHolder enum_,
                       std::string attr) {
        auto* self = module.deref().Make<EnumRef>(
            std::move(span), &enum_.deref(), std::move(attr));
        return EnumRefHolder(self, module.module());
      }))
      .def(py::init([](ModuleHolder module, Span span, TypeDefHolder enum_,
                       std::string attr) {
        auto* self = module.deref().Make<EnumRef>(
            std::move(span), &enum_.deref(), std::move(attr));
        return EnumRefHolder(self, module.module());
      }))
      // TODO(leary): 2020-08-31 Rename from "value" to "attr".
      .def_property_readonly(
          "value", [](EnumRefHolder self) { return self.deref().attr(); })
      .def_property_readonly("enum", [](EnumRefHolder self) {
        return AstNodeHolder(ToAstNode(self.deref().enum_def()), self.module());
      });

  // Ternary
  py::class_<TernaryHolder, ExprHolder>(m, "Ternary")
      .def(py::init([](ModuleHolder module, Span span, ExprHolder test,
                       ExprHolder consequent, ExprHolder alternate) {
        auto* self = module.deref().Make<Ternary>(
            std::move(span), &test.deref(), &consequent.deref(),
            &alternate.deref());
        return TernaryHolder(self, module.module());
      }))
      .def_property_readonly("test",
                             [](TernaryHolder self) {
                               return ExprHolder(self.deref().test(),
                                                 self.module());
                             })
      .def_property_readonly("consequent",
                             [](TernaryHolder self) {
                               return ExprHolder(self.deref().consequent(),
                                                 self.module());
                             })
      .def_property_readonly("alternate", [](TernaryHolder self) {
        return ExprHolder(self.deref().alternate(), self.module());
      });

  // class Let
  py::class_<LetHolder, ExprHolder>(m, "Let")
      .def(py::init([](ModuleHolder module, Span span,
                       NameDefTreeHolder name_def_tree,
                       absl::optional<TypeAnnotationHolder> type,
                       ExprHolder rhs, ExprHolder body,
                       absl::optional<ConstantDefHolder> const_def) {
             ConstantDef* const_def_ptr = nullptr;
             if (const_def) {
               const_def_ptr = &const_def->deref();
             }
             TypeAnnotation* type_ptr = nullptr;
             if (type) {
               type_ptr = &type->deref();
             }
             auto* self = module.deref().Make<Let>(
                 std::move(span), &name_def_tree.deref(), type_ptr,
                 &rhs.deref(), &body.deref(), const_def_ptr);
             return LetHolder(self, module.module());
           }),
           py::arg("module"), py::arg("span"), py::arg("name_def_tree"),
           py::arg("type"), py::arg("rhs"), py::arg("body"), py::arg("const"))
      .def_property_readonly("name_def_tree",
                             [](LetHolder self) {
                               return NameDefTreeHolder(
                                   self.deref().name_def_tree(), self.module());
                             })
      .def_property_readonly(
          "const",
          [](LetHolder self) -> absl::optional<ConstantDefHolder> {
            ConstantDef* constant_def = self.deref().constant_def();
            if (constant_def == nullptr) {
              return absl::nullopt;
            }
            return ConstantDefHolder(constant_def, self.module());
          })
      .def_property_readonly(
          "type_",
          [](LetHolder self) -> absl::optional<TypeAnnotationHolder> {
            if (self.deref().type() == nullptr) {
              return absl::nullopt;
            }
            return TypeAnnotationHolder(self.deref().type(), self.module());
          })
      .def_property_readonly("rhs",
                             [](LetHolder self) {
                               return ExprHolder(self.deref().rhs(),
                                                 self.module());
                             })
      .def_property_readonly("body", [](LetHolder self) {
        return ExprHolder(self.deref().body(), self.module());
      });

  // XlsTuple
  py::class_<XlsTupleHolder, ExprHolder>(m, "XlsTuple")
      .def(py::init(
          [](ModuleHolder module, Span span, std::vector<ExprHolder> members) {
            auto* self = module.deref().Make<XlsTuple>(std::move(span),
                                                       Unwrap<Expr*>(members));
            return XlsTupleHolder(self, module.module());
          }))
      .def("__len__",
           [](XlsTupleHolder self) { return self.deref().members().size(); })
      .def_property_readonly("members", [](XlsTupleHolder self) {
        return WrapExprs(self.deref().members(), self.module());
      });

  // class Array
  py::class_<ArrayHolder, ExprHolder>(m, "Array")
      .def(py::init([](ModuleHolder module, Span span,
                       std::vector<ExprHolder> members,
                       bool has_ellipsis) -> ArrayHolder {
             auto* self = module.deref().Make<Array>(
                 std::move(span), Unwrap<Expr*>(members), has_ellipsis);
             return ArrayHolder(self, module.module());
           }),
           py::arg("module"), py::arg("span"), py::arg("members"),
           py::arg("has_ellipsis"))
      .def_property(
          "type_",
          [](ArrayHolder self) -> absl::optional<TypeAnnotationHolder> {
            TypeAnnotation* t = self.deref().type();
            if (t == nullptr) {
              return absl::nullopt;
            }
            return TypeAnnotationHolder(t, self.module());
          },
          [](ArrayHolder self, TypeAnnotationHolder value) {
            self.deref().set_type(&value.deref());
          })
      .def_property_readonly(
          "has_ellipsis",
          [](ArrayHolder self) { return self.deref().has_ellipsis(); })
      .def_property_readonly("members", [](ArrayHolder self) {
        return WrapExprs(self.deref().members(), self.module());
      });

  py::class_<ConstantArrayHolder, ArrayHolder>(m, "ConstantArray")
      .def(py::init([](ModuleHolder module, Span span,
                       std::vector<ExprHolder> members, bool has_ellipsis) {
             auto* self = module.deref().Make<ConstantArray>(
                 std::move(span), Unwrap<Expr*>(members), has_ellipsis);
             return ConstantArrayHolder(self, module.module());
           }),
           py::arg("module"), py::arg("span"), py::arg("members"),
           py::arg("has_ellipsis"));

  py::class_<AttrHolder, ExprHolder>(m, "Attr")
      .def(py::init([](ModuleHolder module, Span span, ExprHolder lhs,
                       NameDefHolder attr) {
        auto* self = module.deref().Make<Attr>(std::move(span), &lhs.deref(),
                                               &attr.deref());
        return AttrHolder(self, module.module());
      }))
      .def_property_readonly("lhs",
                             [](AttrHolder self) {
                               return ExprHolder(self.deref().lhs(),
                                                 self.module());
                             })
      .def_property_readonly("attr", [](AttrHolder self) {
        return NameDefHolder(self.deref().attr(), self.module());
      });

  py::class_<IndexHolder, ExprHolder>(m, "Index")
      .def(py::init([](ModuleHolder module, Span span, ExprHolder lhs,
                       AstNodeHolder rhs_ast) {
        // TODO(leary): 2020-08-26 Not sure how to return a status as an
        // exception from this routine.
        IndexRhs rhs = AstNodeToIndexRhs(&rhs_ast.deref()).value();
        auto* self =
            module.deref().Make<Index>(std::move(span), &lhs.deref(), rhs);
        return IndexHolder(self, module.module());
      }))
      .def_property_readonly(
          "index",
          [](IndexHolder self) {
            return AstNodeHolder(ToAstNode(self.deref().rhs()), self.module());
          })
      .def_property_readonly("lhs", [](IndexHolder self) {
        return ExprHolder(self.deref().lhs(), self.module());
      });

  // class Match
  py::class_<MatchHolder, ExprHolder>(m, "Match")
      .def(py::init([](ModuleHolder module, Span span, ExprHolder matched,
                       std::vector<MatchArmHolder> arms) {
        auto* self = module.deref().Make<Match>(
            std::move(span), &matched.deref(), Unwrap<MatchArm*>(arms));
        return MatchHolder(self, module.module());
      }))
      .def_property_readonly("matched",
                             [](MatchHolder self) {
                               return ExprHolder(self.deref().matched(),
                                                 self.module());
                             })
      .def_property_readonly("arms", [](MatchHolder self) {
        return Wrap<MatchArm*>(self.deref().arms(), self.module());
      });

  // class MatchArm
  py::class_<MatchArmHolder, AstNodeHolder>(m, "MatchArm")
      .def(py::init([](ModuleHolder module, Span span,
                       const std::vector<NameDefTreeHolder>& patterns,
                       ExprHolder expr) {
             auto* self = module.deref().Make<MatchArm>(
                 std::move(span), Unwrap<NameDefTree*>(patterns),
                 &expr.deref());
             return MatchArmHolder(self, module.module());
           }),
           py::arg("module"), py::arg("span"), py::arg("patterns"),
           py::arg("expr"))
      .def_property_readonly(
          "span", [](MatchArmHolder self) { return self.deref().span(); })
      .def_property_readonly("expr",
                             [](MatchArmHolder self) {
                               return ExprHolder(self.deref().expr(),
                                                 self.module());
                             })
      .def_property_readonly("patterns", [](MatchArmHolder self) {
        return Wrap<NameDefTree*>(self.deref().patterns(), self.module());
      });

  py::class_<WhileHolder, ExprHolder>(m, "While")
      .def(py::init([](ModuleHolder module, Span span) {
        auto* self = module.deref().Make<While>(std::move(span));
        return WhileHolder(self, module.module());
      }))
      .def_property(
          "span", [](WhileHolder self) { return self.deref().span(); },
          [](WhileHolder self, const Span& span) {
            self.deref().set_span(span);
          })
      .def_property(
          "test",
          [](WhileHolder self) {
            return ExprHolder(self.deref().test(), self.module());
          },
          [](WhileHolder self, ExprHolder e) {
            self.deref().set_test(&e.deref());
          })
      .def_property(
          "body",
          [](WhileHolder self) {
            return ExprHolder(self.deref().body(), self.module());
          },
          [](WhileHolder self, ExprHolder e) {
            self.deref().set_body(&e.deref());
          })
      .def_property(
          "init",
          [](WhileHolder self) {
            return ExprHolder(self.deref().init(), self.module());
          },
          [](WhileHolder self, ExprHolder e) {
            self.deref().set_init(&e.deref());
          });

  // class For
  py::class_<ForHolder, ExprHolder>(m, "For")
      .def(py::init([](ModuleHolder module, Span span, NameDefTreeHolder names,
                       TypeAnnotationHolder type, ExprHolder iterable,
                       ExprHolder body, ExprHolder init) {
        auto* self = module.deref().Make<For>(std::move(span), &names.deref(),
                                              &type.deref(), &iterable.deref(),
                                              &body.deref(), &init.deref());
        return ForHolder(self, module.module());
      }))
      .def_property_readonly("init",
                             [](ForHolder self) {
                               return ExprHolder(self.deref().init(),
                                                 self.module());
                             })
      .def_property_readonly("type_",
                             [](ForHolder self) {
                               return TypeAnnotationHolder(self.deref().type(),
                                                           self.module());
                             })
      .def_property_readonly("body",
                             [](ForHolder self) {
                               return ExprHolder(self.deref().body(),
                                                 self.module());
                             })
      .def_property_readonly("iterable",
                             [](ForHolder self) {
                               return ExprHolder(self.deref().iterable(),
                                                 self.module());
                             })
      .def_property_readonly("names", [](ForHolder self) {
        return NameDefTreeHolder(self.deref().names(), self.module());
      });

  // enum NumberKind
  py::enum_<NumberKind>(m, "NumberKind")
      .value("BOOL", NumberKind::kBool)
      .value("CHARACTER", NumberKind::kCharacter)
      .value("OTHER", NumberKind::kOther)
      .export_values();

  // class Number
  py::class_<NumberHolder, ExprHolder>(m, "Number")
      .def(py::init([](ModuleHolder module, Span span, std::string text,
                       NumberKind number_kind,
                       absl::optional<TypeAnnotationHolder> type) {
             if (number_kind == NumberKind::kBool && text != "true" &&
                 text != "false") {
               throw py::value_error("Invalid text for boolean number: \"" +
                                     text + "\"");
             }
             TypeAnnotation* type_ptr = nullptr;
             if (type) {
               type_ptr = &type->deref();
             }
             auto* self =
                 module.deref().Make<Number>(span, text, number_kind, type_ptr);
             return NumberHolder(self, module.module());
           }),
           py::arg("module"), py::arg("span"), py::arg("text"),
           py::arg("number_kind") = NumberKind::kOther,
           py::arg("type_") = absl::nullopt)
      .def_property(
          "type_",
          [](NumberHolder self) -> absl::optional<TypeAnnotationHolder> {
            TypeAnnotation* ptr = self.deref().type();
            if (ptr == nullptr) {
              return absl::nullopt;
            }
            return TypeAnnotationHolder(ptr, self.module());
          },
          [](NumberHolder self, absl::optional<TypeAnnotationHolder> type) {
            if (type) {
              self.deref().set_type(&type->deref());
            } else {
              self.deref().set_type(nullptr);
            }
          })
      .def_property_readonly(
          "kind", [](NumberHolder self) { return self.deref().kind(); })
      .def_property_readonly(
          "value", [](NumberHolder self) { return self.deref().text(); });

  // class ConstantDef
  // TODO(leary): 2020-08-27 Rename to ConstantDef.
  py::class_<ConstantDefHolder, AstNodeHolder>(m, "Constant")
      .def(py::init([](ModuleHolder module, NameDefHolder name_def,
                       ExprHolder expr) {
        auto* self =
            module.deref().Make<ConstantDef>(&name_def.deref(), &expr.deref());
        return ConstantDefHolder(self, module.module());
      }))
      .def_property_readonly("value",
                             [](ConstantDefHolder self) {
                               return ExprHolder(self.deref().value(),
                                                 self.module());
                             })
      // TODO(leary): 2020-09-04 Rename to name_def.
      .def_property_readonly("name", [](ConstantDefHolder self) {
        return NameDefHolder(self.deref().name_def(), self.module());
      });

  py::enum_<UnopKind>(m, "UnopKind")
      .value("INV", UnopKind::kInvert)
      .value("NEG", UnopKind::kNegate)
      .export_values()
      .def(py::init([](absl::string_view s) { return UnopKindFromString(s); }));

  py::class_<UnopHolder, ExprHolder>(m, "Unop")
      .def(py::init(
          [](ModuleHolder module, Span span, UnopKind kind, ExprHolder arg) {
            auto* self =
                module.deref().Make<Unop>(std::move(span), kind, &arg.deref());
            return UnopHolder(self, module.module());
          }))
      .def_property_readonly("operand",
                             [](UnopHolder self) {
                               return ExprHolder(self.deref().operand(),
                                                 self.module());
                             })
      .def_property_readonly(
          "kind", [](UnopHolder self) { return self.deref().kind(); });

  py::enum_<BinopKind>(m, "BinopKind")
#define VALUE(A, B, ...) .value(B, BinopKind::A)
      XLS_DSLX_BINOP_KIND_EACH(VALUE)
#undef VALUE
          .export_values()
          .def_property_readonly(
              "value", [](BinopKind kind) { return BinopKindFormat(kind); })
          .def(py::init(
              [](absl::string_view s) { return BinopKindFromString(s); }));

  py::class_<BinopHolder, ExprHolder>(m, "Binop")
      .def(py::init([](ModuleHolder module, Span span, BinopKind kind,
                       ExprHolder lhs, ExprHolder rhs) {
        auto* self = module.deref().Make<Binop>(std::move(span), kind,
                                                &lhs.deref(), &rhs.deref());
        return BinopHolder(self, module.module());
      }))
      .def("__str__", [](BinopHolder self) { return self.deref().ToString(); })
      .def_property_readonly("lhs",
                             [](BinopHolder self) {
                               return ExprHolder(self.deref().lhs(),
                                                 self.module());
                             })
      .def_property_readonly("rhs",
                             [](BinopHolder self) {
                               return ExprHolder(self.deref().rhs(),
                                                 self.module());
                             })
      .def_property_readonly(
          "kind", [](BinopHolder self) { return self.deref().kind(); });
}  // NOLINT(readability/fn_size)

}  // namespace xls::dslx
