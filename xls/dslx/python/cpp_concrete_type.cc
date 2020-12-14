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
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor_pybind_caster.h"
#include "xls/dslx/concrete_type.h"
#include "xls/dslx/python/cpp_ast.h"

namespace py = pybind11;

namespace xls::dslx {

PYBIND11_MODULE(cpp_concrete_type, m) {
  py::class_<ConcreteTypeDim>(m, "ConcreteTypeDim")
      .def(py::init(
          [](const ParametricExpression& e) { return ConcreteTypeDim(e); }))
      .def(py::init([](int64 value) { return ConcreteTypeDim(value); }))
      .def("__add__", &ConcreteTypeDim::Add)
      .def("__mul__", &ConcreteTypeDim::Mul)
      .def("__eq__", [](const ConcreteTypeDim& self,
                        const ConcreteTypeDim& other) { return self == other; })
      .def("__eq__",
           [](const ConcreteTypeDim& self,
              const absl::variant<int64, const ParametricExpression*>& other) {
             return self == other;
           })
      .def_property_readonly(
          "value",
          [](const ConcreteTypeDim& self)
              -> absl::variant<int64, const ParametricExpression*> {
            const ConcreteTypeDim::Variant& value = self.value();
            if (absl::holds_alternative<int64>(value)) {
              return absl::get<int64>(value);
            }
            if (absl::holds_alternative<std::unique_ptr<ParametricExpression>>(
                    value)) {
              return absl::get<std::unique_ptr<ParametricExpression>>(value)
                  .get();
            }
            XLS_LOG(FATAL) << "Unhandled ConcreteTypeDim variant.";
          },
          py::return_value_policy::reference_internal);

  // class ConcreteType
  py::class_<ConcreteType>(m, "ConcreteType")
      .def("__str__", &ConcreteType::ToString)
      .def("__repr__", &ConcreteType::ToRepr)
      .def("__eq__", &ConcreteType::operator==)
      .def("__ne__", &ConcreteType::operator!=)
      .def("has_enum", &ConcreteType::HasEnum)
      .def("get_total_bit_count", &ConcreteType::GetTotalBitCount)
      .def("get_debug_type_name", &ConcreteType::GetDebugTypeName)
      .def("compatible_with", &ConcreteType::CompatibleWith)
      .def("get_all_dims",
           [](const ConcreteType& self) {
             std::vector<ConcreteTypeDim> dims = self.GetAllDims();
             py::tuple results(dims.size());
             for (int64 i = 0; i < dims.size(); ++i) {
               results[i] = std::move(dims[i]);
             }
             return results;
           })
      .def("is_nil", &ConcreteType::IsNil);

  // class TupleType
  py::class_<TupleType, ConcreteType>(m, "TupleType")
      // UnnamedMembers
      .def(py::init([](const std::vector<const ConcreteType*>& unnamed) {
             std::vector<std::unique_ptr<ConcreteType>> members;
             for (const auto* type : unnamed) {
               members.push_back(type->CloneToUnique());
             }
             auto result = absl::make_unique<TupleType>(
                 TupleType::Members(std::move(members)), nullptr);
             XLS_CHECK(!result->is_named());
             return result;
           }),
           py::arg("members"))
      // NamedMembers
      .def(
          py::init(
              [](const std::vector<std::pair<std::string, const ConcreteType*>>&
                     named,
                 absl::optional<StructDefHolder> struct_def) {
                std::vector<TupleType::NamedMember> members;
                for (const auto& [name, type] : named) {
                  members.push_back(
                      TupleType::NamedMember{name, type->CloneToUnique()});
                }
                auto result = absl::make_unique<TupleType>(
                    TupleType::Members(std::move(members)),
                    struct_def.has_value() ? &struct_def->deref() : nullptr);
                XLS_CHECK(result->is_named());
                return result;
              }),
          py::arg("members"), py::arg("struct") = absl::nullopt)
      .def(
          "get_tuple_member",
          [](const TupleType& self, int64 i) {
            return self.GetUnnamedMembers()[i];
          },
          py::return_value_policy::reference_internal)
      .def("get_nominal_type",
           [](const TupleType& t) -> absl::optional<StructDefHolder> {
             if (StructDef* s = t.nominal_type()) {
               return StructDefHolder(s, s->owner()->shared_from_this());
             }
             return absl::nullopt;
           })
      .def("get_tuple_length", &TupleType::size)
      .def("get_unnamed_members",
           [](const TupleType& t) {
             std::vector<std::unique_ptr<ConcreteType>> v;
             for (const ConcreteType* e : t.GetUnnamedMembers()) {
               v.push_back(e->CloneToUnique());
             }
             return v;
           })
      .def("has_named_member", &TupleType::HasNamedMember)
      .def(
          "get_member_type_by_name",
          [](const TupleType& self, absl::string_view target) {
            absl::optional<const ConcreteType*> t =
                self.GetMemberTypeByName(target);
            if (t.has_value()) {
              return t.value();
            }
            throw py::key_error(
                absl::StrCat("Tuple has no member with name ", target));
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly("size", &TupleType::size)
      .def_property_readonly("named", &TupleType::is_named)
      .def_property_readonly(
          "members",
          [](const TupleType& t)
              -> absl::variant<std::vector<std::pair<
                                   std::string, std::unique_ptr<ConcreteType>>>,
                               std::vector<std::unique_ptr<ConcreteType>>> {
            if (t.is_named()) {
              std::vector<std::pair<std::string, std::unique_ptr<ConcreteType>>>
                  v;
              for (const auto& m :
                   absl::get<TupleType::NamedMembers>(t.members())) {
                v.push_back({m.name, m.type->CloneToUnique()});
              }
              return v;
            }
            std::vector<std::unique_ptr<ConcreteType>> v;
            for (const ConcreteType* t : t.GetUnnamedMembers()) {
              v.push_back(t->CloneToUnique());
            }
            return v;
          })
      .def_property_readonly(
          "tuple_names", [](const TupleType& t) -> absl::StatusOr<py::tuple> {
            XLS_ASSIGN_OR_RETURN(std::vector<std::string> names,
                                 t.GetMemberNames());
            py::tuple result(names.size());
            for (int64 i = 0; i < names.size(); ++i) {
              result[i] = names[i];
            }
            return result;
          });

  // class ArrayType
  py::class_<ArrayType, ConcreteType>(m, "ArrayType")
      .def(py::init([](const ConcreteType& elem_type, int64 size) {
        return absl::make_unique<ArrayType>(elem_type.CloneToUnique(),
                                            ConcreteTypeDim(size));
      }))
      .def(py::init(
          [](const ConcreteType& elem_type, const ParametricExpression& e) {
            return absl::make_unique<ArrayType>(elem_type.CloneToUnique(),
                                                ConcreteTypeDim(e));
          }))
      .def(py::init(
          [](const ConcreteType& elem_type, const ConcreteTypeDim& size) {
            return absl::make_unique<ArrayType>(elem_type.CloneToUnique(),
                                                size.Clone());
          }))
      .def("get_element_type", &ArrayType::element_type,
           py::return_value_policy::reference_internal)
      .def_property_readonly("element_type", &ArrayType::element_type,
                             py::return_value_policy::reference_internal)
      .def_property_readonly(
          "size", [](const ArrayType& self) { return self.size().Clone(); });

  // class BitsType
  py::class_<BitsType, ConcreteType>(m, "BitsType")
      .def(py::init([](bool is_signed, int64 size) {
             return absl::make_unique<BitsType>(is_signed,
                                                ConcreteTypeDim(size));
           }),
           py::arg("signed"), py::arg("size"))
      .def(py::init([](bool is_signed, const ParametricExpression& size) {
             return absl::make_unique<BitsType>(is_signed,
                                                ConcreteTypeDim(size));
           }),
           py::arg("signed"), py::arg("size"))
      .def(py::init([](bool is_signed, const ConcreteTypeDim& size) {
             return absl::make_unique<BitsType>(is_signed, size);
           }),
           py::arg("signed"), py::arg("size"))
      .def("get_signedness", &BitsType::is_signed)
      .def("to_ubits", &BitsType::ToUBits)
      .def_property_readonly(
          "size", [](const BitsType& self) { return self.size().Clone(); })
      .def_property_readonly("signed", &BitsType::is_signed);

  // class EnumType
  py::class_<EnumType, ConcreteType>(m, "EnumType")
      .def(py::init([](EnumDefHolder enum_, const ConcreteTypeDim& bit_count) {
        return EnumType(&enum_.deref(), bit_count.Clone());
      }))
      .def(py::init([](EnumDefHolder enum_, int64 bit_count) {
        return EnumType(&enum_.deref(), ConcreteTypeDim(bit_count));
      }))
      .def("get_nominal_type",
           [](const EnumType& t) -> absl::optional<EnumDefHolder> {
             if (EnumDef* e = t.nominal_type()) {
               return EnumDefHolder(e, e->owner()->shared_from_this());
             }
             return absl::nullopt;
           })
      .def_property_readonly("signed", &EnumType::signedness)
      .def_property_readonly(
          "size", [](const EnumType& self) { return self.size().Clone(); });

  py::class_<FunctionType, ConcreteType>(m, "FunctionType")
      .def(py::init([](const std::vector<const ConcreteType*>& params,
                       const ConcreteType* return_type) {
             std::vector<std::unique_ptr<ConcreteType>> owned_params;
             for (const ConcreteType* param : params) {
               owned_params.push_back(param->CloneToUnique());
             }
             return absl::make_unique<FunctionType>(
                 std::move(owned_params), return_type->CloneToUnique());
           }),
           py::return_value_policy::reference_internal)
      .def_property_readonly(
          "params",
          [](const FunctionType& self) {
            py::tuple t(self.GetParamCount());
            std::vector<const ConcreteType*> params = self.GetParams();
            for (int64 i = 0; i < params.size(); ++i) {
              t[i] = params[i]->CloneToUnique();
            }
            return t;
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly("return_type", &FunctionType::return_type,
                             py::return_value_policy::reference_internal);

  m.def("is_ubits", &IsUBits);
  m.def("is_sbits", &IsSBits);

  auto cls = m.attr("ConcreteType");
  cls.attr("U32") = BitsType(false, 32);
  cls.attr("U8") = BitsType(false, 8);
  cls.attr("U1") = BitsType(false, 1);
  cls.attr("NIL") = absl::make_unique<TupleType>(TupleType::Members{});
}

}  // namespace xls::dslx
