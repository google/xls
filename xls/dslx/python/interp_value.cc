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

#include "xls/dslx/interp_value.h"

#include "absl/base/casts.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor_pybind_caster.h"
#include "xls/dslx/ir_converter.h"
#include "xls/dslx/python/cpp_ast.h"
#include "xls/ir/ir_parser.h"

namespace py = pybind11;

namespace xls::dslx {

struct InterpValuePickler {
  // Note: normally we'd use a struct for this, but then we wouldn't get the
  // nice auto-conversion that pybind11 provides and we'd have to register the
  // type explicitly. Since this should go away once everything is ported to
  // C++, we leave it for now.
  using State = std::tuple<InterpValueTag, absl::optional<Bits>,
                           absl::optional<std::vector<InterpValue>>>;

  static State Pickle(const InterpValue& self) {
    InterpValueTag tag = self.tag();
    absl::optional<Bits> bits;
    if (self.HasBits()) {
      bits = self.GetBitsOrDie();
    }
    absl::optional<std::vector<InterpValue>> values;
    if (self.HasValues()) {
      values = self.GetValuesOrDie();
    }
    return std::make_tuple(tag, bits, values);
  }
  static InterpValue Unpickle(const State& state) {
    InterpValue::Payload payload;
    const absl::optional<Bits>& bits = std::get<1>(state);
    if (bits.has_value()) {
      payload = bits.value();
    } else {
      const auto& values = std::get<2>(state);
      XLS_CHECK(values.has_value());
      payload = values.value();
    }
    return InterpValue(std::get<0>(state), payload);
  }
};

PYBIND11_MODULE(interp_value, m) {
  ImportStatusModule();

  py::enum_<InterpValueTag>(m, "Tag")
      .value("UBITS", InterpValueTag::kUBits)
      .value("SBITS", InterpValueTag::kSBits)
      .value("ARRAY", InterpValueTag::kArray)
      .value("TUPLE", InterpValueTag::kTuple)
      .value("ENUM", InterpValueTag::kEnum)
      .value("FUNCTION", InterpValueTag::kFunction);

  py::enum_<Builtin>(m, "Builtin").def("to_name", &BuiltinToString);

  m.def("get_builtin",
        [](absl::string_view name) { return BuiltinFromString(name); });

  m.attr("Builtin").attr("get") = m.attr("get_builtin");

  py::class_<InterpValue>(m, "Value")
      .def(py::pickle(&InterpValuePickler::Pickle,
                      &InterpValuePickler::Unpickle))
      .def("__eq__", &InterpValue::Eq)
      .def("__ne__", &InterpValue::Ne)
      .def("__str__", [](const InterpValue& self) { return self.ToString(); })
      .def("__repr__", &InterpValue::ToHumanString)
      .def("__len__", &InterpValue::GetLength)
      .def("gt", &InterpValue::Gt)
      .def("ge", &InterpValue::Ge)
      .def("lt", &InterpValue::Lt)
      .def("le", &InterpValue::Le)
      .def("ne",
           [](const InterpValue& self, const InterpValue& other) {
             return InterpValue::MakeBool(self.Ne(other));
           })
      .def("eq",
           [](const InterpValue& self, const InterpValue& other) {
             return InterpValue::MakeBool(self.Eq(other));
           })
      .def("to_human_str", &InterpValue::ToHumanString)
      .def("bitwise_negate", &InterpValue::BitwiseNegate)
      .def("bitwise_xor", &InterpValue::BitwiseXor)
      .def("bitwise_or", &InterpValue::BitwiseOr)
      .def("bitwise_and", &InterpValue::BitwiseAnd)
      .def("arithmetic_negate", &InterpValue::ArithmeticNegate)
      .def("add_with_carry", &InterpValue::AddWithCarry)
      .def("shll", &InterpValue::Shll)
      .def("shrl", &InterpValue::Shrl)
      .def("shra", &InterpValue::Shra)
      .def("add", &InterpValue::Add)
      .def("floordiv", &InterpValue::FloorDiv)
      .def("mul", &InterpValue::Mul)
      .def("sub", &InterpValue::Sub)
      .def("scmp", &InterpValue::SCmp)
      .def("index", &InterpValue::Index)
      .def("index",
           [](const InterpValue& self, uint64 i) {
             return self.Index(InterpValue::MakeUBits(/*bit_count=*/64, i));
           })
      .def("update", &InterpValue::Update)
      .def("update",
           [](const InterpValue& self, uint64 i, const InterpValue& value) {
             return self.Update(InterpValue::MakeU64(i), value);
           })
      .def("flatten", &InterpValue::Flatten)
      .def("slice", &InterpValue::Slice)
      .def("slice",
           [](const InterpValue& self, uint64 start, uint64 length) {
             return self.Slice(InterpValue::MakeU64(start),
                               InterpValue::MakeU64(length));
           })
      .def("one_hot", &InterpValue::OneHot)
      .def("concat", &InterpValue::Concat)
      .def("get_bits", &InterpValue::GetBits)
      .def("get_bit_count", &InterpValue::GetBitCount)
      .def("get_bit_value_uint64", &InterpValue::GetBitValueUint64)
      .def("get_bit_value_int64", &InterpValue::GetBitValueInt64)
      .def("get_bit_value_check_sign", &InterpValue::GetBitValueCheckSign)
      .def("sign_ext", &InterpValue::SignExt)
      .def("zero_ext", &InterpValue::ZeroExt)
      .def("get_type",
           [](const InterpValue& self) -> absl::optional<EnumDefHolder> {
             if (self.type() != nullptr) {
               return EnumDefHolder(self.type(),
                                    self.type()->owner()->shared_from_this());
             }
             return absl::nullopt;
           })
      .def("get_elements", &InterpValue::GetValues)
      .def("is_builtin_function", &InterpValue::IsBuiltinFunction)
      .def_property_readonly("tag", &InterpValue::tag)
      .def("is_bits", &InterpValue::IsBits)
      .def("is_ubits", &InterpValue::IsUBits)
      .def("is_sbits", &InterpValue::IsSBits)
      .def("is_enum", &InterpValue::IsEnum)
      .def("is_function", &InterpValue::IsFunction)
      .def("is_array", &InterpValue::IsArray)
      .def("is_tuple", &InterpValue::IsTuple)
      .def("is_true", &InterpValue::IsTrue)
      .def("is_false", &InterpValue::IsFalse)
      .def("is_nil_tuple", &InterpValue::IsNilTuple)
      // Factories.
      .def_static("make_bool", &InterpValue::MakeBool, py::arg("value"))
      .def_static("make_ubits", &InterpValue::MakeUBits, py::arg("bit_count"),
                  py::arg("value"))
      .def_static("make_sbits", &InterpValue::MakeSBits, py::arg("bit_count"),
                  py::arg("value"))
      .def_static("make_bits", &InterpValue::MakeBits, py::arg("tag"),
                  py::arg("bits"))
      .def_static("make_u32", &InterpValue::MakeU32, py::arg("value"))
      .def_static("make_array", &InterpValue::MakeArray, py::arg("elements"))
      .def_static("make_tuple", &InterpValue::MakeTuple, py::arg("elements"))
      .def_static("make_enum",
                  [](Bits value, EnumDefHolder enum_ast) {
                    return InterpValue::MakeEnum(value, &enum_ast.deref());
                  })
      .def_static("make_nil", []() { return InterpValue::MakeTuple({}); });

  m.def("interp_value_from_string",
        [](absl::string_view s) -> absl::StatusOr<InterpValue> {
          XLS_ASSIGN_OR_RETURN(Value v, Parser::ParseTypedValue(s));
          return ValueToInterpValue(v);
        });
}

}  // namespace xls::dslx
