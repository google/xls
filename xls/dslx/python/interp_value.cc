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
#include "xls/ir/ir_parser.h"

namespace py = pybind11;

namespace xls::dslx {

struct InterpValuePickler {
  // Note: normally we'd use a struct for this, but then we wouldn't get the
  // nice auto-conversion that pybind11 provides and we'd have to register the
  // type explicitly. Since this should go away once everything is ported to
  // C++, we leave it for now.
  using State = std::tuple<int64_t, absl::optional<Bits>,
                           absl::optional<std::vector<InterpValue>>>;

  static State Pickle(const InterpValue& self) {
    InterpValueTag tag = self.tag();
    int64_t tag_value = static_cast<int64_t>(tag);
    absl::optional<Bits> bits;
    if (self.HasBits()) {
      bits = self.GetBitsOrDie();
    }
    absl::optional<std::vector<InterpValue>> values;
    if (self.HasValues()) {
      values = self.GetValuesOrDie();
    }
    return std::make_tuple(tag_value, bits, values);
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
    return InterpValue(static_cast<InterpValueTag>(std::get<0>(state)),
                       payload);
  }
};

PYBIND11_MODULE(interp_value, m) {
  ImportStatusModule();

  // Required to be able to pickle IR `Bits` inside of `InterpValue`s.
  py::module::import("xls.ir.python.bits");

  py::class_<InterpValue>(m, "Value")
      .def(py::pickle(&InterpValuePickler::Pickle,
                      &InterpValuePickler::Unpickle))
      .def("__eq__", &InterpValue::Eq)
      .def("__ne__", &InterpValue::Ne)
      .def("__str__", [](const InterpValue& self) { return self.ToString(); })
      .def("__repr__", &InterpValue::ToHumanString)
      .def("eq",
           [](const InterpValue& self, const InterpValue& other) {
             return InterpValue::MakeBool(self.Eq(other));
           })
      .def("to_human_str", &InterpValue::ToHumanString)
      .def("to_ir_str",
           [](const InterpValue& self) -> absl::StatusOr<std::string> {
             XLS_ASSIGN_OR_RETURN(xls::Value value, self.ConvertToIr());
             return value.ToString(FormatPreference::kHex);
           })
      .def("is_true", &InterpValue::IsTrue)
      .def("is_false", &InterpValue::IsFalse)
      .def("to_signed",
           [](const InterpValue& v) -> absl::StatusOr<InterpValue> {
             if (v.IsSigned()) {
               return v;
             }
             XLS_ASSIGN_OR_RETURN(Bits b, v.GetBits());
             return InterpValue::MakeSigned(std::move(b));
           })
      .def_static("make_tuple", &InterpValue::MakeTuple, py::arg("elements"))
      .def_static("make_array", &InterpValue::MakeArray, py::arg("elements"));

  m.def("interp_value_from_ir_string",
        [](absl::string_view s) -> absl::StatusOr<InterpValue> {
          XLS_ASSIGN_OR_RETURN(Value v, Parser::ParseTypedValue(s));
          return ValueToInterpValue(v);
        });
}

}  // namespace xls::dslx
