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

#include "xls/ir/bits.h"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11_abseil/statusor_caster.h"
#include "xls/common/status/import_status_module.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits_ops.h"

namespace py = pybind11;

namespace xls {

static Bits BitsFromPyInt(py::int_ x, int64_t bit_count) {
  bool negative = false;
  py::int_ zero(0);
  if (PyObject_RichCompareBool(x.ptr(), zero.ptr(), Py_LT)) {
    negative = true;
    x = py::reinterpret_steal<py::int_>(PyNumber_Negative(x.ptr()));
  }
  py::int_ one(1);
  InlineBitmap bitmap(bit_count);
  for (int64_t i = 0; i < bit_count; ++i) {
    auto low_bit =
        py::reinterpret_steal<py::int_>(PyNumber_And(x.ptr(), one.ptr()));
    x = py::reinterpret_steal<py::int_>(PyNumber_Rshift(x.ptr(), one.ptr()));
    bitmap.Set(i, PyObject_RichCompareBool(low_bit.ptr(), one.ptr(), Py_EQ));
  }
  auto result = Bits::FromBitmap(std::move(bitmap));
  if (negative) {
    result = bits_ops::Negate(result);
  }
  return result;
}

static py::int_ BitsToPyInt(Bits x, bool is_signed) {
  py::int_ result;
  py::int_ sixty_four(64);
  int64_t word_count = CeilOfRatio(x.bit_count(), int64_t{64});
  bool input_sign = x.msb();
  if (is_signed && input_sign) {
    x = bits_ops::Negate(x);
  }
  for (int64_t i = 0; i < word_count; ++i) {
    uint64_t word = x.WordToUint64(word_count - i - 1).value();
    result = py::reinterpret_steal<py::int_>(
        PyNumber_Lshift(result.ptr(), sixty_four.ptr()));
    result = py::reinterpret_steal<py::int_>(
        PyNumber_Or(result.ptr(), py::int_(word).ptr()));
  }
  if (is_signed && input_sign) {
    result = -result;
  }
  return result;
}

PYBIND11_MODULE(bits, m) {
  ImportStatusModule();

  py::class_<Bits>(m, "Bits")
      .def(py::init<int64_t>(), py::arg("bit_count"))
      .def("__eq__", &Bits::operator==)
      .def("__ne__", &Bits::operator!=)
      .def("__or__",
           [](const Bits& self, const Bits& other) { return self | other; })
      .def("__xor__",
           [](const Bits& self, const Bits& other) { return self ^ other; })
      .def("__and__",
           [](const Bits& self, const Bits& other) -> absl::StatusOr<Bits> {
             if (self.bit_count() != other.bit_count()) {
               return absl::InvalidArgumentError(absl::StrFormat(
                   "Same bit count is required for lhs and rhs; got %d vs %d",
                   self.bit_count(), other.bit_count()));
             }
             return self & other;
           })
      .def("__invert__", [](const Bits& self) { return ~self; })
      .def(py::pickle(
          [](const Bits& self) {
            py::int_ value = BitsToPyInt(self, /*is_signed=*/false);
            return std::make_tuple(value, self.bit_count());
          },
          [](std::tuple<py::int_, int64_t> t) {
            return BitsFromPyInt(/*x=*/std::get<0>(t),
                                 /*bit_count=*/std::get<1>(t));
          }))
      .def("__rshift__",
           [](const Bits& self, const Bits& amount) -> absl::StatusOr<Bits> {
             XLS_ASSIGN_OR_RETURN(int64_t amount64, amount.ToInt64());
             amount64 = amount64 < 0 ? self.bit_count() : amount64;
             return bits_ops::ShiftRightLogical(self, amount64);
           })
      .def("__rshift__",
           [](const Bits& self, int64_t amount64) {
             amount64 = amount64 < 0 ? self.bit_count() : amount64;
             return bits_ops::ShiftRightLogical(self, amount64);
           })
      // TODO(leary): 2020-10-15 Switch to get_bit_count or make a readonly
      // property.
      .def("bit_count", &Bits::bit_count)
      .def("word_to_uint", &Bits::WordToUint64, py::arg("word_number") = 0)
      .def("get_msb", &Bits::msb)
      .def("get_mask_bits",
           [](const Bits& self) { return ~Bits(self.bit_count()); })
      .def("is_zero", &Bits::IsZero)
      .def("zero_ext", &bits_ops::ZeroExtend, py::arg("new_bit_count"))
      .def("reverse", [](const Bits& self) { return bits_ops::Reverse(self); })
      .def("bitwise_negate", [](const Bits& self) { return ~self; })
      .def("concat",
           [](const Bits& self, const Bits& other) {
             return bits_ops::Concat({self, other});
           })
      .def("slice", [](const Bits& self, int64_t start,
                       int64_t width) { return self.Slice(start, width); })
      .def("get_lsb_index",
           [](const Bits& self, int64_t i) -> absl::StatusOr<Bits> {
             if (i >= self.bit_count()) {
               return absl::InvalidArgumentError(
                   absl::StrFormat("Bit index %d is out of range; size %d", i,
                                   self.bit_count()));
             }
             return UBits(/*value=*/self.Get(i), /*bit_count=*/1);
           })
      .def("to_uint",
           [](const Bits& self) {
             return BitsToPyInt(self, /*is_signed=*/false);
           })
      .def("to_int", [](const Bits& self) {
        return BitsToPyInt(self, /*is_signed=*/true);
      });

  m.def("min_bit_count_unsigned", &Bits::MinBitCountUnsigned);

  m.def("UBits", &UBitsWithStatus, py::arg("value"), py::arg("bit_count"));
  m.def("SBits", &SBitsWithStatus, py::arg("value"), py::arg("bit_count"));

  m.def("from_long", &BitsFromPyInt, py::arg("value"), py::arg("bit_count"));
  m.def("concat_all", [](const std::vector<Bits>& elements) {
    return bits_ops::Concat(elements);
  });
}

}  // namespace xls
