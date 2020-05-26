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

#include "xls/ir/function_builder.h"

#include <memory>

#include "pybind11/pybind11.h"
#include "xls/common/python/absl_casters.h"
#include "xls/common/status/statusor_pybind_caster.h"
#include "xls/ir/package.h"
#include "xls/ir/python/wrapper_types.h"

namespace py = pybind11;

namespace xls {

PYBIND11_MODULE(function_builder, m) {
  py::module::import("xls.ir.python.bits");
  py::module::import("xls.ir.python.function");
  py::module::import("xls.ir.python.lsb_or_msb");
  py::module::import("xls.ir.python.package");
  py::module::import("xls.ir.python.source_location");
  py::module::import("xls.ir.python.type");
  py::module::import("xls.ir.python.value");

  py::class_<BValueHolder>(m, "BValue")
      .def("__str__", PyWrap(&BValue::ToString))
      .def("get_builder", PyWrap(&BValue::builder))
      .def("get_type", PyWrap(&BValue::GetType));

  // Explicitly select overload when pybind11 can't infer it.
  BValue (FunctionBuilder::*add_or)(
      BValue, BValue, absl::optional<SourceLocation>) = &FunctionBuilder::Or;
  BValue (FunctionBuilder::*add_nary_or)(absl::Span<const BValue>,
                                         absl::optional<SourceLocation>) =
      &FunctionBuilder::Or;
  BValue (FunctionBuilder::*add_literal_bits)(
      Bits, absl::optional<SourceLocation>) = &FunctionBuilder::Literal;
  BValue (FunctionBuilder::*add_literal_value)(
      Value, absl::optional<SourceLocation>) = &FunctionBuilder::Literal;
  BValue (FunctionBuilder::*add_sel)(BValue, BValue, BValue,
                                     absl::optional<SourceLocation>) =
      &FunctionBuilder::Select;
  BValue (FunctionBuilder::*add_sel_multi)(
      BValue, absl::Span<const BValue>, absl::optional<BValue>,
      absl::optional<SourceLocation>) = &FunctionBuilder::Select;
  BValue (FunctionBuilder::*add_smul)(
      BValue, BValue, absl::optional<SourceLocation>) = &FunctionBuilder::SMul;
  BValue (FunctionBuilder::*add_umul)(
      BValue, BValue, absl::optional<SourceLocation>) = &FunctionBuilder::UMul;
  BValue (FunctionBuilder::*match_true)(
      absl::Span<const BValue>, absl::Span<const BValue>, BValue,
      absl::optional<SourceLocation>) = &FunctionBuilder::MatchTrue;

  py::class_<FunctionBuilderHolder>(m, "FunctionBuilder")
      .def(py::init<absl::string_view, PackageHolder>(), py::arg("name"),
           py::arg("package"))

      .def_property_readonly("name", PyWrap(&FunctionBuilder::name))

      .def("add_param", PyWrap(&FunctionBuilder::Param), py::arg("name"),
           py::arg("type"), py::arg("loc") = absl::nullopt)

      .def("add_shra", PyWrap(&FunctionBuilder::Shra), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = absl::nullopt)
      .def("add_shrl", PyWrap(&FunctionBuilder::Shrl), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = absl::nullopt)
      .def("add_shll", PyWrap(&FunctionBuilder::Shll), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = absl::nullopt)
      .def("add_or", PyWrap(add_or), py::arg("lhs"), py::arg("rhs"),
           py::arg("loc") = absl::nullopt)
      .def("add_nary_or", PyWrap(add_nary_or), py::arg("operands"),
           py::arg("loc") = absl::nullopt)
      .def("add_xor", PyWrap(&FunctionBuilder::Xor), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = absl::nullopt)
      .def("add_and", PyWrap(&FunctionBuilder::And), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = absl::nullopt)
      .def("add_smul", PyWrap(add_smul), py::arg("lhs"), py::arg("rhs"),
           py::arg("loc") = absl::nullopt)
      .def("add_umul", PyWrap(add_umul), py::arg("lhs"), py::arg("rhs"),
           py::arg("loc") = absl::nullopt)
      .def("add_udiv", PyWrap(&FunctionBuilder::UDiv), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = absl::nullopt)
      .def("add_sub", PyWrap(&FunctionBuilder::Subtract), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = absl::nullopt)
      .def("add_add", PyWrap(&FunctionBuilder::Add), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = absl::nullopt)

      .def("add_concat", PyWrap(&FunctionBuilder::Concat), py::arg("operands"),
           py::arg("loc") = absl::nullopt)

      .def("add_ule", PyWrap(&FunctionBuilder::ULe), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = absl::nullopt)
      .def("add_ult", PyWrap(&FunctionBuilder::ULt), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = absl::nullopt)
      .def("add_uge", PyWrap(&FunctionBuilder::UGe), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = absl::nullopt)
      .def("add_ugt", PyWrap(&FunctionBuilder::UGt), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = absl::nullopt)

      .def("add_sle", PyWrap(&FunctionBuilder::SLe), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = absl::nullopt)
      .def("add_slt", PyWrap(&FunctionBuilder::SLt), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = absl::nullopt)
      .def("add_sge", PyWrap(&FunctionBuilder::SGe), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = absl::nullopt)
      .def("add_sgt", PyWrap(&FunctionBuilder::SGt), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = absl::nullopt)

      .def("add_eq", PyWrap(&FunctionBuilder::Eq), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = absl::nullopt)
      .def("add_ne", PyWrap(&FunctionBuilder::Ne), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = absl::nullopt)

      .def("add_neg", PyWrap(&FunctionBuilder::Negate), py::arg("x"),
           py::arg("loc") = absl::nullopt)
      .def("add_not", PyWrap(&FunctionBuilder::Not), py::arg("x"),
           py::arg("loc") = absl::nullopt)
      .def("add_clz", PyWrap(&FunctionBuilder::Clz), py::arg("x"),
           py::arg("loc") = absl::nullopt)
      .def("add_ctz", PyWrap(&FunctionBuilder::Ctz), py::arg("x"),
           py::arg("loc") = absl::nullopt)

      .def("add_one_hot", PyWrap(&FunctionBuilder::OneHot), py::arg("arg"),
           py::arg("lsb_is_prio"), py::arg("loc") = absl::nullopt)
      .def("add_one_hot_sel", PyWrap(&FunctionBuilder::OneHotSelect),
           py::arg("selector"), py::arg("cases"),
           py::arg("loc") = absl::nullopt)

      .def("add_literal_bits", PyWrap(add_literal_bits), py::arg("bits"),
           py::arg("loc") = absl::nullopt)
      .def("add_literal_value", PyWrap(add_literal_value), py::arg("value"),
           py::arg("loc") = absl::nullopt)

      .def("add_sel", PyWrap(add_sel), py::arg("selector"), py::arg("on_true"),
           py::arg("on_false"), py::arg("loc") = absl::nullopt)
      .def("add_sel_multi", PyWrap(add_sel_multi), py::arg("selector"),
           py::arg("cases"), py::arg("default_value"),
           py::arg("loc") = absl::nullopt)
      .def("add_match_true", PyWrap(match_true), py::arg("case_clauses"),
           py::arg("case_values"), py::arg("default_value"),
           py::arg("loc") = absl::nullopt)

      .def("add_tuple", PyWrap(&FunctionBuilder::Tuple), py::arg("elements"),
           py::arg("loc") = absl::nullopt)
      .def("add_array", PyWrap(&FunctionBuilder::Array), py::arg("elements"),
           py::arg("element_type"), py::arg("loc") = absl::nullopt)

      .def("add_tuple_index", PyWrap(&FunctionBuilder::TupleIndex),
           py::arg("arg"), py::arg("idx"), py::arg("loc") = absl::nullopt)

      .def("add_counted_for", PyWrap(&FunctionBuilder::CountedFor),
           py::arg("init_value"), py::arg("trip_count"), py::arg("stride"),
           py::arg("body"), py::arg("invariant_args"),
           py::arg("loc") = absl::nullopt)

      .def("add_map", PyWrap(&FunctionBuilder::Map), py::arg("operand"),
           py::arg("to_apply"), py::arg("loc") = absl::nullopt)

      .def("add_invoke", PyWrap(&FunctionBuilder::Invoke), py::arg("args"),
           py::arg("to_apply"), py::arg("loc") = absl::nullopt)

      .def("add_array_index", PyWrap(&FunctionBuilder::ArrayIndex),
           py::arg("arg"), py::arg("idx"), py::arg("loc") = absl::nullopt)
      .def("add_array_update", PyWrap(&FunctionBuilder::ArrayUpdate),
           py::arg("arg"), py::arg("idx"), py::arg("update_value"),
           py::arg("loc") = absl::nullopt)
      .def("add_reverse", PyWrap(&FunctionBuilder::Reverse), py::arg("arg"),
           py::arg("loc") = absl::nullopt)
      .def("add_identity", PyWrap(&FunctionBuilder::Identity), py::arg("arg"),
           py::arg("loc") = absl::nullopt)
      .def("add_signext", PyWrap(&FunctionBuilder::SignExtend), py::arg("arg"),
           py::arg("new_bit_count"), py::arg("loc") = absl::nullopt)
      .def("add_zeroext", PyWrap(&FunctionBuilder::ZeroExtend), py::arg("arg"),
           py::arg("new_bit_count"), py::arg("loc") = absl::nullopt)
      .def("add_bit_slice", PyWrap(&FunctionBuilder::BitSlice), py::arg("arg"),
           py::arg("start"), py::arg("width"), py::arg("loc") = absl::nullopt)

      .def("add_and_reduce", PyWrap(&FunctionBuilder::AndReduce),
           py::arg("operand"), py::arg("loc") = absl::nullopt)
      .def("add_or_reduce", PyWrap(&FunctionBuilder::OrReduce),
           py::arg("operand"), py::arg("loc") = absl::nullopt)
      .def("add_xor_reduce", PyWrap(&FunctionBuilder::XorReduce),
           py::arg("operand"), py::arg("loc") = absl::nullopt)

      .def("build", PyWrap(&FunctionBuilder::Build));
}

}  // namespace xls
