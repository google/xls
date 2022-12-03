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

#include "xls/ir/function_builder.h"

#include <memory>

#include "pybind11/pybind11.h"
#include "pybind11_abseil/absl_casters.h"
#include "pybind11_abseil/statusor_caster.h"
#include "xls/common/status/import_status_module.h"
#include "xls/ir/package.h"
#include "xls/ir/python/wrapper_types.h"

namespace py = pybind11;

namespace xls {

// FunctionBuilder is a derived type (derived from BuilderBase) which does not
// play nice with PyWrap. Specifically, the methods defined in the base class
// (BuilderBase) result in PrWrap attempting to create a wrapper for a
// BuilderBase object which has no defined wrapper. Create a special wrapper for
// wrapping of FunctionBuilder methods for this purpose.
template <typename ReturnT, typename T, typename... Args>
auto FbPyWrap(ReturnT (T::*method_pointer)(Args...) const) {
  return PyWrapHelper<decltype(method_pointer), ReturnT, FunctionBuilder,
                      Args...>(method_pointer);
}

template <typename ReturnT, typename T, typename... Args>
auto FbPyWrap(ReturnT (T::*method_pointer)(Args...)) {
  return PyWrapHelper<decltype(method_pointer), ReturnT, FunctionBuilder,
                      Args...>(method_pointer);
}

PYBIND11_MODULE(function_builder, m) {
  ImportStatusModule();

  py::module::import("xls.ir.python.bits");
  py::module::import("xls.ir.python.function");
  py::module::import("xls.ir.python.lsb_or_msb");
  py::module::import("xls.ir.python.package");
  py::module::import("xls.ir.python.source_location");
  py::module::import("xls.ir.python.type");
  py::module::import("xls.ir.python.value");

  py::class_<BValueHolder>(m, "BValue")
      .def("__str__", PyWrap(&BValue::ToString))
      .def("get_type", PyWrap(&BValue::GetType))
      .def("set_name", PyWrap(&BValue::SetName), py::arg("name"))
      .def("has_assigned_name", PyWrap(&BValue::HasAssignedName))
      .def("get_name", PyWrap(&BValue::GetName));

  // -- Explicitly select overload when pybind11 can't infer it.
  // or
  BValue (FunctionBuilder::*add_or)(BValue, BValue, const SourceInfo&,
                                    std::string_view) = &FunctionBuilder::Or;
  BValue (FunctionBuilder::*add_nary_or)(absl::Span<const BValue>,
                                         const SourceInfo&, std::string_view) =
      &FunctionBuilder::Or;
  // xor
  BValue (FunctionBuilder::*add_xor)(BValue, BValue, const SourceInfo&,
                                     std::string_view) = &FunctionBuilder::Xor;
  BValue (FunctionBuilder::*add_nary_xor)(
      absl::Span<const BValue>, const SourceInfo&, std::string_view) =
      &FunctionBuilder::Xor;
  // and
  BValue (FunctionBuilder::*add_and)(BValue, BValue, const SourceInfo&,
                                     std::string_view) = &FunctionBuilder::And;
  BValue (FunctionBuilder::*add_nary_and)(
      absl::Span<const BValue>, const SourceInfo&, std::string_view) =
      &FunctionBuilder::And;

  BValue (FunctionBuilder::*add_literal_bits)(
      Bits, const SourceInfo&, std::string_view) = &FunctionBuilder::Literal;
  BValue (FunctionBuilder::*add_literal_value)(
      Value, const SourceInfo&, std::string_view) = &FunctionBuilder::Literal;
  BValue (FunctionBuilder::*add_sel)(BValue, BValue, BValue, const SourceInfo&,
                                     std::string_view) =
      &FunctionBuilder::Select;
  BValue (FunctionBuilder::*add_sel_multi)(
      BValue, absl::Span<const BValue>, std::optional<BValue>,
      const SourceInfo&, std::string_view) = &FunctionBuilder::Select;
  BValue (FunctionBuilder::*add_smul)(BValue, BValue, const SourceInfo&,
                                      std::string_view) =
      &FunctionBuilder::SMul;
  BValue (FunctionBuilder::*add_umul)(BValue, BValue, const SourceInfo&,
                                      std::string_view) =
      &FunctionBuilder::UMul;
  BValue (FunctionBuilder::*match_true)(
      absl::Span<const BValue>, absl::Span<const BValue>, BValue,
      const SourceInfo&, std::string_view) = &FunctionBuilder::MatchTrue;

  py::class_<FunctionBuilderHolder>(m, "FunctionBuilder")
      .def(py::init<std::string_view, PackageHolder>(), py::arg("name"),
           py::arg("package"))

      .def_property_readonly("name", FbPyWrap(&FunctionBuilder::name))

      .def("add_param", FbPyWrap(&FunctionBuilder::Param), py::arg("name"),
           py::arg("type"), py::arg("loc") = SourceInfo())

      .def("add_shra", FbPyWrap(&FunctionBuilder::Shra), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_shrl", FbPyWrap(&FunctionBuilder::Shrl), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_shll", FbPyWrap(&FunctionBuilder::Shll), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = SourceInfo(), py::arg("name") = "")
      // -- Bitwise operations.
      // or
      .def("add_or", FbPyWrap(add_or), py::arg("lhs"), py::arg("rhs"),
           py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_nary_or", FbPyWrap(add_nary_or), py::arg("operands"),
           py::arg("loc") = SourceInfo(), py::arg("name") = "")
      // xor
      .def("add_xor", FbPyWrap(add_xor), py::arg("lhs"), py::arg("rhs"),
           py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_nary_xor", FbPyWrap(add_nary_xor), py::arg("operands"),
           py::arg("loc") = SourceInfo(), py::arg("name") = "")
      // and
      .def("add_and", FbPyWrap(add_and), py::arg("lhs"), py::arg("rhs"),
           py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_nary_and", FbPyWrap(add_nary_and), py::arg("operands"),
           py::arg("loc") = SourceInfo(), py::arg("name") = "")

      .def("add_smul", PyWrap(add_smul), py::arg("lhs"), py::arg("rhs"),
           py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_umul", PyWrap(add_umul), py::arg("lhs"), py::arg("rhs"),
           py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_udiv", FbPyWrap(&FunctionBuilder::UDiv), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_sub", FbPyWrap(&FunctionBuilder::Subtract), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_add", FbPyWrap(&FunctionBuilder::Add), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = SourceInfo(), py::arg("name") = "")

      .def("add_concat", FbPyWrap(&FunctionBuilder::Concat),
           py::arg("operands"), py::arg("loc") = SourceInfo(),
           py::arg("name") = "")

      .def("add_ule", FbPyWrap(&FunctionBuilder::ULe), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_ult", FbPyWrap(&FunctionBuilder::ULt), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_uge", FbPyWrap(&FunctionBuilder::UGe), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_ugt", FbPyWrap(&FunctionBuilder::UGt), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = SourceInfo(), py::arg("name") = "")

      .def("add_sle", FbPyWrap(&FunctionBuilder::SLe), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_slt", FbPyWrap(&FunctionBuilder::SLt), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_sge", FbPyWrap(&FunctionBuilder::SGe), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_sgt", FbPyWrap(&FunctionBuilder::SGt), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = SourceInfo(), py::arg("name") = "")

      .def("add_eq", FbPyWrap(&FunctionBuilder::Eq), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_ne", FbPyWrap(&FunctionBuilder::Ne), py::arg("lhs"),
           py::arg("rhs"), py::arg("loc") = SourceInfo(), py::arg("name") = "")

      .def("add_neg", FbPyWrap(&FunctionBuilder::Negate), py::arg("x"),
           py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_not", FbPyWrap(&FunctionBuilder::Not), py::arg("x"),
           py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_clz", FbPyWrap(&FunctionBuilder::Clz), py::arg("x"),
           py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_ctz", FbPyWrap(&FunctionBuilder::Ctz), py::arg("x"),
           py::arg("loc") = SourceInfo(), py::arg("name") = "")

      .def("add_one_hot", FbPyWrap(&FunctionBuilder::OneHot), py::arg("arg"),
           py::arg("lsb_is_prio"), py::arg("loc") = SourceInfo(),
           py::arg("name") = "")
      .def("add_one_hot_sel", FbPyWrap(&FunctionBuilder::OneHotSelect),
           py::arg("selector"), py::arg("cases"), py::arg("loc") = SourceInfo(),
           py::arg("name") = "")
      .def("add_priority_sel", FbPyWrap(&FunctionBuilder::PrioritySelect),
           py::arg("selector"), py::arg("cases"), py::arg("loc") = SourceInfo(),
           py::arg("name") = "")

      .def("add_literal_bits", PyWrap(add_literal_bits), py::arg("bits"),
           py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_literal_value", PyWrap(add_literal_value), py::arg("value"),
           py::arg("loc") = SourceInfo(), py::arg("name") = "")

      .def("add_sel", PyWrap(add_sel), py::arg("selector"), py::arg("on_true"),
           py::arg("on_false"), py::arg("loc") = SourceInfo(),
           py::arg("name") = "")
      .def("add_sel_multi", PyWrap(add_sel_multi), py::arg("selector"),
           py::arg("cases"), py::arg("default_value"),
           py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_match_true", PyWrap(match_true), py::arg("case_clauses"),
           py::arg("case_values"), py::arg("default_value"),
           py::arg("loc") = SourceInfo(), py::arg("name") = "")

      .def("add_after_all", FbPyWrap(&FunctionBuilder::AfterAll),
           py::arg("dependencies"), py::arg("loc") = SourceInfo(),
           py::arg("name") = "")

      .def("add_tuple", FbPyWrap(&FunctionBuilder::Tuple), py::arg("elements"),
           py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_array", FbPyWrap(&FunctionBuilder::Array), py::arg("elements"),
           py::arg("element_type"), py::arg("loc") = SourceInfo(),
           py::arg("name") = "")

      .def("add_tuple_index", FbPyWrap(&FunctionBuilder::TupleIndex),
           py::arg("arg"), py::arg("idx"), py::arg("loc") = SourceInfo(),
           py::arg("name") = "")

      .def("add_counted_for", FbPyWrap(&FunctionBuilder::CountedFor),
           py::arg("init_value"), py::arg("trip_count"), py::arg("stride"),
           py::arg("body"), py::arg("invariant_args"),
           py::arg("loc") = SourceInfo(), py::arg("name") = "")

      .def("add_map", FbPyWrap(&FunctionBuilder::Map), py::arg("operand"),
           py::arg("to_apply"), py::arg("loc") = SourceInfo(),
           py::arg("name") = "")

      .def("add_invoke", FbPyWrap(&FunctionBuilder::Invoke), py::arg("args"),
           py::arg("to_apply"), py::arg("loc") = SourceInfo(),
           py::arg("name") = "")

      .def("add_array_index", FbPyWrap(&FunctionBuilder::ArrayIndex),
           py::arg("arg"), py::arg("idx"), py::arg("loc") = SourceInfo(),
           py::arg("name") = "")
      .def("add_array_update", FbPyWrap(&FunctionBuilder::ArrayUpdate),
           py::arg("arg"), py::arg("idx"), py::arg("update_value"),
           py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_array_concat", FbPyWrap(&FunctionBuilder::ArrayConcat),
           py::arg("operands"), py::arg("loc") = SourceInfo(),
           py::arg("name") = "")
      .def("add_reverse", FbPyWrap(&FunctionBuilder::Reverse), py::arg("arg"),
           py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_identity", FbPyWrap(&FunctionBuilder::Identity), py::arg("arg"),
           py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_signext", FbPyWrap(&FunctionBuilder::SignExtend),
           py::arg("arg"), py::arg("new_bit_count"),
           py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_zeroext", FbPyWrap(&FunctionBuilder::ZeroExtend),
           py::arg("arg"), py::arg("new_bit_count"),
           py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_bit_slice", FbPyWrap(&FunctionBuilder::BitSlice),
           py::arg("arg"), py::arg("start"), py::arg("width"),
           py::arg("loc") = SourceInfo(), py::arg("name") = "")
      .def("add_dynamic_bit_slice", FbPyWrap(&FunctionBuilder::DynamicBitSlice),
           py::arg("arg"), py::arg("start"), py::arg("width"),
           py::arg("loc") = SourceInfo(), py::arg("name") = "")

      .def("add_and_reduce", FbPyWrap(&FunctionBuilder::AndReduce),
           py::arg("operand"), py::arg("loc") = SourceInfo(),
           py::arg("name") = "")
      .def("add_or_reduce", FbPyWrap(&FunctionBuilder::OrReduce),
           py::arg("operand"), py::arg("loc") = SourceInfo(),
           py::arg("name") = "")
      .def("add_xor_reduce", FbPyWrap(&FunctionBuilder::XorReduce),
           py::arg("operand"), py::arg("loc") = SourceInfo(),
           py::arg("name") = "")

      .def("build", FbPyWrap(&FunctionBuilder::Build));
}

}  // namespace xls
