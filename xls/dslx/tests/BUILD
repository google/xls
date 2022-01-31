# Copyright 2020 The XLS Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Interpreter-based standalone tests.

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "get_mangled_ir_symbol",
    "xls_benchmark_ir",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_opt_ir_test",
    "xls_dslx_test",
    "xls_eval_ir_test",
)

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

xls_dslx_test(
    name = "tuple_with_array_member_dslx_test",
    srcs = ["tuple_with_array_member.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir(
    name = "tuple_with_array_member",
    srcs = ["tuple_with_array_member.x"],
)

xls_dslx_opt_ir_test(
    name = "tuple_with_array_member_test",
    dep = ":tuple_with_array_member",
)

# -- binops

# TODO(rspringer): 2022-01-19: Can't use the bytecode interpreter due to a
# parametric invocation (the right TypeInfo isn't used for lookup).
xls_dslx_opt_ir(
    name = "binops",
    srcs = ["binops.x"],
    ir_file = "binops.ir",
)

xls_dslx_test(
    name = "binops_dslx_test",
    srcs = ["binops.x"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode due to lack of
    # parametrics support.
)

# TODO: 2021-05-28 Add xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "binops_eval_ir_test",
    src = ":binops.ir",
)

xls_benchmark_ir(
    name = "binops_benchmark_ir",
    src = ":binops.ir",
)

# -- comparisons

xls_dslx_opt_ir(
    name = "comparisons",
    srcs = ["comparisons.x"],
)

xls_dslx_test(
    name = "comparisons_dslx_test",
    srcs = ["comparisons.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

# -- constexpr
xls_dslx_library(
    name = "constexpr_dslx",
    srcs = ["constexpr.x"],
)

xls_dslx_opt_ir(
    name = "constexpr",
    srcs = ["constexpr.x"],
)

xls_dslx_test(
    name = "constexpr_dslx_test",
    srcs = ["constexpr.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir_test(
    name = "constexpr_test",
    dep = ":constexpr",
)

# -- constexpr_attrs

xls_dslx_opt_ir(
    name = "constexpr_attrs",
    srcs = ["constexpr_attrs.x"],
    deps = [":constexpr_dslx"],
)

xls_dslx_test(
    name = "constexpr_attrs_dslx_test",
    srcs = ["constexpr_attrs.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
    deps = [":constexpr_dslx"],
)

# -- fallible_constexpr

xls_dslx_test(
    name = "fallible_constexpr_dslx_test",
    srcs = ["fallible_constexpr.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

# -- compound_eq

xls_dslx_opt_ir(
    name = "compound_eq",
    srcs = ["compound_eq.x"],
)

xls_dslx_test(
    name = "compound_eq_dslx_test",
    srcs = ["compound_eq.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_test(
    name = "constexpr_clog2_dslx_test",
    srcs = ["constexpr_clog2.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir(
    name = "constexpr_clog2",
    srcs = ["constexpr_clog2.x"],
)

xls_dslx_opt_ir_test(
    name = "constexpr_clog2_test",
    dep = ":constexpr_clog2",
)

xls_dslx_test(
    name = "constexpr_imported_clog2_dslx_test",
    srcs = ["constexpr_imported_clog2.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
    deps = [":constexpr_dslx"],
)

xls_dslx_opt_ir(
    name = "constexpr_imported_clog2",
    srcs = ["constexpr_imported_clog2.x"],
    deps = [":constexpr_dslx"],
)

xls_dslx_opt_ir_test(
    name = "constexpr_imported_clog2_test",
    dep = ":constexpr_imported_clog2",
)

xls_dslx_test(
    name = "constexpr_types_via_invocations_dslx_test",
    srcs = ["constexpr_types_via_invocations.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir(
    name = "constexpr_types_via_invocations",
    srcs = ["constexpr_types_via_invocations.x"],
)

xls_dslx_opt_ir_test(
    name = "constexpr_types_via_invocations_test",
    dep = ":constexpr_types_via_invocations",
)

xls_dslx_test(
    name = "constexpr_types_via_invocations_inline_dslx_test",
    srcs = ["constexpr_types_via_invocations_inline.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir(
    name = "constexpr_types_via_invocations_inline",
    srcs = ["constexpr_types_via_invocations_inline.x"],
)

xls_dslx_opt_ir_test(
    name = "constexpr_types_via_invocations_inline_test",
    dep = ":constexpr_types_via_invocations_inline",
)

xls_dslx_test(
    name = "constexpr_types_via_stdlib_invocation_dslx_test",
    srcs = ["constexpr_types_via_stdlib_invocation.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir(
    name = "constexpr_types_via_stdlib_invocation",
    srcs = ["constexpr_types_via_stdlib_invocation.x"],
)

xls_dslx_opt_ir_test(
    name = "constexpr_types_via_stdlib_invocation_test",
    dep = ":constexpr_types_via_stdlib_invocation",
)

xls_dslx_test(
    name = "constexpr_slice_dslx_test",
    srcs = ["constexpr_slice.x"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode due to lack of
    # parametrics support.
)

xls_dslx_opt_ir(
    name = "constexpr_slice",
    srcs = ["constexpr_slice.x"],
)

xls_dslx_opt_ir(
    name = "explicit_parametric",
    srcs = ["explicit_parametric.x"],
)

xls_dslx_test(
    name = "explicit_parametric_dslx_test",
    srcs = ["explicit_parametric.x"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode due to lack of
    # parametrics support.
)

xls_dslx_opt_ir(
    name = "explicit_parametric_simple",
    srcs = ["explicit_parametric_simple.x"],
)

xls_dslx_test(
    name = "explicit_parametric_simple_dslx_test",
    srcs = ["explicit_parametric_simple.x"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode due to lack of
    # parametrics support.
)

xls_dslx_opt_ir(
    name = "explicit_parametric_reduced",
    srcs = ["explicit_parametric_reduced.x"],
)

xls_dslx_test(
    name = "explicit_parametric_reduced_dslx_test",
    srcs = ["explicit_parametric_reduced.x"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode due to lack of
    # parametrics support.
)

xls_dslx_opt_ir(
    name = "parametric_dot_product",
    srcs = ["parametric_dot_product.x"],
)

xls_dslx_opt_ir_test(
    name = "parametric_dot_product_test",
    dep = ":parametric_dot_product",
)

xls_dslx_test(
    name = "string_conversion_dslx_test",
    srcs = ["string_conversion.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir(
    name = "string_conversion",
    srcs = ["string_conversion.x"],
)

xls_dslx_opt_ir_test(
    name = "string_conversion_test",
    dep = ":string_conversion",
)

xls_dslx_test(
    name = "import_constexpr_dslx_test",
    srcs = ["import_constexpr.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
    deps = [":constexpr_dslx"],
)

xls_dslx_opt_ir(
    name = "import_constexpr",
    srcs = ["import_constexpr.x"],
    deps = [":constexpr_dslx"],
)

xls_dslx_opt_ir_test(
    name = "import_constexpr_test",
    dep = ":import_constexpr",
)

# TODO(rspringer): 2022-01-19: Can't run bytecode due to missing "map"
# builtin.
xls_dslx_opt_ir(
    name = "map",
    srcs = ["map.x"],
)

xls_dslx_opt_ir_test(
    name = "map_test",
    dep = ":map",
)

xls_dslx_test(
    name = "multiplies_dslx_test",
    srcs = ["multiplies.x"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode yet due to needing
    # ColonRef invocations.
)

xls_dslx_opt_ir(
    name = "multiplies",
    srcs = ["multiplies.x"],
)

xls_dslx_opt_ir_test(
    name = "multiplies_test",
    dep = ":multiplies",
)

xls_dslx_test(
    name = "match_sample_dslx_test",
    srcs = ["match_sample.x"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode yet due to needing
    # match support.
)

xls_dslx_opt_ir(
    name = "match_sample",
    srcs = ["match_sample.x"],
)

xls_dslx_opt_ir_test(
    name = "match_sample_test",
    dep = ":match_sample",
)

xls_dslx_test(
    name = "match_const_dslx_test",
    srcs = ["match_const.x"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode yet due to needing
    # match support.
)

xls_dslx_opt_ir(
    name = "match_const",
    srcs = ["match_const.x"],
)

xls_dslx_opt_ir_test(
    name = "match_const_test",
    dep = ":match_const",
)

xls_dslx_test(
    name = "match_multi_dslx_test",
    srcs = ["match_multi.x"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode yet due to needing
    # match support.
)

xls_dslx_opt_ir(
    name = "match_multi",
    srcs = ["match_multi.x"],
)

xls_dslx_opt_ir_test(
    name = "match_multi_test",
    dep = ":match_multi",
)

xls_dslx_library(
    name = "number_of_imported_type_import_dslx",
    srcs = ["number_of_imported_type_import.x"],
)

xls_dslx_opt_ir(
    name = "number_of_imported_type",
    srcs = ["number_of_imported_type.x"],
    deps = [":number_of_imported_type_import_dslx"],
)

xls_dslx_test(
    name = "number_of_imported_type_dslx_test",
    srcs = ["number_of_imported_type.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
    deps = [":number_of_imported_type_import_dslx"],
)

xls_dslx_opt_ir_test(
    name = "number_of_imported_type_test",
    dep = ":number_of_imported_type",
)

xls_dslx_library(
    name = "mod_struct_point_dslx",
    srcs = ["mod_struct_point.x"],
)

xls_dslx_opt_ir(
    name = "type_aliasing_importer",
    srcs = ["type_aliasing_importer.x"],
    deps = [":mod_struct_point_dslx"],
)

xls_dslx_test(
    name = "type_aliasing_importer_dslx_test",
    srcs = ["type_aliasing_importer.x"],
    dslx_test_args = {
        "bytecode": "True",
    },
    deps = [":mod_struct_point_dslx"],
)

xls_dslx_opt_ir(
    name = "parametric_call_parametric",
    srcs = ["parametric_call_parametric.x"],
)

xls_dslx_test(
    name = "parametric_call_parametric_dslx_test",
    srcs = ["parametric_call_parametric.x"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode due to needing parametric
    # support.
)

xls_dslx_library(
    name = "parametric_import_dslx",
    srcs = ["parametric_import.x"],
)

xls_dslx_opt_ir(
    name = "parametric_importer",
    srcs = ["parametric_importer.x"],
    deps = [":parametric_import_dslx"],
)

xls_dslx_test(
    name = "parametric_importer_dslx_test",
    srcs = ["parametric_importer.x"],
    deps = [":parametric_import_dslx"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode due to needing parametric
    # support.
)

xls_dslx_opt_ir(
    name = "signed_comparisons",
    srcs = ["signed_comparisons.x"],
)

xls_dslx_test(
    name = "signed_comparisons_dslx_test",
    srcs = ["signed_comparisons.x"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode yet due to needing
    # ColonRef invocations.
)

xls_dslx_test(
    name = "subtract_to_negative_dslx_test",
    srcs = ["subtract_to_negative.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir(
    name = "subtract_to_negative",
    srcs = ["subtract_to_negative.x"],
)

xls_dslx_opt_ir_test(
    name = "subtract_to_negative_test",
    dep = ":subtract_to_negative",
)

xls_dslx_opt_ir(
    name = "trace",
    srcs = ["trace.x"],
)

xls_dslx_opt_ir_test(
    name = "trace_test",
    dep = ":trace",
)

xls_dslx_opt_ir(
    name = "trace_fmt_hello_opt_ir",
    srcs = ["trace.x"],
)

xls_dslx_opt_ir_test(
    name = "trace_fmt_hello_opt_ir_test",
    dep = ":trace_fmt_hello_opt_ir",
)

# Note: This test is enough to keep the example code compiling,
# but does not currently check the output.
# TODO(https://github.com/google/xls/issues/489): 2021-09-02
# Check the trace output of this test
xls_dslx_test(
    name = "trace_fmt_hello_test",
    srcs = ["trace_fmt_hello.x"],
    dslx_test_args = {"compare": "interpreter"},
)

xls_dslx_test(
    name = "array_of_typedef_in_tuple_dslx_test",
    srcs = ["array_of_typedef_in_tuple.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir(
    name = "array_of_typedef_in_tuple",
    srcs = ["array_of_typedef_in_tuple.x"],
)

xls_dslx_opt_ir_test(
    name = "array_of_typedef_in_tuple_test",
    dep = ":array_of_typedef_in_tuple",
)

xls_dslx_opt_ir(
    name = "array_of_tuple",
    srcs = ["array_of_tuple.x"],
)

xls_dslx_test(
    name = "array_of_tuple_dslx_test",
    srcs = ["array_of_tuple.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir(
    name = "array_slice",
    srcs = ["array_slice.x"],
)

xls_dslx_test(
    name = "array_slice_dslx_test",
    srcs = ["array_slice.x"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode yet due to needing
    # array_slice builtin support.
)

# TODO(leary): 2019-07-24 Missing support for array of tuples eliding cast node.
xls_dslx_test(
    name = "parametric_array_of_tuples_dslx_test",
    srcs = ["parametric_array_of_tuples.x"],
    dslx_test_args = {
        "compare": "none",
    },
)

xls_dslx_test(
    name = "parametric_functions_dslx_test",
    srcs = ["parametric_functions.x"],
    dslx_test_args = {
        "compare": "none",
    },
)

xls_dslx_opt_ir(
    name = "parametric_value_as_nested_loop_bound",
    srcs = ["parametric_value_as_nested_loop_bound.x"],
)

xls_dslx_opt_ir_test(
    name = "parametric_value_as_nested_loop_bound_test",
    dep = ":parametric_value_as_nested_loop_bound",
)

xls_dslx_opt_ir(
    name = "derived_parametric_value_as_nested_loop_bound",
    srcs = ["derived_parametric_value_as_nested_loop_bound.x"],
)

xls_dslx_opt_ir_test(
    name = "derived_parametric_value_as_nested_loop_bound_test",
    dep = ":derived_parametric_value_as_nested_loop_bound",
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "bit_slice_dslx_test",
    srcs = ["bit_slice.x"],
    dslx_test_args = {
        "compare": "none",
    },
    # TODO(rspringer): 2022-01-19: Can't run bytecode yet due to needing
    # bit_slice builtin support.
)

xls_dslx_test(
    name = "bit_slice_update_dslx_test",
    srcs = ["bit_slice_update.x"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode yet due to needing
    # bit_slice_update builtin support.
)

xls_dslx_opt_ir(
    name = "bit_slice_update",
    srcs = ["bit_slice_update.x"],
)

xls_dslx_opt_ir_test(
    name = "bit_slice_update_test",
    dep = ":bit_slice_update",
)

xls_dslx_test(
    name = "const_slice_limit_dslx_test",
    srcs = ["const_slice_limit.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir(
    name = "const_slice_limit",
    srcs = ["const_slice_limit.x"],
)

xls_dslx_opt_ir_test(
    name = "const_slice_limit_test",
    dep = ":const_slice_limit",
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "bit_slice_syntax_dslx_test",
    srcs = ["bit_slice_syntax.x"],
    dslx_test_args = {
        "compare": "none",
    },
    # TODO(rspringer): 2022-01-19: Can't run bytecode due to needing parametrics
    # support.
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "tuple_indexing_dslx_test",
    srcs = ["tuple_indexing.x"],
    dslx_test_args = {
        "bytecode": "true",
        "compare": "none",
    },
)

xls_dslx_test(
    name = "clz_dslx_test",
    srcs = ["clz.x"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode due to missing clz builtin.
)

xls_dslx_opt_ir(
    name = "clz",
    srcs = ["clz.x"],
)

xls_dslx_opt_ir_test(
    name = "clz_test",
    dep = ":clz",
)

xls_dslx_test(
    name = "ctz_dslx_test",
    srcs = ["ctz.x"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode due to missing ctz builtin.
)

xls_dslx_opt_ir(
    name = "ctz",
    srcs = ["ctz.x"],
)

xls_dslx_opt_ir_test(
    name = "ctz_test",
    dep = ":ctz",
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "one_hot_dslx_test",
    srcs = ["one_hot.x"],
    dslx_test_args = {
        "compare": "none",
    },
    # TODO(rspringer): 2022-01-19: Can't run bytecode due to missing one_hot
    # builtin.
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "one_hot_sel",
    srcs = ["one_hot_sel.x"],
    dslx_test_args = {
        "compare": "none",
    },
    # TODO(rspringer): 2022-01-19: Can't run bytecode due to missing one_hot_sel
    # builtin.
)

xls_dslx_opt_ir(
    name = "reductions",
    srcs = ["reductions.x"],
    ir_file = "reductions.ir",
)

xls_dslx_test(
    name = "reductions_dslx_test",
    srcs = ["reductions.x"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode due to missing reductions
    # builtins.
)

# TODO: 2021-05-28 Add xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "reductions_eval_ir_test",
    src = ":reductions.ir",
)

xls_benchmark_ir(
    name = "reductions_benchmark_ir",
    src = ":reductions.ir",
)

xls_dslx_test(
    name = "signed_number_type_dslx_test",
    srcs = ["signed_number_type.x"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode due to missing assert_lt
    # builtin.
)

xls_dslx_opt_ir(
    name = "signed_number_type",
    srcs = ["signed_number_type.x"],
)

xls_dslx_opt_ir_test(
    name = "signed_number_type_test",
    dep = ":signed_number_type",
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "numerical_conversions_dslx_test",
    srcs = ["numerical_conversions.x"],
    dslx_test_args = {
        "bytecode": "true",
        "compare": "none",
    },
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "numerical_conversions2_dslx_test",
    srcs = ["numerical_conversions2.x"],
    dslx_test_args = {
        "bytecode": "true",
        "compare": "none",
    },
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "signex_builtin_dslx_test",
    srcs = ["signex_builtin.x"],
    dslx_test_args = {
        "compare": "none",
    },
    # TODO(rspringer): 2022-01-19: Can't run bytecode due to missing signex
    # builtins.
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "enum_values_dslx_test",
    srcs = ["enum_values.x"],
    dslx_test_args = {
        "bytecode": "true",
        "compare": "none",
    },
)

# TODO(leary); 2019-10-24 Enable when we have ConcatArray op.
xls_dslx_test(
    name = "casts_dslx_test",
    srcs = ["casts.x"],
    dslx_test_args = {
        "bytecode": "true",
        "compare": "none",
    },
)

# TODO(leary); 2019-10-24 Enable when we have ConcatArray op.
xls_dslx_test(
    name = "cast_to_array_dslx_test",
    srcs = ["cast_to_array.x"],
    dslx_test_args = {
        "compare": "none",
    },
)

xls_dslx_test(
    name = "parametric_with_comparison_dslx_test",
    srcs = ["parametric_with_comparison.x"],
    dslx_test_args = {
        "compare": "none",
    },
    # TODO(rspringer): 2022-01-19: Can't run bytecode due to missing parametrics
    # support.
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "parametric_smul_dslx_test",
    srcs = ["parametric_smul.x"],
    dslx_test_args = {
        "compare": "none",
    },
    # TODO(rspringer): 2022-01-19: Can't run bytecode due to missing parametrics
    # support.
)

xls_dslx_test(
    name = "basic_struct_dslx_test",
    srcs = ["basic_struct.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir(
    name = "basic_struct",
    srcs = ["basic_struct.x"],
)

xls_dslx_opt_ir_test(
    name = "basic_struct_test",
    dep = ":basic_struct",
)

xls_dslx_test(
    name = "basic_struct_parameter_dslx_test",
    srcs = ["basic_struct_parameter.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir(
    name = "basic_struct_parameter",
    srcs = ["basic_struct_parameter.x"],
)

xls_dslx_opt_ir_test(
    name = "basic_struct_parameter_test",
    dep = ":basic_struct_parameter",
)

xls_dslx_test(
    name = "basic_struct_attr_dslx_test",
    srcs = ["basic_struct_attr.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir(
    name = "basic_struct_attr",
    srcs = ["basic_struct_attr.x"],
)

xls_dslx_opt_ir_test(
    name = "basic_struct_attr_test",
    dep = ":basic_struct_attr",
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "struct_equality_dslx_test",
    srcs = ["struct_equality.x"],
    dslx_test_args = {
        "bytecode": "true",
        "compare": "none",
    },
)

xls_dslx_opt_ir(
    name = "basic_struct_update",
    srcs = ["basic_struct_update.x"],
)

xls_dslx_test(
    name = "basic_struct_update_dslx_test",
    srcs = ["basic_struct_update.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir_test(
    name = "basic_struct_update_test",
    dep = ":basic_struct_update",
)

xls_dslx_test(
    name = "struct_splat_update_dslx_test",
    srcs = ["struct_splat_update.x"],
    dslx_test_args = {
        "bytecode": "True",
    },
)

xls_dslx_opt_ir(
    name = "struct_splat_update",
    srcs = ["struct_splat_update.x"],
)

xls_dslx_opt_ir_test(
    name = "struct_splat_update_test",
    dep = ":struct_splat_update",
)

xls_dslx_test(
    name = "struct_splat_update_to_const_dslx_test",
    srcs = ["struct_splat_update_to_const.x"],
    dslx_test_args = {
        "bytecode": "True",
    },
)

xls_dslx_opt_ir(
    name = "struct_splat_update_to_const",
    srcs = ["struct_splat_update_to_const.x"],
)

xls_dslx_opt_ir_test(
    name = "struct_splat_update_to_const_test",
    dep = ":struct_splat_update_to_const",
)

xls_dslx_test(
    name = "empty_struct_splat_update_dslx_test",
    srcs = ["empty_struct_splat_update.x"],
    dslx_test_args = {
        "bytecode": "True",
    },
)

xls_dslx_opt_ir(
    name = "empty_struct_splat_update",
    srcs = ["empty_struct_splat_update.x"],
)

xls_dslx_opt_ir_test(
    name = "empty_struct_splat_update_test",
    dep = ":empty_struct_splat_update",
)

xls_dslx_test(
    name = "basic_struct_array_dslx_test",
    srcs = ["basic_struct_array.x"],
)

xls_dslx_opt_ir(
    name = "basic_struct_array",
    srcs = ["basic_struct_array.x"],
)

xls_dslx_opt_ir_test(
    name = "basic_struct_array_test",
    dep = ":basic_struct_array",
)

xls_dslx_test(
    name = "basic_enum_array_dslx_test",
    srcs = ["basic_enum_array.x"],
)

xls_dslx_opt_ir(
    name = "basic_enum_array",
    srcs = ["basic_enum_array.x"],
)

xls_dslx_opt_ir_test(
    name = "basic_enum_array_test",
    dep = ":basic_enum_array",
)

xls_dslx_test(
    name = "basic_array_dslx_test",
    srcs = ["basic_array.x"],
)

xls_dslx_opt_ir(
    name = "basic_array",
    srcs = ["basic_array.x"],
)

xls_dslx_opt_ir_test(
    name = "basic_array_test",
    dep = ":basic_array",
)

xls_dslx_test(
    name = "std_lsb_dslx_test",
    srcs = ["std_lsb.x"],
)

xls_dslx_opt_ir(
    name = "std_lsb",
    srcs = ["std_lsb.x"],
)

xls_dslx_opt_ir_test(
    name = "std_lsb_test",
    dep = ":std_lsb",
)

xls_dslx_test(
    name = "std_alias_dslx_test",
    srcs = ["std_alias.x"],
)

xls_dslx_opt_ir(
    name = "std_alias",
    srcs = ["std_alias.x"],
)

xls_dslx_opt_ir_test(
    name = "std_alias_test",
    dep = ":std_alias",
)

filegroup(
    name = "mod_imported_file",
    srcs = ["mod_imported.x"],
)

xls_dslx_library(
    name = "mod_imported_dslx",
    srcs = ["mod_imported.x"],
)

xls_dslx_test(
    name = "mod_importer_dslx_test",
    srcs = ["mod_importer.x"],
    deps = [":mod_imported_dslx"],
)

xls_dslx_opt_ir(
    name = "mod_importer",
    srcs = ["mod_importer.x"],
    deps = [":mod_imported_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_importer_test",
    dep = ":mod_importer",
)

xls_dslx_library(
    name = "mod_imported_typedef_dslx",
    srcs = ["mod_imported_typedef.x"],
)

xls_dslx_test(
    name = "mod_importer_typedef_dslx_test",
    srcs = ["mod_importer_typedef.x"],
    deps = [":mod_imported_typedef_dslx"],
)

xls_dslx_opt_ir(
    name = "mod_importer_typedef",
    srcs = ["mod_importer_typedef.x"],
    deps = [":mod_imported_typedef_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_importer_typedef_test",
    dep = ":mod_importer_typedef",
)

xls_dslx_library(
    name = "mod_imported_struct_of_enum_dslx",
    srcs = [":mod_imported_struct_of_enum.x"],
)

xls_dslx_test(
    name = "mod_importer_struct_of_enum_dslx_test",
    srcs = ["mod_importer_struct_of_enum.x"],
    deps = [":mod_imported_struct_of_enum_dslx"],
)

xls_dslx_opt_ir(
    name = "mod_importer_struct_of_enum",
    srcs = ["mod_importer_struct_of_enum.x"],
    deps = [":mod_imported_struct_of_enum_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_importer_struct_of_enum_test",
    dep = ":mod_importer_struct_of_enum",
)

xls_dslx_test(
    name = "mod_importer_simple_dslx_test",
    srcs = ["mod_importer_simple.x"],
    deps = [":mod_imported_dslx"],
)

xls_dslx_opt_ir(
    name = "mod_importer_simple",
    srcs = ["mod_importer_simple.x"],
    deps = [":mod_imported_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_importer_simple_test",
    dep = ":mod_importer_simple",
)

xls_dslx_test(
    name = "mod_struct_importer_dslx_test",
    srcs = ["mod_struct_importer.x"],
    deps = [":mod_imported_dslx"],
)

xls_dslx_opt_ir(
    name = "mod_struct_importer",
    srcs = ["mod_struct_importer.x"],
    deps = [":mod_imported_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_struct_importer_test",
    dep = ":mod_struct_importer",
)

xls_dslx_test(
    name = "mod_enum_importer_dslx_test",
    srcs = ["mod_enum_importer.x"],
    deps = [":mod_imported_dslx"],
)

xls_dslx_opt_ir(
    name = "mod_enum_importer",
    srcs = ["mod_enum_importer.x"],
    deps = [":mod_imported_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_enum_importer_test",
    dep = ":mod_enum_importer",
)

xls_dslx_library(
    name = "mod_parametric_id_dslx",
    srcs = [
        "mod_parametric_id.x",
        "mod_parametric_id_indirect.x",
    ],
)

xls_dslx_test(
    name = "mod_parametric_id_user_dslx_test",
    srcs = ["mod_parametric_id_user.x"],
    deps = [":mod_parametric_id_dslx"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode due to lack of parametrics
    # support.
)

xls_dslx_opt_ir(
    name = "mod_parametric_id_user",
    srcs = ["mod_parametric_id_user.x"],
    deps = [":mod_parametric_id_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_parametric_id_user_test",
    dep = ":mod_parametric_id_user",
)

xls_dslx_library(
    name = "mod_parametric_id_with_default",
    srcs = ["mod_parametric_id_with_default.x"],
)

xls_dslx_test(
    name = "parametric_id_with_default_user_test",
    srcs = ["parametric_id_with_default_user.x"],
    deps = [":mod_parametric_id_with_default"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode due to lack of parametrics
    # support.
)

xls_dslx_opt_ir(
    name = "parametric_id_with_default_user_opt_ir",
    srcs = ["parametric_id_with_default_user.x"],
    deps = [":mod_parametric_id_with_default"],
)

xls_dslx_library(
    name = "mod_imported_lsb_dslx",
    srcs = ["mod_imported_lsb.x"],
)

xls_dslx_test(
    name = "call_imported_fn_simple_dslx_test",
    srcs = ["call_imported_fn_simple.x"],
    deps = [":mod_imported_lsb_dslx"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode due to lack of ColonRef
    # invocation support.
)

xls_dslx_opt_ir(
    name = "call_imported_fn_simple",
    srcs = ["call_imported_fn_simple.x"],
    deps = [":mod_imported_lsb_dslx"],
)

xls_dslx_opt_ir_test(
    name = "call_imported_fn_simple_test",
    dep = ":call_imported_fn_simple",
)

xls_dslx_library(
    name = "mod_imported_aliases_dslx",
    srcs = ["mod_imported_aliases.x"],
)

xls_dslx_test(
    name = "mod_typedef_importer_dslx_test",
    srcs = ["mod_typedef_importer.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
    deps = [":mod_imported_aliases_dslx"],
)

xls_dslx_opt_ir(
    name = "mod_typedef_importer",
    srcs = ["mod_typedef_importer.x"],
    deps = [":mod_imported_aliases_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_typedef_importer_test",
    dep = ":mod_typedef_importer",
)

xls_dslx_test(
    name = "array_ellipsis_dslx_test",
    srcs = ["array_ellipsis.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir(
    name = "array_ellipsis",
    srcs = ["array_ellipsis.x"],
)

xls_dslx_opt_ir_test(
    name = "array_ellipsis_test",
    dep = ":array_ellipsis",
)

xls_dslx_test(
    name = "local_const_value_dslx_test",
    srcs = ["local_const_value.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir(
    name = "local_const_value",
    srcs = ["local_const_value.x"],
)

xls_dslx_opt_ir_test(
    name = "local_const_value_test",
    dep = ":local_const_value",
)

xls_dslx_test(
    name = "const_value_as_nested_loop_bound_dslx_test",
    srcs = ["const_value_as_nested_loop_bound.x"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of For node
    # handling.
)

xls_dslx_opt_ir(
    name = "const_value_as_nested_loop_bound",
    srcs = ["const_value_as_nested_loop_bound.x"],
)

xls_dslx_opt_ir_test(
    name = "const_value_as_nested_loop_bound_test",
    dep = ":const_value_as_nested_loop_bound",
)

xls_dslx_test(
    name = "local_const_value_in_parametric_dslx_test",
    srcs = ["local_const_value_in_parametric.x"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of
    # parametrics support.
)

xls_dslx_opt_ir(
    name = "local_const_value_in_parametric",
    srcs = ["local_const_value_in_parametric.x"],
)

xls_dslx_opt_ir_test(
    name = "local_const_value_in_parametric_test",
    dep = ":local_const_value_in_parametric",
)

xls_dslx_test(
    name = "constant_array_unannotated_type_dslx_test",
    srcs = ["constant_array_unannotated_type.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir(
    name = "constant_array_unannotated_type",
    srcs = ["constant_array_unannotated_type.x"],
)

xls_dslx_opt_ir_test(
    name = "constant_array_unannotated_type_test",
    dep = ":constant_array_unannotated_type",
)

xls_dslx_test(
    name = "reverse_dslx_test",
    srcs = ["reverse.x"],
    # TODO(rspringer): Cannot run bytecode due to lack of "rev" builtin support.
)

xls_dslx_opt_ir(
    name = "reverse",
    srcs = ["reverse.x"],
)

xls_dslx_opt_ir_test(
    name = "reverse_test",
    dep = ":reverse",
)

xls_dslx_test(
    name = "typedef_multidim_array_dslx_test",
    srcs = ["typedef_multidim_array.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir(
    name = "typedef_multidim_array",
    srcs = ["typedef_multidim_array.x"],
)

xls_dslx_opt_ir_test(
    name = "typedef_multidim_array_test",
    dep = ":typedef_multidim_array",
)

xls_dslx_test(
    name = "parametric_multidim_array_dslx_test",
    srcs = ["parametric_multidim_array.x"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of
    # parametrics support.
)

xls_dslx_opt_ir(
    name = "parametric_multidim_array",
    srcs = ["parametric_multidim_array.x"],
    opt_ir_args = {
        "entry": get_mangled_ir_symbol(
            "parametric_multidim_array",
            "id",
            (6, 3, 2),
        ),
    },
)

xls_dslx_opt_ir_test(
    name = "parametric_multidim_array_test",
    dep = ":parametric_multidim_array",
)

xls_dslx_test(
    name = "for_sans_type_annotation_dslx_test",
    srcs = ["for_sans_type_annotation.x"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of
    # parametrics support.
)

xls_dslx_opt_ir(
    name = "for_sans_type_annotation",
    srcs = ["for_sans_type_annotation.x"],
)

xls_dslx_opt_ir_test(
    name = "for_sans_type_annotation_test",
    dep = ":for_sans_type_annotation",
)

xls_dslx_test(
    name = "derived_parametrics_dslx_test",
    srcs = ["derived_parametrics.x"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of
    # parametrics support.
)

xls_dslx_opt_ir(
    name = "derived_parametrics",
    srcs = ["derived_parametrics.x"],
)

xls_dslx_opt_ir_test(
    name = "derived_parametrics_test",
    dep = ":derived_parametrics",
)

xls_dslx_test(
    name = "parametric_struct_dslx_test",
    srcs = ["parametric_struct.x"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of
    # parametrics support.
)

xls_dslx_opt_ir(
    name = "parametric_struct",
    srcs = ["parametric_struct.x"],
)

xls_dslx_opt_ir_test(
    name = "parametric_struct_test",
    dep = ":parametric_struct",
)

xls_dslx_test(
    name = "parametric_struct_simple_dslx_test",
    srcs = ["parametric_struct_simple.x"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of
    # parametrics support.
)

xls_dslx_opt_ir(
    name = "parametric_struct_simple",
    srcs = ["parametric_struct_simple.x"],
)

xls_dslx_test(
    name = "test_f_calls_parametric_f_dslx_test",
    srcs = ["test_f_calls_parametric_f.x"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of
    # parametrics support.
)

xls_dslx_opt_ir(
    name = "test_f_calls_parametric_f",
    srcs = ["test_f_calls_parametric_f.x"],
    opt_ir_args = {
        "entry": get_mangled_ir_symbol("test_f_calls_parametric_f", "f"),
    },
)

xls_dslx_opt_ir_test(
    name = "test_f_calls_parametric_f_test",
    dep = ":test_f_calls_parametric_f",
)

xls_dslx_test(
    name = "array_concat_dslx_test",
    srcs = ["array_concat.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir(
    name = "array_concat",
    srcs = ["array_concat.x"],
)

xls_dslx_opt_ir_test(
    name = "array_concat_test",
    dep = ":array_concat",
)

xls_dslx_test(
    name = "for_up_to_const_dslx_test",
    srcs = ["for_up_to_const.x"],
    # TODO(rspringer): Cannot run bytyecode due to lack of "for" node support.
)

xls_dslx_opt_ir(
    name = "for_up_to_const",
    srcs = ["for_up_to_const.x"],
    opt_ir_args = {
        "entry": get_mangled_ir_symbol("for_up_to_const", "f"),
    },
)

xls_dslx_opt_ir_test(
    name = "for_up_to_const_test",
    dep = ":for_up_to_const",
)

xls_dslx_test(
    name = "for_up_to_const_local_dslx_test",
    srcs = ["for_up_to_const_local.x"],
    # TODO(rspringer): Cannot run bytyecode due to lack of "for" node support.
)

xls_dslx_opt_ir(
    name = "for_up_to_const_local",
    srcs = ["for_up_to_const_local.x"],
)

xls_dslx_opt_ir_test(
    name = "for_up_to_const_local_test",
    dep = ":for_up_to_const_local",
)

xls_dslx_test(
    name = "map_of_stdlib_parametric_dslx_test",
    srcs = ["map_of_stdlib_parametric.x"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of
    # parametrics support.
)

xls_dslx_opt_ir(
    name = "map_of_stdlib_parametric",
    srcs = ["map_of_stdlib_parametric.x"],
)

xls_dslx_opt_ir_test(
    name = "map_of_stdlib_parametric_test",
    dep = ":map_of_stdlib_parametric",
)

xls_dslx_library(
    name = "mod_imported_array_dslx",
    srcs = ["mod_imported_array.x"],
)

xls_dslx_test(
    name = "map_multi_module_dslx_test",
    srcs = ["map_multi_module.x"],
    deps = [
        ":mod_imported_array_dslx",
        ":mod_imported_lsb_dslx",
    ],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of
    # ColonRef invocation support.
)

xls_dslx_opt_ir(
    name = "map_multi_module",
    srcs = ["map_multi_module.x"],
    deps = [
        ":mod_imported_array_dslx",
        ":mod_imported_lsb_dslx",
    ],
)

xls_dslx_opt_ir_test(
    name = "map_multi_module_test",
    dep = ":map_multi_module",
)

xls_dslx_opt_ir(
    name = "parametric_const_ref",
    srcs = ["parametric_const_ref.x"],
)

# Only a parametric entry point, so no benchmark.
xls_dslx_test(
    name = "parametric_const_ref_dslx_test",
    srcs = ["parametric_const_ref.x"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of
    # parametrics support.
)

xls_dslx_test(
    name = "array_with_binary_dim_dslx_test",
    srcs = ["array_with_binary_dim.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir(
    name = "array_with_binary_dim",
    srcs = ["array_with_binary_dim.x"],
)

xls_dslx_opt_ir_test(
    name = "array_with_binary_dim_test",
    dep = ":array_with_binary_dim",
)

xls_dslx_library(
    name = "mod_simple_enum_dslx",
    srcs = ["mod_simple_enum.x"],
)

xls_dslx_test(
    name = "mod_simple_enum_importer_dslx_test",
    srcs = ["mod_simple_enum_importer.x"],
    deps = [":mod_simple_enum_dslx"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of Match
    # node support.
)

xls_dslx_opt_ir(
    name = "mod_simple_enum_importer",
    srcs = ["mod_simple_enum_importer.x"],
    deps = [":mod_simple_enum_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_simple_enum_importer_test",
    dep = ":mod_simple_enum_importer",
)

xls_dslx_test(
    name = "mod_enum_use_in_for_match_importer_dslx_test",
    srcs = ["mod_enum_use_in_for_match_importer.x"],
    deps = [":mod_simple_enum_dslx"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of Match
    # node support.
)

xls_dslx_opt_ir(
    name = "mod_enum_use_in_for_match_importer",
    srcs = ["mod_enum_use_in_for_match_importer.x"],
    deps = [":mod_simple_enum_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_enum_use_in_for_match_importer_test",
    dep = ":mod_enum_use_in_for_match_importer",
)

xls_dslx_opt_ir(
    name = "mod_enum_fully_qualified_match_importer",
    srcs = ["mod_enum_fully_qualified_match_importer.x"],
    ir_file = "mod_enum_fully_qualified_match_importer.ir",
    deps = [":mod_simple_enum_dslx"],
)

# TODO(rspringer): 2021-06-29 We can't lower fail!/assert to Z3, so we can't do
# opt/unopt equivalence testing yet.
xls_dslx_test(
    name = "mod_enum_fully_qualified_match_importer_dslx_test",
    srcs = ["mod_enum_fully_qualified_match_importer.x"],
    deps = [":mod_simple_enum_dslx"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of Match
    # node support.
)

xls_eval_ir_test(
    name = "mod_enum_fully_qualified_match_importer_eval_ir_test",
    src = ":mod_enum_fully_qualified_match_importer.ir",
)

xls_benchmark_ir(
    name = "mod_enum_fully_qualified_match_importer_benchmark_ir",
    src = ":mod_enum_fully_qualified_match_importer.ir",
)

xls_dslx_library(
    name = "mod_simple_const_dslx",
    srcs = ["mod_simple_const.x"],
)

xls_dslx_test(
    name = "mod_const_importer_dslx_test",
    srcs = ["mod_const_importer.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
    deps = [":mod_simple_const_dslx"],
)

xls_dslx_opt_ir(
    name = "mod_const_importer",
    srcs = ["mod_const_importer.x"],
    deps = [":mod_simple_const_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_const_importer_test",
    dep = ":mod_const_importer",
)

xls_dslx_test(
    name = "importer_const_in_range_dslx_test",
    srcs = ["importer_const_in_range.x"],
    deps = [":mod_simple_const_dslx"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode due to lack of "For" node
    # support.
)

xls_dslx_opt_ir(
    name = "importer_const_in_range",
    srcs = ["importer_const_in_range.x"],
    deps = [":mod_simple_const_dslx"],
)

xls_dslx_opt_ir_test(
    name = "importer_const_in_range_test",
    dep = ":importer_const_in_range",
)

xls_dslx_test(
    name = "importer_const_in_return_type_dslx_test",
    srcs = ["importer_const_in_return_type.x"],
    deps = [":mod_simple_const_dslx"],
)

xls_dslx_opt_ir(
    name = "importer_const_in_return_type",
    srcs = ["importer_const_in_return_type.x"],
    deps = [":mod_simple_const_dslx"],
)

xls_dslx_opt_ir_test(
    name = "importer_const_in_return_type_test",
    dep = ":importer_const_in_return_type",
)

xls_dslx_test(
    name = "mod_const_in_type_alias_dslx_test",
    srcs = ["mod_const_in_type_alias.x"],
    dslx_test_args = {
        "bytecode": "True",
    },
    deps = [":mod_simple_const_dslx"],
)

xls_dslx_opt_ir(
    name = "mod_const_in_type_alias",
    srcs = ["mod_const_in_type_alias.x"],
    deps = [":mod_simple_const_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_const_in_type_alias_test",
    dep = ":mod_const_in_type_alias",
)

xls_dslx_test(
    name = "mod_const_in_type_alias_expr_dslx_test",
    srcs = ["mod_const_in_type_alias_expr.x"],
    dslx_test_args = {
        "bytecode": "True",
    },
    deps = [":mod_simple_const_dslx"],
)

xls_dslx_opt_ir(
    name = "mod_const_in_type_alias_expr",
    srcs = ["mod_const_in_type_alias_expr.x"],
    deps = [":mod_simple_const_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_const_in_type_alias_expr_test",
    dep = ":mod_const_in_type_alias_expr",
)

xls_dslx_library(
    name = "mod_simple_const_enum_dslx",
    srcs = ["mod_simple_const_enum.x"],
)

xls_dslx_test(
    name = "mod_const_enum_importer_dslx_test",
    srcs = ["mod_const_enum_importer.x"],
    dslx_test_args = {
        "bytecode": "True",
    },
    deps = [":mod_simple_const_enum_dslx"],
)

xls_dslx_opt_ir(
    name = "mod_const_enum_importer",
    srcs = ["mod_const_enum_importer.x"],
    deps = [":mod_simple_const_enum_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_const_enum_importer_test",
    dep = ":mod_const_enum_importer",
)

xls_dslx_test(
    name = "mod_simple_enum_alias_importer_dslx_test",
    srcs = ["mod_simple_enum_alias_importer.x"],
    deps = [":mod_simple_enum_dslx"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode due to lack of "Match"
    # node support.
)

xls_dslx_opt_ir(
    name = "mod_simple_enum_alias_importer",
    srcs = ["mod_simple_enum_alias_importer.x"],
    deps = [":mod_simple_enum_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_simple_enum_alias_importer_test",
    dep = ":mod_simple_enum_alias_importer",
)

xls_dslx_library(
    name = "mod_simple_const_array_of_enums_dslx",
    srcs = ["mod_simple_const_array_of_enums.x"],
)

xls_dslx_test(
    name = "mod_const_array_of_enum_importer_dslx_test",
    srcs = ["mod_const_array_of_enum_importer.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
    deps = [":mod_simple_const_array_of_enums_dslx"],
)

xls_dslx_opt_ir(
    name = "mod_const_array_of_enum_importer",
    srcs = ["mod_const_array_of_enum_importer.x"],
    deps = [":mod_simple_const_array_of_enums_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_const_array_of_enum_importer_test",
    dep = ":mod_const_array_of_enum_importer",
)

xls_dslx_test(
    name = "simple_enum_usage_dslx_test",
    srcs = ["simple_enum_usage.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir(
    name = "simple_enum_usage",
    srcs = ["simple_enum_usage.x"],
)

xls_dslx_opt_ir_test(
    name = "simple_enum_usage_test",
    dep = ":simple_enum_usage",
)

xls_dslx_test(
    name = "for_rotate_dslx_test",
    srcs = ["for_rotate.x"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of "For"
    # node handling and ColonRef invocation support.
)

xls_dslx_opt_ir(
    name = "for_rotate",
    srcs = ["for_rotate.x"],
)

xls_dslx_opt_ir_test(
    name = "for_rotate_test",
    dep = ":for_rotate",
)

xls_dslx_test(
    name = "for_enum_ref_dslx_test",
    srcs = ["for_enum_ref.x"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of "For"
    # node handling.
)

xls_dslx_opt_ir(
    name = "for_enum_ref",
    srcs = ["for_enum_ref.x"],
)

xls_dslx_opt_ir_test(
    name = "for_enum_ref_test",
    dep = ":for_enum_ref",
)

xls_dslx_opt_ir(
    name = "update_within_parametric",
    srcs = ["update_within_parametric.x"],
)

# Only a parametric entry point, so no benchmark.
xls_dslx_test(
    name = "update_within_parametric_dslx_test",
    srcs = ["update_within_parametric.x"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of
    # parametrics support.
)

# -- width_slice_to_parametric

xls_dslx_test(
    name = "width_slice_to_parametric_dslx_test",
    srcs = ["width_slice_to_parametric.x"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of
    # parametrics support.
)

xls_dslx_opt_ir(
    name = "width_slice_to_parametric",
    srcs = ["width_slice_to_parametric.x"],
)

xls_dslx_opt_ir_test(
    name = "width_slice_to_parametric_test",
    dep = ":width_slice_to_parametric",
)

# -- mask_bits

xls_dslx_opt_ir(
    name = "mask_bits",
    srcs = ["mask_bits.x"],
)

# Only a parametric entry point, so no benchmark.
xls_dslx_test(
    name = "mask_bits_dslx_test",
    srcs = ["mask_bits.x"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of
    # parametrics support.
)

# -- function_called_in_derived_parametric_simple

xls_dslx_test(
    name = "function_called_in_derived_parametric_simple_dslx_test",
    srcs = ["function_called_in_derived_parametric_simple.x"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of
    # parametrics support.
)

xls_dslx_opt_ir(
    name = "function_called_in_derived_parametric_simple",
    srcs = ["function_called_in_derived_parametric_simple.x"],
)

xls_dslx_opt_ir_test(
    name = "function_called_in_derived_parametric_simple_test",
    dep = ":function_called_in_derived_parametric_simple",
)

# -- parametric_enum_value

xls_dslx_test(
    name = "parametric_enum_value_dslx_test",
    srcs = ["parametric_enum_value.x"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of
    # parametrics support.
)

xls_dslx_opt_ir(
    name = "parametric_enum_value",
    srcs = ["parametric_enum_value.x"],
)

xls_dslx_opt_ir_test(
    name = "parametric_enum_value_test",
    dep = ":parametric_enum_value",
)

# -- non_parametric_called_solely_from_parametric

xls_dslx_test(
    name = "non_parametric_called_solely_from_parametric_dslx_test",
    srcs = ["non_parametric_called_solely_from_parametric.x"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of
    # parametrics support.
)

xls_dslx_opt_ir(
    name = "non_parametric_called_solely_from_parametric",
    srcs = ["non_parametric_called_solely_from_parametric.x"],
)

xls_dslx_opt_ir_test(
    name = "non_parametric_called_solely_from_parametric_test",
    dep = ":non_parametric_called_solely_from_parametric",
)

xls_dslx_opt_ir(
    name = "function_called_in_derived_parametric_struct_simple",
    srcs = ["function_called_in_derived_parametric_struct_simple.x"],
)

xls_dslx_opt_ir_test(
    name = "function_called_in_derived_parametric_struct_simple_test",
    dep = ":function_called_in_derived_parametric_struct_simple",
)

# -- identifiers_with_ticks

xls_dslx_test(
    name = "identifiers_with_ticks_dslx_test",
    srcs = ["identifiers_with_ticks.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir(
    name = "identifiers_with_ticks",
    srcs = ["identifiers_with_ticks.x"],
)

xls_dslx_opt_ir_test(
    name = "identifiers_with_ticks_test",
    dep = ":identifiers_with_ticks",
)

# -- bitslice_with_constexpr_bounds

xls_dslx_test(
    name = "bitslice_with_constexpr_bounds_dslx_test",
    srcs = ["bitslice_with_constexpr_bounds.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir(
    name = "bitslice_with_constexpr_bounds",
    srcs = ["bitslice_with_constexpr_bounds.x"],
)

xls_dslx_opt_ir_test(
    name = "bitslice_with_constexpr_bounds_test",
    dep = ":bitslice_with_constexpr_bounds",
)

# -- bitslice_negated_constexpr_bounds

xls_dslx_test(
    name = "bitslice_negated_constexpr_bounds_dslx_test",
    srcs = ["bitslice_negated_constexpr_bounds.x"],
    dslx_test_args = {
        "bytecode": "True",
    },
)

xls_dslx_opt_ir(
    name = "bitslice_negated_constexpr_bounds",
    srcs = ["bitslice_negated_constexpr_bounds.x"],
    opt_ir_args = {
        "entry": get_mangled_ir_symbol(
            "bitslice_negated_constexpr_bounds",
            "h",
        ),
    },
)

xls_dslx_opt_ir_test(
    name = "bitslice_negated_constexpr_bounds_test",
    dep = ":bitslice_negated_constexpr_bounds",
)

# -- enum_value_from_const

xls_dslx_test(
    name = "enum_value_from_const_dslx_test",
    srcs = ["enum_value_from_const.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir(
    name = "enum_value_from_const",
    srcs = ["enum_value_from_const.x"],
)

xls_dslx_opt_ir_test(
    name = "enum_value_from_const_test",
    dep = ":enum_value_from_const",
)

# -- mod_imported_const_struct_dslx

xls_dslx_library(
    name = "mod_imported_const_struct_dslx",
    srcs = ["mod_imported_const_struct.x"],
)

xls_dslx_test(
    name = "mod_importer_access_const_struct_dslx_test",
    srcs = ["mod_importer_access_const_struct.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
    deps = [":mod_imported_const_struct_dslx"],
)

xls_dslx_opt_ir(
    name = "mod_importer_access_const_struct",
    srcs = ["mod_importer_access_const_struct.x"],
    deps = [":mod_imported_const_struct_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_importer_access_const_struct_test",
    dep = ":mod_importer_access_const_struct",
)

xls_dslx_library(
    name = "mod_imported_access_const_struct_dslx",
    srcs = ["mod_imported_access_const_struct.x"],
    deps = [":mod_imported_const_struct_dslx"],
)

xls_dslx_test(
    name = "mod_importer_access_const_struct_indirect_dslx_test",
    srcs = ["mod_importer_access_const_struct_indirect.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
    deps = [":mod_imported_access_const_struct_dslx"],
)

xls_dslx_opt_ir(
    name = "mod_importer_access_const_struct_indirect",
    srcs = ["mod_importer_access_const_struct_indirect.x"],
    deps = [":mod_imported_access_const_struct_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_importer_access_const_struct_indirect_test",
    dep = ":mod_importer_access_const_struct_indirect",
)

xls_dslx_test(
    name = "nest_alias_const_array_dslx_test",
    srcs = ["nest_alias_const_array.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir(
    name = "nest_alias_const_array",
    srcs = ["nest_alias_const_array.x"],
)

xls_dslx_opt_ir_test(
    name = "nest_alias_const_array_test",
    dep = ":nest_alias_const_array",
)

xls_dslx_test(
    name = "nparametric_used_in_for_dslx_test",
    srcs = ["parametric_used_in_for.x"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode due to lack of
    # parametrics support.
)

xls_dslx_opt_ir(
    name = "parametric_used_in_for",
    srcs = ["parametric_used_in_for.x"],
)

xls_dslx_opt_ir_test(
    name = "parametric_used_in_for_test",
    dep = ":parametric_used_in_for",
)

xls_dslx_test(
    name = "parametric_to_value_dslx_test",
    srcs = ["parametric_to_value.x"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode due to lack of
    # parametrics support.
)

xls_dslx_opt_ir(
    name = "parametric_to_value",
    srcs = ["parametric_to_value.x"],
)

xls_dslx_opt_ir_test(
    name = "parametric_to_value_test",
    dep = ":parametric_to_value",
)

xls_dslx_test(
    name = "constexpr_with_typed_stuff_after_dslx_test",
    srcs = ["constexpr_with_typed_stuff_after.x"],
    dslx_test_args = {
        "bytecode": "true",
    },
)

xls_dslx_opt_ir(
    name = "constexpr_with_typed_stuff_after",
    srcs = ["constexpr_with_typed_stuff_after.x"],
)

xls_dslx_opt_ir_test(
    name = "constexpr_with_typed_stuff_after_test",
    dep = ":constexpr_with_typed_stuff_after",
)

xls_dslx_test(
    name = "for_up_to_constexpr_cast_dslx_test",
    srcs = ["for_up_to_constexpr_cast.x"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of "For"
    # node support.
)

xls_dslx_opt_ir(
    name = "for_up_to_constexpr_cast",
    srcs = ["for_up_to_constexpr_cast.x"],
)

xls_dslx_opt_ir_test(
    name = "for_up_to_constexpr_cast_test",
    dep = ":for_up_to_constexpr_cast",
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "bits_concat_dslx_test",
    srcs = ["bits_concat.x"],
    dslx_test_args = {
        "bytecode": "true",
        "compare": "none",
    },
)

xls_dslx_test(
    name = "local_const_value_in_parametric_match_dslx_test",
    srcs = ["local_const_value_in_parametric_match.x"],
)

xls_dslx_opt_ir(
    name = "local_const_value_in_parametric_match",
    srcs = ["local_const_value_in_parametric_match.x"],
)

xls_dslx_opt_ir_test(
    name = "local_const_value_in_parametric_match_test",
    dep = ":local_const_value_in_parametric_match",
)

xls_dslx_test(
    name = "let_match_tuple_dslx_test",
    srcs = ["let_match_tuple.x"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of "Match"
    # node support.
)

xls_dslx_opt_ir(
    name = "let_match_tuple",
    srcs = ["let_match_tuple.x"],
    ir_file = "let_match_tuple.ir",
)

# TODO(https://github.com/google/xls/issues/400): This produces a zero-bit
# concate which is currently not supported by Z3 translation. Add
# xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "let_match_tuple_eval_ir_test",
    src = ":let_match_tuple.ir",
)

xls_benchmark_ir(
    name = "let_match_tuple_benchmark_ir",
    src = ":let_match_tuple.ir",
)

# -- map_parametric_with_default

xls_dslx_test(
    name = "map_parametric_with_default_dslx_test",
    srcs = ["map_parametric_with_default.x"],
    # TODO(rspringer): 2022-01-19: Can't run bytecode due to lack of
    # parametrics support.
)

# -- gate

xls_dslx_test(
    name = "gate_dslx_test",
    srcs = ["gate.x"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack support
    # for the gate! builtin.
)

# -- local_const_value_non_bitstype

xls_dslx_test(
    name = "local_const_value_non_bitstype_dslx_test",
    srcs = ["local_const_value_non_bitstype.x"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of "For"
    # node support.
)

xls_dslx_test(
    name = "send_if_recv_if_test",
    srcs = ["send_if_recv_if.x"],
    # TODO(rspringer): 2022-01-19: Cannot run bytecode due to lack of "SendIf"
    # and "RecvIf" node support.
)

xls_dslx_opt_ir(
    name = "send_if_recv_if",
    srcs = ["send_if_recv_if.x"],
    ir_conv_args = {
        "entry": "main",
    },
)
