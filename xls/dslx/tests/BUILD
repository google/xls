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
    "xls_dslx_ir",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_opt_ir_test",
    "xls_dslx_test",
    "xls_ir_opt_ir",
)
load("//xls/dslx/tests:build_defs.bzl", "dslx_lang_test")
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

bzl_library(
    name = "build_defs",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        "//xls/build_rules:xls_build_defs_bzl",
    ],
)

dslx_lang_test(name = "tuple_with_array_member")

dslx_lang_test(
    name = "array_of_imported_struct",
    dslx_deps = [":mod_imported_dslx"],
)

dslx_lang_test(
    name = "binops",
    test_ir_equivalence = False,
)

dslx_lang_test(
    name = "comparisons",
    convert_to_ir = False,
)

dslx_lang_test(name = "constexpr")

dslx_lang_test(
    name = "constexpr_attrs",
    convert_to_ir = False,
    dslx_deps = [":constexpr_dslx"],
)

dslx_lang_test(
    name = "builtin_type_max",
    dslx_deps = [":number_of_imported_type_import_dslx"],
)

dslx_lang_test(name = "fallible_constexpr")

dslx_lang_test(name = "comments")

dslx_lang_test(
    name = "compound_eq",
    # Compound eq not supported yet in z3.
    test_ir_equivalence = False,
)

dslx_lang_test(name = "constexpr_clog2")

dslx_lang_test(
    name = "constexpr_imported_clog2",
    dslx_deps = [":constexpr_dslx"],
)

dslx_lang_test(name = "constexpr_types_via_invocations")

dslx_lang_test(name = "constexpr_types_via_invocations_inline")

dslx_lang_test(name = "constexpr_types_via_stdlib_invocation")

dslx_lang_test(name = "constexpr_slice")

dslx_lang_test(
    name = "explicit_parametric",
    dslx_entry = "instantiates_indirect_foo",
)

dslx_lang_test(
    name = "explicit_parametric_simple",
    # No meaningful entry point to convert.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "explicit_parametric_reduced",
    # No meaningful entry point to convert.
    convert_to_ir = False,
)

dslx_lang_test(name = "parametric_dot_product")

dslx_lang_test(name = "string_conversion")

dslx_lang_test(name = "array_concat_str")

dslx_lang_test(name = "array_slice_str")

dslx_lang_test(name = "array_update_str")

dslx_lang_test(
    name = "import_constexpr",
    dslx_deps = [":constexpr_dslx"],
)

dslx_lang_test(name = "map")

dslx_lang_test(name = "multiplies")

dslx_lang_test(name = "match_sample")

dslx_lang_test(name = "match_const")

dslx_lang_test(name = "match_multi")

xls_dslx_library(
    name = "number_of_imported_type_import_dslx",
    srcs = ["number_of_imported_type_import.x"],
)

dslx_lang_test(
    name = "number_of_imported_type",
    dslx_deps = [":number_of_imported_type_import_dslx"],
)

# Note: library defined for importing.
xls_dslx_library(
    name = "mod_struct_point_dslx",
    srcs = ["mod_struct_point.x"],
)

dslx_lang_test(
    name = "type_aliasing_importer",
    dslx_deps = [":mod_struct_point_dslx"],
)

dslx_lang_test(
    name = "parametric_call_parametric",
    # No meaningful entry function to convert.
    convert_to_ir = False,
)

xls_dslx_library(
    name = "parametric_import_dslx",
    srcs = ["parametric_import.x"],
)

dslx_lang_test(
    name = "parametric_importer",
    # No meaningful entry function to convert.
    convert_to_ir = False,
    dslx_deps = [":parametric_import_dslx"],
)

dslx_lang_test(
    name = "signed_comparisons",
    # No meaningful entry function to convert.
    convert_to_ir = False,
)

dslx_lang_test(name = "struct_as_parametric")

dslx_lang_test(name = "subtract_to_negative")

dslx_lang_test(name = "trace")

# Note: This test is enough to keep the example code compiling,
# but does not currently check the output.
# TODO(https://github.com/google/xls/issues/489): 2021-09-02
# Check the trace output of this test
xls_dslx_test(
    name = "trace_fmt_hello_test",
    srcs = ["trace_fmt_hello.x"],
    dslx_test_args = {"compare": "interpreter"},
)

dslx_lang_test(name = "array_of_typedef_in_tuple")

dslx_lang_test(
    name = "array_of_tuple",
    # No meaningful function to convert, all tests.
    convert_to_ir = False,
)

dslx_lang_test(name = "array_slice")

dslx_lang_test(name = "array_update")

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

dslx_lang_test(
    name = "parametric_value_as_nested_loop_bound",
)

dslx_lang_test(
    name = "derived_parametric_value_as_nested_loop_bound",
)

# Note: no meaningful function to convert to IR.
xls_dslx_test(
    name = "bit_slice_dslx_test",
    srcs = ["bit_slice.x"],
    dslx_test_args = {
        "compare": "none",
    },
)

dslx_lang_test(name = "bit_slice_update")

dslx_lang_test(name = "const_slice_limit")

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "bit_slice_syntax_dslx_test",
    srcs = ["bit_slice_syntax.x"],
    dslx_test_args = {
        "compare": "none",
    },
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "tuple_indexing_dslx_test",
    srcs = ["tuple_indexing.x"],
    dslx_test_args = {
        "compare": "none",
    },
)

dslx_lang_test(name = "clz")

dslx_lang_test(name = "ctz")

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "one_hot_dslx_test",
    srcs = ["one_hot.x"],
    dslx_test_args = {
        "compare": "none",
    },
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "one_hot_sel",
    srcs = ["one_hot_sel.x"],
    dslx_test_args = {
        "compare": "none",
    },
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "priority_sel",
    srcs = ["priority_sel.x"],
)

dslx_lang_test(name = "reductions")

dslx_lang_test(name = "signed_number_type")

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "numerical_conversions_dslx_test",
    srcs = ["numerical_conversions.x"],
    dslx_test_args = {
        "compare": "none",
    },
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "numerical_conversions2_dslx_test",
    srcs = ["numerical_conversions2.x"],
    dslx_test_args = {
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
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "enum_values_dslx_test",
    srcs = ["enum_values.x"],
    dslx_test_args = {
        "compare": "none",
    },
)

# TODO(leary); 2019-10-24 Enable when we have ConcatArray op.
xls_dslx_test(
    name = "casts_dslx_test",
    srcs = ["casts.x"],
    dslx_test_args = {
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
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "parametric_smul_dslx_test",
    srcs = ["parametric_smul.x"],
    dslx_test_args = {
        "compare": "none",
    },
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "mulp_test",
    srcs = ["mulp.x"],
)

dslx_lang_test(name = "basic_struct")

dslx_lang_test(name = "basic_struct_parameter")

dslx_lang_test(name = "basic_struct_attr")

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "struct_equality_dslx_test",
    srcs = ["struct_equality.x"],
    dslx_test_args = {
        "compare": "none",
    },
)

dslx_lang_test(name = "basic_struct_update")

dslx_lang_test(name = "struct_splat_update")

dslx_lang_test(name = "struct_splat_update_to_const")

dslx_lang_test(
    name = "empty_struct_splat_update",
    warnings_as_errors = False,
)

dslx_lang_test(name = "basic_struct_array")

dslx_lang_test(name = "basic_enum_array")

dslx_lang_test(name = "basic_array")

dslx_lang_test(name = "std_lsb")

dslx_lang_test(name = "std_alias")

filegroup(
    name = "mod_imported_file",
    srcs = ["mod_imported.x"],
)

# Note: library defined for importing.
xls_dslx_library(
    name = "mod_imported_dslx",
    srcs = ["mod_imported.x"],
)

dslx_lang_test(
    name = "mod_importer",
    dslx_deps = [":mod_imported_dslx"],
)

xls_dslx_library(
    name = "mod_imported_typedef_dslx",
    srcs = ["mod_imported_typedef.x"],
)

dslx_lang_test(
    name = "mod_importer_typedef",
    dslx_deps = [":mod_imported_typedef_dslx"],
)

# Library defined to be imported.
xls_dslx_library(
    name = "mod_imported_struct_of_enum_dslx",
    srcs = [":mod_imported_struct_of_enum.x"],
)

dslx_lang_test(
    name = "mod_importer_struct_of_enum",
    dslx_deps = [":mod_imported_struct_of_enum_dslx"],
)

dslx_lang_test(
    name = "mod_importer_simple",
    dslx_deps = [":mod_imported_dslx"],
)

dslx_lang_test(
    name = "mod_struct_importer",
    dslx_deps = [":mod_imported_dslx"],
)

dslx_lang_test(
    name = "mod_enum_importer",
    dslx_deps = [":mod_imported_dslx"],
)

xls_dslx_library(
    name = "mod_parametric_id_dslx",
    srcs = [
        "mod_parametric_id.x",
        "mod_parametric_id_indirect.x",
    ],
)

dslx_lang_test(
    name = "mod_parametric_id_user",
    dslx_deps = [":mod_parametric_id_dslx"],
)

xls_dslx_opt_ir(
    name = "mod_parametric_id_user",
    srcs = ["mod_parametric_id_user.x"],
    dslx_top = "main",
    deps = [":mod_parametric_id_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_parametric_id_user_test",
    dep = ":mod_parametric_id_user",
)

# Library for use in importing a parametric identity function.
xls_dslx_library(
    name = "mod_parametric_id_with_default",
    srcs = ["mod_parametric_id_with_default.x"],
)

dslx_lang_test(
    name = "parametric_id_with_default_user",
    dslx_deps = [":mod_parametric_id_with_default"],
)

xls_dslx_library(
    name = "mod_imported_lsb_dslx",
    srcs = ["mod_imported_lsb.x"],
)

dslx_lang_test(
    name = "call_imported_fn_simple",
    dslx_deps = [":mod_imported_lsb_dslx"],
)

xls_dslx_library(
    name = "mod_imported_aliases_dslx",
    srcs = ["mod_imported_aliases.x"],
)

dslx_lang_test(
    name = "mod_typedef_importer",
    dslx_deps = [":mod_imported_aliases_dslx"],
)

dslx_lang_test(name = "array_ellipsis")

dslx_lang_test(name = "local_const_value")

dslx_lang_test(name = "const_value_as_nested_loop_bound")

dslx_lang_test(name = "local_const_value_in_parametric")

dslx_lang_test(name = "constant_array_unannotated_type")

dslx_lang_test(name = "reverse")

dslx_lang_test(name = "typedef_multidim_array")

dslx_lang_test(
    name = "parametric_multidim_array",
    dslx_entry = "id_6_3_2",
)

dslx_lang_test(name = "for_nonzero_start")

dslx_lang_test(name = "for_sans_type_annotation")

dslx_lang_test(name = "derived_parametrics")

dslx_lang_test(name = "parametric_struct")

dslx_lang_test(
    name = "parametric_struct_simple",
    # No meaningful entry point to convert.
    convert_to_ir = False,
)

dslx_lang_test(name = "test_f_calls_parametric_f")

dslx_lang_test(name = "array_concat")

dslx_lang_test(name = "for_up_to_const")

dslx_lang_test(name = "for_up_to_const_local")

dslx_lang_test(name = "map_of_stdlib_parametric")

# Library meant to be imported.
xls_dslx_library(
    name = "mod_imported_array_dslx",
    srcs = ["mod_imported_array.x"],
)

dslx_lang_test(
    name = "map_multi_module",
    dslx_deps = [
        ":mod_imported_array_dslx",
        ":mod_imported_lsb_dslx",
    ],
)

dslx_lang_test(
    name = "parametric_const_ref",
    # Only a parametric entry point, so no IR conversion.
    convert_to_ir = False,
)

dslx_lang_test(name = "array_with_binary_dim")

xls_dslx_library(
    name = "mod_simple_enum_dslx",
    srcs = ["mod_simple_enum.x"],
)

dslx_lang_test(
    name = "mod_simple_enum_importer",
    dslx_deps = [":mod_simple_enum_dslx"],
)

dslx_lang_test(
    name = "mod_enum_use_in_for_match_importer",
    dslx_deps = [":mod_simple_enum_dslx"],
)

dslx_lang_test(
    name = "mod_enum_fully_qualified_match_importer",
    dslx_deps = [":mod_simple_enum_dslx"],
    # TODO(rspringer): 2021-06-29 We can't lower fail!/assert to Z3, so we
    # can't do opt/unopt equivalence testing yet.
    test_ir_equivalence = False,
)

xls_dslx_library(
    name = "mod_simple_const_dslx",
    srcs = ["mod_simple_const.x"],
)

dslx_lang_test(
    name = "mod_const_importer",
    dslx_deps = [":mod_simple_const_dslx"],
)

xls_dslx_opt_ir(
    name = "mod_const_importer",
    srcs = ["mod_const_importer.x"],
    dslx_top = "main",
    deps = [":mod_simple_const_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_const_importer_test",
    dep = ":mod_const_importer",
)

dslx_lang_test(
    name = "importer_const_in_range",
    dslx_deps = [":mod_simple_const_dslx"],
)

dslx_lang_test(
    name = "importer_const_in_return_type",
    dslx_deps = [":mod_simple_const_dslx"],
)

dslx_lang_test(
    name = "mod_const_in_type_alias",
    dslx_deps = [":mod_simple_const_dslx"],
)

dslx_lang_test(
    name = "mod_const_in_type_alias_expr",
    dslx_deps = [":mod_simple_const_dslx"],
)

xls_dslx_library(
    name = "mod_simple_const_enum_dslx",
    srcs = ["mod_simple_const_enum.x"],
)

dslx_lang_test(
    name = "mod_const_enum_importer",
    dslx_deps = [":mod_simple_const_enum_dslx"],
)

dslx_lang_test(
    name = "mod_simple_enum_alias_importer",
    dslx_deps = [":mod_simple_enum_dslx"],
)

xls_dslx_library(
    name = "mod_simple_const_array_of_enums_dslx",
    srcs = ["mod_simple_const_array_of_enums.x"],
)

dslx_lang_test(
    name = "mod_const_array_of_enum_importer",
    dslx_deps = [":mod_simple_const_array_of_enums_dslx"],
)

xls_dslx_opt_ir(
    name = "mod_const_array_of_enum_importer",
    srcs = ["mod_const_array_of_enum_importer.x"],
    dslx_top = "main",
    deps = [":mod_simple_const_array_of_enums_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_const_array_of_enum_importer_test",
    dep = ":mod_const_array_of_enum_importer",
)

dslx_lang_test(name = "simple_enum_usage")

xls_dslx_opt_ir(
    name = "simple_enum_usage",
    srcs = ["simple_enum_usage.x"],
    dslx_top = "main",
)

xls_dslx_opt_ir_test(
    name = "simple_enum_usage_test",
    dep = ":simple_enum_usage",
)

dslx_lang_test(name = "for_rotate")

dslx_lang_test(name = "for_enum_ref")

dslx_lang_test(
    name = "update_within_parametric",
    # Only a parametric entry point, so nothing meaningful to convert.
    convert_to_ir = False,
)

dslx_lang_test(name = "width_slice_to_parametric")

dslx_lang_test(
    name = "mask_bits",
    # Only a parametric entry point, so nothing meaningful to convert.
    convert_to_ir = False,
)

dslx_lang_test(name = "function_called_in_derived_parametric_simple")

dslx_lang_test(name = "parametric_enum_value")

dslx_lang_test(name = "non_parametric_called_solely_from_parametric")

dslx_lang_test(name = "function_called_in_derived_parametric_struct_simple")

dslx_lang_test(name = "identifiers_with_ticks")

dslx_lang_test(name = "bitslice_with_constexpr_bounds")

dslx_lang_test(name = "bitslice_negated_constexpr_bounds")

dslx_lang_test(name = "enum_value_from_const")

# -- mod_imported_const_struct_dslx

xls_dslx_library(
    name = "mod_imported_const_struct_dslx",
    srcs = ["mod_imported_const_struct.x"],
)

dslx_lang_test(
    name = "mod_importer_access_const_struct",
    dslx_deps = [":mod_imported_const_struct_dslx"],
)

xls_dslx_library(
    name = "mod_imported_access_const_struct_dslx",
    srcs = ["mod_imported_access_const_struct.x"],
    deps = [":mod_imported_const_struct_dslx"],
)

dslx_lang_test(
    name = "mod_importer_access_const_struct_indirect",
    dslx_deps = [":mod_imported_access_const_struct_dslx"],
)

dslx_lang_test(name = "nest_alias_const_array")

dslx_lang_test(name = "parametric_used_in_for")

dslx_lang_test(name = "parametric_to_value")

dslx_lang_test(name = "constexpr_with_typed_stuff_after")

dslx_lang_test(name = "for_up_to_constexpr_cast")

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "bits_concat_dslx_test",
    srcs = ["bits_concat.x"],
    dslx_test_args = {
        "compare": "none",
    },
)

dslx_lang_test(name = "local_const_value_in_parametric_match")

dslx_lang_test(
    name = "let_match_tuple",
    # TODO(https://github.com/google/xls/issues/400): 2022-02-03 This fails on
    # the missing conversion to a Z3 representation for a 0-bit concat.
    test_ir_equivalence = False,
)

dslx_lang_test(
    name = "map_parametric_with_default",
    # TODO(https://github.com/google/xls/issues/400): Unable to convert to Z3
    # due to zero-bit value.
    test_ir_equivalence = False,
)

dslx_lang_test(
    name = "gate",
    # TODO(https://github.com/google/xls/issues/566): 2022-02-03 gate
    # conversion to Z3 not implemented.
    test_ir_equivalence = False,
)

dslx_lang_test(name = "local_const_value_non_bitstype")

dslx_lang_test(
    name = "proc_channel_arrays",
    # No normal 'function' entry point, it is a test proc.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "proc_network",
    # No normal 'function' entry point, it is a test proc.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "proc_smoke",
    # No normal 'function' entry point, it is a test proc.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "proc_two_level",
    # No normal 'function' entry point, it is a test proc.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "send_if_recv_if",
    convert_to_ir = False,
    dslx_entry = "main",
)

xls_dslx_ir(
    name = "send_if_recv_if_ir",
    dslx_top = "main",
    library = "send_if_recv_if_dslx",
)

dslx_lang_test(
    name = "non_blocking_receives",
    convert_to_ir = False,
    dslx_entry = "proc_main",
)

xls_dslx_ir(
    name = "non_blocking_receives_ir",
    dslx_top = "proc_main",
    library = "non_blocking_receives_dslx",
)

xls_ir_opt_ir(
    name = "non_blocking_receives_opt_ir",
    src = ":non_blocking_receives_ir.ir",
    top = "__non_blocking_receives__proc_main_0_next",
)

dslx_lang_test(
    name = "non_blocking_receive_if",
    convert_to_ir = False,
    dslx_entry = "Tester",
)

xls_dslx_ir(
    name = "non_blocking_receive_if_ir",
    dslx_top = "Main",
    library = "non_blocking_receive_if_dslx",
)

xls_ir_opt_ir(
    name = "non_blocking_receive_if_opt_ir",
    src = ":non_blocking_receive_if_ir.ir",
    top = "__non_blocking_receive_if__Main_0_next",
)

exports_files([
    "mod_const_enum_importer.x",
    "mod_simple_const_enum.x",
])
