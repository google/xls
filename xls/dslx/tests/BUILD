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

xls_dslx_opt_ir(
    name = "tuple_with_array_member",
    src = "tuple_with_array_member.x",
)

xls_dslx_opt_ir_test(
    name = "tuple_with_array_member_test",
    dep = ":tuple_with_array_member",
)

xls_dslx_opt_ir(
    name = "binops",
    src = "binops.x",
    ir_file = "binops.ir",
)

# TODO: 2021-05-28 Add xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "binops_eval_ir_test",
    src = "binops.ir",
)

xls_benchmark_ir(
    name = "binops_benchmark_ir",
    src = "binops.ir",
)

xls_dslx_opt_ir(
    name = "comparisons",
    src = "comparisons.x",
)

xls_dslx_test(
    name = "comparisons_test",
    src = "comparisons.x",
)

xls_dslx_opt_ir(
    name = "constexpr",
    src = "constexpr.x",
)

xls_dslx_opt_ir_test(
    name = "constexpr_test",
    dep = ":constexpr",
)

xls_dslx_library(
    name = "constexpr_dslx",
    srcs = ["constexpr.x"],
)

xls_dslx_opt_ir(
    name = "constexpr_attrs",
    src = "constexpr_attrs.x",
    deps = [":constexpr_dslx"],
)

xls_dslx_test(
    name = "constexpr_attrs_test",
    src = "constexpr_attrs.x",
    deps = [":constexpr_dslx"],
)

xls_dslx_opt_ir(
    name = "compound_eq",
    src = "compound_eq.x",
)

xls_dslx_test(
    name = "compound_eq_test",
    src = "compound_eq.x",
)

xls_dslx_opt_ir(
    name = "constexpr_clog2",
    src = "constexpr_clog2.x",
)

xls_dslx_opt_ir_test(
    name = "constexpr_clog2_test",
    dep = ":constexpr_clog2",
)

xls_dslx_opt_ir(
    name = "constexpr_imported_clog2",
    src = "constexpr_imported_clog2.x",
    deps = [":constexpr_dslx"],
)

xls_dslx_opt_ir_test(
    name = "constexpr_imported_clog2_test",
    dep = ":constexpr_imported_clog2",
)

xls_dslx_opt_ir(
    name = "constexpr_types_via_invocations",
    src = "constexpr_types_via_invocations.x",
)

xls_dslx_opt_ir_test(
    name = "constexpr_types_via_invocations_test",
    dep = ":constexpr_types_via_invocations",
)

xls_dslx_opt_ir(
    name = "constexpr_types_via_invocations_inline",
    src = "constexpr_types_via_invocations_inline.x",
)

xls_dslx_opt_ir_test(
    name = "constexpr_types_via_invocations_inline_test",
    dep = ":constexpr_types_via_invocations_inline",
)

xls_dslx_opt_ir(
    name = "constexpr_types_via_stdlib_invocation",
    src = "constexpr_types_via_stdlib_invocation.x",
)

xls_dslx_opt_ir_test(
    name = "constexpr_types_via_stdlib_invocation_test",
    dep = ":constexpr_types_via_stdlib_invocation",
)

xls_dslx_opt_ir(
    name = "constexpr_slice",
    src = "constexpr_slice.x",
)

xls_dslx_test(
    name = "constexpr_slice_test",
    src = "constexpr_slice.x",
)

xls_dslx_opt_ir(
    name = "explicit_parametric",
    src = "explicit_parametric.x",
)

xls_dslx_test(
    name = "explicit_parametric_test",
    src = "explicit_parametric.x",
)

xls_dslx_opt_ir(
    name = "explicit_parametric_simple",
    src = "explicit_parametric_simple.x",
)

xls_dslx_test(
    name = "explicit_parametric_simple_test",
    src = "explicit_parametric_simple.x",
)

xls_dslx_opt_ir(
    name = "explicit_parametric_reduced",
    src = "explicit_parametric_reduced.x",
)

xls_dslx_test(
    name = "explicit_parametric_reduced_test",
    src = "explicit_parametric_reduced.x",
)

xls_dslx_opt_ir(
    name = "parametric_dot_product",
    src = "parametric_dot_product.x",
)

xls_dslx_opt_ir_test(
    name = "parametric_dot_product_test",
    dep = ":parametric_dot_product",
)

xls_dslx_opt_ir(
    name = "string_conversion",
    src = "string_conversion.x",
)

xls_dslx_opt_ir_test(
    name = "string_conversion_test",
    dep = ":string_conversion",
)

xls_dslx_opt_ir(
    name = "import_constexpr",
    src = "import_constexpr.x",
    deps = ["constexpr_dslx"],
)

xls_dslx_opt_ir_test(
    name = "import_constexpr_test",
    dep = ":import_constexpr",
)

xls_dslx_opt_ir(
    name = "map",
    src = "map.x",
)

xls_dslx_opt_ir_test(
    name = "map_test",
    dep = ":map",
)

xls_dslx_opt_ir(
    name = "multiplies",
    src = "multiplies.x",
)

xls_dslx_opt_ir_test(
    name = "multiplies_test",
    dep = ":multiplies",
)

xls_dslx_opt_ir(
    name = "match_sample",
    src = "match_sample.x",
)

xls_dslx_opt_ir_test(
    name = "match_sample_test",
    dep = ":match_sample",
)

xls_dslx_opt_ir(
    name = "match_const",
    src = "match_const.x",
)

xls_dslx_opt_ir_test(
    name = "match_const_test",
    dep = ":match_const",
)

xls_dslx_opt_ir(
    name = "match_multi",
    src = "match_multi.x",
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
    src = "number_of_imported_type.x",
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
    src = "type_aliasing_importer.x",
    deps = ["mod_struct_point_dslx"],
)

xls_dslx_test(
    name = "type_aliasing_importer_test",
    src = "type_aliasing_importer.x",
    deps = ["mod_struct_point_dslx"],
)

xls_dslx_opt_ir(
    name = "parametric_call_parametric",
    src = "parametric_call_parametric.x",
)

xls_dslx_test(
    name = "parametric_call_parametric_test",
    src = "parametric_call_parametric.x",
)

xls_dslx_library(
    name = "parametric_import_dslx",
    srcs = ["parametric_import.x"],
)

xls_dslx_opt_ir(
    name = "parametric_importer",
    src = "parametric_importer.x",
    deps = ["parametric_import_dslx"],
)

xls_dslx_test(
    name = "parametric_importer_test",
    src = "parametric_importer.x",
    deps = ["parametric_import_dslx"],
)

xls_dslx_opt_ir(
    name = "signed_comparisons",
    src = "signed_comparisons.x",
)

xls_dslx_test(
    name = "signed_comparisons_test",
    src = "signed_comparisons.x",
)

xls_dslx_opt_ir(
    name = "subtract_to_negative",
    src = "subtract_to_negative.x",
)

xls_dslx_opt_ir_test(
    name = "subtract_to_negative_test",
    dep = ":subtract_to_negative",
)

xls_dslx_opt_ir(
    name = "trace",
    src = "trace.x",
)

xls_dslx_opt_ir_test(
    name = "trace_test",
    dep = ":trace",
)

xls_dslx_opt_ir(
    name = "array_of_typedef_in_tuple",
    src = "array_of_typedef_in_tuple.x",
)

xls_dslx_opt_ir_test(
    name = "array_of_typedef_in_tuple_test",
    dep = ":array_of_typedef_in_tuple",
)

xls_dslx_opt_ir(
    name = "array_of_tuple",
    src = "array_of_tuple.x",
)

xls_dslx_test(
    name = "array_of_tuple_test",
    src = "array_of_tuple.x",
)

xls_dslx_opt_ir(
    name = "array_slice",
    src = "array_slice.x",
)

xls_dslx_test(
    name = "array_slice_test",
    src = "array_slice.x",
)

# TODO(leary): 2019-07-24 Missing support for array of tuples eliding cast node.
xls_dslx_test(
    name = "parametric_array_of_tuples",
    src = "parametric_array_of_tuples.x",
    dslx_test_args = {
        "compare": "none",
    },
)

xls_dslx_test(
    name = "parametric_functions",
    src = "parametric_functions.x",
    dslx_test_args = {
        "compare": "none",
    },
)

xls_dslx_opt_ir(
    name = "parametric_value_as_nested_loop_bound",
    src = "parametric_value_as_nested_loop_bound.x",
)

xls_dslx_opt_ir_test(
    name = "parametric_value_as_nested_loop_bound_test",
    dep = ":parametric_value_as_nested_loop_bound",
)

xls_dslx_opt_ir(
    name = "derived_parametric_value_as_nested_loop_bound",
    src = "derived_parametric_value_as_nested_loop_bound.x",
)

xls_dslx_opt_ir_test(
    name = "derived_parametric_value_as_nested_loop_bound_test",
    dep = ":derived_parametric_value_as_nested_loop_bound",
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "bit_slice",
    src = "bit_slice.x",
    dslx_test_args = {
        "compare": "none",
    },
)

xls_dslx_opt_ir(
    name = "bit_slice_update",
    src = "bit_slice_update.x",
)

xls_dslx_opt_ir_test(
    name = "bit_slice_update_test",
    dep = ":bit_slice_update",
)

xls_dslx_opt_ir(
    name = "const_slice_limit",
    src = "const_slice_limit.x",
)

xls_dslx_opt_ir_test(
    name = "const_slice_limit_test",
    dep = ":const_slice_limit",
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "bit_slice_syntax",
    src = "bit_slice_syntax.x",
    dslx_test_args = {
        "compare": "none",
    },
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "tuple_indexing",
    src = "tuple_indexing.x",
    dslx_test_args = {
        "compare": "none",
    },
)

xls_dslx_opt_ir(
    name = "clz",
    src = "clz.x",
)

xls_dslx_opt_ir_test(
    name = "clz_test",
    dep = ":clz",
)

xls_dslx_opt_ir(
    name = "ctz",
    src = "ctz.x",
)

xls_dslx_opt_ir_test(
    name = "ctz_test",
    dep = ":ctz",
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "one_hot",
    src = "one_hot.x",
    dslx_test_args = {
        "compare": "none",
    },
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "one_hot_sel",
    src = "one_hot_sel.x",
    dslx_test_args = {
        "compare": "none",
    },
)

xls_dslx_opt_ir(
    name = "reductions",
    src = "reductions.x",
    ir_file = "reductions.ir",
)

# TODO: 2021-05-28 Add xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "reductions_eval_ir_test",
    src = "reductions.ir",
)

xls_benchmark_ir(
    name = "reductions_benchmark_ir",
    src = "reductions.ir",
)

xls_dslx_opt_ir(
    name = "signed_number_type",
    src = "signed_number_type.x",
)

xls_dslx_opt_ir_test(
    name = "signed_number_type_test",
    dep = ":signed_number_type",
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "numerical_conversions",
    src = "numerical_conversions.x",
    dslx_test_args = {
        "compare": "none",
    },
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "numerical_conversions2",
    src = "numerical_conversions2.x",
    dslx_test_args = {
        "compare": "none",
    },
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "signex_builtin",
    src = "signex_builtin.x",
    dslx_test_args = {
        "compare": "none",
    },
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "enum_values",
    src = "enum_values.x",
    dslx_test_args = {
        "compare": "none",
    },
)

# TODO(leary); 2019-10-24 Enable when we have ConcatArray op.
xls_dslx_test(
    name = "casts",
    src = "casts.x",
    dslx_test_args = {
        "compare": "none",
    },
)

# TODO(leary); 2019-10-24 Enable when we have ConcatArray op.
xls_dslx_test(
    name = "cast_to_array",
    src = "cast_to_array.x",
    dslx_test_args = {
        "compare": "none",
    },
)

xls_dslx_test(
    name = "parametric_with_comparison",
    src = "parametric_with_comparison.x",
    dslx_test_args = {
        "compare": "none",
    },
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "parametric_smul",
    src = "parametric_smul.x",
    dslx_test_args = {
        "compare": "none",
    },
)

xls_dslx_opt_ir(
    name = "basic_struct",
    src = "basic_struct.x",
)

xls_dslx_opt_ir_test(
    name = "basic_struct_test",
    dep = ":basic_struct",
)

xls_dslx_opt_ir(
    name = "basic_struct_parameter",
    src = "basic_struct_parameter.x",
)

xls_dslx_opt_ir_test(
    name = "basic_struct_parameter_test",
    dep = ":basic_struct_parameter",
)

xls_dslx_opt_ir(
    name = "basic_struct_attr",
    src = "basic_struct_attr.x",
)

xls_dslx_opt_ir_test(
    name = "basic_struct_attr_test",
    dep = ":basic_struct_attr",
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "struct_equality",
    src = "struct_equality.x",
    dslx_test_args = {
        "compare": "none",
    },
)

xls_dslx_opt_ir(
    name = "basic_struct_update",
    src = "basic_struct_update.x",
)

xls_dslx_opt_ir_test(
    name = "basic_struct_update_test",
    dep = ":basic_struct_update",
)

xls_dslx_opt_ir(
    name = "struct_splat_update",
    src = "struct_splat_update.x",
)

xls_dslx_opt_ir_test(
    name = "struct_splat_update_test",
    dep = ":struct_splat_update",
)

xls_dslx_opt_ir(
    name = "struct_splat_update_to_const",
    src = "struct_splat_update_to_const.x",
)

xls_dslx_opt_ir_test(
    name = "struct_splat_update_to_const_test",
    dep = ":struct_splat_update_to_const",
)

xls_dslx_opt_ir(
    name = "empty_struct_splat_update",
    src = "empty_struct_splat_update.x",
)

xls_dslx_opt_ir_test(
    name = "empty_struct_splat_update_test",
    dep = ":empty_struct_splat_update",
)

xls_dslx_opt_ir(
    name = "basic_struct_array",
    src = "basic_struct_array.x",
)

xls_dslx_opt_ir_test(
    name = "basic_struct_array_test",
    dep = ":basic_struct_array",
)

xls_dslx_opt_ir(
    name = "basic_enum_array",
    src = "basic_enum_array.x",
)

xls_dslx_opt_ir_test(
    name = "basic_enum_array_test",
    dep = ":basic_enum_array",
)

xls_dslx_opt_ir(
    name = "basic_array",
    src = "basic_array.x",
)

xls_dslx_opt_ir_test(
    name = "basic_array_test",
    dep = ":basic_array",
)

xls_dslx_opt_ir(
    name = "std_lsb",
    src = "std_lsb.x",
)

xls_dslx_opt_ir_test(
    name = "std_lsb_test",
    dep = ":std_lsb",
)

xls_dslx_opt_ir(
    name = "std_alias",
    src = "std_alias.x",
)

xls_dslx_opt_ir_test(
    name = "std_alias_test",
    dep = ":std_alias",
)

xls_dslx_library(
    name = "mod_imported_dslx",
    srcs = ["mod_imported.x"],
)

xls_dslx_opt_ir(
    name = "mod_importer",
    src = "mod_importer.x",
    deps = ["mod_imported_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_importer_test",
    dep = ":mod_importer",
)

xls_dslx_library(
    name = "mod_imported_typedef_dslx",
    srcs = ["mod_imported_typedef.x"],
)

filegroup(
    name = "mod_imported_file",
    srcs = ["mod_imported.x"],
)

xls_dslx_opt_ir(
    name = "mod_importer_typedef",
    src = "mod_importer_typedef.x",
    deps = ["mod_imported_typedef_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_importer_typedef_test",
    dep = ":mod_importer_typedef",
)

xls_dslx_library(
    name = "mod_imported_struct_of_enum_dslx",
    srcs = ["mod_imported_struct_of_enum.x"],
)

xls_dslx_opt_ir(
    name = "mod_importer_struct_of_enum",
    src = "mod_importer_struct_of_enum.x",
    deps = ["mod_imported_struct_of_enum_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_importer_struct_of_enum_test",
    dep = ":mod_importer_struct_of_enum",
)

xls_dslx_opt_ir(
    name = "mod_importer_simple",
    src = "mod_importer_simple.x",
    deps = ["mod_imported_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_importer_simple_test",
    dep = ":mod_importer_simple",
)

xls_dslx_opt_ir(
    name = "mod_struct_importer",
    src = "mod_struct_importer.x",
    deps = ["mod_imported_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_struct_importer_test",
    dep = ":mod_struct_importer",
)

xls_dslx_opt_ir(
    name = "mod_enum_importer",
    src = "mod_enum_importer.x",
    deps = ["mod_imported_dslx"],
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

xls_dslx_opt_ir(
    name = "mod_parametric_id_user",
    src = "mod_parametric_id_user.x",
    deps = ["mod_parametric_id_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_parametric_id_user_test",
    dep = ":mod_parametric_id_user",
)

xls_dslx_library(
    name = "mod_imported_lsb_dslx",
    srcs = ["mod_imported_lsb.x"],
)

xls_dslx_opt_ir(
    name = "call_imported_fn_simple",
    src = "call_imported_fn_simple.x",
    deps = ["mod_imported_lsb_dslx"],
)

xls_dslx_opt_ir_test(
    name = "call_imported_fn_simple_test",
    dep = ":call_imported_fn_simple",
)

xls_dslx_library(
    name = "mod_imported_aliases_dslx",
    srcs = ["mod_imported_aliases.x"],
)

xls_dslx_opt_ir(
    name = "mod_typedef_importer",
    src = "mod_typedef_importer.x",
    deps = ["mod_imported_aliases_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_typedef_importer_test",
    dep = ":mod_typedef_importer",
)

xls_dslx_opt_ir(
    name = "array_ellipsis",
    src = "array_ellipsis.x",
)

xls_dslx_opt_ir_test(
    name = "array_ellipsis_test",
    dep = ":array_ellipsis",
)

xls_dslx_opt_ir(
    name = "local_const_value",
    src = "local_const_value.x",
)

xls_dslx_opt_ir_test(
    name = "local_const_value_test",
    dep = ":local_const_value",
)

xls_dslx_opt_ir(
    name = "const_value_as_nested_loop_bound",
    src = "const_value_as_nested_loop_bound.x",
)

xls_dslx_opt_ir_test(
    name = "const_value_as_nested_loop_bound_test",
    dep = ":const_value_as_nested_loop_bound",
)

xls_dslx_opt_ir(
    name = "local_const_value_in_parametric",
    src = "local_const_value_in_parametric.x",
)

xls_dslx_opt_ir_test(
    name = "local_const_value_in_parametric_test",
    dep = ":local_const_value_in_parametric",
)

xls_dslx_opt_ir(
    name = "constant_array_unannotated_type",
    src = "constant_array_unannotated_type.x",
)

xls_dslx_opt_ir_test(
    name = "constant_array_unannotated_type_test",
    dep = ":constant_array_unannotated_type",
)

xls_dslx_opt_ir(
    name = "reverse",
    src = "reverse.x",
)

xls_dslx_opt_ir_test(
    name = "reverse_test",
    dep = ":reverse",
)

xls_dslx_opt_ir(
    name = "typedef_multidim_array",
    src = "typedef_multidim_array.x",
)

xls_dslx_opt_ir_test(
    name = "typedef_multidim_array_test",
    dep = ":typedef_multidim_array",
)

xls_dslx_opt_ir(
    name = "parametric_multidim_array",
    src = "parametric_multidim_array.x",
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

xls_dslx_opt_ir(
    name = "for_sans_type_annotation",
    src = "for_sans_type_annotation.x",
)

xls_dslx_opt_ir_test(
    name = "for_sans_type_annotation_test",
    dep = ":for_sans_type_annotation",
)

xls_dslx_opt_ir(
    name = "derived_parametrics",
    src = "derived_parametrics.x",
)

xls_dslx_opt_ir_test(
    name = "derived_parametrics_test",
    dep = ":derived_parametrics",
)

xls_dslx_opt_ir(
    name = "parametric_struct",
    src = "parametric_struct.x",
)

xls_dslx_opt_ir_test(
    name = "parametric_struct_test",
    dep = ":parametric_struct",
)

xls_dslx_opt_ir(
    name = "parametric_struct_simple",
    src = "parametric_struct_simple.x",
)

# Only a parametric entry point, so no benchmark.
xls_dslx_test(
    name = "parametric_struct_simple_test",
    src = "parametric_struct_simple.x",
)

xls_dslx_opt_ir(
    name = "test_f_calls_parametric_f",
    src = "test_f_calls_parametric_f.x",
    opt_ir_args = {
        "entry": get_mangled_ir_symbol("test_f_calls_parametric_f", "f"),
    },
)

xls_dslx_opt_ir_test(
    name = "test_f_calls_parametric_f_test",
    dep = ":test_f_calls_parametric_f",
)

xls_dslx_opt_ir(
    name = "array_concat",
    src = "array_concat.x",
)

xls_dslx_opt_ir_test(
    name = "array_concat_test",
    dep = ":array_concat",
)

xls_dslx_opt_ir(
    name = "for_up_to_const",
    src = "for_up_to_const.x",
    opt_ir_args = {
        "entry": get_mangled_ir_symbol("for_up_to_const", "f"),
    },
)

xls_dslx_opt_ir_test(
    name = "for_up_to_const_test",
    dep = ":for_up_to_const",
)

xls_dslx_opt_ir(
    name = "for_up_to_const_local",
    src = "for_up_to_const_local.x",
)

xls_dslx_opt_ir_test(
    name = "for_up_to_const_local_test",
    dep = ":for_up_to_const_local",
)

xls_dslx_opt_ir(
    name = "map_of_stdlib_parametric",
    src = "map_of_stdlib_parametric.x",
)

xls_dslx_opt_ir_test(
    name = "map_of_stdlib_parametric_test",
    dep = ":map_of_stdlib_parametric",
)

xls_dslx_library(
    name = "mod_imported_array_dslx",
    srcs = ["mod_imported_array.x"],
)

xls_dslx_opt_ir(
    name = "map_multi_module",
    src = "map_multi_module.x",
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
    src = "parametric_const_ref.x",
)

# Only a parametric entry point, so no benchmark.
xls_dslx_test(
    name = "parametric_const_ref_test",
    src = "parametric_const_ref.x",
)

xls_dslx_opt_ir(
    name = "array_with_binary_dim",
    src = "array_with_binary_dim.x",
)

xls_dslx_opt_ir_test(
    name = "array_with_binary_dim_test",
    dep = ":array_with_binary_dim",
)

xls_dslx_library(
    name = "mod_simple_enum_dslx",
    srcs = ["mod_simple_enum.x"],
)

xls_dslx_opt_ir(
    name = "mod_simple_enum_importer",
    src = "mod_simple_enum_importer.x",
    deps = [":mod_simple_enum_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_simple_enum_importer_test",
    dep = ":mod_simple_enum_importer",
)

xls_dslx_opt_ir(
    name = "mod_enum_use_in_for_match_importer",
    src = "mod_enum_use_in_for_match_importer.x",
    deps = [":mod_simple_enum_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_enum_use_in_for_match_importer_test",
    dep = ":mod_enum_use_in_for_match_importer",
)

xls_dslx_opt_ir(
    name = "mod_enum_fully_qualified_match_importer",
    src = "mod_enum_fully_qualified_match_importer.x",
    deps = [":mod_simple_enum_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_enum_fully_qualified_match_importer_test",
    dep = ":mod_enum_fully_qualified_match_importer",
)

xls_dslx_library(
    name = "mod_simple_const_dslx",
    srcs = ["mod_simple_const.x"],
)

xls_dslx_opt_ir(
    name = "mod_const_importer",
    src = "mod_const_importer.x",
    deps = [":mod_simple_const_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_const_importer_test",
    dep = ":mod_const_importer",
)

xls_dslx_opt_ir(
    name = "importer_const_in_range",
    src = "importer_const_in_range.x",
    deps = [":mod_simple_const_dslx"],
)

xls_dslx_opt_ir_test(
    name = "importer_const_in_range_test",
    dep = ":importer_const_in_range",
)

xls_dslx_opt_ir(
    name = "importer_const_in_return_type",
    src = "importer_const_in_return_type.x",
    deps = [":mod_simple_const_dslx"],
)

xls_dslx_opt_ir_test(
    name = "importer_const_in_return_type_test",
    dep = ":importer_const_in_return_type",
)

xls_dslx_opt_ir(
    name = "mod_const_in_type_alias",
    src = "mod_const_in_type_alias.x",
    deps = [":mod_simple_const_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_const_in_type_alias_test",
    dep = ":mod_const_in_type_alias",
)

xls_dslx_opt_ir(
    name = "mod_const_in_type_alias_expr",
    src = "mod_const_in_type_alias_expr.x",
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

xls_dslx_opt_ir(
    name = "mod_const_enum_importer",
    src = "mod_const_enum_importer.x",
    deps = [":mod_simple_const_enum_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_const_enum_importer_test",
    dep = ":mod_const_enum_importer",
)

xls_dslx_opt_ir(
    name = "mod_simple_enum_alias_importer",
    src = "mod_simple_enum_alias_importer.x",
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

xls_dslx_opt_ir(
    name = "mod_const_array_of_enum_importer",
    src = "mod_const_array_of_enum_importer.x",
    deps = [":mod_simple_const_array_of_enums_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_const_array_of_enum_importer_test",
    dep = ":mod_const_array_of_enum_importer",
)

xls_dslx_opt_ir(
    name = "simple_enum_usage",
    src = "simple_enum_usage.x",
)

xls_dslx_opt_ir_test(
    name = "simple_enum_usage_test",
    dep = ":simple_enum_usage",
)

xls_dslx_opt_ir(
    name = "for_rotate",
    src = "for_rotate.x",
)

xls_dslx_opt_ir_test(
    name = "for_rotate_test",
    dep = ":for_rotate",
)

xls_dslx_opt_ir(
    name = "for_enum_ref",
    src = "for_enum_ref.x",
)

xls_dslx_opt_ir_test(
    name = "for_enum_ref_test",
    dep = ":for_enum_ref",
)

xls_dslx_opt_ir(
    name = "update_within_parametric",
    src = "update_within_parametric.x",
)

# Only a parametric entry point, so no benchmark.
xls_dslx_test(
    name = "update_within_parametric_test",
    src = "update_within_parametric.x",
)

xls_dslx_opt_ir(
    name = "width_slice_to_parametric",
    src = "width_slice_to_parametric.x",
)

xls_dslx_opt_ir_test(
    name = "width_slice_to_parametric_test",
    dep = ":width_slice_to_parametric",
)

xls_dslx_opt_ir(
    name = "array_size_all_bits_set_ellipsis",
    src = "array_size_all_bits_set_ellipsis.x",
)

xls_dslx_opt_ir_test(
    name = "array_size_all_bits_set_ellipsis_test",
    dep = ":array_size_all_bits_set_ellipsis",
)

xls_dslx_opt_ir(
    name = "mask_bits",
    src = "mask_bits.x",
)

# Only a parametric entry point, so no benchmark.
xls_dslx_test(
    name = "mask_bits_test",
    src = "mask_bits.x",
)

xls_dslx_opt_ir(
    name = "function_called_in_derived_parametric_simple",
    src = "function_called_in_derived_parametric_simple.x",
)

xls_dslx_opt_ir_test(
    name = "function_called_in_derived_parametric_simple_test",
    dep = ":function_called_in_derived_parametric_simple",
)

xls_dslx_opt_ir(
    name = "non_parametric_called_solely_from_parametric",
    src = "non_parametric_called_solely_from_parametric.x",
)

xls_dslx_opt_ir_test(
    name = "non_parametric_called_solely_from_parametric_test",
    dep = ":non_parametric_called_solely_from_parametric",
)

xls_dslx_opt_ir(
    name = "function_called_in_derived_parametric_struct_simple",
    src = "function_called_in_derived_parametric_struct_simple.x",
)

xls_dslx_opt_ir_test(
    name = "function_called_in_derived_parametric_struct_simple_test",
    dep = ":function_called_in_derived_parametric_struct_simple",
)

xls_dslx_opt_ir(
    name = "identifiers_with_ticks",
    src = "identifiers_with_ticks.x",
)

xls_dslx_opt_ir_test(
    name = "identifiers_with_ticks_test",
    dep = ":identifiers_with_ticks",
)

# TODO(https://github.com/google/xls/issues/246) Need const values to be
# evaluated at typechecking time so they can interact with the type system.
#dslx_test(
#    name = "bitslice_with_constexpr_bounds",
#    srcs = ["bitslice_with_constexpr_bounds.x"],
#)

xls_dslx_opt_ir(
    name = "enum_value_from_const",
    src = "enum_value_from_const.x",
)

xls_dslx_opt_ir_test(
    name = "enum_value_from_const_test",
    dep = ":enum_value_from_const",
)

xls_dslx_library(
    name = "mod_imported_const_struct_dslx",
    srcs = ["mod_imported_const_struct.x"],
)

xls_dslx_opt_ir(
    name = "mod_importer_access_const_struct",
    src = "mod_importer_access_const_struct.x",
    deps = [":mod_imported_const_struct_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_importer_access_const_struct_test",
    dep = ":mod_importer_access_const_struct",
)

xls_dslx_library(
    name = "mod_imported_access_const_struct_dslx",
    srcs = ["mod_imported_access_const_struct.x"],
    deps = [
        ":mod_imported_const_struct_dslx",
    ],
)

xls_dslx_opt_ir(
    name = "mod_importer_access_const_struct_indirect",
    src = "mod_importer_access_const_struct_indirect.x",
    deps = [
        ":mod_imported_access_const_struct_dslx",
    ],
)

xls_dslx_opt_ir_test(
    name = "mod_importer_access_const_struct_indirect_test",
    dep = ":mod_importer_access_const_struct_indirect",
)

xls_dslx_opt_ir(
    name = "nest_alias_const_array",
    src = "nest_alias_const_array.x",
)

xls_dslx_opt_ir_test(
    name = "nest_alias_const_array_test",
    dep = ":nest_alias_const_array",
)

xls_dslx_opt_ir(
    name = "parametric_used_in_for",
    src = "parametric_used_in_for.x",
)

xls_dslx_opt_ir_test(
    name = "parametric_used_in_for_test",
    dep = ":parametric_used_in_for",
)

xls_dslx_opt_ir(
    name = "parametric_to_value",
    src = "parametric_to_value.x",
)

xls_dslx_opt_ir_test(
    name = "parametric_to_value_test",
    dep = ":parametric_to_value",
)

xls_dslx_opt_ir(
    name = "constexpr_with_typed_stuff_after",
    src = "constexpr_with_typed_stuff_after.x",
)

xls_dslx_opt_ir_test(
    name = "constexpr_with_typed_stuff_after_test",
    dep = ":constexpr_with_typed_stuff_after",
)

xls_dslx_opt_ir(
    name = "for_up_to_constexpr_cast",
    src = "for_up_to_constexpr_cast.x",
)

xls_dslx_opt_ir_test(
    name = "for_up_to_constexpr_cast_test",
    dep = ":for_up_to_constexpr_cast",
)

# Note: no meaningful function to convert.
xls_dslx_test(
    name = "bits_concat",
    src = "bits_concat.x",
    dslx_test_args = {
        "compare": "none",
    },
)

xls_dslx_opt_ir(
    name = "local_const_value_in_parametric_match",
    src = "local_const_value_in_parametric_match.x",
)

xls_dslx_opt_ir_test(
    name = "local_const_value_in_parametric_match_test",
    dep = ":local_const_value_in_parametric_match",
)

xls_dslx_opt_ir(
    name = "let_match_tuple",
    src = "let_match_tuple.x",
    ir_file = "let_match_tuple.ir",
)

# TODO(https://github.com/google/xls/issues/400): This produces a zero-bit
# concate which is currently not supported by Z3 translation. Add
# xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "let_match_tuple_eval_ir_test",
    src = "let_match_tuple.ir",
)

xls_benchmark_ir(
    name = "let_match_tuple_benchmark_ir",
    src = "let_match_tuple.ir",
)
