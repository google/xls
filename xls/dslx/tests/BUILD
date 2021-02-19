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

load("//xls/build_rules:build_defs.bzl", "dslx_test")

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

dslx_test(
    name = "tuple_with_array_member",
    srcs = ["tuple_with_array_member.x"],
)

dslx_test(
    name = "binops",
    srcs = ["binops.x"],
    prove_unopt_eq_opt = False,
)

dslx_test(
    name = "comparisons",
    srcs = ["comparisons.x"],
    generate_benchmark = False,
    prove_unopt_eq_opt = False,
)

dslx_test(
    name = "constexpr",
    srcs = ["constexpr.x"],
)

dslx_test(
    name = "constexpr_clog2",
    srcs = ["constexpr_clog2.x"],
)

dslx_test(
    name = "constexpr_imported_clog2",
    srcs = ["constexpr_imported_clog2.x"],
    deps = ["constexpr.x"],
)

dslx_test(
    name = "explicit_parametric",
    srcs = ["explicit_parametric.x"],
    generate_benchmark = False,
    prove_unopt_eq_opt = False,
)

dslx_test(
    name = "explicit_parametric_simple",
    srcs = ["explicit_parametric_simple.x"],
    generate_benchmark = False,
    prove_unopt_eq_opt = False,
)

dslx_test(
    name = "explicit_parametric_reduced",
    srcs = ["explicit_parametric_reduced.x"],
    generate_benchmark = False,
    prove_unopt_eq_opt = False,
)

dslx_test(
    name = "import_constexpr",
    srcs = ["import_constexpr.x"],
    deps = ["constexpr.x"],
)

dslx_test(
    name = "map",
    srcs = ["map.x"],
)

dslx_test(
    name = "multiplies",
    srcs = ["multiplies.x"],
)

dslx_test(
    name = "match_sample",
    srcs = ["match_sample.x"],
)

dslx_test(
    name = "match_const",
    srcs = ["match_const.x"],
)

dslx_test(
    name = "match_multi",
    srcs = ["match_multi.x"],
)

dslx_test(
    name = "type_aliasing_importer",
    srcs = ["type_aliasing_importer.x"],
    generate_benchmark = False,
    prove_unopt_eq_opt = False,
    deps = ["mod_struct_point.x"],
)

dslx_test(
    name = "parametric_call_parametric",
    srcs = ["parametric_call_parametric.x"],
    generate_benchmark = False,
    prove_unopt_eq_opt = False,
)

dslx_test(
    name = "parametric_importer",
    srcs = ["parametric_importer.x"],
    generate_benchmark = False,
    prove_unopt_eq_opt = False,
    deps = ["parametric_import.x"],
)

dslx_test(
    name = "signed_comparisons",
    srcs = ["signed_comparisons.x"],
    generate_benchmark = False,
    prove_unopt_eq_opt = False,
)

dslx_test(
    name = "subtract_to_negative",
    srcs = ["subtract_to_negative.x"],
)

dslx_test(
    name = "trace",
    srcs = ["trace.x"],
)

dslx_test(
    name = "array_of_typedef_in_tuple",
    srcs = ["array_of_typedef_in_tuple.x"],
)

dslx_test(
    name = "array_of_tuple",
    srcs = ["array_of_tuple.x"],
    # Only a test function.
    generate_benchmark = False,
    prove_unopt_eq_opt = False,
)

dslx_test(
    name = "parametric_array_of_tuples",
    srcs = ["parametric_array_of_tuples.x"],
    # TODO(leary): 2019-07-24 Missing support for array of tuples eliding cast
    # node.
    convert_ir = False,
)

dslx_test(
    name = "parametric_functions",
    srcs = ["parametric_functions.x"],
    convert_ir = False,
)

dslx_test(
    name = "bit_slice",
    srcs = ["bit_slice.x"],
    # Note: no meaningful function to convert.
    convert_ir = False,
)

dslx_test(
    name = "bit_slice_update",
    srcs = ["bit_slice_update.x"],
)

dslx_test(
    name = "bit_slice_syntax",
    srcs = ["bit_slice_syntax.x"],
    # Note: no meaningful function to convert.
    convert_ir = False,
)

dslx_test(
    name = "tuple_indexing",
    srcs = ["tuple_indexing.x"],
    # Note: no meaningful function to convert.
    convert_ir = False,
)

dslx_test(
    name = "clz",
    srcs = ["clz.x"],
)

dslx_test(
    name = "ctz",
    srcs = ["ctz.x"],
)

dslx_test(
    name = "one_hot",
    srcs = ["one_hot.x"],
    # Note: no meaningful function to convert.
    convert_ir = False,
)

dslx_test(
    name = "one_hot_sel",
    srcs = ["one_hot_sel.x"],
    # Note: no meaningful function to convert.
    convert_ir = False,
)

dslx_test(
    name = "reductions",
    srcs = ["reductions.x"],
    prove_unopt_eq_opt = False,
)

dslx_test(
    name = "signed_number_type",
    srcs = ["signed_number_type.x"],
)

dslx_test(
    name = "numerical_conversions",
    srcs = ["numerical_conversions.x"],
    # Note: no meaningful function to convert.
    convert_ir = False,
)

dslx_test(
    name = "numerical_conversions2",
    srcs = ["numerical_conversions2.x"],
    # Note: no meaningful function to convert.
    convert_ir = False,
)

dslx_test(
    name = "signex_builtin",
    srcs = ["signex_builtin.x"],
    # Note: no meaningful function to convert.
    convert_ir = False,
)

dslx_test(
    name = "enum_values",
    srcs = ["enum_values.x"],
    # Note: no meaningful function to convert.
    convert_ir = False,
)

dslx_test(
    name = "casts",
    srcs = ["casts.x"],
    # TODO(leary); 2019-10-24 Enable when we have ConcatArray op.
    convert_ir = False,
)

dslx_test(
    name = "cast_to_array",
    srcs = ["cast_to_array.x"],
    # TODO(leary); 2019-10-24 Enable when we have ConcatArray op.
    convert_ir = False,
)

dslx_test(
    name = "parametric_with_comparison",
    srcs = ["parametric_with_comparison.x"],
    convert_ir = False,
)

dslx_test(
    name = "parametric_smul",
    srcs = ["parametric_smul.x"],
    # Note: no meaningful function to convert.
    convert_ir = False,
)

dslx_test(
    name = "basic_struct",
    srcs = ["basic_struct.x"],
)

dslx_test(
    name = "basic_struct_parameter",
    srcs = ["basic_struct_parameter.x"],
)

dslx_test(
    name = "basic_struct_attr",
    srcs = ["basic_struct_attr.x"],
)

dslx_test(
    name = "struct_equality",
    srcs = ["struct_equality.x"],
    # Note: no meaningful function to convert.
    convert_ir = False,
)

dslx_test(
    name = "basic_struct_update",
    srcs = ["basic_struct_update.x"],
)

dslx_test(
    name = "struct_splat_update",
    srcs = ["struct_splat_update.x"],
)

dslx_test(
    name = "empty_struct_splat_update",
    srcs = ["empty_struct_splat_update.x"],
)

dslx_test(
    name = "basic_struct_array",
    srcs = ["basic_struct_array.x"],
)

dslx_test(
    name = "basic_enum_array",
    srcs = ["basic_enum_array.x"],
)

dslx_test(
    name = "basic_array",
    srcs = ["basic_array.x"],
)

dslx_test(
    name = "std_lsb",
    srcs = ["std_lsb.x"],
)

dslx_test(
    name = "std_alias",
    srcs = ["std_alias.x"],
)

dslx_test(
    name = "mod_importer",
    srcs = ["mod_importer.x"],
    deps = ["mod_imported.x"],
)

dslx_test(
    name = "mod_importer_typedef",
    srcs = ["mod_importer_typedef.x"],
    deps = ["mod_imported_typedef.x"],
)

dslx_test(
    name = "mod_importer_struct_of_enum",
    srcs = ["mod_importer_struct_of_enum.x"],
    deps = ["mod_imported_struct_of_enum.x"],
)

dslx_test(
    name = "mod_importer_simple",
    srcs = ["mod_importer_simple.x"],
    deps = ["mod_imported.x"],
)

dslx_test(
    name = "mod_struct_importer",
    srcs = ["mod_struct_importer.x"],
    deps = ["mod_imported.x"],
)

dslx_test(
    name = "mod_enum_importer",
    srcs = ["mod_enum_importer.x"],
    deps = ["mod_imported.x"],
)

dslx_test(
    name = "mod_typedef_importer",
    srcs = ["mod_typedef_importer.x"],
    deps = ["mod_imported_aliases.x"],
)

dslx_test(
    name = "array_ellipsis",
    srcs = ["array_ellipsis.x"],
)

dslx_test(
    name = "local_const_value",
    srcs = ["local_const_value.x"],
)

dslx_test(
    name = "constant_array_unannotated_type",
    srcs = ["constant_array_unannotated_type.x"],
)

dslx_test(
    name = "reverse",
    srcs = ["reverse.x"],
)

dslx_test(
    name = "typedef_multidim_array",
    srcs = ["typedef_multidim_array.x"],
)

dslx_test(
    name = "parametric_multidim_array",
    srcs = ["parametric_multidim_array.x"],
    entry = "__parametric_multidim_array__id_6_3_2",
)

dslx_test(
    name = "derived_parametrics",
    srcs = ["derived_parametrics.x"],
)

dslx_test(
    name = "parametric_struct",
    srcs = ["parametric_struct.x"],
)

dslx_test(
    name = "test_f_calls_parametric_f",
    srcs = ["test_f_calls_parametric_f.x"],
    entry = "__test_f_calls_parametric_f__f",
)

dslx_test(
    name = "array_concat",
    srcs = ["array_concat.x"],
    # TODO(hjmontero): 10-08-2020 array comparison via JIT is incorrect.
    compare_jit = False,
)

dslx_test(
    name = "for_up_to_const",
    srcs = ["for_up_to_const.x"],
    entry = "__for_up_to_const__f",
)

dslx_test(
    name = "map_of_stdlib_parametric",
    srcs = ["map_of_stdlib_parametric.x"],
)

dslx_test(
    name = "parametric_const_ref",
    srcs = ["parametric_const_ref.x"],
    # Only a parametric entry point, so no benchmark.
    generate_benchmark = False,
    prove_unopt_eq_opt = False,
)

dslx_test(
    name = "array_with_binary_dim",
    srcs = ["array_with_binary_dim.x"],
)

dslx_test(
    name = "mod_simple_enum_importer",
    srcs = ["mod_simple_enum_importer.x"],
    deps = ["mod_simple_enum.x"],
)

dslx_test(
    name = "mod_enum_use_in_for_match_importer",
    srcs = ["mod_enum_use_in_for_match_importer.x"],
    deps = ["mod_simple_enum.x"],
)

dslx_test(
    name = "mod_enum_fully_qualified_match_importer",
    srcs = ["mod_enum_fully_qualified_match_importer.x"],
    deps = ["mod_simple_enum.x"],
)

dslx_test(
    name = "mod_const_importer",
    srcs = ["mod_const_importer.x"],
    deps = ["mod_simple_const.x"],
)

dslx_test(
    name = "mod_const_enum_importer",
    srcs = ["mod_const_enum_importer.x"],
    deps = ["mod_simple_const_enum.x"],
)

dslx_test(
    name = "mod_simple_enum_alias_importer",
    srcs = ["mod_simple_enum_alias_importer.x"],
    deps = ["mod_simple_enum.x"],
)

dslx_test(
    name = "mod_simple_const_array_of_enums_importer",
    srcs = ["mod_const_array_of_enum_importer.x"],
    deps = ["mod_simple_const_array_of_enums.x"],
)

dslx_test(
    name = "simple_enum_usage",
    srcs = ["simple_enum_usage.x"],
)

dslx_test(
    name = "for_rotate",
    srcs = ["for_rotate.x"],
)

dslx_test(
    name = "for_enum_ref",
    srcs = ["for_enum_ref.x"],
)

dslx_test(
    name = "update_within_parametric",
    srcs = ["update_within_parametric.x"],
    # Only a parametric entry point, so no benchmark.
    generate_benchmark = False,
    prove_unopt_eq_opt = False,
)

dslx_test(
    name = "width_slice_to_parametric",
    srcs = ["width_slice_to_parametric.x"],
)

dslx_test(
    name = "array_size_all_bits_set_ellipsis",
    srcs = ["array_size_all_bits_set_ellipsis.x"],
)

dslx_test(
    name = "mask_bits",
    srcs = ["mask_bits.x"],
    # Only a parametric entry point, so no benchmark.
    generate_benchmark = False,
    prove_unopt_eq_opt = False,
)

dslx_test(
    name = "function_called_in_derived_parametric_simple",
    srcs = ["function_called_in_derived_parametric_simple.x"],
)

dslx_test(
    name = "non_parametric_called_solely_from_parametric",
    srcs = ["non_parametric_called_solely_from_parametric.x"],
)

dslx_test(
    name = "function_called_in_derived_parametric_struct_simple",
    srcs = ["function_called_in_derived_parametric_struct_simple.x"],
)

dslx_test(
    name = "identifiers_with_ticks",
    srcs = ["identifiers_with_ticks.x"],
)

# TODO(https://github.com/google/xls/issues/246) Need const values to be
# evaluated at typechecking time so they can interact with the type system.
#dslx_test(
#    name = "bitslice_with_constexpr_bounds",
#    srcs = ["bitslice_with_constexpr_bounds.x"],
#)

dslx_test(
    name = "enum_value_from_const",
    srcs = ["enum_value_from_const.x"],
)
