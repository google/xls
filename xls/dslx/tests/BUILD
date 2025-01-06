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

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@bazel_skylib//rules:build_test.bzl", "build_test")
load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_dslx_ir",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_opt_ir_test",
    "xls_ir_opt_ir",
)
load("//xls/dslx:strip_comments.bzl", "dslx_strip_comments")
load("//xls/dslx/tests:build_defs.bzl", "dslx_lang_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
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

dslx_lang_test(name = "two_plus_two")

# Simple unit tests with no meaningful main entry point.
dslx_lang_test(
    name = "simple_invocation",
    convert_to_ir = False,
)

dslx_lang_test(
    name = "simple_cast",
    convert_to_ir = False,
)

dslx_lang_test(
    name = "simple_subtract",
    convert_to_ir = False,
)

dslx_lang_test(
    name = "simple_add_overflow",
    convert_to_ir = False,
)

dslx_lang_test(name = "builtin_zip")

dslx_lang_test(
    name = "simple_mod",
    # Some delay models do not characterize mod.
    benchmark_ir = False,
)

dslx_lang_test(
    name = "simple_div",
    # Some delay models do not characterize div.
    benchmark_ir = False,
)

dslx_lang_test(
    name = "simple_inequality",
    convert_to_ir = False,
)

dslx_lang_test(
    name = "simple_array_equality",
    convert_to_ir = False,
)

dslx_lang_test(name = "simple_const_assert")

dslx_lang_test(
    name = "module_level_const_assert",
    # Note: no meaningful function to convert.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "module_level_const_assert_that_calls_fn",
    # No meaningful entry point to convert.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "wide_ashr",
    convert_to_ir = False,
)

dslx_lang_test(
    name = "pad_bits_via_concat",
    convert_to_ir = False,
)

dslx_lang_test(
    name = "derived_parametric",
    convert_to_ir = False,
)

dslx_lang_test(
    name = "derived_parametric_struct",
    convert_to_ir = False,
)

dslx_lang_test(
    name = "struct_with_const_sized_array",
    convert_to_ir = False,
)

dslx_lang_test(
    name = "typedef_array",
    convert_to_ir = False,
)

dslx_lang_test(
    name = "slice_builtin",
    convert_to_ir = False,
)

dslx_lang_test(
    name = "destructure",
    convert_to_ir = False,
)

dslx_lang_test(
    name = "wide_shifts",
    convert_to_ir = False,
)

dslx_lang_test(
    name = "bitslice_syntax",
    convert_to_ir = False,
)

dslx_lang_test(
    name = "destructure_wildcard",
    convert_to_ir = False,
)

dslx_lang_test(
    name = "bool_not",
    convert_to_ir = False,
)

dslx_lang_test(
    name = "bool_literal",
    convert_to_ir = False,
)

dslx_lang_test(
    name = "parametric_invocation",
    convert_to_ir = False,
)

dslx_lang_test(name = "parametric_signedness")

dslx_lang_test(name = "xn_sizeof")

dslx_lang_test(name = "xn_type_equivalence")

dslx_lang_test(name = "xn_signedness_properties")

dslx_lang_test(name = "xn_slice_bounds")

dslx_lang_test(name = "xn_widening_cast")

dslx_lang_test(
    name = "parametric_shift",
    # TODO(leary): 2023-08-14 Runs into "cannot translate zero length bitvector
    # with uses".
    test_ir_equivalence = False,
)

dslx_lang_test(
    name = "for_over_range_u8",
    convert_to_ir = False,
)

dslx_lang_test(
    name = "parametric_binding",
    convert_to_ir = False,
)

dslx_lang_test(
    name = "tree_binding",
    convert_to_ir = False,
)

dslx_lang_test(
    name = "array_index",
    convert_to_ir = False,
)

dslx_lang_test(
    name = "tuple_index",
    convert_to_ir = False,
)

dslx_lang_test(
    name = "for_over_array",
    convert_to_ir = False,
)

dslx_lang_test(name = "zero_macro")

dslx_lang_test(name = "all_ones_macro")

dslx_lang_test(name = "tuple_with_array_member")

dslx_lang_test(
    name = "array_of_imported_struct",
    dslx_deps = [":mod_imported_dslx"],
)

dslx_lang_test(name = "local_type_alias")

dslx_lang_test(
    name = "block_with_trailing_semi",
    # We intenionally have unused vars.
    warnings_as_errors = False,
)

dslx_lang_test(
    name = "binops",
    test_ir_equivalence = False,
)

dslx_lang_test(
    name = "comparisons",
    convert_to_ir = False,
)

dslx_lang_test(
    name = "constexpr",
)

dslx_lang_test(
    name = "constexpr_attrs",
    convert_to_ir = False,
    dslx_deps = [":constexpr_dslx"],
)

dslx_lang_test(
    name = "builtin_type_max",
    dslx_deps = [":number_of_imported_type_import_dslx"],
)

dslx_lang_test(
    name = "builtin_type_zero",
    dslx_deps = [":number_of_imported_type_import_dslx"],
)

dslx_lang_test(
    name = "fallible_constexpr",
)

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

dslx_lang_test(name = "constexpr_pad_via_slice")

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

dslx_lang_test(
    name = "parametric_dot_product",
)

dslx_lang_test(name = "parametric_plus_global_as_dimension")

dslx_lang_test(
    name = "parametric_range",
    # Note: no meaningful function to convert to IR.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "for_over_range",
    # No meaningful entry point to convert.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "enumerate",
    # No meaningful entry point to convert.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "character_conversion",
    # TODO: https://github.com/google/xls/issues/1526 - fails opportunistic_postcondition in autofmt
    # due to mismatch between /0 and /x00.
    test_autofmt = False,
)

dslx_lang_test(
    name = "string_conversion",
    # TODO(leary): 2023-10-20 Mangles characters / rewrites strings.
    test_autofmt = False,
)

dslx_lang_test(name = "array_concat_str")

dslx_lang_test(name = "array_slice_str")

dslx_lang_test(name = "array_update_str")

dslx_lang_test(
    name = "import_constexpr",
    dslx_deps = [":constexpr_dslx"],
)

dslx_lang_test(
    name = "array_of_imported_parametric_struct",
    dslx_deps = [":parametric_import_dslx"],
)

dslx_lang_test(
    name = "import_enum_alias",
    dslx_deps = [":mod_enum_importer_dslx"],
)

dslx_lang_test(name = "map")

dslx_lang_test(name = "multiplies")

dslx_lang_test(name = "match_sample")

dslx_lang_test(name = "match_const")

dslx_lang_test(name = "match_multi")

dslx_lang_test(name = "match_range_or_wildcard")

dslx_lang_test(
    name = "match_overlapping_range_and_value",
)

dslx_lang_test(
    name = "match_arms_with_tokens",
    # No meaningful entry function to convert.
    convert_to_ir = False,
)

dslx_lang_test(name = "elseif_sample")

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

dslx_lang_test(name = "impl")

dslx_lang_test(name = "subtract_to_negative")

dslx_lang_test(name = "trace")

# Note: Trace output is checked in xls/dslx:interpreter_test and similar
# trace_fmt tests.
dslx_lang_test(
    name = "trace_fmt_hello",
    # No trace op Z3 translation.
    test_ir_equivalence = False,
)

# Note: Trace output is checked in xls/dslx:interpreter_test and similar
# trace_fmt tests.
dslx_lang_test(
    name = "trace_fmt_struct",
    # No trace op Z3 translation.
    test_ir_equivalence = False,
)

dslx_lang_test(name = "array_of_typedef_in_tuple")

dslx_lang_test(
    name = "array_of_tuple",
    # No meaningful entry function to convert.
    convert_to_ir = False,
)

dslx_lang_test(name = "array_slice")

dslx_lang_test(
    name = "parametric_proc",
    # No meaningful entry function to convert.
    convert_to_ir = False,
)

dslx_lang_test(name = "array_update")

dslx_lang_test(
    name = "parametric_array_of_tuples",
)

dslx_lang_test(
    name = "parametric_functions",
)

dslx_lang_test(
    name = "array_literal_ellipsis",
)

dslx_lang_test(
    name = "cover",
    # cover! not supported yet in translation to z3.
    test_ir_equivalence = False,
)

dslx_lang_test(
    name = "cover_in_loop",
    # cover! not supported yet in translation to z3.
    test_ir_equivalence = False,
)

dslx_lang_test(
    name = "parametric_issue_727",
    # Note: no meaningful function to convert to IR.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "const_array_of_enum_refs",
    # Note: no meaningful function to convert to IR.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "parametric_value_as_nested_loop_bound",
)

dslx_lang_test(
    name = "derived_parametric_value_as_nested_loop_bound",
)

dslx_lang_test(
    name = "bit_slice_update",
)

dslx_lang_test(name = "const_slice_limit")

dslx_lang_test(
    name = "bit_slice_syntax",
    # Note: no meaningful function to convert.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "tuple_indexing",
    # Note: no meaningful function to convert.
    convert_to_ir = False,
)

dslx_lang_test(name = "clz")

dslx_lang_test(name = "ctz")

dslx_lang_test(name = "decode")

dslx_lang_test(name = "encode")

dslx_lang_test(
    name = "one_hot",
    # Note: no meaningful function to convert.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "one_hot_sel",
    # Note: no meaningful function to convert.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "priority_sel",
    # Note: no meaningful function to convert.
    convert_to_ir = False,
)

dslx_lang_test(name = "reductions")

dslx_lang_test(name = "signed_number_type")

dslx_lang_test(
    name = "numerical_conversions",
    # Note: no meaningful function to convert.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "numerical_conversions2",
    # Note: no meaningful function to convert.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "signex_builtin",
    dslx_deps = [":mod_imported_typedef_dslx"],
)

dslx_lang_test(
    name = "builtin_array_rev",
)

dslx_lang_test(
    name = "builtin_array_size",
    # Note: no meaningful entry function to convert.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "enum_values",
    # Note: no meaningful function to convert.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "casts",
    # Note: no meaningful function to convert.
    convert_to_ir = False,
)

dslx_lang_test(name = "casts_to_xn")

dslx_lang_test(
    name = "cast_to_array",
    # Note: no meaningful function to convert.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "parametric_with_comparison",
)

dslx_lang_test(
    name = "parametric_smul",
    # Note: no meaningful function to convert.
    convert_to_ir = False,
)

dslx_lang_test(name = "mulp")

dslx_lang_test(name = "basic_struct")

dslx_lang_test(name = "basic_struct_parameter")

dslx_lang_test(name = "basic_struct_attr")

dslx_lang_test(name = "basic_sv_alias_attr")

dslx_lang_test(name = "basic_sv_enum_attr")

dslx_lang_test(name = "basic_sv_struct_attr")

dslx_lang_test(name = "basic_sv_multiline_struct_attr")

dslx_lang_test(
    name = "struct_equality",
    # TODO(https://github.com/google/xls/issues/1083): No struct equality
    # translation to Z3 is currently implemented.
    test_ir_equivalence = False,
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
    name = "mod_u16_type_alias_dslx",
    srcs = ["mod_u16_type_alias.x"],
)

dslx_lang_test(
    name = "import_u16_type_alias",
    dslx_deps = [":mod_u16_type_alias_dslx"],
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
    name = "mod_parametric_id_user_opt_ir",
    srcs = ["mod_parametric_id_user.x"],
    dslx_top = "main",
    deps = [":mod_parametric_id_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_parametric_id_user_opt_ir_test",
    dep = ":mod_parametric_id_user_opt_ir",
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

dslx_lang_test(name = "local_const_value_as_nested_loop_bound")

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

dslx_lang_test(name = "nil_tuple_accumulator")

dslx_lang_test(name = "derived_parametrics")

dslx_lang_test(
    name = "derived_parametrics2",
    # No meaningful entry point to convert.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "parametric_struct",
)

dslx_lang_test(
    name = "parametric_struct_simple",
    # No meaningful entry point to convert.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "test_f_calls_parametric_f",
)

dslx_lang_test(name = "array_concat")

dslx_lang_test(
    name = "assert_true_predicate",
    # TODO(https://github.com/google/xls/issues/1346): 2021-06-29 We can't
    # lower IR assert to Z3, so we can't do opt/unopt equivalence testing yet.
    test_ir_equivalence = False,
)

dslx_lang_test(
    name = "assert_false_in_false_conditional",
    # TODO(https://github.com/google/xls/issues/1346): 2021-06-29 We can't
    # lower IR assert to Z3, so we can't do opt/unopt equivalence testing yet.
    test_ir_equivalence = False,
)

dslx_lang_test(
    name = "assert_false_in_true_conditional",
    # TODO(https://github.com/google/xls/issues/1346): 2021-06-29 We can't
    # lower IR assert to Z3, so we can't do opt/unopt equivalence testing yet.
    test_ir_equivalence = False,
)

dslx_lang_test(
    name = "quickcheck_fn_with_fail",
    # Primarily a #[quickcheck] function.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "quickcheck_value_is_one_hot",
    # Primarily a #[quickcheck] function.
    convert_to_ir = False,
)

dslx_lang_test(name = "for_up_to_const")

dslx_lang_test(name = "for_up_to_const_local")

dslx_lang_test(name = "map_of_stdlib_parametric")

dslx_lang_test(
    name = "map_consecutive",
    # Primarily a #[test] function.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "simple_map_assert_eq",
    # Primarily a #[test] function.
    convert_to_ir = False,
)

dslx_lang_test(name = "simplest_map")

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
    name = "parametric_const_in_parametric_type_expr",
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
    name = "zero_macro_imported_enum",
    dslx_deps = [":mod_simple_enum_dslx"],
)

dslx_lang_test(
    name = "mod_enum_fully_qualified_match_importer",
    dslx_deps = [":mod_simple_enum_dslx"],
    # TODO(https://github.com/google/xls/issues/1346): 2021-06-29 We can't
    # lower IR assert to Z3, so we can't do opt/unopt equivalence testing yet.
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
    name = "mod_const_importer_opt_ir",
    srcs = ["mod_const_importer.x"],
    dslx_top = "main",
    deps = [":mod_simple_const_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_const_importer_test",
    dep = ":mod_const_importer_opt_ir",
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
    name = "mod_const_array_of_enum_importer_opt_ir",
    srcs = ["mod_const_array_of_enum_importer.x"],
    dslx_top = "main",
    deps = [":mod_simple_const_array_of_enums_dslx"],
)

xls_dslx_opt_ir_test(
    name = "mod_const_array_of_enum_importer_opt_ir_test",
    dep = ":mod_const_array_of_enum_importer_opt_ir",
)

dslx_lang_test(name = "simple_enum_usage")

xls_dslx_opt_ir(
    name = "simple_enum_usage_opt_ir",
    srcs = ["simple_enum_usage.x"],
    dslx_top = "main",
)

xls_dslx_opt_ir_test(
    name = "simple_enum_usage_opt_ir_test",
    dep = ":simple_enum_usage_opt_ir",
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

dslx_lang_test(
    name = "const_struct_array",
    # TODO(https://github.com/google/xls/issues/1084): 2023-07-25 We cannot yet
    # convert the array_size builtin to IR.
    convert_to_ir = False,
)

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

dslx_lang_test(
    name = "constexpr_with_typed_stuff_after",
    # TODO(leary): 2023-10-20 Puts inline comment in a bad spot.
    test_autofmt = False,
)

dslx_lang_test(name = "for_up_to_constexpr_cast")

dslx_lang_test(
    name = "bits_concat",
    # Note: no meaningful function to convert.
    convert_to_ir = False,
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

dslx_lang_test(name = "gh_1197")

dslx_lang_test(
    name = "proc_counter_issue_947",
    # No normal 'function' entry point, it is a test proc.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "proc_channel_arrays",
    # No normal 'function' entry point, it is a test proc.
    convert_to_ir = False,
    # TODO(leary): 2023-10-20 Arrays of channels are being formatted
    # incorrectly.
    test_autofmt = False,
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
    # TODO(leary): 2023-10-20 Comment adorning proc next fn is being deleted.
    test_autofmt = False,
)

dslx_lang_test(
    name = "parametric_proc_with_const",
    # No meaningful entry function to convert.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "proc_with_const",
    # No normal 'function' entry point, it is a test proc.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "proc_two_level",
    # No normal 'function' entry point, it is a test proc.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "proc_with_alias",
    # No normal 'function' entry point, it is a test proc.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "proc_with_alias_from_parametric",
    # No normal 'function' entry point, it is a test proc.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "proc_match_arms_with_recvs",
    # No normal 'function' entry point, it is a test proc.
    convert_to_ir = False,
)

dslx_lang_test(
    name = "send_if_recv_if",
    convert_to_ir = False,
    dslx_entry = "main",
    # TODO(leary): 2023-10-20 Crashes! (assertion failure)
    test_autofmt = False,
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
    # TODO(leary): 2023-10-20 Crashes! (assertion failure)
    test_autofmt = False,
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

dslx_lang_test(
    name = "non_blocking_receive_if_2",
    convert_to_ir = False,
    dslx_entry = "Tester",
)

dslx_lang_test(name = "parametric_builtin_in_module_level_const")

dslx_lang_test(name = "map_module_level_const")

dslx_lang_test(
    name = "three_level_proc_hierarchy",
    convert_to_ir = False,
    dslx_entry = "my_top",
)

exports_files(glob(include = ["*.x"]))

build_test(
    name = "dslx_have_targets",
    # Note: we exclude `mod_` prefixed `.x` files because those are, by
    # convention, just used as "modules with entities for inclusion" i.e. for
    # testing module import facilities, and don't contain significant
    # functionality on their own.
    targets = [x[:-len(".x")] + "_dslx" for x in glob(["*.x"]) if not x.startswith("mod_")],
)

filegroup(
    name = "x_files",
    srcs = glob(["*.x"]),
    visibility = ["//xls:xls_internal"],
)

dslx_strip_comments(
    name = "x_files_stripped",
    srcs = [":x_files"],
)
