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

# Optimization passes, pass managers.

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "standard_pipeline",
    srcs = ["standard_pipeline.cc"],
    hdrs = ["standard_pipeline.h"],
    deps = [
        ":arith_simplification_pass",
        ":array_simplification_pass",
        ":bdd_cse_pass",
        ":bdd_simplification_pass",
        ":bit_slice_simplification_pass",
        ":boolean_simplification_pass",
        ":canonicalization_pass",
        ":concat_simplification_pass",
        ":constant_folding_pass",
        ":cse_pass",
        ":dce_pass",
        ":dfe_pass",
        ":identity_removal_pass",
        ":inlining_pass",
        ":literal_uncommoning_pass",
        ":map_inlining_pass",
        ":narrowing_pass",
        ":passes",
        ":reassociation_pass",
        ":select_simplification_pass",
        ":strength_reduction_pass",
        ":table_switch_pass",
        ":tuple_simplification_pass",
        ":unroll_pass",
        ":verifier_checker",
        "@com_google_absl//absl/status:statusor",
        "//xls/scheduling:pipeline_scheduling_pass",
        "//xls/scheduling:scheduling_checker",
        "//xls/scheduling:scheduling_pass",
    ],
)

cc_library(
    name = "verifier_checker",
    srcs = ["verifier_checker.cc"],
    hdrs = ["verifier_checker.h"],
    deps = [
        ":passes",
        "@com_google_absl//absl/status",
        "//xls/ir",
    ],
)

cc_test(
    name = "passes_test",
    srcs = ["passes_test.cc"],
    deps = [
        ":passes",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:casts",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/examples:sample_packages",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_parser",
        "//xls/ir:type",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "identity_removal_pass_test",
    srcs = ["identity_removal_pass_test.cc"],
    deps = [
        ":dce_pass",
        ":identity_removal_pass",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "arith_simplification_pass_test",
    srcs = ["arith_simplification_pass_test.cc"],
    deps = [
        ":arith_simplification_pass",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "standard_pipeline_test",
    srcs = ["standard_pipeline_test.cc"],
    deps = [
        ":arith_simplification_pass",
        ":dce_pass",
        ":dump_pass",
        ":standard_pipeline",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/examples:sample_packages",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_parser",
        "//xls/ir:ir_test_base",
        "//xls/ir:type",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "canonicalization_pass_test",
    srcs = ["canonicalization_pass_test.cc"],
    deps = [
        ":canonicalization_pass",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "unroll_pass",
    srcs = ["unroll_pass.cc"],
    hdrs = ["unroll_pass.h"],
    deps = [
        ":passes",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/status:status_macros",
        "//xls/ir",
    ],
)

cc_test(
    name = "unroll_pass_test",
    srcs = ["unroll_pass_test.cc"],
    deps = [
        ":dce_pass",
        ":unroll_pass",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_parser",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "inlining_pass",
    srcs = ["inlining_pass.cc"],
    hdrs = ["inlining_pass.h"],
    deps = [
        ":passes",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common/status:status_macros",
        "//xls/ir",
    ],
)

cc_library(
    name = "literal_uncommoning_pass",
    srcs = ["literal_uncommoning_pass.cc"],
    hdrs = ["literal_uncommoning_pass.h"],
    deps = [
        ":passes",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/status:status_macros",
        "//xls/ir",
    ],
)

cc_library(
    name = "map_inlining_pass",
    srcs = ["map_inlining_pass.cc"],
    hdrs = ["map_inlining_pass.h"],
    deps = [
        ":pass_base",
        ":passes",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "//xls/common:math_util",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir:function_builder",
    ],
)

cc_test(
    name = "map_inlining_pass_test",
    srcs = ["map_inlining_pass_test.cc"],
    deps = [
        ":map_inlining_pass",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_parser",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "cse_pass",
    srcs = ["cse_pass.cc"],
    hdrs = ["cse_pass.h"],
    deps = [
        ":passes",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:op",
    ],
)

cc_library(
    name = "constant_folding_pass",
    srcs = ["constant_folding_pass.cc"],
    hdrs = ["constant_folding_pass.h"],
    deps = [
        ":passes",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
    ],
)

cc_library(
    name = "bit_slice_simplification_pass",
    srcs = ["bit_slice_simplification_pass.cc"],
    hdrs = ["bit_slice_simplification_pass.h"],
    deps = [
        ":passes",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits_ops",
    ],
)

cc_library(
    name = "concat_simplification_pass",
    srcs = ["concat_simplification_pass.cc"],
    hdrs = ["concat_simplification_pass.h"],
    deps = [
        ":passes",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits_ops",
        "//xls/ir:node_util",
        "//xls/ir:op",
    ],
)

cc_library(
    name = "boolean_simplification_pass",
    srcs = ["boolean_simplification_pass.cc"],
    hdrs = ["boolean_simplification_pass.h"],
    deps = [
        ":passes",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits_ops",
        "//xls/ir:node_util",
        "//xls/netlist:logical_effort",
    ],
)

cc_library(
    name = "arith_simplification_pass",
    srcs = ["arith_simplification_pass.cc"],
    hdrs = ["arith_simplification_pass.h"],
    deps = [
        ":passes",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits_ops",
        "//xls/ir:node_util",
        "//xls/ir:value_helpers",
    ],
)

cc_library(
    name = "dfe_pass",
    srcs = ["dfe_pass.cc"],
    hdrs = ["dfe_pass.h"],
    deps = [
        ":passes",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/logging",
        "//xls/ir",
    ],
)

cc_library(
    name = "canonicalization_pass",
    srcs = ["canonicalization_pass.cc"],
    hdrs = ["canonicalization_pass.h"],
    deps = [
        ":passes",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits_ops",
    ],
)

cc_library(
    name = "dump_pass",
    srcs = ["dump_pass.cc"],
    hdrs = ["dump_pass.h"],
    deps = [
        ":passes",
        "@com_google_absl//absl/status:statusor",
        "//xls/ir",
    ],
)

cc_library(
    name = "passes",
    srcs = ["passes.cc"],
    hdrs = ["passes.h"],
    deps = [
        ":pass_base",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir",
    ],
)

cc_library(
    name = "identity_removal_pass",
    srcs = ["identity_removal_pass.cc"],
    hdrs = ["identity_removal_pass.h"],
    deps = [
        ":passes",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/status:status_macros",
        "//xls/ir",
    ],
)

cc_library(
    name = "dce_pass",
    srcs = ["dce_pass.cc"],
    hdrs = ["dce_pass.h"],
    deps = [
        ":passes",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:op",
        "//xls/ir:type",
    ],
)

cc_library(
    name = "tuple_simplification_pass",
    srcs = ["tuple_simplification_pass.cc"],
    hdrs = ["tuple_simplification_pass.h"],
    deps = [
        ":passes",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
    ],
)

cc_library(
    name = "strength_reduction_pass",
    srcs = ["strength_reduction_pass.cc"],
    hdrs = [
        "strength_reduction_pass.h",
    ],
    deps = [
        ":passes",
        ":query_engine",
        ":ternary_query_engine",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:node_util",
    ],
)

cc_library(
    name = "ternary_query_engine",
    srcs = ["ternary_query_engine.cc"],
    hdrs = ["ternary_query_engine.h"],
    deps = [
        ":query_engine",
        ":ternary_evaluator",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "//xls/common/status:status_macros",
        "//xls/data_structures:leaf_type_tree",
        "//xls/ir",
        "//xls/ir:abstract_node_evaluator",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
    ],
)

cc_library(
    name = "select_simplification_pass",
    srcs = ["select_simplification_pass.cc"],
    hdrs = ["select_simplification_pass.h"],
    deps = [
        ":passes",
        ":post_dominator_analysis",
        ":ternary_query_engine",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common:visitor",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/logging:vlog_is_on",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/data_structures:algorithm",
        "//xls/ir",
        "//xls/ir:bits_ops",
        "//xls/ir:node_util",
    ],
)

cc_library(
    name = "ternary_evaluator",
    hdrs = ["ternary_evaluator.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "//xls/common/logging",
        "//xls/ir:abstract_evaluator",
        "//xls/ir:bits",
        "//xls/ir:ternary",
    ],
)

cc_library(
    name = "post_dominator_analysis",
    srcs = ["post_dominator_analysis.cc"],
    hdrs = ["post_dominator_analysis.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "//xls/common/status:ret_check",
        "//xls/ir",
    ],
)

cc_library(
    name = "bdd_function",
    srcs = ["bdd_function.cc"],
    hdrs = ["bdd_function.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/data_structures:binary_decision_diagram",
        "//xls/data_structures:leaf_type_tree",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:abstract_evaluator",
        "//xls/ir:abstract_node_evaluator",
        "//xls/ir:op",
    ],
)

cc_library(
    name = "query_engine",
    srcs = ["query_engine.cc"],
    hdrs = ["query_engine.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:variant",
        "//xls/common/logging",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:ternary",
    ],
)

cc_library(
    name = "bdd_query_engine",
    srcs = ["bdd_query_engine.cc"],
    hdrs = ["bdd_query_engine.h"],
    deps = [
        ":bdd_function",
        ":query_engine",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/data_structures:binary_decision_diagram",
        "//xls/ir",
        "//xls/ir:bits",
    ],
)

cc_library(
    name = "bdd_simplification_pass",
    srcs = ["bdd_simplification_pass.cc"],
    hdrs = ["bdd_simplification_pass.h"],
    deps = [
        ":bdd_query_engine",
        ":passes",
        ":post_dominator_analysis",
        ":query_engine",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:optional",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/logging:vlog_is_on",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
    ],
)

cc_library(
    name = "pass_base",
    hdrs = ["pass_base.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:status_macros",
        "//xls/ir",
    ],
)

cc_library(
    name = "narrowing_pass",
    srcs = ["narrowing_pass.cc"],
    hdrs = ["narrowing_pass.h"],
    deps = [
        ":passes",
        ":query_engine",
        ":ternary_query_engine",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:node_util",
        "//xls/ir:op",
    ],
)

cc_library(
    name = "bdd_cse_pass",
    srcs = ["bdd_cse_pass.cc"],
    hdrs = ["bdd_cse_pass.h"],
    deps = [
        ":bdd_function",
        ":passes",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimator",
        "//xls/delay_model:delay_estimators",
        "//xls/ir",
    ],
)

cc_library(
    name = "reassociation_pass",
    srcs = ["reassociation_pass.cc"],
    hdrs = ["reassociation_pass.h"],
    deps = [
        ":pass_base",
        ":passes",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common:math_util",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits_ops",
        "//xls/ir:node_util",
        "//xls/ir:op",
    ],
)

cc_library(
    name = "table_switch_pass",
    srcs = ["table_switch_pass.cc"],
    hdrs = ["table_switch_pass.h"],
    deps = [
        ":passes",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
    ],
)

cc_library(
    name = "array_simplification_pass",
    srcs = ["array_simplification_pass.cc"],
    hdrs = ["array_simplification_pass.h"],
    deps = [
        ":passes",
        ":ternary_query_engine",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/logging",
        "//xls/ir",
        "//xls/ir:bits_ops",
        "//xls/ir:type",
        "//xls/ir:value_helpers",
    ],
)

cc_test(
    name = "inlining_pass_test",
    srcs = ["inlining_pass_test.cc"],
    deps = [
        ":dce_pass",
        ":inlining_pass",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_parser",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "literal_uncommoning_pass_test",
    srcs = ["literal_uncommoning_pass_test.cc"],
    deps = [
        ":literal_uncommoning_pass",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "cse_pass_test",
    srcs = ["cse_pass_test.cc"],
    deps = [
        ":cse_pass",
        ":dce_pass",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "constant_folding_pass_test",
    srcs = ["constant_folding_pass_test.cc"],
    deps = [
        ":constant_folding_pass",
        ":dce_pass",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_test_base",
        "//xls/ir:value",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "bit_slice_simplification_pass_test",
    srcs = ["bit_slice_simplification_pass_test.cc"],
    deps = [
        ":bit_slice_simplification_pass",
        ":dce_pass",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "//xls/ir:value",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "boolean_simplification_pass_test",
    srcs = ["boolean_simplification_pass_test.cc"],
    deps = [
        ":boolean_simplification_pass",
        ":dce_pass",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/ir:bits_ops",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "concat_simplification_pass_test",
    srcs = ["concat_simplification_pass_test.cc"],
    deps = [
        ":bdd_cse_pass",
        ":bdd_simplification_pass",
        ":bit_slice_simplification_pass",
        ":concat_simplification_pass",
        ":dce_pass",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "//xls/ir:value",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "tuple_simplification_pass_test",
    srcs = ["tuple_simplification_pass_test.cc"],
    deps = [
        ":dce_pass",
        ":tuple_simplification_pass",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "strength_reduction_pass_test",
    srcs = ["strength_reduction_pass_test.cc"],
    deps = [
        ":dce_pass",
        ":strength_reduction_pass",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "ternary_query_engine_test",
    srcs = ["ternary_query_engine_test.cc"],
    deps = [
        ":ternary_evaluator",
        ":ternary_query_engine",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "select_simplification_pass_test",
    srcs = ["select_simplification_pass_test.cc"],
    deps = [
        ":select_simplification_pass",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "ternary_evaluator_test",
    srcs = ["ternary_evaluator_test.cc"],
    deps = [
        ":ternary_evaluator",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "dce_pass_test",
    srcs = ["dce_pass_test.cc"],
    deps = [
        ":dce_pass",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "post_dominator_analysis_test",
    srcs = ["post_dominator_analysis_test.cc"],
    deps = [
        ":post_dominator_analysis",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "bdd_function_test",
    srcs = ["bdd_function_test.cc"],
    tags = ["optonly"],
    deps = [
        ":bdd_function",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/examples:sample_packages",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:random_value",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "bdd_query_engine_test",
    srcs = ["bdd_query_engine_test.cc"],
    deps = [
        ":bdd_query_engine",
        "@com_google_absl//absl/container:inlined_vector",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/data_structures:binary_decision_diagram",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "query_engine_test",
    srcs = ["query_engine_test.cc"],
    deps = [
        ":bdd_query_engine",
        ":query_engine",
        ":ternary_evaluator",
        ":ternary_query_engine",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits_ops",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "bdd_simplification_pass_test",
    srcs = ["bdd_simplification_pass_test.cc"],
    deps = [
        ":bdd_cse_pass",
        ":bdd_simplification_pass",
        ":bit_slice_simplification_pass",
        ":concat_simplification_pass",
        ":dce_pass",
        ":select_simplification_pass",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "narrowing_pass_test",
    srcs = ["narrowing_pass_test.cc"],
    deps = [
        ":narrowing_pass",
        ":pass_base",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "dfe_pass_test",
    srcs = ["dfe_pass_test.cc"],
    deps = [
        ":dfe_pass",
        ":pass_base",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "bdd_cse_pass_test",
    srcs = ["bdd_cse_pass_test.cc"],
    deps = [
        ":bdd_cse_pass",
        ":pass_base",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "reassociation_pass_test",
    srcs = ["reassociation_pass_test.cc"],
    deps = [
        ":reassociation_pass",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "array_simplification_pass_test",
    srcs = ["array_simplification_pass_test.cc"],
    deps = [
        ":array_simplification_pass",
        ":constant_folding_pass",
        ":dce_pass",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_parser",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "table_switch_pass_test",
    srcs = ["table_switch_pass_test.cc"],
    deps = [
        ":pass_base",
        ":table_switch_pass",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:ir_matcher",
        "//xls/ir:ir_parser",
        "//xls/ir:ir_test_base",
        "@com_google_googletest//:gtest",
    ],
)
