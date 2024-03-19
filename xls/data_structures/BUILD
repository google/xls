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

# Auxiliary data structures.

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "union_find_map",
    hdrs = ["union_find_map.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_test(
    name = "union_find_map_test",
    srcs = ["union_find_map_test.cc"],
    deps = [
        ":union_find_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
    ],
)

cc_library(
    name = "graph_contraction",
    hdrs = ["graph_contraction.h"],
    deps = [
        ":union_find_map",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "//xls/common/logging",
    ],
)

cc_test(
    name = "graph_contraction_test",
    srcs = ["graph_contraction_test.cc"],
    deps = [
        ":graph_contraction",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
    ],
)

cc_library(
    name = "graph_coloring",
    hdrs = ["graph_coloring.h"],
    deps = [
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/logging",
        "@z3//:api",
    ],
)

cc_test(
    name = "graph_coloring_test",
    srcs = ["graph_coloring_test.cc"],
    deps = [
        ":graph_coloring",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
    ],
)

cc_library(
    name = "maximum_clique",
    hdrs = ["maximum_clique.h"],
    deps = [
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/logging",
        "@com_google_ortools//ortools/linear_solver",
    ],
)

cc_test(
    name = "maximum_clique_test",
    srcs = ["maximum_clique_test.cc"],
    deps = [
        ":maximum_clique",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/random:distributions",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
    ],
)

cc_library(
    name = "transitive_closure",
    hdrs = ["transitive_closure.h"],
    deps = [
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/logging",
    ],
)

cc_test(
    name = "transitive_closure_test",
    srcs = ["transitive_closure_test.cc"],
    deps = [
        ":transitive_closure",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
    ],
)

cc_library(
    name = "inline_bitmap",
    hdrs = ["inline_bitmap.h"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "//xls/common:bits_util",
        "//xls/common:endian",
        "//xls/common:math_util",
        "//xls/common:test_macros",
        "//xls/common/logging",
    ],
)

cc_library(
    name = "leaf_type_tree",
    srcs = ["leaf_type_tree.cc"],
    hdrs = ["leaf_type_tree.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir:type",
    ],
)

cc_library(
    name = "algorithm",
    hdrs = ["algorithm.h"],
    deps = [
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "binary_decision_diagram",
    srcs = ["binary_decision_diagram.cc"],
    hdrs = ["binary_decision_diagram.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:strong_int",
        "//xls/common/logging",
    ],
)

cc_library(
    name = "min_cut",
    srcs = ["min_cut.cc"],
    hdrs = ["min_cut.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common:strong_int",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
    ],
)

cc_library(
    name = "path_cut",
    srcs = ["path_cut.cc"],
    hdrs = ["path_cut.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common:strong_int",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
    ],
)

cc_library(
    name = "binary_search",
    srcs = ["binary_search.cc"],
    hdrs = ["binary_search.h"],
    deps = [
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
    ],
)

cc_library(
    name = "strongly_connected_components",
    hdrs = ["strongly_connected_components.h"],
    deps = [
        "@com_google_absl//absl/container:btree",
    ],
)

cc_library(
    name = "submodular",
    hdrs = ["submodular.h"],
    deps = [
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
        "//xls/common/logging",
    ],
)

cc_library(
    name = "union_find",
    hdrs = ["union_find.h"],
    deps = [
        ":union_find_map",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "//xls/common/logging",
    ],
)

cc_test(
    name = "inline_bitmap_test",
    srcs = ["inline_bitmap_test.cc"],
    deps = [
        ":inline_bitmap",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
    ],
)

cc_test(
    name = "leaf_type_tree_test",
    srcs = ["leaf_type_tree_test.cc"],
    deps = [
        ":leaf_type_tree",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:ir_parser",
        "//xls/ir:type",
    ],
)

cc_test(
    name = "algorithm_test",
    srcs = ["algorithm_test.cc"],
    deps = [
        ":algorithm",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
    ],
)

cc_test(
    name = "binary_decision_diagram_test",
    srcs = ["binary_decision_diagram_test.cc"],
    deps = [
        ":binary_decision_diagram",
        "@com_google_absl//absl/container:inlined_vector",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/ir:bits",
        "//xls/ir:format_preference",
    ],
)

cc_test(
    name = "min_cut_test",
    srcs = ["min_cut_test.cc"],
    deps = [
        ":min_cut",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/random:mocking_bit_gen",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common:random_util",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
    ],
)

cc_test(
    name = "path_cut_test",
    srcs = ["path_cut_test.cc"],
    deps = [
        ":path_cut",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:strong_int",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/logging",
    ],
)

cc_test(
    name = "binary_search_test",
    srcs = ["binary_search_test.cc"],
    deps = [
        ":binary_search",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
    ],
)

cc_test(
    name = "strongly_connected_components_test",
    srcs = ["strongly_connected_components_test.cc"],
    deps = [
        ":strongly_connected_components",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
    ],
)

cc_test(
    name = "submodular_test",
    srcs = ["submodular_test.cc"],
    deps = [
        ":submodular",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
    ],
)
