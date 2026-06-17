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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "union_find_map",
    hdrs = ["union_find_map.h"],
    deps = [
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log:check",
    ],
)

cc_test(
    name = "union_find_map_test",
    srcs = ["union_find_map_test.cc"],
    deps = [
        ":union_find_map",
        "//xls/common:xls_gunit_main",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "graph_contraction",
    hdrs = ["graph_contraction.h"],
    deps = [
        ":union_find_map",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log:check",
    ],
)

cc_test(
    name = "graph_contraction_test",
    srcs = ["graph_contraction_test.cc"],
    deps = [
        ":graph_contraction",
        "//xls/common:xls_gunit_main",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/types:variant",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "graph_coloring",
    hdrs = ["graph_coloring.h"],
    deps = [
        "@abseil-cpp//absl/container:btree",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/strings:str_format",
        "@z3//:api",
    ],
)

cc_test(
    name = "graph_coloring_test",
    srcs = ["graph_coloring_test.cc"],
    deps = [
        ":graph_coloring",
        "//xls/common:xls_gunit_main",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "transitive_closure",
    srcs = ["transitive_closure.cc"],
    hdrs = ["transitive_closure.h"],
    deps = [
        ":inline_bitmap",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_test(
    name = "transitive_closure_test",
    srcs = ["transitive_closure_test.cc"],
    deps = [
        ":inline_bitmap",
        ":transitive_closure",
        "//xls/common:math_util",
        "//xls/common:xls_gunit_main",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/random:bit_gen_ref",
        "@abseil-cpp//absl/random:distributions",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@google_benchmark//:benchmark",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "inline_bitmap",
    srcs = ["inline_bitmap.cc"],
    hdrs = ["inline_bitmap.h"],
    deps = [
        "//xls/common:bits_util",
        "//xls/common:math_util",
        "@abseil-cpp//absl/base",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_prod",
    ],
)

cc_library(
    name = "leaf_type_tree",
    srcs = ["leaf_type_tree.cc"],
    hdrs = ["leaf_type_tree.h"],
    deps = [
        "//xls/common:math_util",
        "//xls/common:visitor",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir:type",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_library(
    name = "algorithm",
    hdrs = ["algorithm.h"],
    deps = [
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_library(
    name = "binary_decision_diagram",
    srcs = ["binary_decision_diagram.cc"],
    hdrs = ["binary_decision_diagram.h"],
    deps = [
        ":inline_bitmap",
        "//xls/common:math_util",
        "//xls/common:strong_int",
        "//xls/common/status:ret_check",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/hash",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/log:vlog_is_on",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
        "@cppitertools",
    ],
)

cc_library(
    name = "binary_search",
    srcs = ["binary_search.cc"],
    hdrs = ["binary_search.h"],
    deps = [
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
    ],
)

cc_library(
    name = "strongly_connected_components",
    hdrs = ["strongly_connected_components.h"],
    deps = [
        "@abseil-cpp//absl/container:btree",
    ],
)

cc_library(
    name = "submodular",
    hdrs = ["submodular.h"],
    deps = [
        "@abseil-cpp//absl/container:btree",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/random:distributions",
    ],
)

cc_library(
    name = "union_find",
    hdrs = ["union_find.h"],
    deps = [
        ":union_find_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/types:variant",
    ],
)

cc_test(
    name = "inline_bitmap_test",
    srcs = ["inline_bitmap_test.cc"],
    deps = [
        ":inline_bitmap",
        "//xls/common:xls_gunit_main",
        "//xls/common/fuzzing:fuzztest",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "@abseil-cpp//absl/types:span",
        "@cppitertools",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "leaf_type_tree_test",
    srcs = ["leaf_type_tree_test.cc"],
    deps = [
        ":leaf_type_tree",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:ir_parser",
        "//xls/ir:type",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "algorithm_test",
    srcs = ["algorithm_test.cc"],
    deps = [
        ":algorithm",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "binary_decision_diagram_test",
    srcs = ["binary_decision_diagram_test.cc"],
    deps = [
        ":binary_decision_diagram",
        "//xls/common:xls_gunit_main",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status:status_matchers",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "binary_search_test",
    srcs = ["binary_search_test.cc"],
    deps = [
        ":binary_search",
        "//xls/common:xls_gunit_main",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "strongly_connected_components_test",
    srcs = ["strongly_connected_components_test.cc"],
    deps = [
        ":strongly_connected_components",
        "//xls/common:xls_gunit_main",
        "@abseil-cpp//absl/container:btree",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "submodular_test",
    srcs = ["submodular_test.cc"],
    deps = [
        ":submodular",
        "//xls/common:xls_gunit_main",
        "@abseil-cpp//absl/container:btree",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "union_find_test",
    srcs = ["union_find_test.cc"],
    deps = [
        ":union_find",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)
