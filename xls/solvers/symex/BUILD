# Copyright 2026 The XLS Authors
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

load(
    "@rules_cc//cc:cc_library.bzl",
    "cc_library",
)
load(
    "@rules_cc//cc:cc_test.bzl",
    "cc_test",
)
load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_dslx_ir",
    "xls_dslx_library",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "symbolic_path",
    srcs = ["symbolic_path.cc"],
    hdrs = ["symbolic_path.h"],
    deps = [
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/solvers:z3_utils",
        "@abseil-cpp//absl/cleanup",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
        "@z3//:api",
    ],
)

cc_test(
    name = "symbolic_path_test",
    srcs = ["symbolic_path_test.cc"],
    deps = [
        ":symbolic_path",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:value",
        "//xls/solvers:z3_utils",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest",
        "@z3//:api",
    ],
)

cc_library(
    name = "z3_semantics_encoder",
    srcs = ["z3_semantics_encoder.cc"],
    hdrs = ["z3_semantics_encoder.h"],
    deps = [
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:op",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/solvers:z3_utils",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@z3//:api",
    ],
)

cc_test(
    name = "z3_semantics_encoder_test",
    srcs = ["z3_semantics_encoder_test.cc"],
    deps = [
        ":z3_semantics_encoder",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:function_builder",
        "//xls/ir:value",
        "//xls/solvers:z3_utils",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest",
        "@z3//:api",
    ],
)

cc_library(
    name = "symex_engine",
    hdrs = ["symex_engine.h"],
    deps = [
        ":symbolic_path",
        "//xls/ir",
        "@abseil-cpp//absl/status:statusor",
        "@z3//:api",
    ],
)

cc_library(
    name = "cfg_symex_engine",
    srcs = ["cfg_symex_engine.cc"],
    hdrs = ["cfg_symex_engine.h"],
    deps = [
        ":symbolic_path",
        ":symex_engine",
        ":z3_semantics_encoder",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/solvers:z3_utils",
        "@abseil-cpp//absl/cleanup",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
        "@z3//:api",
    ],
)

cc_test(
    name = "cfg_symex_engine_test",
    srcs = ["cfg_symex_engine_test.cc"],
    deps = [
        ":cfg_symex_engine",
        ":symex_engine",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
        "@z3//:api",
    ],
)

cc_test(
    name = "symex_e2e_test",
    srcs = ["symex_e2e_test.cc"],
    data = [
        ":testdata/execute_alu.ir",
    ],
    deps = [
        ":cfg_symex_engine",
        ":symbolic_path",
        ":symex_engine",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest",
        "@z3//:api",
    ],
)

xls_dslx_library(
    name = "execute_alu_dslx",
    srcs = ["testdata/execute_alu.x"],
)

xls_dslx_ir(
    name = "execute_alu_ir",
    dslx_top = "execute_alu",
    ir_file = "testdata/execute_alu.ir",
    library = ":execute_alu_dslx",
)
