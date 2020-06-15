# Copyright 2020 Google LLC
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

load("@xls_pip_deps//:requirements.bzl", "requirement")

# pytype binary and test
load("//xls/examples:list_filegroup_files.bzl", "list_filegroup_files")
load("@bazel_skylib//rules:build_test.bzl", "build_test")

# cc_proto_library is used in this file
load("@io_bazel_rules_closure//closure:defs.bzl", "closure_js_binary", "closure_js_library")

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "ir_examples",
    srcs = [
        "//xls/examples:ir_examples",
        "//xls/modules:ir_examples",
    ],
)

list_filegroup_files(
    name = "ir_examples_file_list",
    src = ":ir_examples",
    out = "ir_examples_file_list.txt",
)

py_binary(
    name = "app",
    srcs = ["app.py"],
    data = glob([
        "templates/*.tmpl",
    ]) + [
        "ir_examples_file_list.txt",
        ":ir_examples",
        ":js_compiled",
        "ir_graph.css",
        "third_party_js.txt",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("Flask"),
        requirement("Jinja2"),
        requirement("Werkzeug"),
        requirement("click"),
        requirement("itsdangerous"),
        "//xls/common:runfiles",
        "//xls/common/python:init_xls",
        "//xls/ir/visualization/python:ir_to_json",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)

# A test which tests the UI is a lot of work to make. For now, at least test
# that the visualization app builds.
build_test(
    name = "app_build_test",
    targets = [":app"],
)

cc_proto_library(
    name = "ir_for_visualization_cc_proto",
    deps = [":ir_for_visualization_proto"],
)

proto_library(
    name = "ir_for_visualization_proto",
    srcs = ["ir_for_visualization.proto"],
)

cc_library(
    name = "ir_to_json",
    srcs = ["ir_to_json.cc"],
    hdrs = ["ir_to_json.h"],
    deps = [
        ":ir_for_visualization_cc_proto",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor",
        "//xls/delay_model:analyze_critical_path",
        "//xls/delay_model:delay_estimator",
        "//xls/delay_model:delay_estimators",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:op",
        "//xls/passes:bdd_query_engine",
        "//xls/scheduling:pipeline_schedule",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "ir_to_json_test",
    srcs = ["ir_to_json_test.cc"],
    deps = [
        ":ir_to_json",
        "//xls/common/logging",
        "//xls/common/status:matchers",
        "//xls/delay_model:delay_estimator",
        "//xls/delay_model:delay_estimators",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

closure_js_library(
    name = "js_lib",
    srcs = glob(["*.js"]),
    # Turn off compiler checks because the compiler is unhappy with jquery
    # (e.g., '$' is undefined). Compiler checks are enabled when building
    # 'js_compiled'.
    suppress = [
        "JSC_STRICT_INEXISTENT_PROPERTY",
        "JSC_UNKNOWN_EXPR_TYPE",
        "JSC_DEPRECATED_PROP_REASON",
        "JSC_UNDEFINED_VARIABLE",
        "JSC_POSSIBLE_INEXISTENT_PROPERTY",
        "JSC_UNRECOGNIZED_TYPE_ERROR",
        "JSC_UNUSED_LOCAL_ASSIGNMENT",
    ],
)

closure_js_binary(
    name = "js_compiled",
    compilation_level = "SIMPLE_OPTIMIZATIONS",
    entry_points = ["xls.IrVisualization"],
    deps = [
        ":js_lib",
    ],
)
