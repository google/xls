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

load("@bazel_skylib//rules:build_test.bzl", "build_test")

# pytype binary and test
load("@io_bazel_rules_closure//closure:defs.bzl", "closure_js_binary", "closure_js_library")
load("@xls_pip_deps//:requirements.bzl", "requirement")
load("//xls/examples:list_filegroup_files.bzl", "list_filegroup_files")
# js_binary used in this file
# cc_proto_library is used in this file

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "ir_examples",
    srcs = [
        "//xls/dslx/stdlib:ir_examples",
        "//xls/examples:ir_examples",
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
    data = [
        "ir_examples_file_list.txt",
        "ir_graph.css",
        "templates/splash.tmpl",
        "third_party_js.txt",
        ":ir_examples",
        ":ir_to_json_main",
        ":js_compiled",
        "//xls/tools:opt_main",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("Flask"),
        requirement("Jinja2"),
        requirement("click"),
        requirement("itsdangerous"),
        requirement("MarkupSafe"),
        requirement("Werkzeug"),
        "//xls/common:runfiles",
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
    name = "visualization_cc_proto",
    deps = [":visualization_proto"],
)

proto_library(
    name = "visualization_proto",
    srcs = ["visualization.proto"],
)

cc_library(
    name = "ir_to_proto",
    srcs = ["ir_to_proto.cc"],
    hdrs = ["ir_to_proto.h"],
    deps = [
        ":visualization_cc_proto",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/delay_model:analyze_critical_path",
        "//xls/delay_model:delay_estimator",
        "//xls/ir",
        "//xls/ir:op",
        "//xls/passes:bdd_query_engine",
        "//xls/scheduling:pipeline_schedule",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_test(
    name = "ir_to_proto_test",
    srcs = ["ir_to_proto_test.cc"],
    data = glob([
        "testdata/ir_to_html_test_*",
    ]),
    deps = [
        ":ir_to_proto",
        ":visualization_cc_proto",
        "//xls/common:golden_files",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/delay_model:delay_estimator",
        "//xls/delay_model:delay_estimators",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:scheduling_options",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "ir_to_json",
    srcs = ["ir_to_json.cc"],
    hdrs = ["ir_to_json.h"],
    deps = [
        ":ir_to_proto",
        ":visualization_cc_proto",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimator",
        "//xls/ir",
        "//xls/scheduling:pipeline_schedule",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "ir_to_json_test",
    srcs = ["ir_to_json_test.cc"],
    data = glob([
        "testdata/ir_to_html_test_*",
    ]),
    deps = [
        ":ir_to_json",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/delay_model:delay_estimator",
        "//xls/delay_model:delay_estimators",
        "//xls/ir",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:scheduling_options",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_binary(
    name = "ir_to_json_main",
    srcs = ["ir_to_json_main.cc"],
    deps = [
        ":ir_to_json",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimator",
        "//xls/delay_model:delay_estimators",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:run_pipeline_schedule",
        "//xls/scheduling:scheduling_options",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_binary(
    name = "ir_to_csvs_main",
    srcs = ["ir_to_csvs_main.cc"],
    deps = [
        ":ir_to_proto",
        ":visualization_cc_proto",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/delay_model:delay_estimator",
        "//xls/delay_model:delay_estimators",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:run_pipeline_schedule",
        "//xls/scheduling:scheduling_options",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_riegeli//riegeli/bytes:fd_writer",
        "@com_google_riegeli//riegeli/csv:csv_record",
        "@com_google_riegeli//riegeli/csv:csv_writer",
    ],
)

closure_js_library(
    name = "externs",
    srcs = ["externs.js"],
)

closure_js_library(
    name = "js_lib",
    srcs = [
        "graph_view.js",
        "ir_graph.js",
        "ir_visualizer.js",
        "selectable_graph.js",
        "ui.js",
    ],
    # Turn off unknown expression type compiler checks because the cytoscape
    # externs file is very sparse and most interactions with cytoscape result in
    # an unknown expression type.
    suppress = [
        "JSC_POSSIBLE_INEXISTENT_PROPERTY",
        "JSC_STRICT_INEXISTENT_PROPERTY",
        "JSC_UNKNOWN_EXPR_TYPE",
        "JSC_UNRECOGNIZED_TYPE_ERROR",
    ],
    deps = [":externs"],
)

closure_js_binary(
    name = "js_compiled",
    compilation_level = "SIMPLE_OPTIMIZATIONS",
    entry_points = ["xls.ui"],
    deps = [
        ":js_lib",
    ],
)
