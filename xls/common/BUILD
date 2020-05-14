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

# Common utilities shared among XLA subfolders.

# pytype library

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "integral_types",
    hdrs = ["integral_types.h"],
)

cc_library(
    name = "bits_util",
    hdrs = ["bits_util.h"],
    deps = [
        "//xls/common/logging",
    ],
)

cc_library(
    name = "casts",
    hdrs = ["casts.h"],
)

cc_test(
    name = "casts_test",
    srcs = ["casts_test.cc"],
    deps = [
        ":casts",
        "@com_google_absl//absl/base",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cleanup",
    hdrs = ["cleanup.h"],
    deps = [
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "indent",
    srcs = ["indent.cc"],
    hdrs = ["indent.h"],
    deps = [
        ":integral_types",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "init_xls",
    srcs = ["init_xls.cc"],
    hdrs = ["init_xls.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "@com_google_absl//absl/flags:parse",
    ],
)

cc_library(
    name = "math_util",
    srcs = ["math_util.cc"],
    hdrs = ["math_util.h"],
    deps = ["//xls/common/logging"],
)

cc_test(
    name = "math_util_test",
    srcs = ["math_util_test.cc"],
    deps = [
        ":math_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "module_initializer",
    hdrs = ["module_initializer.h"],
)

cc_test(
    name = "module_initializer_test",
    srcs = ["module_initializer_test.cc"],
    deps = [
        ":module_initializer",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "source_location",
    hdrs = ["source_location.h"],
    deps = [
        "@com_google_absl//absl/base:config",
    ],
)

cc_test(
    name = "source_location_test",
    size = "small",
    srcs = ["source_location_test.cc"],
    deps = [
        ":source_location",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "subprocess",
    srcs = ["subprocess.cc"],
    hdrs = ["subprocess.h"],
    deps = [
        ":strerror",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common/file:file_descriptor",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:status_macros",
        "//xls/common/status:statusor",
    ],
)

cc_test(
    name = "subprocess_test",
    srcs = ["subprocess_test.cc"],
    deps = [
        ":subprocess",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "strerror",
    srcs = ["strerror.cc"],
    hdrs = ["strerror.h"],
)

cc_test(
    name = "strerror_test",
    srcs = ["strerror_test.cc"],
    deps = [
        ":strerror",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "strong_int",
    hdrs = ["strong_int.h"],
    deps = [
        ":integral_types",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "strong_int_test",
    srcs = ["strong_int_test.cc"],
    deps = [
        ":integral_types",
        ":strong_int",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/hash:hash_testing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "symbolized_stacktrace",
    srcs = ["symbolized_stacktrace.cc"],
    hdrs = ["symbolized_stacktrace.h"],
    deps = [
        "@com_google_absl//absl/debugging:stacktrace",
        "@com_google_absl//absl/debugging:symbolize",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "symbolized_stacktrace_test",
    srcs = ["symbolized_stacktrace_test.cc"],
    deps = [
        ":symbolized_stacktrace",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "visitor",
    hdrs = ["visitor.h"],
)

cc_test(
    name = "visitor_test",
    srcs = ["visitor_test.cc"],
    deps = [
        ":visitor",
        "@com_google_absl//absl/types:variant",
        "@com_google_googletest//:gtest_main",
    ],
)

py_library(
    name = "xls_error",
    srcs = ["xls_error.py"],
)

py_library(
    name = "memoize",
    srcs = ["memoize.py"],
)

cc_library(
    name = "iterator_range",
    hdrs = ["iterator_range.h"],
    deps = [
        "@com_google_absl//absl/base:config",
    ],
)

py_library(
    name = "runfiles",
    srcs = ["runfiles.py"],
    deps = [
        "@rules_python//python/runfiles",
    ],
)

py_library(
    name = "test_base",
    srcs = ["test_base.py"],
    deps = [
        "@com_google_absl_py//absl/testing:absltest",
        "//xls/common/python:init_xls",
    ],
)

py_library(
    name = "gfile",
    srcs = ["gfile.py"],
)
