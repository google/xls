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

# Common utilities shared among XLA subfolders.

# pytype test and library

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "bits_util",
    hdrs = ["bits_util.h"],
    deps = [
        "//xls/common/logging",
    ],
)

cc_library(
    name = "proto_adaptor_utils",
    hdrs = ["proto_adaptor_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "proto_adaptor_utils_test",
    srcs = ["proto_adaptor_utils_test.cc"],
    deps = [
        ":proto_adaptor_utils",
        ":xls_gunit_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "case_converters",
    srcs = ["case_converters.cc"],
    hdrs = ["case_converters.h"],
    deps = [
        "@com_google_absl//absl/strings",
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
        ":xls_gunit_main",
        "@com_google_absl//absl/base",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "indent",
    srcs = ["indent.cc"],
    hdrs = ["indent.h"],
    deps = [
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
        "//xls/common/logging:vlog_is_on",
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
        ":xls_gunit_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "module_initializer",
    srcs = ["module_initializer.inc"],
    hdrs = ["module_initializer.h"],
)

cc_test(
    name = "module_initializer_test",
    srcs = ["module_initializer_test.cc"],
    deps = [
        ":module_initializer",
        ":xls_gunit_main",
        "@com_google_googletest//:gtest",
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
        ":xls_gunit_main",
        "@com_google_googletest//:gtest",
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
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common/file:file_descriptor",
        "//xls/common/logging",
        "//xls/common/logging:log_lines",
        "//xls/common/status:status_macros",
    ],
)

cc_test(
    name = "subprocess_test",
    srcs = ["subprocess_test.cc"],
    deps = [
        ":subprocess",
        ":xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "strerror",
    srcs = ["strerror.cc"],
    hdrs = ["strerror.h"],
    deps = ["@com_google_absl//absl/strings:str_format"],
)

cc_test(
    name = "strerror_test",
    srcs = ["strerror_test.cc"],
    deps = [
        ":strerror",
        ":xls_gunit_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "strong_int",
    hdrs = ["strong_int.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "string_to_int",
    srcs = ["string_to_int.cc"],
    hdrs = ["string_to_int.h"],
    deps = [
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
    ],
)

cc_test(
    name = "string_to_int_test",
    srcs = ["string_to_int_test.cc"],
    deps = [
        ":string_to_int",
        ":xls_gunit_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "strong_int_test",
    srcs = ["strong_int_test.cc"],
    deps = [
        ":strong_int",
        ":xls_gunit_main",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/hash:hash_testing",
        "@com_google_googletest//:gtest",
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
        ":xls_gunit_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "thread",
    srcs = ["thread.inc"],
    hdrs = ["thread.h"],
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
        ":xls_gunit_main",
        "@com_google_absl//absl/types:variant",
        "@com_google_googletest//:gtest",
    ],
)

py_library(
    name = "xls_error",
    srcs = ["xls_error.py"],
)

py_library(
    name = "memoize",
    srcs = ["memoize.py"],
    srcs_version = "PY2AND3",
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
    srcs_version = "PY3",
    deps = [
        "@rules_python//python/runfiles",
    ],
)

py_library(
    name = "multiprocess",
    srcs = ["multiprocess.py"],
    srcs_version = "PY3",
    deps = ["@com_google_absl_py//absl:app"],
)

py_test(
    name = "runfiles_test",
    srcs = ["runfiles_test.py"],
    data = [
        "testdata/bar.txt",
        "testdata/foo.txt",
    ],
    python_version = "PY3",
    deps = [
        ":runfiles",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "test_base",
    srcs = ["test_base.py"],
    srcs_version = "PY2AND3",
    deps = [
        "@com_google_absl_py//absl/testing:absltest",
        "//xls/common/python:init_xls",
    ],
)

cc_library(
    name = "update_golden_files",
    testonly = True,
    hdrs = ["update_golden_files.inc"],
)

py_library(
    name = "gfile",
    srcs = ["gfile.py"],
)

py_library(
    name = "check_simulator",
    srcs = ["check_simulator.py"],
)

py_library(
    name = "revision",
    srcs = ["revision.py"],
)

cc_library(
    name = "xls_gunit_main",
    testonly = True,
    srcs = ["xls_gunit_main.cc"],
    visibility = ["//xls:xls_utility_users"],
    deps = [
        ":init_xls",
        "@com_google_googletest//:gtest",
    ],
)
