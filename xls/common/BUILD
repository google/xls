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
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

# Load proto_library
# cc_proto_library is used in this file

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
    name = "timeout_support",
    srcs = ["timeout_support.cc"],
    hdrs = ["timeout_support.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "iter_util",
    hdrs = ["iter_util.h"],
    deps = ["@com_google_absl//absl/types:span"],
)

cc_test(
    name = "iter_util_test",
    srcs = ["iter_util_test.cc"],
    deps = [
        ":iter_util",
        ":xls_gunit_main",
        "@com_google_absl//absl/types:span",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "bits_util",
    hdrs = ["bits_util.h"],
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "random_util",
    hdrs = ["random_util.h"],
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/random:bit_gen_ref",
    ],
)

cc_library(
    name = "proto_adaptor_utils",
    hdrs = ["proto_adaptor_utils.h"],
)

cc_library(
    name = "proto_test_utils",
    testonly = True,
    hdrs = ["proto_test_utils.h"],
    deps = [
        "@com_google_absl//absl/strings:str_format",
        "@com_google_protobuf//:differencer",
        "@com_google_protobuf//:protobuf",
        "@googletest//:gtest",
    ],
)

proto_library(
    name = "matcher_test_messages_proto",
    srcs = ["testdata/matcher_test_messages.proto"],
)

cc_proto_library(
    name = "matcher_test_messages_cc_proto",
    deps = [":matcher_test_messages_proto"],
)

cc_test(
    name = "proto_test_utils_test",
    srcs = ["proto_test_utils_test.cc"],
    deps = [
        ":matcher_test_messages_cc_proto",
        ":proto_test_utils",
        ":xls_gunit_main",
        "@com_google_protobuf//:protobuf",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "proto_adaptor_utils_test",
    srcs = ["proto_adaptor_utils_test.cc"],
    deps = [
        ":proto_adaptor_utils",
        ":xls_gunit_main",
        "@googletest//:gtest",
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
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "xls_gunit_sample",
    srcs = ["xls_gunit_sample.cc"],
    deps = [
        ":xls_gunit_main",
        "@googletest//:gtest",
    ],
)

py_test(
    name = "xls_gunit_sample_test",
    srcs = ["xls_gunit_sample_test.py"],
    data = [":xls_gunit_sample"],
    deps = [
        ":runfiles",
        "@abseil-py//absl/testing:absltest",
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
    name = "gunit_init_xls",
    testonly = True,
    srcs = ["gunit_init_xls.cc"],
    hdrs = ["gunit_init_xls.h"],
    deps = [
        ":init_xls",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "benchmark_support",
    testonly = True,
    srcs = ["benchmark_support.cc"],
    hdrs = ["benchmark_support.h"],
    visibility = ["//xls:xls_utility_users"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@google_benchmark//:benchmark",
    ],
)

cc_library(
    name = "init_xls",
    srcs = ["init_xls.cc"],
    hdrs = [
        "build_embed.h",
        "init_xls.h",
    ],
    linkstamp = "build_embed.cc",
    visibility = ["//xls:xls_utility_users"],
    deps = [
        "@com_google_absl//absl/flags:config",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/flags:usage",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:flags",
        "@com_google_absl//absl/log:initialize",
    ],
)

cc_library(
    name = "exit_status",
    srcs = ["exit_status.cc"],
    hdrs = ["exit_status.h"],
    visibility = ["//xls:xls_utility_users"],
    deps = ["@com_google_absl//absl/status"],
)

cc_library(
    name = "math_util",
    srcs = ["math_util.cc"],
    hdrs = ["math_util.h"],
    visibility = ["//xls:xls_utility_users"],
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "math_util_test",
    srcs = ["math_util_test.cc"],
    deps = [
        ":math_util",
        ":xls_gunit_main",
        "//xls/common/fuzzing:fuzztest",
        "@com_google_absl//absl/base:core_headers",
        "@googletest//:gtest",
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
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "source_location",
    hdrs = ["source_location.h"],
)

cc_test(
    name = "source_location_test",
    size = "small",
    srcs = ["source_location_test.cc"],
    deps = [
        ":source_location",
        ":xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_binary(
    name = "subprocess_helper",
    srcs = ["subprocess_helper.cc"],
)

cc_library(
    name = "subprocess",
    srcs = ["subprocess.cc"],
    hdrs = ["subprocess.h"],
    data = [":subprocess_helper"],
    deps = [
        ":strerror",
        ":thread",
        "//xls/common/file:file_descriptor",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging:log_lines",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "subprocess_test",
    srcs = ["subprocess_test.cc"],
    deps = [
        ":subprocess",
        ":xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@googletest//:gtest",
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
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "strong_int",
    hdrs = ["strong_int.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/meta:type_traits",
    ],
)

cc_library(
    name = "string_to_int",
    srcs = ["string_to_int.cc"],
    hdrs = ["string_to_int.h"],
    deps = [
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "string_to_int_test",
    srcs = ["string_to_int_test.cc"],
    deps = [
        ":string_to_int",
        ":xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@googletest//:gtest",
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
        "@googletest//:gtest",
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
        "@googletest//:gtest",
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
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "type_traits_utils",
    hdrs = ["type_traits_utils.h"],
)

cc_test(
    name = "type_traits_utils_test",
    srcs = ["type_traits_utils_test.cc"],
    deps = [
        ":type_traits_utils",
        ":xls_gunit_main",
        "@com_google_absl//absl/types:span",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "to_string_utils",
    srcs = ["to_string_utils.cc"],
    hdrs = ["to_string_utils.h"],
    deps = [
        ":type_traits_utils",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "to_string_utils_test",
    srcs = ["to_string_utils_test.cc"],
    deps = [
        ":to_string_utils",
        ":xls_gunit_main",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "comparison_utils",
    hdrs = ["comparison_utils.h"],
    deps = [
        ":to_string_utils",
        ":type_traits_utils",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "comparison_utils_test",
    srcs = ["comparison_utils_test.cc"],
    deps = [
        ":comparison_utils",
        ":xls_gunit_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "golden_files_test",
    srcs = ["golden_files_test.cc"],
    data = glob([
        "testdata/golden_files_*",
    ]),
    deps = [
        ":golden_files",
        ":xls_gunit_main",
        "@com_google_absl//absl/strings:str_format",
        "@googletest//:gtest",
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
)

cc_library(
    name = "test_macros",
    hdrs = ["test_macros.h"],
)

py_library(
    name = "runfiles",
    srcs = ["runfiles.py"],
    deps = [
        "@rules_python//python/runfiles",
    ],
)

py_library(
    name = "multiprocess",
    srcs = ["multiprocess.py"],
    deps = ["@abseil-py//absl:app"],
)

py_test(
    name = "runfiles_test",
    srcs = ["runfiles_test.py"],
    data = [
        "testdata/bar.txt",
        "testdata/foo.txt",
    ],
    deps = [
        ":runfiles",
        "@abseil-py//absl/testing:absltest",
    ],
)

py_library(
    name = "test_base",
    srcs = ["test_base.py"],
    deps = ["@abseil-py//absl/testing:absltest"],
)

cc_library(
    name = "update_golden_files_flag",
    testonly = True,
    hdrs = ["update_golden_files_flag.inc"],
    deps = ["@com_google_absl//absl/flags:flag"],
)

py_library(
    name = "gfile",
    srcs = ["gfile.py"],
)

cc_library(
    name = "revision",
    srcs = ["revision.cc"],
    hdrs = ["revision.h"],
)

cc_library(
    name = "xls_gunit_main",
    testonly = True,
    srcs = ["xls_gunit_main.cc"],
    visibility = ["//xls:xls_utility_users"],
    deps = [
        ":benchmark_support",
        ":gunit_init_xls",
        "@com_google_fuzztest//fuzztest:init_fuzztest",
        "@googletest//:gtest",
    ],
    alwayslink = True,
)

cc_library(
    name = "golden_files",
    testonly = True,
    srcs = ["golden_files.cc"],
    hdrs = ["golden_files.h"],
    deps = [
        ":source_location",
        ":update_golden_files_flag",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:matchers",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "endian",
    hdrs = ["endian.h"],
    deps = ["@com_google_absl//absl/base:config"],
)

cc_library(
    name = "stopwatch",
    srcs = ["stopwatch.cc"],
    hdrs = ["stopwatch.h"],
    deps = [
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "stopwatch_test",
    srcs = ["stopwatch_test.cc"],
    deps = [
        ":stopwatch",
        ":xls_gunit_main",
        "//xls/common/fuzzing:fuzztest",
        "@com_google_absl//absl/time",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "undeclared_outputs",
    testonly = True,
    srcs = ["undeclared_outputs.cc"],
    hdrs = ["undeclared_outputs.h"],
)
