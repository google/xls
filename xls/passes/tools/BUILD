# Copyright 2025 The XLS Authors
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

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@protobuf//bazel:proto_library.bzl", "proto_library")
load("@protobuf//bazel:py_proto_library.bzl", "py_proto_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//xls/build_rules:py_oss_defs.bzl", "pytype_strict_binary")
load("//xls/build_rules:xls_build_defs.bzl", "xls_diff_test")
load(
    "//xls/build_rules:xls_pass_rules.bzl",
    "xls_pass",
    "xls_pass_registry",
)
load(
    "//xls/passes/tools:generate_documentation_rule.bzl",
    "xls_generate_documentation",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

proto_library(
    name = "pass_documentation_proto",
    srcs = ["pass_documentation.proto"],
)

py_proto_library(
    name = "pass_documentation_py_pb2",
    deps = [":pass_documentation_proto"],
)

cc_proto_library(
    name = "pass_documentation_cc_proto",
    deps = [":pass_documentation_proto"],
)

exports_files([
    "pass_registration.cc.tmpl",
    "pipeline_registration.cc.tmpl",
])

# This generates the passes_list.md file.
xls_generate_documentation(
    name = "rebuild_documentation",
    codelink_format = "http://github.com/google/xls/tree/main%s",
    # Specifically generate for the oss passes regardless of what the default passes are.
    passes = "//xls/passes:oss_optimization_passes",
    tags = [
        "noasan",
        "nomsan",
    ],
    template = "passes.md.tmpl",
)

# TODO(allight): Making this runnable with asan/msan would be nice but not particularly important.
xls_diff_test(
    name = "passes_md_up_to_date_test",
    failure_message = """Documentation passes_list.md is out of date.

Please update it using dev_tools/rebuild_golden_files.sh.""",
    file = ":rebuild_documentation",
    golden = "//docs_src:passes_list.md",
    tags = [
        "noasan",
        "nomsan",
    ],
)

####################################################
# Implementation tools for 'rebuild_documentation' #
####################################################

bzl_library(
    name = "generate_documentation_rule_bzl",
    srcs = ["generate_documentation_rule.bzl"],
    visibility = ["//xls:xls_internal"],
    deps = [
        "//xls/build_rules:xls_providers_bzl",
        "//xls/build_rules:xls_utilities_bzl",
        "@rules_cc//cc/common",
    ],
)

# NB This is intentionally a library with a 'main' etc since we want to link in
# the passes later.
# DO NOT CHANGE TO A CC_BINARY!
cc_library(
    name = "generate_documentation_proto_main",
    srcs = ["generate_documentation_proto_main.cc"],
    deps = [
        ":generate_documentation_proto",
        ":pass_documentation_cc_proto",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        # NB The pass registry is intentionally not included. The regenerate
        # documentation rule will link the appropriate pass registry to ensure
        # we get the expected documentation list out.
        # "//xls/passes",
        "//xls/passes:optimization_pass_registry",
    ],
    alwayslink = True,
)

cc_library(
    name = "generate_documentation_proto",
    srcs = ["generate_documentation_proto.cc"],
    hdrs = ["generate_documentation_proto.h"],
    data = ["@llvm-project//clang:builtin_headers_gen"],
    deps = [
        ":pass_documentation_cc_proto",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/passes:optimization_pass",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:vlog_is_on",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
        "@llvm-project//clang:ast",
        "@llvm-project//clang:ast_matchers",
        "@llvm-project//clang:basic",
        "@llvm-project//clang:frontend",
        "@llvm-project//clang:tooling",
        "@llvm-project//llvm:Support",
    ],
)

pytype_strict_binary(
    name = "generate_documentation_md",
    srcs = ["generate_documentation_md.py"],
    deps = [
        ":pass_documentation_py_pb2",
        "//xls/passes:optimization_pass_pipeline_py_pb2",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
        "@xls_pip_deps//jinja2",
    ],
)

cc_library(
    name = "passes_profile",
    srcs = ["passes_profile.cc"],
    hdrs = ["passes_profile.h"],
    deps = [
        "//xls/common:module_initializer",
        "//xls/common:stopwatch",
        "//xls/common/file:filesystem",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@abseil-cpp//absl/types:span",
        "@cppitertools",
        "@pprof//:profile_cc_proto",
    ],
)

xls_pass(
    name = "test_multi_header_pass",
    srcs = ["test_multi_header_pass.cc"],
    hdrs = [
        "test_multi_header_helper.h",
        "test_multi_header_pass.h",
    ],
    pass_class = "xls::TestMultiHeaderPass",
    visibility = ["//visibility:private"],
    deps = [
        "//xls/ir",
        "//xls/passes:optimization_pass",
        "//xls/passes:pass_base",
        "@abseil-cpp//absl/status:statusor",
    ],
)

xls_pass_registry(
    name = "test_multi_header_passes",
    passes = [":test_multi_header_pass"],
    visibility = ["//visibility:private"],
)

xls_generate_documentation(
    name = "test_multi_header_documentation",
    codelink_format = "https://example.com%s",
    passes = ":test_multi_header_passes",
    tags = [
        "noasan",
        "nomsan",
    ],
    template = "passes.md.tmpl",
    visibility = ["//visibility:private"],
)

cc_test(
    name = "generate_documentation_proto_test",
    srcs = ["generate_documentation_proto_test.cc"],
    data = [":test_multi_header_documentation"],
    tags = [
        "noasan",
        "nomsan",
    ],
    deps = [
        ":generate_documentation_proto",
        ":pass_documentation_cc_proto",
        "//xls/common:proto_test_utils",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/file:temp_directory",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/passes:optimization_pass",
        "//xls/passes:optimization_pass_registry",
        "//xls/passes:pass_base",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest",
    ],
)
