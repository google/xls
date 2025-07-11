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

# cc_proto_library is used in this file
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@bazel_skylib//rules:diff_test.bzl", "diff_test")

# pytype binary only
load("@rules_python//python:proto.bzl", "py_proto_library")
load("@xls_pip_deps//:requirements.bzl", "requirement")

# Load proto_library
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
#
# TODO(https://github.com/google/xls/issues/2556) This has complicated
# requirements on libcxx being from the same version of llvm as the clang
# depenedencies. Internally at google this works automatically but in OSS this
# doesn't work.
xls_generate_documentation(
    name = "rebuild_documentation",
    codelink_format = "http://github.com/google/xls/tree/main%s",
    # Specifically generate for the oss passes regardless of what the default passes are.
    passes = "//xls/passes:oss_optimization_passes",
    tags = [
        "manual",
        "noasan",
        "nomsan",
    ],  # Requirements on libcxx being explicit in include path.
    template = "passes.md.tmpl",
)

_DOC_PREFIX = ""

_DOC_PATH = "docs_src/passes_list.md"

_BUILD_CMD = "bazelisk"

_BUILD_BIN = "bazel-bin"

_TARGET = "//xls/passes/tools:rebuild_documentation"

_GOLD_PATH = "xls/passes/tools/passes_documentation.rebuild_documentation.md"

_FAILURE_MESSAGE = """Documentation passes_list.md is out of date. Please update it.


To Update run:

```
$ {BUILD_CMD} build {TARGET}
$ cp {BUILD_BIN}/{GOLD_PATH} {DOC_PREFIX}{DOC_PATH}/passes_list.md
```

And update the CL.
""".format(
    BUILD_BIN = _BUILD_BIN,
    BUILD_CMD = _BUILD_CMD,
    DOC_PATH = _DOC_PATH,
    DOC_PREFIX = _DOC_PREFIX,
    GOLD_PATH = _GOLD_PATH,
    TARGET = _TARGET,
)

# TODO(allight): Running this automatically would be nice.
# TODO(allight): Making this runnable with asan/msan would be nice but not particularly important.
diff_test(
    name = "passes_md_up_to_date_test",
    failure_message = _FAILURE_MESSAGE,
    file1 = ":rebuild_documentation",
    file2 = "//docs_src:passes_list.md",
    tags = [
        "manual",
        "noasan",
        "nomsan",
    ],  # Requirements on libcxx being explicit in include path.
)

####################################################
# Implementation tools for 'rebuild_documentation' #
####################################################

bzl_library(
    name = "generate_documentation_rule_bzl",
    srcs = ["generate_documentation_rule.bzl"],
    visibility = ["//xls:xls_internal"],
    deps = ["//xls/build_rules:xls_providers_bzl"],
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
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        # NB The pass registry is intentionally not included. The regenerate
        # documentation rule will link the appropriate pass registry to ensure
        # we get the expected documentation list out.
        # "//xls/passes",
        "//xls/passes:optimization_pass_registry",
    ],
    alwayslink = True,
)

# TODO(https://github.com/google/xls/issues/2556) This has complicated
# requirements on libcxx being from the same version of llvm as the clang
# depenedencies. Internally at google this works automatically but in OSS this
# doesn't work. If we could add libcxx headers to data like the clang
# builtin_headers this would be simple enough to implement.
cc_library(
    name = "generate_documentation_proto",
    srcs = ["generate_documentation_proto.cc"],
    hdrs = ["generate_documentation_proto.h"],
    data = ["@llvm-project//clang:builtin_headers_gen"],
    deps = [
        ":pass_documentation_cc_proto",
        "//xls/common:iterator_range",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/passes:optimization_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//clang:ast",
        "@llvm-project//clang:ast_matchers",
        "@llvm-project//clang:basic",
        "@llvm-project//clang:frontend",
        "@llvm-project//clang:tooling",
        "@llvm-project//llvm:Support",
    ],
)

py_binary(
    name = "generate_documentation_md",
    srcs = ["generate_documentation_md.py"],
    deps = [
        ":pass_documentation_py_pb2",
        requirement("Jinja2"),
        requirement("MarkupSafe"),
        "//xls/passes:optimization_pass_pipeline_py_pb2",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
    ],
)
