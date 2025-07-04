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
load("//xls/passes/tools:generate_documentation_rule.bzl", "xls_generate_documentation")

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

cc_binary(
    name = "generate_documentation_proto_main",
    srcs = ["generate_documentation_proto_main.cc"],
    deps = [
        ":generate_documentation_proto",
        ":pass_documentation_cc_proto",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "//xls/passes",
        "//xls/passes:optimization_pass_registry",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "generate_documentation_proto",
    srcs = ["generate_documentation_proto.cc"],
    hdrs = ["generate_documentation_proto.h"],
    deps = [
        ":pass_documentation_cc_proto",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/passes:optimization_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/iterator:range",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
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
    data = [":passes.md.tmpl"],
    deps = [
        ":pass_documentation_py_pb2",
        requirement("Jinja2"),
        requirement("MarkupSafe"),
        "//xls/common:runfiles",
        "//xls/passes:optimization_pass_pipeline_py_pb2",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
    ],
)

xls_generate_documentation(
    name = "rebuild_documentation",
    codelink_format = "http://github.com/google/xls/tree/main%s",
)

bzl_library(
    name = "generate_documentation_rule_bzl",
    srcs = ["generate_documentation_rule.bzl"],
    visibility = ["//visibility:private"],
    deps = ["//xls/build_rules:xls_providers_bzl"],
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

diff_test(
    name = "passes_md_up_to_date_test",
    failure_message = _FAILURE_MESSAGE,
    file1 = ":rebuild_documentation",
    file2 = "//docs_src:passes_list.md",
)
