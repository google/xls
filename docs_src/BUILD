# Copyright 2021 The XLS Authors
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

# Targets for documentation files.

# pytype tests are present in this file

load("@xls_pip_deps//:requirements.bzl", "requirement")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(
    ["bazel_rules_macros.md"],
    visibility = ["//xls:xls_internal"],
)

# All files that have a ```dslx ... in them.
filegroup(
    name = "dslx_documentation_files",
    srcs = [
        "dslx_ffi.md",
        "dslx_reference.md",
        "dslx_std.md",
        "dslx_type_system.md",
        "floating_point.md",
    ],
)

filegroup(
    name = "all_markdown",
    srcs = glob(["**/*.md"]),
    data = [
        "//docs_src/tutorials:all_files",
    ],
)

py_test(
    name = "check_nav",
    srcs = ["check_nav.py"],
    data = [
        ":all_markdown",
        "//:mkdocs.yml",
    ],
    deps = [
        requirement("pyyaml"),
        "//xls/common:runfiles",
        "@com_google_absl_py//absl:app",
    ],
)
