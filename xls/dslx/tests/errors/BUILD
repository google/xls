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

# Tests for DSLX files that have various forms of errors.

# pytype tests are present in this file
load("//xls/dslx:strip_comments.bzl", "dslx_strip_comments")

package(
    default_applicable_licenses = ["//:license"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

py_test(
    name = "error_modules_test",
    srcs = ["error_modules_test.py"],
    data = glob(["*.x"]) + [
        "//xls/dslx:interpreter_main",
    ],
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
        "@abseil-py//absl/logging",
    ],
)

filegroup(
    name = "x_files",
    srcs = glob(["*.x"]),
    visibility = ["//visibility:private"],
)

dslx_strip_comments(
    name = "x_files_stripped",
    srcs = [":x_files"],
    visibility = ["//xls:xls_internal"],
)
