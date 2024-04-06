# Copyright 2023 The XLS Authors
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

# DSLX module example

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_dslx_library",
    "xls_dslx_test",
)

package(
    default_applicable_licenses = ["//:license"],
    licenses = ["notice"],
)

xls_dslx_library(
    name = "capitalize_module",
    srcs = [
        "capitalize.x",
        "some_caps.x",
        "some_caps_streaming.x",
    ],
)

xls_dslx_test(
    name = "capitalize_tests",
    library = ":capitalize_module",
)

filegroup(
    name = "x_files",
    srcs = glob(["*.x"]),
    visibility = ["//visibility:private"],
)
