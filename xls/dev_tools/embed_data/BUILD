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

# Templates for embeding binary data into a c file. Needs a build file to make it a visible target.

# pytype binary only
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@xls_pip_deps//:requirements.bzl", "requirement")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

py_binary(
    name = "create_source_files",
    srcs = ["create_source_files.py"],
    data = [
        ":embedded_data.cc.tmpl",
        ":embedded_data.h.tmpl",
    ],
    deps = [
        requirement("Jinja2"),
        requirement("MarkupSafe"),
        "//xls/common:runfiles",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
    ],
)

# Compatibility shim for internal cc runtime dependencies. DO NOT REMOVE.
#
# Due to version differences in build tooling a specific dependency is needed
# when built internally that is not yet available in OSS. For now just use the
# :compat_libc_dep as the depenedency name and in oss replace it with this empty
# library.
#
# TODO(b/324174208): Once rules_cc has more toolchain on demand support remove
# and replace with real solution.
cc_library(
    name = "compat_nothing_lib",
    srcs = [],
    hdrs = [],
)

# Alias for toolchain info internally. Remove once able.
alias(
    name = "compat_libc_dep",
    actual = ":compat_nothing_lib",
)
