# Copyright 2024 The XLS Authors
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

load("@rules_pkg//pkg:mappings.bzl", "pkg_files", "strip_prefix")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load(":dist.bzl", "xls_release")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

string_flag(
    name = "version",
    build_setting_default = "dev",
)

string_flag(
    name = "os",
    build_setting_default = "unknown",
)

string_flag(
    name = "arch",
    build_setting_default = "unknown",
)

xls_release(
    name = "xls_package_vars",
    arch = ":arch",
    os = ":os",
    version = ":version",
)

pkg_files(
    name = "xls_stdlib_dslx",
    srcs = [
        "//xls/dslx/stdlib:x_files",
    ],
    # use forward `/` to prevent copybara transformation.
    prefix = "/xls/dslx/stdlib",
    strip_prefix = strip_prefix.from_pkg(),
)

pkg_tar(
    name = "xls_dist_tar",
    srcs = [
        "THIRD_PARTY_NOTICES.txt",
        ":xls_stdlib_dslx",
        "//:LICENSE",
        "//xls/dslx:interpreter_main",
        "//xls/dslx/ir_convert:ir_converter_main",
        "//xls/tools:codegen_main",
        "//xls/tools:opt_main",
        "//xls/tools:proto_to_dslx_main",
    ],
    extension = "tar.gz",
    package_dir = "xls-{version}-{os}-{arch}",
    package_file_name = "xls-{version}-{os}-{arch}.tar.gz",
    package_variables = ":xls_package_vars",
    stamp = 1,
)
