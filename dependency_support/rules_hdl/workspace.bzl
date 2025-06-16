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

"""Loads the rules_hdl package which contains Bazel rules for other tools that
XLS uses."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    # Required to support rules_hdl, 0.7.0 release is current as of 2022-05-09.
    http_archive(
        name = "rules_pkg",
        sha256 = "eea0f59c28a9241156a47d7a8e32db9122f3d50b505fae0f33de6ce4d9b61834",
        urls = [
            "https://github.com/bazelbuild/rules_pkg/releases/download/0.8.0/rules_pkg-0.8.0.tar.gz",
            "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.8.0/rules_pkg-0.8.0.tar.gz",
        ],
    )

    # Required to support rules_hdl.
    http_archive(
        name = "rules_7zip",
        strip_prefix = "rules_7zip-e00b15d3cb76b78ddc1c15e7426eb1d1b7ddaa3e",
        urls = ["https://github.com/zaucy/rules_7zip/archive/e00b15d3cb76b78ddc1c15e7426eb1d1b7ddaa3e.zip"],
        sha256 = "fd9e99f6ccb9e946755f9bc444abefbdd1eedb32c372c56dcacc7eb486aed178",
    )

    # Required to support rules_hdl.
    http_archive(
        name = "rules_m4",
        sha256 = "10ce41f150ccfbfddc9d2394ee680eb984dc8a3dfea613afd013cfb22ea7445c",
        urls = ["https://github.com/jmillikin/rules_m4/releases/download/v0.2.3/rules_m4-v0.2.3.tar.xz"],
    )

    # Required to support rules_hdl.
    http_archive(
        name = "rules_flex",
        sha256 = "8929fedc40909d19a4b42548d0785f796c7677dcef8b5d1600b415e5a4a7749f",
        urls = ["https://github.com/jmillikin/rules_flex/releases/download/v0.2.1/rules_flex-v0.2.1.tar.xz"],
    )

    # Required to support rules_hdl.
    http_archive(
        name = "rules_bison",
        sha256 = "2279183430e438b2dc77cacd7b1dbb63438971b2411406570f1ddd920b7c9145",
        urls = ["https://github.com/jmillikin/rules_bison/releases/download/v0.2.2/rules_bison-v0.2.2.tar.xz"],
    )

    # Current as of 2025-06-14
    git_hash = "defad2a7719421672377a73d8befad0e4016c34b"
    archive_sha256 = "d5e211913c152e2eb50ac2a3aeb89a785d9ad890f0154546b2635bee0897403c"

    maybe(
        http_archive,
        name = "rules_hdl",
        sha256 = archive_sha256,
        strip_prefix = "bazel_rules_hdl-%s" % git_hash,
        urls = [
            "https://github.com/hdl/bazel_rules_hdl/archive/%s.tar.gz" % git_hash,
        ],
    )
