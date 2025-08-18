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
    # See MODULE.bazel for the bzlmod-enabled deps.

    # Required to support rules_hdl.
    http_archive(
        name = "rules_7zip",
        strip_prefix = "rules_7zip-e00b15d3cb76b78ddc1c15e7426eb1d1b7ddaa3e",
        urls = ["https://github.com/zaucy/rules_7zip/archive/e00b15d3cb76b78ddc1c15e7426eb1d1b7ddaa3e.zip"],
        sha256 = "fd9e99f6ccb9e946755f9bc444abefbdd1eedb32c372c56dcacc7eb486aed178",
    )

    # Current as of 2025-08-13
    git_hash = "5223ac62cf6f56667ae9bfd2b7a24064252acf66"
    archive_sha256 = "02589270abb13ff114b166d67010c8df096c9f14ddd5d189044e2068afe2f00c"

    maybe(
        http_archive,
        name = "rules_hdl",
        sha256 = archive_sha256,
        strip_prefix = "bazel_rules_hdl-%s" % git_hash,
        urls = [
            "https://github.com/hdl/bazel_rules_hdl/archive/%s.tar.gz" % git_hash,
        ],
    )
