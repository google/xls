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

    # Commit on 2023-07-07, current as of 2023-07-07.
    git_hash = "83b51f4b10e0e8970f88b087b5fab3f90eb89834"
    git_sha256 = "506d4e7ad746cf3fbeeb2e073667a940612bdd2a258ed9fe637d0c9f7d5c8930"

    maybe(
        http_archive,
        name = "rules_hdl",
        sha256 = git_sha256,
        strip_prefix = "bazel_rules_hdl-%s" % git_hash,
        urls = [
            "https://github.com/antmicro/bazel_rules_hdl/archive/%s.tar.gz" % git_hash,
        ],
    )
