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

    # Commit on 2022-12-16, current as of 2022-12-20
    git_hash = "c6ba9ffadf9b25be83f8300650f4447386b07fea"
    git_sha256 = "8d7c271505f571fc4707c2a8cbe2f9ab40df2c6ad74ed3965f4a0397ead82940"

    maybe(
        http_archive,
        name = "rules_hdl",
        sha256 = git_sha256,
        strip_prefix = "bazel_rules_hdl-%s" % git_hash,
        urls = [
            "https://github.com/hdl/bazel_rules_hdl/archive/%s.tar.gz" % git_hash,
        ],
    )
