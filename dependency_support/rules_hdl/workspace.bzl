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
    maybe(
        http_archive,
        name = "rules_hdl",
        sha256 = "3149a553a98c9b4bfe31eb00363dd535820b22abed568ba0c60957ff7ec63e2a",
        strip_prefix = "bazel_rules_hdl-06705603bf3456bcabc28008d19ea5da22359fe5",
        urls = [
            "https://github.com/per-gron/bazel_rules_hdl/archive/06705603bf3456bcabc28008d19ea5da22359fe5.tar.gz",  # 2021-04-14
        ],
    )
