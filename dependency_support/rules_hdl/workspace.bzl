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
        sha256 = "818026634fa14f00c020b679ea0b26066ed544a9b9a14a4a586d9f317a70e0e0",
        strip_prefix = "bazel_rules_hdl-ce8d2c758d985de988aba430e6db599c28b245a7",
        urls = [
            "https://github.com/hdl/bazel_rules_hdl/archive/ce8d2c758d985de988aba430e6db599c28b245a7.tar.gz",  # 2021-06-22
        ],
    )
