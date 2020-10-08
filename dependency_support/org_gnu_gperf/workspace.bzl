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

"""Loads the gperf perfect hash function generator, used by iverilog."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        http_archive,
        name = "org_gnu_gperf",
        urls = [
            "http://ftp.acc.umu.se/mirror/gnu.org/gnu/gperf/gperf-3.1.tar.gz",
            "http://ftp.gnu.org/gnu/gperf/gperf-3.1.tar.gz",
        ],
        strip_prefix = "gperf-3.1",
        sha256 = "588546b945bba4b70b6a3a616e80b4ab466e3f33024a352fc2198112cdbb3ae2",
        build_file = Label("//dependency_support:org_gnu_gperf/bundled.BUILD.bazel"),
    )
