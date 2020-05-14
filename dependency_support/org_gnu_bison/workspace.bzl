# Copyright 2020 Google LLC
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

"""Loads the Bison parser generator, used by iverilog."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        http_archive,
        name = "org_gnu_bison",
        urls = [
            "http://ftp.acc.umu.se/mirror/gnu.org/gnu/bison/bison-3.5.tar.xz",
            "http://ftp.gnu.org/gnu/bison/bison-3.5.tar.xz",
        ],
        strip_prefix = "bison-3.5",
        sha256 = "55e4a023b1b4ad19095a5f8279f0dc048fa29f970759cea83224a6d5e7a3a641",
        build_file = Label("//dependency_support:org_gnu_bison/bundled.BUILD.bazel"),
    )
