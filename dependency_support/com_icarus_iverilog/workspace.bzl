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

"""Loads the libedit library, used by iverilog (it poses as GNU readline)."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        http_archive,
        name = "com_icarus_iverilog",
        urls = [
            "https://github.com/steveicarus/iverilog/archive/v10_3.tar.gz",
        ],
        strip_prefix = "iverilog-10_3",
        sha256 = "4b884261645a73b37467242d6ae69264fdde2e7c4c15b245d902531efaaeb234",
        build_file = Label("//dependency_support:com_icarus_iverilog/bundled.BUILD.bazel"),
    )
