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

"""Loads Project Trellis which documents the bitstream for Lattice ECP5 devices."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        http_archive,
        name = "prjtrellis",
        urls = [
            "https://github.com/SymbiFlow/prjtrellis/archive/315969591a12af11b02b4aa9c1dca9066e1be8e2.zip",  # 2020-05-18
        ],
        strip_prefix = "prjtrellis-315969591a12af11b02b4aa9c1dca9066e1be8e2",
        sha256 = "7f6eaf88ab386e8976e3bf085d0189a9153c19ad9866dea3920c995a7f717bbd",
        build_file = Label("//dependency_support/prjtrellis:bundled.BUILD.bazel"),
    )
