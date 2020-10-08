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

"""Loads the Icestorm project that documents the bitstream format of Lattice iCE40."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        http_archive,
        name = "at_clifford_icestorm",
        urls = [
            "https://github.com/cliffordwolf/icestorm/archive/cd2610e0fa1c6a90e8e4e4cfe06db1b474e752bb.zip",  # 2020-06-02
        ],
        strip_prefix = "icestorm-cd2610e0fa1c6a90e8e4e4cfe06db1b474e752bb",
        sha256 = "e8d12796091f5988097459450de20e4a59c873ca2fa580fef2f560c5543a1738",
        build_file = Label("//dependency_support:at_clifford_icestorm/bundled.BUILD.bazel"),
    )
