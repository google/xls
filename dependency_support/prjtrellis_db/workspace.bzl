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

"""Loads the Project Trellis database with bitstream documentation for Lattice ECP5 devices."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        http_archive,
        name = "prjtrellis_db",
        urls = [
            "https://github.com/SymbiFlow/prjtrellis-db/archive/c137076fdd8bfca3d2bf9cdacda9983dbbec599a.zip",  # 2020-06-03
        ],
        strip_prefix = "prjtrellis-db-c137076fdd8bfca3d2bf9cdacda9983dbbec599a",
        sha256 = "01d3d3906a7f690bd05f97b87bf196f602c659e0dadc77ebbccdd1acf1e362ca",
        build_file = Label("//dependency_support/prjtrellis_db:bundled.BUILD.bazel"),
    )
