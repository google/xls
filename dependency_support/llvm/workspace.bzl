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

"""Loads LLVM."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    """Initialize the llvm-project repository."""

    # Updated 2023/09/15 to match internal version.
    LLVM_COMMIT = "3d51010a3350660160981c6b8e624dcc87c208a3"
    LLVM_SHA256 = "efbca707a6eb1c714b849de120309070eef282660c0f4be5b68efef62cc95cf5"

    maybe(
        http_archive,
        name = "llvm-raw",
        build_file_content = "# empty",
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-" + LLVM_COMMIT,
        urls = ["https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT)],
    )
