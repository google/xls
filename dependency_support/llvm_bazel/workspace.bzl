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

"""Loads llvm-bazel and llvm projects."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    """Initialize the llvm-project repository."""

    # Untagged version, post 14.0.6.
    LLVM_COMMIT = "a9a60f20e6cc80855864b8f559073bc31f34554b"
    LLVM_SHA256 = "0b9ee39ff10ae2ef77eb6ac1ea00679530347763edc2b19a77970771afb761d0"

    maybe(
        http_archive,
        name = "llvm-bazel",
        build_file_content = "#empty",
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-" + LLVM_COMMIT,
        url = "https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
    )
