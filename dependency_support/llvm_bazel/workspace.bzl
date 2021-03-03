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
    """Initialize the llvm-bazel (and llvm-project) repository."""
    LLVM_COMMIT = "7f086d74c347750c1da619058eb5b3e79c2fae14"  # 2021/03/02
    LLVM_BAZEL_TAG = "llvm-project-%s" % (LLVM_COMMIT,)
    LLVM_BAZEL_SHA256 = "4acee9aa70309542a2a806e4584750c07ead0ffbc5fcd9deb0ff05ed64d7ae21"
    LLVM_SHA256 = "49590fd9975bab23a347859d945a8dcae1db35a1c32bd0d9387271a6b8058106"

    maybe(
        http_archive,
        name = "llvm-bazel",
        sha256 = LLVM_BAZEL_SHA256,
        strip_prefix = "llvm-bazel-{tag}/llvm-bazel".format(tag = LLVM_BAZEL_TAG),
        url = "https://github.com/google/llvm-bazel/archive/{tag}.tar.gz".format(tag = LLVM_BAZEL_TAG),
    )

    maybe(
        http_archive,
        name = "llvm-project-raw",
        build_file_content = "#empty",
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-" + LLVM_COMMIT,
        url = "https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
    )
