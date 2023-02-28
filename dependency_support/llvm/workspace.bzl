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

    # Updated 2023/02/26 to match internal version.
    LLVM_COMMIT = "bc8de519ea6d47a6d1602508eb24594493e9912e"
    LLVM_SHA256 = "0a4ff4ac46126a17b6a0b26c96c5511acb9438c66c91759cfecf4a629babeaf2"

    maybe(
        http_archive,
        name = "llvm-raw",
        build_file_content = "# empty",
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-" + LLVM_COMMIT,
        urls = ["https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT)],
    )
