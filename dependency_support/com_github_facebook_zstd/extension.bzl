# Copyright 2026 The XLS Authors
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

"""Module extension for zstd; used in C++ tests of the ZSTD Module

Required (rather than the BCR version) while decodecorpus is not exposed in BCR."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _zstd_extension_impl(
        module_ctx):  # @unused
    # Version fdfb2aff released on 2024-07-31
    # https://github.com/facebook/zstd/commit/fdfb2aff39dc498372d8c9e5f2330b692fea9794
    # Updated 2024-08-08
    http_archive(
        name = "zstd",
        sha256 = "9ace5a1b3c477048c6e034fe88d2abb5d1402ced199cae8e9eef32fdc32204df",
        strip_prefix = "zstd-fdfb2aff39dc498372d8c9e5f2330b692fea9794",
        urls = ["https://github.com/facebook/zstd/archive/fdfb2aff39dc498372d8c9e5f2330b692fea9794.zip"],
        build_file = Label("//dependency_support/com_github_facebook_zstd:bundled.BUILD.bazel"),
    )

zstd_extension = module_extension(
    implementation = _zstd_extension_impl,
)
