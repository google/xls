# Copyright 2025 The XLS Authors
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

"""Perfetto extension for XLS."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# TODO: bazelbuild/bazel-central-registry#1020 - revisit when perfetto is in BCR.
def _perfetto_extension_impl(_):
    INTEGRITY = "sha256-GHI0lDmCDiQQA6mGLLafTVNZJsQ31TFrSc5KggYTU50="
    URL = "https://github.com/google/perfetto/archive/refs/tags/v52.0.tar.gz"
    http_archive(
        name = "perfetto",
        integrity = INTEGRITY,
        url = URL,
        strip_prefix = "perfetto-52.0",
    )

    # perfetto_cfg is a new_local_repository using a path relative to the top-level WORKSPACE file.
    # We replicate this with an http_archive using a bigger strip_prefix.
    http_archive(
        name = "perfetto_cfg",
        integrity = INTEGRITY,
        url = URL,
        strip_prefix = "perfetto-52.0/bazel/standalone",
        build_file_content = "# empty BUILD to make a bazel package",
    )

perfetto_extension = module_extension(
    implementation = _perfetto_extension_impl,
)
