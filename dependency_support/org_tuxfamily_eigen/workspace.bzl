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

"""Loads the Eigen linear algebra library."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        new_git_repository,
        name = "org_tuxfamily_eigen",
        commit = "21ae2afd4edaa1b69782c67a54182d34efe43f9c",  # 2020-06-02
        remote = "https://gitlab.com/libeigen/eigen.git",
        shallow_since = "1544551075 +0100",
        build_file = Label("//dependency_support/org_tuxfamily_eigen:bundled.BUILD.bazel"),
    )
