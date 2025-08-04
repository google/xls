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

"""Loads the pprof package and adds build rule for the one proto we need from it."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    # Current as of 2025-08-04
    git_hash = "6e76a2b096b5fa52e4bb3f7f7a357bd6e6b3b7b1"
    archive_sha256 = "a539183a563e820ff189dfae8dd3b3690cbb376cf3f297031392307bfe72350b"

    http_archive(
        name = "pprof",
        sha256 = archive_sha256,
        strip_prefix = "pprof-%s" % git_hash,
        build_file = Label("//dependency_support/pprof:bundled.BUILD.bazel"),
        urls = [
            "https://github.com/google/pprof/archive/%s.tar.gz" % git_hash,
        ],
    )
