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

"""Loads the m4 macro processor, used by Bison."""

load("//dependency_support:repo.bzl", "xls_http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        xls_http_archive,
        name = "org_gnu_m4",
        urls = [
            "http://ftp.acc.umu.se/mirror/gnu.org/gnu/m4/m4-1.4.18.tar.xz",
            "http://ftp.gnu.org/gnu/m4/m4-1.4.18.tar.xz",
        ],
        strip_prefix = "m4-1.4.18",
        sha256 = "f2c1e86ca0a404ff281631bdc8377638992744b175afb806e25871a24a934e07",
        build_file = Label("//dependency_support:org_gnu_m4/bundled.BUILD.bazel"),
        system_build_file = "@//dependency_support/systemlibs:org_gnu_m4.BUILD",
    )
