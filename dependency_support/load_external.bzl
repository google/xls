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

"""Provides helper that loads external repositories with third-party code."""

# TODO: https://github.com/google/xls/issues/931 - all the remaining toplevel projects we need should move to MODULE.bazel,
# somewhat dependent on what becomes available in https://registry.bazel.build/.
# Eventual goal that none of this is needed anymore and the file can be removed.

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//dependency_support/llvm:workspace.bzl", repo_llvm = "repo")
load("//dependency_support/rules_hdl:workspace.bzl", repo_rules_hdl = "repo")

def load_external_repositories():
    """Loads external repositories with third-party code."""

    # Note: there are more direct dependencies than are explicitly listed here.
    #
    # By letting direct dependencies be satisfied by transitive WORKSPACE
    # setups we let the transitive dependency versions "float" to satisfy their
    # package requirements, so that we can bump our dependency versions here
    # more easily without debugging/resolving unnecessary conflicts.
    #
    # This situation will change when XLS moves to bzlmod. See
    # https://github.com/google/xls/issues/865 and
    # https://github.com/google/xls/issues/931#issue-1667228764 for more
    # information / background.

    repo_llvm()
    repo_rules_hdl()

    # Released on 2024-09-24, current as of 2024-10-01
    http_archive(
        name = "rules_python",
        sha256 = "ca77768989a7f311186a29747e3e95c936a41dffac779aff6b443db22290d913",
        strip_prefix = "rules_python-0.36.0",
        url = "https://github.com/bazelbuild/rules_python/releases/download/0.36.0/rules_python-0.36.0.tar.gz",
    )

    http_archive(
        name = "z3",
        urls = ["https://github.com/Z3Prover/z3/archive/z3-4.14.1.tar.gz"],
        integrity = "sha256-gaAsLGTGTWw98jP1kYa5VieZCtoMTC/JAcnCWnByZyo=",
        strip_prefix = "z3-z3-4.14.1",
        build_file = Label("//dependency_support/z3:bundled.BUILD.bazel"),
    )

    # Release 2024-02-23, current as of 2024-06-26
    http_archive(
        name = "io_bazel_rules_closure",
        sha256 = "70ef2b4da987bf0d266e663d7c251eac509ff70dd65bba02d41d1e86e840a569",
        strip_prefix = "rules_closure-0.13.0",
        urls = [
            "https://github.com/bazelbuild/rules_closure/archive/0.13.0.tar.gz",
        ],
    )

    # Commit from 2024-02-22, current as of 2024-06-26
    http_archive(
        name = "linenoise",
        sha256 = "839ed407fe0dfa5fd7dd103abfc695dee72fea2840df8d4250ad42b0e64839e8",
        strip_prefix = "linenoise-d895173d679be70bcd8b23041fff3e458e1a3506",
        urls = ["https://github.com/antirez/linenoise/archive/d895173d679be70bcd8b23041fff3e458e1a3506.tar.gz"],
        build_file = Label("//dependency_support/linenoise:bundled.BUILD.bazel"),
    )

    # Used by xlscc. Tagged 2024-02-16 (note: release is lagging tag), current as of 2024-06-26
    http_archive(
        name = "com_github_hlslibs_ac_types",
        urls = ["https://github.com/hlslibs/ac_types/archive/refs/tags/4.8.0.tar.gz"],
        sha256 = "238197203f8c6254a1d6ac6884e89e6f4c060bffb7473d336df4a1fb53ba7fab",
        strip_prefix = "ac_types-4.8.0",
        build_file = Label("//dependency_support/com_github_hlslibs_ac_types:bundled.BUILD.bazel"),
    )

    # Used in C++ tests of the ZSTD Module
    # Version fdfb2aff released on 2024-07-31
    # https://github.com/facebook/zstd/commit/fdfb2aff39dc498372d8c9e5f2330b692fea9794
    # Updated 2024-08-08
    # Note: this exists in BCR, but TODO: include :decodecorpus
    http_archive(
        name = "zstd",
        sha256 = "9ace5a1b3c477048c6e034fe88d2abb5d1402ced199cae8e9eef32fdc32204df",
        strip_prefix = "zstd-fdfb2aff39dc498372d8c9e5f2330b692fea9794",
        urls = ["https://github.com/facebook/zstd/archive/fdfb2aff39dc498372d8c9e5f2330b692fea9794.zip"],
        build_file = Label("//dependency_support/com_github_facebook_zstd:bundled.BUILD.bazel"),
    )
