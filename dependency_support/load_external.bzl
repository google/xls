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

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//dependency_support/boost:workspace.bzl", repo_boost = "repo")
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

    repo_boost()
    repo_llvm()
    repo_rules_hdl()

    # Commit on 2023-02-09
    http_archive(
        name = "pybind11_bazel",
        strip_prefix = "pybind11_bazel-fc56ce8a8b51e3dd941139d329b63ccfea1d304b",
        urls = ["https://github.com/pybind/pybind11_bazel/archive/fc56ce8a8b51e3dd941139d329b63ccfea1d304b.tar.gz"],
        sha256 = "150e2105f9243c445d48f3820b5e4e828ba16c41f91ab424deae1fa81d2d7ac6",
    )

    http_archive(
        name = "six_archive",
        build_file_content = """py_library(
            name = "six",
            visibility = ["//visibility:public"],
            srcs = glob(["*.py"])
        )""",
        sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
        strip_prefix = "six-1.10.0",
        urls = [
            "https://mirror.bazel.build/pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
            "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
        ],
    )

    # Released on 2022-12-27.
    # Current as of 2024-06-26 would be 6.0.2, but that does not work yet
    # with rules_hdl (it assumes rules_proto_toolchains is in repositories.bzl)
    # https://github.com/bazelbuild/rules_proto/releases/tag/5.3.0-21.7
    http_archive(
        name = "rules_proto",
        sha256 = "dc3fb206a2cb3441b485eb1e423165b231235a1ea9b031b4433cf7bc1fa460dd",
        strip_prefix = "rules_proto-5.3.0-21.7",
        urls = [
            "https://github.com/bazelbuild/rules_proto/archive/refs/tags/5.3.0-21.7.tar.gz",
        ],
    )

    # Released on 2024-06-19, current as of 2024-06-26
    http_archive(
        name = "rules_python",
        sha256 = "e3f1cc7a04d9b09635afb3130731ed82b5f58eadc8233d4efb59944d92ffc06f",
        strip_prefix = "rules_python-0.33.2",
        url = "https://github.com/bazelbuild/rules_python/releases/download/0.33.2/rules_python-0.33.2.tar.gz",
    )

    http_archive(
        name = "z3",
        urls = ["https://github.com/Z3Prover/z3/archive/z3-4.12.2.tar.gz"],
        sha256 = "9f58f3710bd2094085951a75791550f547903d75fe7e2fcb373c5f03fc761b8f",
        strip_prefix = "z3-z3-4.12.2",
        build_file = "//dependency_support/z3:bundled.BUILD.bazel",
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
        build_file = "//dependency_support/linenoise:bundled.BUILD.bazel",
    )

    # Needed by fuzztest. Release 2024-05-21, current as of 2024-06-26
    http_archive(
        name = "snappy",
        sha256 = "736aeb64d86566d2236ddffa2865ee5d7a82d26c9016b36218fcc27ea4f09f86",
        build_file = "@com_google_riegeli//third_party:snappy.BUILD",
        strip_prefix = "snappy-1.2.1",
        urls = ["https://github.com/google/snappy/archive/1.2.1.tar.gz"],
    )

    # Needed by fuzztest. Release 2023-08-31, current as of 2024-06-26
    http_archive(
        name = "org_brotli",
        sha256 = "e720a6ca29428b803f4ad165371771f5398faba397edf6778837a18599ea13ff",
        strip_prefix = "brotli-1.1.0",
        urls = ["https://github.com/google/brotli/archive/refs/tags/v1.1.0.tar.gz"],
    )

    # Needed by fuzztest. Commit from 2024-04-18, current as of 2024-06-26
    http_archive(
        name = "highwayhash",
        build_file = "@com_google_riegeli//third_party:highwayhash.BUILD",
        sha256 = "d564c621618ef734e0ae68545f59526e97dfe4912612f80b2b8b9b31b9bb02b5",
        strip_prefix = "highwayhash-f8381f3331d9c56a9792f9b4a35f61c41108c39e",
        urls = ["https://github.com/google/highwayhash/archive/f8381f3331d9c56a9792f9b4a35f61c41108c39e.tar.gz"],
    )

    # Used by xlscc. Tagged 2024-02-16 (note: release is lagging tag), current as of 2024-06-26
    http_archive(
        name = "com_github_hlslibs_ac_types",
        urls = ["https://github.com/hlslibs/ac_types/archive/refs/tags/4.8.0.tar.gz"],
        sha256 = "238197203f8c6254a1d6ac6884e89e6f4c060bffb7473d336df4a1fb53ba7fab",
        strip_prefix = "ac_types-4.8.0",
        build_file = "//dependency_support/com_github_hlslibs_ac_types:bundled.BUILD.bazel",
    )

    # Released 2024-04-25, current as of 2024-06-26
    http_archive(
        name = "platforms",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.10/platforms-0.0.10.tar.gz",
            "https://github.com/bazelbuild/platforms/releases/download/0.0.10/platforms-0.0.10.tar.gz",
        ],
        sha256 = "218efe8ee736d26a3572663b374a253c012b716d8af0c07e842e82f238a0a7ee",
    )

    # Released 2024-05-08, current as of 2024-06-26.
    ORTOOLS_VERSION = "9.10"
    http_archive(
        name = "com_google_ortools",
        urls = ["https://github.com/google/or-tools/archive/refs/tags/v{tag}.tar.gz".format(tag = ORTOOLS_VERSION)],
        sha256 = "e7c27a832f3595d4ae1d7e53edae595d0347db55c82c309c8f24227e675fd378",
        strip_prefix = "or-tools-" + ORTOOLS_VERSION,
    )

    # Updated to head on 2024-03-14
    FUZZTEST_COMMIT = "393ae75c0fca5f9892e73969da5d6bce453ad318"
    http_archive(
        name = "com_google_fuzztest",
        strip_prefix = "fuzztest-" + FUZZTEST_COMMIT,
        url = "https://github.com/google/fuzztest/archive/" + FUZZTEST_COMMIT + ".zip",
        sha256 = "a0558ceb617d78ee93d7e6b62930b4aeebc02f1e5817d4d0dae53699f6f6c352",
        patch_args = ["-p1", "-R"],  # reverse patch until we upgrade bazel to 7.1; see: bazelbuild/bazel#19233.
        patches = ["//dependency_support/com_google_fuzztest:e317d5277e34948ae7048cb5e48309e0288e8df3.patch"],
    )

    # 2022-09-19
    http_archive(
        name = "com_grail_bazel_compdb",
        sha256 = "a3ff6fe238eec8202270dff75580cba3d604edafb8c3408711e82633c153efa8",
        strip_prefix = "bazel-compilation-database-940cedacdb8a1acbce42093bf67f3a5ca8b265f7",
        urls = ["https://github.com/grailbio/bazel-compilation-database/archive/940cedacdb8a1acbce42093bf67f3a5ca8b265f7.tar.gz"],
    )

    # Current as of 2024-08-23
    http_archive(
        name = "verible",
        sha256 = "0f6e2f0cad335c9aca903aaa014f2a55d40c5c4dbfe4666474771f5c08e4a42c",
        strip_prefix = "verible-17e909b09b279e5c1f5cd6a07404691babe3d3c3",
        urls = ["https://github.com/chipsalliance/verible/archive/17e909b09b279e5c1f5cd6a07404691babe3d3c3.tar.gz"],
        patch_args = ["-p1"],
        patches = ["//dependency_support/verible:visibility.patch"],
    )

    # Used in C++ tests of the ZSTD Module
    # Transitive dependency of fuzztest (required by riegeli in fuzztest workspace)
    # Version fdfb2aff released on 2024-07-31
    # https://github.com/facebook/zstd/commit/fdfb2aff39dc498372d8c9e5f2330b692fea9794
    # Updated 2024-08-08
    http_archive(
        name = "zstd",
        sha256 = "9ace5a1b3c477048c6e034fe88d2abb5d1402ced199cae8e9eef32fdc32204df",
        strip_prefix = "zstd-fdfb2aff39dc498372d8c9e5f2330b692fea9794",
        urls = ["https://github.com/facebook/zstd/archive/fdfb2aff39dc498372d8c9e5f2330b692fea9794.zip"],
        build_file = "//dependency_support/com_github_facebook_zstd:bundled.BUILD.bazel",
    )
