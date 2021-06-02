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
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("//dependency_support/boost:workspace.bzl", repo_boost = "repo")
load("//dependency_support/llvm_bazel:workspace.bzl", repo_llvm_bazel = "repo")
load("//dependency_support/org_tuxfamily_eigen:workspace.bzl", repo_eigen = "repo")
load("//dependency_support/rules_hdl:workspace.bzl", repo_rules_hdl = "repo")

def load_external_repositories():
    """Loads external repositories with third-party code."""

    repo_boost()
    repo_eigen()
    repo_llvm_bazel()
    repo_rules_hdl()

    http_archive(
        name = "com_google_googletest",
        urls = ["https://github.com/google/googletest/archive/0eea2e9fc63461761dea5f2f517bd6af2ca024fa.zip"],  # 2020-04-30
        strip_prefix = "googletest-0eea2e9fc63461761dea5f2f517bd6af2ca024fa",
        sha256 = "9463ff914d7c3db02de6bd40a3c412a74e979e3c76eaa89920a49ff8488d6d69",
    )

    # 2021-03-31
    http_archive(
        name = "com_google_absl",
        strip_prefix = "abseil-cpp-20210324.0",
        urls = ["https://github.com/abseil/abseil-cpp/archive/refs/tags/20210324.0.zip"],
        sha256 = "11d6eea257cc9322cc49924cf9584dbe61922bfffe3e7c42e2bce3abc1694a1a",
        patches = ["@com_google_xls//dependency_support/com_google_absl:aarch64_rng_flags.patch"],
    )

    # Protobuf depends on Skylib
    # Load bazel skylib as per
    # https://github.com/bazelbuild/bazel-skylib/releases
    http_archive(
        name = "bazel_skylib",
        urls = [
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.3/bazel-skylib-1.0.3.tar.gz",
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.3/bazel-skylib-1.0.3.tar.gz",
        ],
        sha256 = "1c531376ac7e5a180e0237938a2536de0c54d93f5c278634818e0efc952dd56c",
    )

    git_repository(
        name = "boringssl",
        commit = "14164f6fef47b7ebd97cdb0cea1624eabd6fe6b8",  # 2018-11-26
        remote = "https://github.com/google/boringssl.git",
        shallow_since = "1543277914 +0000",
    )

    http_archive(
        name = "pybind11_bazel",
        strip_prefix = "pybind11_bazel-34206c29f891dbd5f6f5face7b91664c2ff7185c",
        urls = ["https://github.com/pybind/pybind11_bazel/archive/34206c29f891dbd5f6f5face7b91664c2ff7185c.zip"],
        sha256 = "8d0b776ea5b67891f8585989d54aa34869fc12f14bf33f1dc7459458dd222e95",
    )

    http_archive(
        name = "pybind11",
        build_file = "@pybind11_bazel//:pybind11.BUILD",
        strip_prefix = "pybind11-59a2ac2745d8a57ac94c6accced73620d59fb844",
        urls = ["https://github.com/pybind/pybind11/archive/59a2ac2745d8a57ac94c6accced73620d59fb844.tar.gz"],
        sha256 = "3276f79ae507ae578501ffc2833f8e9082092eb99e59763a77d08c3fdea35bec",
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

    # Version: pypi-v0.11.0, 2020/10/27
    http_archive(
        name = "com_google_absl_py",
        strip_prefix = "abseil-py-ddbd7d46d01fa71b0584e948d68fcd1d47bea0c4",
        urls = ["https://github.com/abseil/abseil-py/archive/ddbd7d46d01fa71b0584e948d68fcd1d47bea0c4.zip"],
        sha256 = "c4d112feb36d254de0057b9e67f5423c64908f17219b13f799b47b4deacc279c",
    )

    # Note - use @com_github_google_re2 instead of more canonical
    #        @com_google_re2 for consistency with dependency grpc
    #        which uses @com_github_google_re2.
    #          (see https://github.com/google/xls/issues/234)
    http_archive(
        name = "com_github_google_re2",
        sha256 = "d070e2ffc5476c496a6a872a6f246bfddce8e7797d6ba605a7c8d72866743bf9",
        strip_prefix = "re2-506cfa4bffd060c06ec338ce50ea3468daa6c814",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/re2/archive/506cfa4bffd060c06ec338ce50ea3468daa6c814.tar.gz",
            "https://github.com/google/re2/archive/506cfa4bffd060c06ec338ce50ea3468daa6c814.tar.gz",
        ],
    )

    http_archive(
        name = "rules_python",
        url = "https://github.com/bazelbuild/rules_python/releases/download/0.1.0/rules_python-0.1.0.tar.gz",
        sha256 = "b6d46438523a3ec0f3cead544190ee13223a52f6a6765a29eae7b7cc24cc83a0",
    )

    http_archive(
        name = "z3",
        urls = ["https://github.com/Z3Prover/z3/archive/z3-4.8.7.tar.gz"],
        sha256 = "8c1c49a1eccf5d8b952dadadba3552b0eac67482b8a29eaad62aa7343a0732c3",
        strip_prefix = "z3-z3-4.8.7",
        build_file = "@com_google_xls//dependency_support/z3:bundled.BUILD.bazel",
    )

    http_archive(
        name = "io_bazel_rules_closure",
        sha256 = "7d206c2383811f378a5ef03f4aacbcf5f47fd8650f6abbc3fa89f3a27dd8b176",
        strip_prefix = "rules_closure-0.10.0",
        urls = [
            "https://github.com/bazelbuild/rules_closure/archive/0.10.0.tar.gz",
        ],
    )

    # zlib is added automatically by gRPC, but the zlib BUILD file used by gRPC
    # does not include all the source code (e.g., gzread is missing) which
    # breaks other users of zlib like iverilog. So add zlib explicitly here with
    # a working BUILD file.
    http_archive(
        name = "zlib",
        sha256 = "f5cc4ab910db99b2bdbba39ebbdc225ffc2aa04b4057bc2817f1b94b6978cfc3",
        strip_prefix = "zlib-1.2.11",
        urls = [
            "https://github.com/madler/zlib/archive/v1.2.11.zip",
        ],
        build_file = "@com_google_xls//dependency_support/zlib:bundled.BUILD.bazel",
    )

    http_archive(
        name = "linenoise",
        sha256 = "e7dbebca81b518544bea6622d5cc1a2e6347d080793cb0ba134edc66c3822fd5",
        strip_prefix = "linenoise-97d2850af13c339369093b78abe5265845d78220",
        urls = ["https://github.com/antirez/linenoise/archive/97d2850af13c339369093b78abe5265845d78220.zip"],
        build_file = "@com_google_xls//dependency_support/linenoise:bundled.BUILD.bazel",
    )

    http_archive(
        name = "com_github_grpc_grpc",
        urls = ["https://github.com/grpc/grpc/archive/v1.32.0.tar.gz"],
        sha256 = "f880ebeb2ccf0e47721526c10dd97469200e40b5f101a0d9774eb69efa0bd07a",
        strip_prefix = "grpc-1.32.0",
        patches = ["@com_google_xls//dependency_support/com_github_grpc_grpc:grpc-cython.patch"],
    )
