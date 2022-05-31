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
load("//dependency_support/rules_hdl:workspace.bzl", repo_rules_hdl = "repo")

def load_external_repositories():
    """Loads external repositories with third-party code."""

    repo_boost()
    repo_llvm_bazel()
    repo_rules_hdl()

    http_archive(
        name = "com_google_googletest",
        urls = ["https://github.com/google/googletest/archive/47f819c3ca54fb602f432904443e00a0a1fe2f42.zip"],  # 2021-08-10
        strip_prefix = "googletest-47f819c3ca54fb602f432904443e00a0a1fe2f42",
        sha256 = "db93dd8a793e54a3a429197ecdd94813d280019e2adfa9241c7bc9204842bedc",
    )

    # 2022-1-31
    http_archive(
        name = "com_google_absl",
        urls = ["https://github.com/abseil/abseil-cpp/archive/940c06c25d2953f44310b68eb8aab6114dba11fb.zip"],
        strip_prefix = "abseil-cpp-940c06c25d2953f44310b68eb8aab6114dba11fb",
        sha256 = "0e800799aa64d0b4d354f3ff317bbd5fbf42f3a522ab0456bb749fc8d3b67415",
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
        commit = "4fb158925f7753d80fb858cb0239dff893ef9f15",  # 2021-11-01
        remote = "https://github.com/google/boringssl.git",
        shallow_since = "1635790430 +0000",
    )

    # Commit on 2021-12-03, current as of 2022-05-31
    http_archive(
        name = "pybind11_bazel",
        strip_prefix = "pybind11_bazel-72cbbf1fbc830e487e3012862b7b720001b70672",
        urls = ["https://github.com/pybind/pybind11_bazel/archive/72cbbf1fbc830e487e3012862b7b720001b70672.tar.gz"],
        sha256 = "516c1b3a10d87740d2b7de6f121f8e19dde2c372ecbfe59aef44cd1872c10395",
        patches = ["@com_google_xls//dependency_support/pybind11_bazel:sysconfig_fix.patch"],
    )

    http_archive(
        name = "pybind11",
        build_file = "@pybind11_bazel//:pybind11.BUILD",
        strip_prefix = "pybind11-30eb39ed79d1e2eeff15219ac00773034300a5e6",
        urls = ["https://github.com/pybind/pybind11/archive/30eb39ed79d1e2eeff15219ac00773034300a5e6.tar.gz"],
        sha256 = "2e748579ea1903d55b8abe9d314fd5c3cf9f4e5734c525444e4d2aed146efc9e",
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

    # Released on 2022-04-22, current as of 2022-05-27
    # https://github.com/bazelbuild/rules_python/releases/tag/0.8.1
    http_archive(
        name = "rules_python",
        sha256 = "cdf6b84084aad8f10bf20b46b77cb48d83c319ebe6458a18e9d2cebf57807cdd",
        strip_prefix = "rules_python-0.8.1",
        url = "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.8.1.tar.gz",
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

    # Released on 2022-05-20, current as of 2022-05-31.
    # https://github.com/grpc/grpc/releases/tag/v1.46.3
    http_archive(
        name = "com_github_grpc_grpc",
        urls = ["https://github.com/grpc/grpc/archive/v1.46.3.tar.gz"],
        sha256 = "d6cbf22cb5007af71b61c6be316a79397469c58c82a942552a62e708bce60964",
        strip_prefix = "grpc-1.46.3",
        # repo_mapping = {"@com_github_google_re2": "@com_googlesource_code_re2"},
        # Note: repo mapping doesn't seem to work for gRPC because it
        # explicitly binds the re2 name to the com_googlesource_code_re2 repo.
        # Instead we patch it.
        #repo_mapping = {"@com_googlesource_code_re2": "@com_github_google_re2"},
        patches = ["@com_google_xls//dependency_support/com_github_grpc_grpc:grpc_patch.diff"],
    )

    # Used by xlscc.
    http_archive(
        name = "com_github_hlslibs_ac_types",
        urls = ["https://github.com/hlslibs/ac_types/archive/57d89634cb5034a241754f8f5347803213dabfca.tar.gz"],
        sha256 = "7ab5e2ee4c675ef6895fdd816c32349b3070dc8211b7d412242c66d0c6e8edca",
        strip_prefix = "ac_types-57d89634cb5034a241754f8f5347803213dabfca",
        build_file = "@com_google_xls//dependency_support/com_github_hlslibs_ac_types:bundled.BUILD.bazel",
    )

    git_repository(
        name = "platforms",
        remote = "https://github.com/bazelbuild/platforms.git",
        # Apparently the arguments below are the reproducible form of this tag.
        # tag = "0.0.5",
        commit = "fbd0d188dac49fbcab3d2876a2113507e6fc68e9",
        shallow_since = "1644333305 -0500",
    )

    git_repository(
        name = "com_google_ortools",
        commit = "525162feaadaeef640783b2eaea38cf4b623877f",
        shallow_since = "1647023481 +0100",
        remote = "https://github.com/google/or-tools.git",
        # Removes undesired dependencies like Eigen, BLISS, SCIP
        patches = ["@com_google_xls//dependency_support/com_google_ortools:remove_deps.diff"],
    )
