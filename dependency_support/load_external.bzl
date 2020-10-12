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
load("//dependency_support:repo.bzl", "xls_http_archive")
load("//dependency_support:edu_berkeley_abc/workspace.bzl", repo_abc = "repo")
load("//dependency_support/boost:workspace.bzl", repo_boost = "repo")
load("//dependency_support/org_gnu_bison:workspace.bzl", repo_bison = "repo")
load("//dependency_support:org_sourceware_bzip2/workspace.bzl", repo_bzip2 = "repo")
load("//dependency_support/org_tuxfamily_eigen:workspace.bzl", repo_eigen = "repo")
load("//dependency_support/flex:workspace.bzl", repo_flex = "repo")
load("//dependency_support/org_gnu_gperf:workspace.bzl", repo_gperf = "repo")
load("//dependency_support/at_clifford_icestorm:workspace.bzl", repo_icestorm = "repo")
load("//dependency_support/com_icarus_iverilog:workspace.bzl", repo_iverilog = "repo")
load("//dependency_support/dk_thrysoee_libedit:workspace.bzl", repo_libedit = "repo")
load("//dependency_support:org_sourceware_libffi/workspace.bzl", repo_libffi = "repo")
load("//dependency_support/org_gnu_m4:workspace.bzl", repo_m4 = "repo")
load("//dependency_support/net_invisible_island_ncurses:workspace.bzl", repo_ncurses = "repo")
load("//dependency_support/nextpnr:workspace.bzl", repo_nextpnr = "repo")
load("//dependency_support/prjtrellis:workspace.bzl", repo_prjtrellis = "repo")
load("//dependency_support/prjtrellis_db:workspace.bzl", repo_prjtrellis_db = "repo")
load("//dependency_support:tk_tcl_tcl/workspace.bzl", repo_tcl = "repo")
load("//dependency_support/at_clifford_yosys:workspace.bzl", repo_yosys = "repo")

def load_external_repositories():
    """Loads external repositories with third-party code."""

    repo_abc()
    repo_bison()
    repo_boost()
    repo_bzip2()
    repo_eigen()
    repo_flex()
    repo_gperf()
    repo_icestorm()
    repo_iverilog()
    repo_libedit()
    repo_libffi()
    repo_m4()
    repo_ncurses()
    repo_nextpnr()
    repo_prjtrellis()
    repo_prjtrellis_db()
    repo_tcl()
    repo_yosys()

    http_archive(
        name = "com_google_googletest",
        urls = ["https://github.com/google/googletest/archive/0eea2e9fc63461761dea5f2f517bd6af2ca024fa.zip"],  # 2020-04-30
        strip_prefix = "googletest-0eea2e9fc63461761dea5f2f517bd6af2ca024fa",
        sha256 = "9463ff914d7c3db02de6bd40a3c412a74e979e3c76eaa89920a49ff8488d6d69",
    )

    # 2020-09-09
    http_archive(
        name = "com_google_absl",
        strip_prefix = "abseil-cpp-3c8b5d7587dc8ecf730ce9996c89e156e408c3ed",
        urls = ["https://github.com/abseil/abseil-cpp/archive/3c8b5d7587dc8ecf730ce9996c89e156e408c3ed.zip"],
        sha256 = "2adc8d02700923164a1128afd36d245ac6fd9fe4a10ee936146b89cd9e4b82a5",
    )

    # Protobuf depends on Skylib
    http_archive(
        name = "bazel_skylib",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
        ],
        sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
    )

    git_repository(
        name = "boringssl",
        commit = "14164f6fef47b7ebd97cdb0cea1624eabd6fe6b8",  # 2018-11-26
        remote = "https://github.com/google/boringssl.git",
        shallow_since = "1543277914 +0000",
    )

    xls_http_archive(
        name = "llvm",
        urls = ["https://github.com/llvm/llvm-project/archive/52cae05e087b3d4fd02849fc37c387c720055ffb.tar.gz"],
        sha256 = "84af24a605a9e20d999e65f87a36baeed56773d74ccc0287517f5fdcf44e737a",
        strip_prefix = "llvm-project-52cae05e087b3d4fd02849fc37c387c720055ffb/llvm",
        build_file = "@//dependency_support/llvm:bundled.BUILD.bazel",
        system_build_file = "@//dependency_support/systemlibs:llvm.BUILD",
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
        strip_prefix = "pybind11-4a288ab92839837126e75a862ba07af1b207b084",
        urls = ["https://github.com/pybind/pybind11/archive/4a288ab92839837126e75a862ba07af1b207b084.tar.gz"],
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

    http_archive(
        name = "com_google_absl_py",
        strip_prefix = "abseil-py-06edd9c20592cec39178b94240b5e86f32e19768",
        urls = ["https://github.com/abseil/abseil-py/archive/06edd9c20592cec39178b94240b5e86f32e19768.zip"],
        sha256 = "6ace3cd8921804aaabc37970590edce05c6664901cc98d30010d09f2811dc56f",
    )

    http_archive(
        name = "com_google_re2",
        sha256 = "d070e2ffc5476c496a6a872a6f246bfddce8e7797d6ba605a7c8d72866743bf9",
        strip_prefix = "re2-506cfa4bffd060c06ec338ce50ea3468daa6c814",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/re2/archive/506cfa4bffd060c06ec338ce50ea3468daa6c814.tar.gz",
            "https://github.com/google/re2/archive/506cfa4bffd060c06ec338ce50ea3468daa6c814.tar.gz",
        ],
    )

    http_archive(
        name = "rules_python",
        url = "https://github.com/bazelbuild/rules_python/releases/download/0.0.2/rules_python-0.0.2.tar.gz",
        strip_prefix = "rules_python-0.0.2",
        sha256 = "b5668cde8bb6e3515057ef465a35ad712214962f0b3a314e551204266c7be90c",
    )

    xls_http_archive(
        name = "z3",
        urls = ["https://github.com/Z3Prover/z3/archive/z3-4.8.7.tar.gz"],
        sha256 = "8c1c49a1eccf5d8b952dadadba3552b0eac67482b8a29eaad62aa7343a0732c3",
        strip_prefix = "z3-z3-4.8.7",
        build_file = "@//dependency_support/z3:bundled.BUILD.bazel",
        system_build_file = "@//dependency_support/systemlibs:z3.BUILD",
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
    xls_http_archive(
        name = "zlib",
        sha256 = "f5cc4ab910db99b2bdbba39ebbdc225ffc2aa04b4057bc2817f1b94b6978cfc3",
        strip_prefix = "zlib-1.2.11",
        urls = [
            "https://github.com/madler/zlib/archive/v1.2.11.zip",
        ],
        build_file = "@//dependency_support/zlib:bundled.BUILD.bazel",
        system_build_file = "@//dependency_support/systemlibs:zlib.BUILD",
    )

    http_archive(
        name = "com_github_grpc_grpc",
        urls = ["https://github.com/grpc/grpc/archive/v1.32.0.tar.gz"],
        sha256 = "f880ebeb2ccf0e47721526c10dd97469200e40b5f101a0d9774eb69efa0bd07a",
        strip_prefix = "grpc-1.32.0",
        patches = ["//dependency_support/com_github_grpc_grpc:grpc-cython.patch"],
    )
