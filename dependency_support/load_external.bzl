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

    # Release 2024-01-22, current as of 2024-06-26
    # zlib is added automatically by gRPC, but the zlib BUILD file used by gRPC
    # does not include all the source code (e.g., gzread is missing) which
    # breaks other users of zlib like iverilog. So add zlib explicitly here with
    # a working BUILD file.
    # Needs to be early in this file to make sure this is the version
    # picked -- Version 1.3.x fixes function prototype warnings in c++20.
    http_archive(
        name = "zlib",
        sha256 = "50b24b47bf19e1f35d2a21ff36d2a366638cdf958219a66f30ce0861201760e6",
        strip_prefix = "zlib-1.3.1",
        urls = [
            "https://github.com/madler/zlib/archive/v1.3.1.zip",
        ],
        build_file = "//dependency_support/zlib:bundled.BUILD.bazel",
    )

    # V 1.14.0 (released 2023-08-02, current as of 2024-06-26)
    http_archive(
        name = "com_google_googletest",
        urls = ["https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip"],
        strip_prefix = "googletest-1.14.0",
        sha256 = "1f357c27ca988c3f7c6b4bf68a9395005ac6761f034046e9dde0896e3aba00e4",
    )

    # LTS 20240116.2 (released 2024-04-08, current as of 2024-06-26)
    http_archive(
        name = "com_google_absl",
        urls = ["https://github.com/abseil/abseil-cpp/archive/refs/tags/20240116.2.tar.gz"],
        strip_prefix = "abseil-cpp-20240116.2",
        sha256 = "733726b8c3a6d39a4120d7e45ea8b41a434cdacde401cba500f14236c49b39dc",
    )

    # Released 2024-06-03, current as of 2024-06-26
    # Protobuf depends on Skylib
    # Load bazel skylib as per
    # https://github.com/bazelbuild/bazel-skylib/releases
    http_archive(
        name = "bazel_skylib",
        urls = [
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
        ],
        sha256 = "bc283cdfcd526a52c3201279cda4bc298652efa898b10b4db0837dc51652756f",
    )

    http_archive(
        name = "boringssl",
        # Commit date: 2024-06-24
        # Note for updating: we need to use a commit from the main-with-bazel branch.
        strip_prefix = "boringssl-e6b03733628149a89a1d18b3ef8f39aa1055aba8",
        sha256 = "006596f84d9cc142d9d6c48600cf6208f9d24426943b05e8bcda06e523f69dc8",
        urls = ["https://github.com/google/boringssl/archive/e6b03733628149a89a1d18b3ef8f39aa1055aba8.tar.gz"],
    )

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

    # Version release tag 2023-01-11
    http_archive(
        name = "com_google_absl_py",
        strip_prefix = "abseil-py-1.4.0",
        urls = ["https://github.com/abseil/abseil-py/archive/refs/tags/v1.4.0.tar.gz"],
        sha256 = "0fb3a4916a157eb48124ef309231cecdfdd96ff54adf1660b39c0d4a9790a2c0",
    )

    # Released on 2024-06-01, current as of 2024-06-26
    http_archive(
        name = "com_googlesource_code_re2",
        strip_prefix = "re2-2024-06-01",
        sha256 = "7326c74cddaa90b12090fcfc915fe7b4655723893c960ee3c2c66e85c5504b6c",
        urls = [
            "https://github.com/google/re2/archive/refs/tags/2024-06-01.tar.gz",
        ],
        repo_mapping = {
            "@abseil-cpp": "@com_google_absl",
        },
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

    # Commit from 2024-06-26
    http_archive(
        name = "com_google_riegeli",
        sha256 = "38fd4b6bc24958ae51e1a5a0eb57ce9c3dbbaf5034a78453a4d133597fbf31e4",
        strip_prefix = "riegeli-cb68d579f108c96831b6a7815da43ff24b4e5242",
        url = "https://github.com/google/riegeli/archive/cb68d579f108c96831b6a7815da43ff24b4e5242.tar.gz",
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

    # Released 2024-06-07, current as of 2024-06-26.
    http_archive(
        name = "com_github_grpc_grpc",
        urls = ["https://github.com/grpc/grpc/archive/v1.64.2.tar.gz"],
        patches = ["//dependency_support/com_github_grpc_grpc:0001-Add-absl-status-to-deps.patch"],
        sha256 = "c682fc39baefc6e804d735e6b48141157b7213602cc66dbe0bf375b904d8b5f9",
        strip_prefix = "grpc-1.64.2",
        repo_mapping = {
            "@local_config_python": "@project_python",
            "@system_python": "@project_python",
        },
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

    # Released 2024-05-23, current as of 2024-06-26.
    http_archive(
        name = "com_google_benchmark",
        urls = ["https://github.com/google/benchmark/archive/refs/tags/v1.8.4.tar.gz"],
        sha256 = "3e7059b6b11fb1bbe28e33e02519398ca94c1818874ebed18e504dc6f709be45",
        strip_prefix = "benchmark-1.8.4",
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

    # Released 2024-01-24, current as of 2024-06-26
    http_archive(
        name = "rules_license",
        urls = [
            "https://github.com/bazelbuild/rules_license/releases/download/0.0.8/rules_license-0.0.8.tar.gz",
        ],
        sha256 = "241b06f3097fd186ff468832150d6cc142247dc42a32aaefb56d0099895fd229",
    )

    # 2022-09-19
    http_archive(
        name = "com_grail_bazel_compdb",
        sha256 = "a3ff6fe238eec8202270dff75580cba3d604edafb8c3408711e82633c153efa8",
        strip_prefix = "bazel-compilation-database-940cedacdb8a1acbce42093bf67f3a5ca8b265f7",
        urls = ["https://github.com/grailbio/bazel-compilation-database/archive/940cedacdb8a1acbce42093bf67f3a5ca8b265f7.tar.gz"],
    )

    # Tagged 2024-08-23, current as of 2024-08-24
    VERIBLE_TAG = "v0.0-3756-gda9a0f8c"
    http_archive(
        name = "verible",
        sha256 = "0d45e646ce8cf618c55e614f827aead0377c34035be04b843aee225ea5be4527",
        strip_prefix = "verible-" + VERIBLE_TAG.lstrip('v'),
        urls = ["https://github.com/chipsalliance/verible/archive/refs/tags/" + VERIBLE_TAG + ".tar.gz"],
        patch_args = ["-p1"],
        patches = ["//dependency_support/verible:visibility.patch"],
    )

    # Used by Verible; current as of 2024-06-26
    http_archive(
        name = "jsonhpp",
        build_file = "@verible//bazel:jsonhpp.BUILD",
        sha256 = "0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406",
        strip_prefix = "json-3.11.3",
        urls = [
            "https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz",
        ],
    )

    # Released 2024-06-03, current as of 2024-06-26
    http_archive(
        name = "rules_pkg",
        urls = ["https://github.com/bazelbuild/rules_pkg/releases/download/1.0.0/rules_pkg-1.0.0.tar.gz"],
        sha256 = "cad05f864a32799f6f9022891de91ac78f30e0fa07dc68abac92a628121b5b11",
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
