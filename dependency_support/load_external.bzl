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

    # Release 2024-01-22, current as of 2024-03-14
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
        build_file = "@com_google_xls//dependency_support/zlib:bundled.BUILD.bazel",
    )

    # Released 2023-09-20, current as of 2024-03-14
    http_archive(
        name = "rules_cc",
        urls = ["https://github.com/bazelbuild/rules_cc/releases/download/0.0.9/rules_cc-0.0.9.tar.gz"],
        sha256 = "2037875b9a4456dce4a79d112a8ae885bbc4aad968e6587dca6e64f3a0900cdf",
        strip_prefix = "rules_cc-0.0.9",
    )

    # V 1.14.0 (released 2023-08-02, current as of 2024-03-14)
    http_archive(
        name = "com_google_googletest",
        urls = ["https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip"],
        strip_prefix = "googletest-1.14.0",
        sha256 = "1f357c27ca988c3f7c6b4bf68a9395005ac6761f034046e9dde0896e3aba00e4",
    )

    # LTS 20240116.1 (released 2024-02-12, current as of 2024-03-14)
    http_archive(
        name = "com_google_absl",
        urls = ["https://github.com/abseil/abseil-cpp/archive/refs/tags/20240116.1.tar.gz"],
        strip_prefix = "abseil-cpp-20240116.1",
        sha256 = "3c743204df78366ad2eaf236d6631d83f6bc928d1705dd0000b872e53b73dc6a",
    )

    # Released 2023-11-06, current as of 2024-03-14
    # Protobuf depends on Skylib
    # Load bazel skylib as per
    # https://github.com/bazelbuild/bazel-skylib/releases
    http_archive(
        name = "bazel_skylib",
        urls = [
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.5.0/bazel-skylib-1.5.0.tar.gz",
        ],
        sha256 = "cd55a062e763b9349921f0f5db8c3933288dc8ba4f76dd9416aac68acee3cb94",
    )

    http_archive(
        name = "boringssl",
        # Commit date: 2024-03-13
        # Note for updating: we need to use a commit from the main-with-bazel branch.
        strip_prefix = "boringssl-b84aa830c43eeac47374b2a179063250a39496ef",
        sha256 = "c1b8d25b76d31877066650554c18049fe647f8f996fe3ed2fa61aea171bc34d1",
        urls = ["https://github.com/google/boringssl/archive/b84aa830c43eeac47374b2a179063250a39496ef.tar.gz"],
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

    # Released on 2024-03-01, current as of 2024-03-14
    http_archive(
        name = "com_googlesource_code_re2",
        strip_prefix = "re2-2024-03-01",
        sha256 = "7b2b3aa8241eac25f674e5b5b2e23d4ac4f0a8891418a2661869f736f03f57f4",
        urls = [
            "https://github.com/google/re2/archive/refs/tags/2024-03-01.tar.gz",
        ],
        repo_mapping = {
            "@abseil-cpp": "@com_google_absl",
        },
    )

    # Released on 2022-12-27.
    # Current as of 2024-03-14 would be 6.0.0rc2, but that does not work yet
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

    # Released on 2023-08-22, current as of 2023-09-26
    # https://github.com/bazelbuild/rules_python/releases/tag/0.25.0
    http_archive(
        name = "rules_python",
        sha256 = "5868e73107a8e85d8f323806e60cad7283f34b32163ea6ff1020cf27abef6036",
        strip_prefix = "rules_python-0.25.0",
        url = "https://github.com/bazelbuild/rules_python/releases/download/0.25.0/rules_python-0.25.0.tar.gz",
    )

    http_archive(
        name = "z3",
        urls = ["https://github.com/Z3Prover/z3/archive/z3-4.12.2.tar.gz"],
        sha256 = "9f58f3710bd2094085951a75791550f547903d75fe7e2fcb373c5f03fc761b8f",
        strip_prefix = "z3-z3-4.12.2",
        build_file = "@com_google_xls//dependency_support/z3:bundled.BUILD.bazel",
        # Fix gcc 13.x build failure
        # https://github.com/Z3Prover/z3/issues/6722
        patches = ["@com_google_xls//dependency_support/z3:6723.patch"],
    )

    # Release 2024-02-23, current as of 2023-03-14
    http_archive(
        name = "io_bazel_rules_closure",
        sha256 = "70ef2b4da987bf0d266e663d7c251eac509ff70dd65bba02d41d1e86e840a569",
        strip_prefix = "rules_closure-0.13.0",
        urls = [
            "https://github.com/bazelbuild/rules_closure/archive/0.13.0.tar.gz",
        ],
    )

    # Commit from 2024-02-22
    http_archive(
        name = "linenoise",
        sha256 = "839ed407fe0dfa5fd7dd103abfc695dee72fea2840df8d4250ad42b0e64839e8",
        strip_prefix = "linenoise-d895173d679be70bcd8b23041fff3e458e1a3506",
        urls = ["https://github.com/antirez/linenoise/archive/d895173d679be70bcd8b23041fff3e458e1a3506.tar.gz"],
        build_file = "@com_google_xls//dependency_support/linenoise:bundled.BUILD.bazel",
    )

    # Needed by fuzztest.
    http_archive(
        name = "com_google_riegeli",
        sha256 = "f8386e44e16d044c1d7151c0b553bb7075d79583d4fa9e613a4be452599e0795",
        strip_prefix = "riegeli-411cda7f6aa81f8b8591b04cf141b1decdcc928c",
        url = "https://github.com/google/riegeli/archive/411cda7f6aa81f8b8591b04cf141b1decdcc928c.tar.gz",
    )

    # Needed by fuzztest.
    http_archive(
        name = "net_zstd",
        build_file = "@com_google_riegeli//third_party:net_zstd.BUILD",
        sha256 = "b6c537b53356a3af3ca3e621457751fa9a6ba96daf3aebb3526ae0f610863532",
        strip_prefix = "zstd-1.4.5/lib",
        urls = ["https://github.com/facebook/zstd/archive/v1.4.5.zip"],
    )

    # Needed by fuzztest.
    http_archive(
        name = "snappy",
        build_file = "@com_google_riegeli//third_party:snappy.BUILD",
        sha256 = "38b4aabf88eb480131ed45bfb89c19ca3e2a62daeb081bdf001cfb17ec4cd303",
        strip_prefix = "snappy-1.1.8",
        urls = ["https://github.com/google/snappy/archive/1.1.8.zip"],
    )

    # Needed by fuzztest.
    http_archive(
        name = "org_brotli",
        sha256 = "84a9a68ada813a59db94d83ea10c54155f1d34399baf377842ff3ab9b3b3256e",
        strip_prefix = "brotli-3914999fcc1fda92e750ef9190aa6db9bf7bdb07",
        urls = ["https://github.com/google/brotli/archive/3914999fcc1fda92e750ef9190aa6db9bf7bdb07.zip"],
    )

    # Needed by fuzztest.
    http_archive(
        name = "highwayhash",
        build_file = "@com_google_riegeli//third_party:highwayhash.BUILD",
        sha256 = "cf891e024699c82aabce528a024adbe16e529f2b4e57f954455e0bf53efae585",
        strip_prefix = "highwayhash-276dd7b4b6d330e4734b756e97ccfb1b69cc2e12",
        urls = ["https://github.com/google/highwayhash/archive/276dd7b4b6d330e4734b756e97ccfb1b69cc2e12.zip"],
    )

    # Needed by or-tools.
    http_archive(
        name = "glpk",
        build_file = "@com_google_ortools//bazel:glpk.BUILD.bazel",
        sha256 = "4a1013eebb50f728fc601bdd833b0b2870333c3b3e5a816eeba921d95bec6f15",
        url = "http://ftp.gnu.org/gnu/glpk/glpk-5.0.tar.gz",
    )

    # Needed by or-tools.
    http_archive(
        name = "scip",
        build_file = "@com_google_ortools//bazel:scip.BUILD.bazel",
        patches = ["@com_google_ortools//bazel:scip.patch"],
        patch_args = ["-p1"],
        sha256 = "b6daf54c37d02564b12fb32ec3bb7a105710eb0026adeafc602af4435fa94685",
        strip_prefix = "scip-810",
        url = "https://github.com/scipopt/scip/archive/refs/tags/v810.tar.gz",
    )

    http_archive(
        name = "bliss",
        build_file = "@com_google_ortools//bazel:bliss.BUILD.bazel",
        patches = ["@com_google_ortools//bazel:bliss-0.73.patch"],
        sha256 = "f57bf32804140cad58b1240b804e0dbd68f7e6bf67eba8e0c0fa3a62fd7f0f84",
        url = "https://github.com/google/or-tools/releases/download/v9.0/bliss-0.73.zip",
    )

    # Released on 2023-11-28, current as of 2024-02-01.
    # https://github.com/grpc/grpc/releases/tag/v1.60.0
    http_archive(
        name = "com_github_grpc_grpc",
        urls = ["https://github.com/grpc/grpc/archive/v1.60.0.tar.gz"],
        integrity = "sha256-Q3BouLd307M52pTTSY8dwgZCrJv6dttDq91SIYaxVCs=",
        strip_prefix = "grpc-1.60.0",
        repo_mapping = {
            "@local_config_python": "@python39",
            "@system_python": "@python39",
        },
    )

    # Used by xlscc.
    http_archive(
        name = "com_github_hlslibs_ac_types",
        urls = ["https://github.com/hlslibs/ac_types/archive/57d89634cb5034a241754f8f5347803213dabfca.tar.gz"],
        sha256 = "7ab5e2ee4c675ef6895fdd816c32349b3070dc8211b7d412242c66d0c6e8edca",
        strip_prefix = "ac_types-57d89634cb5034a241754f8f5347803213dabfca",
        build_file = "@com_google_xls//dependency_support/com_github_hlslibs_ac_types:bundled.BUILD.bazel",
    )

    # Released 2023-10-18, current as of 2024-03-14
    http_archive(
        name = "platforms",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.8/platforms-0.0.8.tar.gz",
            "https://github.com/bazelbuild/platforms/releases/download/0.0.8/platforms-0.0.8.tar.gz",
        ],
        sha256 = "8150406605389ececb6da07cbcb509d5637a3ab9a24bc69b1101531367d89d74",
    )

    # Released 2024-02-22, current as of 2024-02-22.
    # TODO(rigge): switch back to stable releases when or-tools cuts a release
    # against abseil w/ VLOG.
    ORTOOLS_COMMIT = "05aa100b904d23da218b6f41bfab9d20b930a3b4"
    http_archive(
        name = "com_google_ortools",
        urls = ["https://github.com/google/or-tools/archive/{commit}.tar.gz".format(commit = ORTOOLS_COMMIT)],
        sha256 = "f0db745dca2da71038f1dffe58319906842d449e4d7f55823495be159d40c7f0",
        strip_prefix = "or-tools-" + ORTOOLS_COMMIT,
        patch_args = ["-p1"],
        # Removes undesired dependencies like Eigen, BLISS, SCIP
        patches = [
            "@com_google_xls//dependency_support/com_google_ortools:0001-Fix-GLPK-Eigen-and-SCIP-deps.patch",
            "@com_google_xls//dependency_support/com_google_ortools:0002-Remove-duplicate-logtostderr-flag.patch",
        ],
    )

    # Released 2023-08-31, current as of 2024-03-14.
    http_archive(
        name = "com_google_benchmark",
        urls = ["https://github.com/google/benchmark/archive/refs/tags/v1.8.3.tar.gz"],
        sha256 = "6bc180a57d23d4d9515519f92b0c83d61b05b5bab188961f36ac7b06b0d9e9ce",
        strip_prefix = "benchmark-1.8.3",
    )

    # Updated to head on 2024-03-14
    FUZZTEST_COMMIT = "393ae75c0fca5f9892e73969da5d6bce453ad318"
    http_archive(
        name = "com_google_fuzztest",
        strip_prefix = "fuzztest-" + FUZZTEST_COMMIT,
        url = "https://github.com/google/fuzztest/archive/" + FUZZTEST_COMMIT + ".zip",
        sha256 = "a0558ceb617d78ee93d7e6b62930b4aeebc02f1e5817d4d0dae53699f6f6c352",
        patch_args = ["-p1", "-R"],  # reverse patch until we upgrade bazel to 7.1; see: bazelbuild/bazel#19233.
        patches = ["@com_google_xls//dependency_support/com_google_fuzztest:e317d5277e34948ae7048cb5e48309e0288e8df3.patch"],
    )

    # Released 2024-01-24, current as of 2024-03-14
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

    # 2023-12-15  Last version compatible with absl with or without VLOG()
    http_archive(
        name = "verible",
        sha256 = "133bb3f7b041ce5009f6bb369ed62f1d6c3760e3ab9b44ab08484a7245d096d3",
        strip_prefix = "verible-0.0-3498-g82ac5189",
        urls = ["https://github.com/chipsalliance/verible/archive/refs/tags/v0.0-3498-g82ac5189.tar.gz"],
        patch_args = ["-p1"],
        patches = ["@com_google_xls//dependency_support/verible:visibility.patch"],
    )

    # Same as Verible as of 2023-05-18
    http_archive(
        name = "jsonhpp",
        build_file = "@verible//bazel:jsonhpp.BUILD",
        sha256 = "081ed0f9f89805c2d96335c3acfa993b39a0a5b4b4cef7edb68dd2210a13458c",
        strip_prefix = "json-3.10.2",
        urls = [
            "https://github.com/nlohmann/json/archive/refs/tags/v3.10.2.tar.gz",
        ],
    )

    # Version 1.4.7 released on 17.12.2020
    # https://github.com/facebook/zstd/releases/tag/v1.4.7
    # Updated 23.11.2023
    http_archive(
        name = "com_github_facebook_zstd",
        sha256 = "192cbb1274a9672cbcceaf47b5c4e9e59691ca60a357f1d4a8b2dfa2c365d757",
        strip_prefix = "zstd-1.4.7",
        urls = ["https://github.com/facebook/zstd/releases/download/v1.4.7/zstd-1.4.7.tar.gz"],
        build_file = "@//dependency_support/com_github_facebook_zstd:bundled.BUILD.bazel",
    )
