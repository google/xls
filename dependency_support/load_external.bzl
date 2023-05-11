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

    http_archive(
        name = "rules_cc",
        urls = ["https://github.com/bazelbuild/rules_cc/releases/download/0.0.5/rules_cc-0.0.5.tar.gz"],
        sha256 = "2004c71f3e0a88080b2bd3b6d3b73b4c597116db9c9a36676d0ffad39b849214",
        strip_prefix = "rules_cc-0.0.5",
    )

    http_archive(
        name = "com_google_googletest",
        urls = ["https://github.com/google/googletest/archive/0e6aac2571eb1753b8855d8d1f592df64d1a4828.zip"],  # 2022-11-14
        strip_prefix = "googletest-0e6aac2571eb1753b8855d8d1f592df64d1a4828",
        sha256 = "77bfecb8d930cbd97e24e7570a3dee9b09bad483aab47c96b7b7efb7d54332ff",
    )

    # LTS 20230125.3 (released 04 May 2023)
    http_archive(
        name = "com_google_absl",
        urls = ["https://github.com/abseil/abseil-cpp/archive/refs/tags/20230125.3.zip"],
        strip_prefix = "abseil-cpp-20230125.3",
        sha256 = "51d676b6846440210da48899e4df618a357e6e44ecde7106f1e44ea16ae8adc7",
    )

    # Protobuf depends on Skylib
    # Load bazel skylib as per
    # https://github.com/bazelbuild/bazel-skylib/releases
    http_archive(
        name = "bazel_skylib",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
        ],
        sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
    )

    http_archive(
        name = "boringssl",
        # Commit date: 2022-09-14
        # Note for updating: we need to use a commit from the main-with-bazel branch.
        strip_prefix = "boringssl-d345d68d5c4b5471290ebe13f090f1fd5b7e8f58",
        sha256 = "482796f369c8655dbda3be801ae98c47916ecd3bff223d007a723fd5f5ecba22",
        urls = ["https://github.com/google/boringssl/archive/d345d68d5c4b5471290ebe13f090f1fd5b7e8f58.zip"],
    )

    # Commit on 2023-02-09
    http_archive(
        name = "pybind11_bazel",
        strip_prefix = "pybind11_bazel-fc56ce8a8b51e3dd941139d329b63ccfea1d304b",
        urls = ["https://github.com/pybind/pybind11_bazel/archive/fc56ce8a8b51e3dd941139d329b63ccfea1d304b.tar.gz"],
        sha256 = "150e2105f9243c445d48f3820b5e4e828ba16c41f91ab424deae1fa81d2d7ac6",
    )

    # Updated 2022-11-29
    http_archive(
        name = "pybind11",
        build_file = "@pybind11_bazel//:pybind11.BUILD",
        strip_prefix = "pybind11-8720cf94d69a3fce302c2e2f7abed464a9e485c6",
        urls = ["https://github.com/pybind/pybind11/archive/8720cf94d69a3fce302c2e2f7abed464a9e485c6.tar.gz"],
        sha256 = "d004efbb63bdbc79f3227b69134e51af67f4747e88c783835040df16d7a2d02c",
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

    # Note - use @com_github_google_re2 instead of more canonical
    #        @com_google_re2 for consistency with dependency grpc
    #        which uses @com_github_google_re2.
    #          (see https://github.com/google/xls/issues/234)
    # Commit from 2023-03-17, current as of 2023-05-08.
    http_archive(
        name = "com_github_google_re2",
        sha256 = "d929e9f7d6d3648f98a9349770569a819d90e81cd8765b46e61bbd1de37ead9c",
        strip_prefix = "re2-578843a516fd1da7084ae46209a75f3613b6065e",
        urls = [
            "https://github.com/google/re2/archive/578843a516fd1da7084ae46209a75f3613b6065e.tar.gz",
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/re2/archive/578843a516fd1da7084ae46209a75f3613b6065e.tar.gz",
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
        urls = ["https://github.com/Z3Prover/z3/archive/z3-4.8.8.tar.gz"],
        sha256 = "6962facdcdea287c5eeb1583debe33ee23043144d0e5308344e6a8ee4503bcff",
        strip_prefix = "z3-z3-4.8.8",
        build_file = "@com_google_xls//dependency_support/z3:bundled.BUILD.bazel",
        # Fix Undefined Behavior (UB) of overflow in mpz::bitwise_not by cherry picking
        # https://github.com/Z3Prover/z3/commit/a96f5a9b425b6f5ba7e8ce1c1a75db6683c4bdc9 and
        # https://github.com/Z3Prover/z3/commit/9ebacd87e2ee8a79adfe128021fbfd444db7857a.
        patches = ["@com_google_xls//dependency_support/z3:mpz_cpp.patch"],
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

    # Released on 2022-12-8, current as of 2022-12-20.
    # https://github.com/grpc/grpc/releases/tag/v1.46.3
    http_archive(
        name = "com_github_grpc_grpc",
        urls = ["https://github.com/grpc/grpc/archive/v1.49.2.tar.gz"],
        sha256 = "cdeb805385fba23242bf87073e68d590c446751e09089f26e5e0b3f655b0f089",
        strip_prefix = "grpc-1.49.2",
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

    http_archive(
        name = "platforms",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.5/platforms-0.0.5.tar.gz",
            "https://github.com/bazelbuild/platforms/releases/download/0.0.5/platforms-0.0.5.tar.gz",
        ],
        sha256 = "379113459b0feaf6bfbb584a91874c065078aa673222846ac765f86661c27407",
    )

    # Released 2023-03-13, current as of 2023-05-08.
    http_archive(
        name = "com_google_ortools",
        urls = ["https://github.com/google/or-tools/archive/refs/tags/v9.6.tar.gz"],
        sha256 = "bc4b07dc9c23f0cca43b1f5c889f08a59c8f2515836b03d4cc7e0f8f2c879234",
        strip_prefix = "or-tools-9.6",
        # Removes undesired dependencies like Eigen, BLISS, SCIP
        patches = [
            "@com_google_xls//dependency_support/com_google_ortools:add_logging_prefix.diff",
            "@com_google_xls//dependency_support/com_google_ortools:no_glpk.diff",
            "@com_google_xls//dependency_support/com_google_ortools:no_scip_or_pdlp.diff",
        ],
    )

    http_archive(
        name = "com_google_benchmark",
        urls = ["https://github.com/google/benchmark/archive/refs/tags/v1.7.0.zip"],
        sha256 = "e0e6a0f2a5e8971198e5d382507bfe8e4be504797d75bb7aec44b5ea368fa100",
        strip_prefix = "benchmark-1.7.0",
    )

    http_archive(
        name = "rapidcheck",
        strip_prefix = "rapidcheck-ce2b602bbe89e9264fb8725cc3f04049951f29cb",
        urls = ["https://github.com/emil-e/rapidcheck/archive/ce2b602bbe89e9264fb8725cc3f04049951f29cb.zip"],
        build_file = "@//dependency_support/rapidcheck:bundled.BUILD.bazel",
        sha256 = "50af34562dfaa6dd183708f4c8ef6cfb7a7ea49d926cfd2c14c2fbd9844b06c8",
    )

    # Updated 2022-11-29
    http_archive(
        name = "pybind11_abseil",
        sha256 = "fe1911341b26cb8f46efe2799fd611fa448e10804d1ab9e4e5c16a0c54e87931",
        strip_prefix = "pybind11_abseil-6776a52004a92528789155b202508750049f584c",
        urls = ["https://github.com/pybind/pybind11_abseil/archive/6776a52004a92528789155b202508750049f584c.zip"],
        patches = ["@com_google_xls//dependency_support/pybind11_abseil:status_module.patch"],
    )

    # Updated 2023-2-1
    http_archive(
        name = "rules_license",
        urls = [
            "https://github.com/bazelbuild/rules_license/releases/download/0.0.4/rules_license-0.0.4.tar.gz",
        ],
        sha256 = "6157e1e68378532d0241ecd15d3c45f6e5cfd98fc10846045509fb2a7cc9e381",
    )

    # 2022-09-19
    http_archive(
        name = "com_grail_bazel_compdb",
        sha256 = "a3ff6fe238eec8202270dff75580cba3d604edafb8c3408711e82633c153efa8",
        strip_prefix = "bazel-compilation-database-940cedacdb8a1acbce42093bf67f3a5ca8b265f7",
        urls = ["https://github.com/grailbio/bazel-compilation-database/archive/940cedacdb8a1acbce42093bf67f3a5ca8b265f7.tar.gz"],
    )
