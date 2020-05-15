# Copyright 2020 Google LLC
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

# pytype binary, test, library
load("//xls/dslx/fuzzer:build_defs.bzl", "generate_crasher_regression_tests")
# cc_proto_library is used in this file

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "cli_helpers",
    srcs = ["cli_helpers.py"],
    srcs_version = "PY3ONLY",
)

py_library(
    name = "ast_generator",
    srcs = ["ast_generator.py"],
    srcs_version = "PY3ONLY",
    deps = [
        "//xls/common:memoize",
        "//xls/dslx:ast",
        "//xls/dslx:bit_helpers",
        "//xls/dslx:scanner",
        "//xls/dslx:span",
    ],
)

py_test(
    name = "ast_generator_test",
    srcs = ["ast_generator_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ast_generator",
        "@com_google_absl_py//absl/testing:absltest",
        "//xls/dslx:fakefs_util",
        "//xls/dslx:parser_helpers",
        "//xls/dslx:span",
        "//xls/dslx:typecheck",
    ],
)

py_library(
    name = "run_fuzz",
    srcs = ["run_fuzz.py"],
    data = [
        ":find_failing_input_main",
        ":sample_runner_main",
        ":summarize_ir_main",
        "//xls/tools:ir_minimizer_main",
    ],
    srcs_version = "PY3ONLY",
    tags = ["optonly"],
    deps = [
        ":ast_generator",
        ":sample",
        ":sample_generator",
        ":sample_runner",
        "@com_google_absl_py//absl/logging",
        "@termcolor_archive//:termcolor",
        "//xls/common:runfiles",
    ],
)

py_test(
    name = "run_fuzz_sanity_test",
    srcs = ["run_fuzz_sanity_test.py"],
    data = glob(["testdata/run_fuzz_test*.x"]),
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ast_generator",
        ":run_fuzz",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "run_fuzz_test",
    srcs = ["run_fuzz_test.py"],
    data = glob(["testdata/run_fuzz_test.*.x"]) + [
        "//xls/tools:parse_ir",
    ],
    python_version = "PY3",
    shard_count = 50,
    srcs_version = "PY3",
    deps = [
        ":ast_generator",
        ":run_fuzz",
        ":sample",
        ":sample_runner",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@com_google_absl_py//absl/flags",
        "//xls/common:runfiles",
    ],
)

py_test(
    name = "run_fuzz_test_no_codegen",
    srcs = ["run_fuzz_test.py"],
    args = ["--codegen=false"],
    data = glob(["testdata/run_fuzz_test_no_codegen*.x"]) + [
        "//xls/tools:parse_ir",
    ],
    main = "run_fuzz_test.py",
    python_version = "PY3",
    shard_count = 50,
    srcs_version = "PY3",
    deps = [
        ":ast_generator",
        ":run_fuzz",
        ":sample",
        ":sample_runner",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@com_google_absl_py//absl/flags",
        "//xls/common:runfiles",
    ],
)

py_test(
    name = "run_fuzz_shll_test",
    srcs = ["run_fuzz_shll_test.py"],
    data = glob(["testdata/run_fuzz_shll_test*.x"]),
    python_version = "PY3",
    shard_count = 50,
    srcs_version = "PY3",
    deps = [
        ":ast_generator",
        ":run_fuzz",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@com_google_absl_py//absl/flags",
        "//xls/common:runfiles",
        "//xls/dslx:ast",
    ],
)

py_binary(
    name = "run_crasher",
    srcs = ["run_crasher.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":run_fuzz",
        ":sample",
        ":sample_runner",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "//xls/common:gfile",
    ],
)

filegroup(
    name = "run_crasher_sh",
    srcs = ["run_crasher.sh"],
)

py_library(
    name = "sample_runner",
    srcs = ["sample_runner.py"],
    data = [
        "//xls/dslx:ir_converter_main",
        "//xls/tools:codegen_main",
        "//xls/tools:eval_ir_main",
        "//xls/tools:opt_main",
        "//xls/tools:simulate_module_main",
    ],
    srcs_version = "PY3ONLY",
    deps = [
        ":sample",
        "@com_google_absl_py//absl/logging",
        "//xls/common:runfiles",
        "//xls/common:xls_error",
        "//xls/dslx:ast",
        "//xls/dslx:concrete_type",
        "//xls/dslx:deduce",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:typecheck",
        "//xls/dslx/interpreter",
        "//xls/dslx/interpreter:value",
        "//xls/dslx/interpreter:value_parser",
        "//xls/ir/python:format_preference",
        "//xls/ir/python:ir_parser",
        "//xls/ir/python:value",
    ],
)

py_test(
    name = "sample_runner_test",
    srcs = ["sample_runner_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":sample",
        ":sample_runner",
        "@com_google_absl_py//absl/testing:absltest",
        "//xls/dslx:concrete_type",
        "//xls/dslx:fakefs_util",
        "//xls/dslx:parser_helpers",
        "//xls/dslx/interpreter:value",
    ],
)

generate_crasher_regression_tests(
    srcs = glob(["crashers/*"]),
    prefix = "xls/dslx/fuzzer",
)

py_library(
    name = "run_fuzz_multiprocess_lib",
    srcs = ["run_fuzz_multiprocess.py"],
    srcs_version = "PY3ONLY",
    deps = [
        ":ast_generator",
        ":run_fuzz",
        ":sample",
        ":sample_generator",
        ":sample_runner",
        "@termcolor_archive//:termcolor",
        "//xls/common:gfile",
    ],
)

py_library(
    name = "sample",
    srcs = ["sample.py"],
    srcs_version = "PY3ONLY",
    deps = [
        "//xls/dslx/interpreter:value",
        "//xls/dslx/interpreter:value_parser",
    ],
)

py_library(
    name = "sample_generator",
    srcs = ["sample_generator.py"],
    srcs_version = "PY3ONLY",
    deps = [
        ":ast_generator",
        ":sample",
        "//xls/dslx:concrete_type",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx/interpreter:value",
    ],
)

py_binary(
    name = "run_fuzz_multiprocess",
    srcs = ["run_fuzz_multiprocess_main.py"],
    main = "run_fuzz_multiprocess_main.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cli_helpers",
        ":ast_generator",
        ":run_fuzz_multiprocess_lib",
        ":sample",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "//xls/common:gfile",
    ],
)

py_binary(
    name = "sample_runner_main",
    srcs = ["sample_runner_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":sample_runner",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)

py_binary(
    name = "run_fuzz_multiprocess_benchmark",
    srcs = ["run_fuzz_multiprocess_benchmark.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ast_generator",
        ":run_fuzz",
        ":run_fuzz_multiprocess_lib",
        ":sample",
        ":sample_generator",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)

py_test(
    name = "sample_test",
    srcs = ["sample_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":sample",
        "@com_google_absl_py//absl/testing:absltest",
        "//xls/dslx/interpreter:value",
    ],
)

py_test(
    name = "sample_generator_test",
    srcs = ["sample_generator_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ast_generator",
        ":sample",
        ":sample_generator",
        "//xls/common:test_base",
        "//xls/dslx:concrete_type",
    ],
)

cc_binary(
    name = "summarize_ir_main",
    srcs = ["summarize_ir_main.cc"],
    deps = [
        ":sample_summary_cc_proto",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:op",
    ],
)

cc_binary(
    name = "read_summary_main",
    srcs = ["read_summary_main.cc"],
    deps = [
        ":sample_summary_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:op",
    ],
)

cc_binary(
    name = "find_failing_input_main",
    srcs = ["find_failing_input_main.cc"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/ir",
        "//xls/ir:ir_interpreter",
        "//xls/ir:ir_parser",
        "//xls/ir:llvm_ir_jit",
        "//xls/ir:value",
    ],
)

py_test(
    name = "find_failing_input_main_test",
    srcs = ["find_failing_input_main_test.py"],
    data = [
        ":find_failing_input_main",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
    ],
)

cc_proto_library(
    name = "sample_summary_cc_proto",
    deps = [":sample_summary_proto"],
)

proto_library(
    name = "sample_summary_proto",
    srcs = ["sample_summary.proto"],
)

py_test(
    name = "run_fuzz_multiprocess_test",
    srcs = ["run_fuzz_multiprocess_test.py"],
    data = [":run_fuzz_multiprocess"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//xls/common:runfiles",
        "//xls/common:test_base",
    ],
)
