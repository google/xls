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

load("@xls_pip_deps//:requirements.bzl", "requirement")

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
        "//xls/dslx:ast_helpers",
        "//xls/dslx:bit_helpers",
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_pos",
        "//xls/dslx/python:scanner",
    ],
)

py_test(
    name = "ast_generator_test",
    srcs = ["ast_generator_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ast_generator",
        "//xls/common:test_base",
        "//xls/dslx:ast_helpers",
        "//xls/dslx:fakefs_test_util",
        "//xls/dslx:parser_helpers",
        "//xls/dslx:span",
        "//xls/dslx/python:cpp_typecheck",
        "//xls/dslx/python:import_routines",
        "//xls/dslx/python:scanner",
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
        requirement("termcolor"),
        "//xls/common:runfiles",
        "@com_google_absl_py//absl/logging",
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
    tags = ["optonly"],
    deps = [
        ":ast_generator",
        ":run_fuzz",
        ":sample",
        ":sample_runner",
        "//xls/common:runfiles",
        "//xls/common:test_base",
        "@com_google_absl_py//absl/flags",
        "@com_google_absl_py//absl/testing:parameterized",
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
    tags = ["optonly"],  # Times out in fastbuild TAP 2020-10-20.
    deps = [
        ":ast_generator",
        ":run_fuzz",
        ":sample",
        ":sample_runner",
        "//xls/common:runfiles",
        "//xls/common:test_base",
        "@com_google_absl_py//absl/flags",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "run_fuzz_shll_test",
    srcs = ["run_fuzz_shll_test.py"],
    data = glob(["testdata/run_fuzz_shll_test*.x"]),
    python_version = "PY3",
    shard_count = 50,
    srcs_version = "PY3",
    tags = ["optonly"],
    deps = [
        ":ast_generator",
        ":run_fuzz",
        "//xls/common:runfiles",
        "//xls/dslx/python:cpp_ast",
        "@com_google_absl_py//absl/flags",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
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
        "//xls/common:gfile",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
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
        "//xls/dslx:cpp_ir_converter_main",
        "//xls/tools:codegen_main",
        "//xls/tools:eval_ir_main",
        "//xls/tools:opt_main",
        "//xls/tools:simulate_module_main",
    ],
    srcs_version = "PY3ONLY",
    deps = [
        ":sample",
        "//xls/common:check_simulator",
        "//xls/common:revision",
        "//xls/common:runfiles",
        "//xls/common:xls_error",
        "//xls/dslx/interpreter:value_parser",
        "//xls/dslx/python:cpp_ast",
        "//xls/dslx/python:cpp_concrete_type",
        "//xls/dslx/python:cpp_type_info",
        "//xls/dslx/python:cpp_typecheck",
        "//xls/dslx/python:import_routines",
        "//xls/dslx/python:interp_value",
        "//xls/dslx/python:interpreter",
        "//xls/ir/python:format_preference",
        "//xls/ir/python:ir_parser",
        "@com_google_absl_py//absl/logging",
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
        "//xls/common:check_simulator",
        "//xls/common:test_base",
        "//xls/dslx:fakefs_test_util",
        "//xls/dslx:parser_helpers",
        "//xls/dslx/python:cpp_concrete_type",
        "//xls/dslx/python:interp_value",
        "//xls/ir/python:bits",
        "@com_google_absl_py//absl/logging",
    ],
)

generate_crasher_regression_tests(
    name = "regression_tests",
    srcs = glob(["crashers/*"]),
    prefix = package_name(),
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
        requirement("termcolor"),
        "//xls/common:gfile",
        "//xls/common:multiprocess",
    ],
)

py_library(
    name = "sample",
    srcs = ["sample.py"],
    srcs_version = "PY3ONLY",
    deps = [
        "//xls/dslx/interpreter:value_parser",
        "//xls/dslx/python:interp_value",
    ],
)

py_library(
    name = "sample_generator",
    srcs = ["sample_generator.py"],
    srcs_version = "PY3ONLY",
    deps = [
        ":ast_generator",
        ":sample",
        "//xls/dslx/python:cpp_concrete_type",
        "//xls/dslx/python:import_routines",
        "//xls/dslx/python:interp_value",
        "//xls/dslx/python:interpreter",
        "//xls/ir/python:bits",
    ],
)

py_binary(
    name = "run_fuzz_multiprocess",
    srcs = ["run_fuzz_multiprocess_main.py"],
    main = "run_fuzz_multiprocess_main.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ast_generator",
        ":cli_helpers",
        ":run_fuzz_multiprocess_lib",
        ":sample",
        requirement("psutil"),
        "//xls/common:gfile",
        "//xls/common:multiprocess",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
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
        "//xls/dslx/python:interp_value",
        "@com_google_absl_py//absl/testing:absltest",
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
        "//xls/dslx/python:cpp_concrete_type",
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
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:op",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_binary(
    name = "find_failing_input_main",
    srcs = ["find_failing_input_main.cc"],
    deps = [
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/logging",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:value",
        "//xls/jit:ir_jit",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
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
