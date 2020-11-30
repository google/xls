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
load("//xls/build_rules:py_proto_library.bzl", "xls_py_proto_library")

package(
    default_visibility = ["//xls/contrib/xlscc:xlscc_internal"],
    licenses = ["notice"],  # Apache 2.0
)

proto_library(
    name = "hls_types_proto",
    srcs = ["hls_types.proto"],
    features = ["-proto_dynamic_mode_static_link"],
)

xls_py_proto_library(
    name = "hls_types_py_pb2",
    srcs = ["hls_types.proto"],
    internal_deps = [":hls_types_proto"],
)

py_library(
    name = "translator",
    srcs = [
        "translator.py",
    ],
    deps = [
        ":hls_types_py_pb2",
        requirement("pycparser"),
        "//xls/common:runfiles",
        "//xls/contrib/xlscc/parse:ext_c_ast",
        "//xls/ir/python:bits",
        "//xls/ir/python:fileno",
        "//xls/ir/python:function_builder",
        "//xls/ir/python:package",
        "//xls/ir/python:source_location",
        "//xls/ir/python:value",
    ],
)

cc_binary(
    name = "mandelbrot_test",
    testonly = 1,
    srcs = [
        "testdata/mandelbrot_test.cc",
    ],
)

py_test(
    name = "translator_test",
    srcs = ["translator_test.py"],
    data = [
        "testdata/mandelbrot_test.cc",
        ":mandelbrot_test",
    ],
    python_version = "PY3",
    shard_count = 30,
    deps = [
        ":translator",
        "//xls/contrib/xlscc/parse:ext_c_ast",
        "//xls/contrib/xlscc/parse:ext_c_parser",
        "//xls/interpreter/python:ir_interpreter",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)
