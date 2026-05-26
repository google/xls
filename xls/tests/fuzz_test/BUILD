# Copyright 2026 The XLS Authors
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

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "dslx_fuzz_test",
    "xls_dslx_library",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

xls_dslx_library(
    name = "fuzz_test_dslx",
    srcs = ["fuzz_tests.x"],
)

dslx_fuzz_test(
    name = "test_arbitrary_fuzz_test",
    library = ":fuzz_test_dslx",
    test_function = "test_arbitrary",
)

dslx_fuzz_test(
    name = "test_no_domains_fuzz_test",
    library = ":fuzz_test_dslx",
    test_function = "test_no_domains",
)

dslx_fuzz_test(
    name = "test_arbitrary_opt_fuzz_test",
    library = ":fuzz_test_dslx",
    opt_ir = True,
    test_function = "test_arbitrary",
)

dslx_fuzz_test(
    name = "test_range_fuzz_test",
    library = ":fuzz_test_dslx",
    test_function = "test_range",
)

dslx_fuzz_test(
    name = "test_element_of_fuzz_test",
    library = ":fuzz_test_dslx",
    test_function = "test_element_of",
)

dslx_fuzz_test(
    name = "test_tuple_fuzz_test",
    library = ":fuzz_test_dslx",
    test_function = "test_tuple",
)

xls_dslx_library(
    name = "enum_tests_dslx",
    srcs = ["enum_tests.x"],
)

dslx_fuzz_test(
    name = "arbitrary_enum_fuzz_test",
    library = ":enum_tests_dslx",
    test_function = "arbitrary_enum",
)

dslx_fuzz_test(
    name = "oneof_enum_fuzz_test",
    library = ":enum_tests_dslx",
    test_function = "oneof_enum",
)

xls_dslx_library(
    name = "array_tests_dslx",
    srcs = ["array_tests.x"],
)

dslx_fuzz_test(
    name = "array_fuzz_test",
    library = ":array_tests_dslx",
    test_function = "arbitrary_array",
)

dslx_fuzz_test(
    name = "array_of_tuples_fuzz_test",
    library = ":array_tests_dslx",
    test_function = "array_of_tuples",
)

dslx_fuzz_test(
    name = "tuple_with_array_fuzz_test",
    library = ":array_tests_dslx",
    test_function = "tuple_with_array",
)

dslx_fuzz_test(
    name = "big_array_fuzz_test",
    library = ":array_tests_dslx",
    test_function = "big_array",
)

dslx_fuzz_test(
    name = "tuple_with_big_array_fuzz_test",
    library = ":array_tests_dslx",
    test_function = "tuple_with_big_array",
)

xls_dslx_library(
    name = "struct_tests_dslx",
    srcs = ["struct_tests.x"],
)

dslx_fuzz_test(
    name = "flat_struct_domain_fuzz_test",
    library = ":struct_tests_dslx",
    test_function = "test_flat_struct_domain",
)

dslx_fuzz_test(
    name = "nested_struct_domain_fuzz_test",
    library = ":struct_tests_dslx",
    test_function = "test_nested_struct_domain",
)
