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

load("//xls/build_rules:iverilog_test.bzl", "iverilog_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

iverilog_test(
    name = "uart_receiver_one_byte_test",
    srcs = [
        "uart_receiver.v",
        "uart_receiver_test.v",
        "xls_assertions.inc",
    ],
    main = "uart_receiver_test.v",
    tick_defines = {
        "TEST_SINGLE_BYTE": 1,
    },
    top = "uart_receiver_test",
)

iverilog_test(
    name = "uart_receiver_two_byte_test",
    srcs = [
        "uart_receiver.v",
        "uart_receiver_test.v",
        "xls_assertions.inc",
    ],
    main = "uart_receiver_test.v",
    top = "uart_receiver_test",
)

# uart_transmitter_two_bytes_*_test variants.

iverilog_test(
    name = "uart_transmitter_two_bytes_early_cpb2_test",
    srcs = [
        "uart_transmitter.v",
        "uart_transmitter_two_bytes_test.v",
        "xls_assertions.inc",
    ],
    main = "uart_transmitter_two_bytes_test.v",
    tick_defines = {
        "PRESENT_BIT_EARLY": 1,
        "CLOCKS_PER_BAUD": 2,
    },
    top = "uart_transmitter_two_bytes_test",
)

iverilog_test(
    name = "uart_transmitter_two_bytes_early_cpb4_test",
    srcs = [
        "uart_transmitter.v",
        "uart_transmitter_two_bytes_test.v",
        "xls_assertions.inc",
    ],
    main = "uart_transmitter_two_bytes_test.v",
    tick_defines = {
        "PRESENT_BIT_EARLY": 1,
        "CLOCKS_PER_BAUD": 4,
    },
    top = "uart_transmitter_two_bytes_test",
)

iverilog_test(
    name = "uart_transmitter_two_bytes_late_cpb2_test",
    srcs = [
        "uart_transmitter.v",
        "uart_transmitter_two_bytes_test.v",
        "xls_assertions.inc",
    ],
    main = "uart_transmitter_two_bytes_test.v",
    tick_defines = {
        "CLOCKS_PER_BAUD": 2,
    },
    top = "uart_transmitter_two_bytes_test",
)

iverilog_test(
    name = "uart_transmitter_two_bytes_late_cpb4_test",
    srcs = [
        "uart_transmitter.v",
        "uart_transmitter_two_bytes_test.v",
        "xls_assertions.inc",
    ],
    main = "uart_transmitter_two_bytes_test.v",
    tick_defines = {
        "CLOCKS_PER_BAUD": 4,
    },
    top = "uart_transmitter_two_bytes_test",
)

# "main" programs.

iverilog_test(
    name = "uart_transmitter_two_bytes_main_wrapper",
    srcs = [
        "uart_transmitter.v",
        "uart_transmitter_two_bytes_main.v",
        "uart_transmitter_two_bytes_main_wrapper.v",
        "xls_assertions.inc",
    ],
    main = "uart_transmitter_two_bytes_main_wrapper.v",
    top = "top_wrapper",
)

iverilog_test(
    name = "ident_8b_main_wrapper",
    srcs = [
        "ident_8b_main.v",
        "ident_8b_main_wrapper.v",
        "uart_receiver.v",
        "uart_transmitter.v",
        "xls_assertions.inc",
    ],
    main = "ident_8b_main_wrapper.v",
    top = "top_wrapper",
)

# wrap_io: add1_8b

genrule(
    name = "wrap_io_ice40_add1_8b",
    srcs = [
        "add1_8b.v",
        "add1_8b.add1_8b.sig",
    ],
    outs = ["wrap_io_ice40_add1_8b.v"],
    cmd = """$(location //xls/contrib/ice40:wrap_io_main) \\
        -include ./xls/contrib/ice40/uncore_rtl/ice40/add1_8b.v \\
        -signature_proto_path $(location add1_8b.add1_8b.sig) \\
        -wrapped_module_name add1_8b \\
        -target_device ice40 > $(OUTS)
        """,
    tools = ["//xls/contrib/ice40:wrap_io_main"],
)

iverilog_test(
    name = "wrap_io_ice40_add1_8b_test",
    srcs = [
        "add1_8b.v",
        "uart_receiver.v",
        "uart_transmitter.v",
        "wrap_io_ice40_testbench.v",
        "xls_assertions.inc",
        "//xls/contrib/ice40/uncore_rtl/ice40:wrap_io_ice40_add1_8b",
    ],
    main = "wrap_io_ice40_testbench.v",
    top = "tb",
)

# wrap_io: repeat_byte_4

genrule(
    name = "wrap_io_ice40_repeat_byte_4",
    srcs = [
        "repeat_byte_4.v",
        "repeat_byte_4.repeat_byte_4.sig",
    ],
    outs = ["wrap_io_ice40_repeat_byte_4.v"],
    cmd = """$(location //xls/contrib/ice40:wrap_io_main) \\
        -include ./xls/contrib/ice40/uncore_rtl/ice40/repeat_byte_4.v \\
        -signature_proto_path $(location repeat_byte_4.repeat_byte_4.sig) \\
        -wrapped_module_name repeat_byte_4 \\
        -target_device ice40 > $(OUTS)
        """,
    tools = ["//xls/contrib/ice40:wrap_io_main"],
)

iverilog_test(
    name = "wrap_io_ice40_repeat_byte_4_test",
    srcs = [
        "repeat_byte_4.v",
        "uart_receiver.v",
        "uart_transmitter.v",
        "wrap_io_ice40_repeat_byte_4_testbench.v",
        "xls_assertions.inc",
        "//xls/contrib/ice40/uncore_rtl/ice40:wrap_io_ice40_repeat_byte_4",
    ],
    main = "wrap_io_ice40_repeat_byte_4_testbench.v",
    top = "tb",
)

cc_library(
    name = "iceprog_includes",
    data = [
        "icestick.pcf",
        "uart_receiver.v",
        "uart_transmitter.v",
    ],
)
