# Copyright 2024 The XLS Authors
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

# pytype tests are present in this file

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

[
    genrule(
        name = "interface_for_%s" % target,
        srcs = ["//xls/contrib/xlscc/examples:%s" % target],
        outs = ["%s.interface.binpb" % target],
        cmd = """
            $(location //xls/dev_tools:extract_interface_main) --binary_proto $< > $@
        """,
        tools = ["//xls/dev_tools:extract_interface_main"],
    )
    for target in [
        "simple_unsigned_pipelined_loop_fsm.ir",
        "simple_unsigned_pipelined_loop_fsm.opt.ir",
        "simple_unsigned_nested_pipelined_loop_fsm.ir",
        "simple_unsigned_nested_pipelined_loop_fsm.opt.ir",
        "simple_pipelined_loop_fsm.ir",
        "simple_pipelined_loop_fsm.opt.ir",
        "simple_backwards_pipelined_loop_fsm.ir",
        "simple_backwards_pipelined_loop_fsm.opt.ir",
        "simple_nested_pipelined_loop_fsm.ir",
        "simple_nested_pipelined_loop_fsm.opt.ir",
    ]
]

py_test(
    name = "artifact_analyzability_test",
    srcs = ["artifact_analyzability_test.py"],
    data = [
        ":simple_backwards_pipelined_loop_fsm.ir.interface.binpb",
        ":simple_backwards_pipelined_loop_fsm.opt.ir.interface.binpb",
        ":simple_nested_pipelined_loop_fsm.ir.interface.binpb",
        ":simple_nested_pipelined_loop_fsm.opt.ir.interface.binpb",
        ":simple_pipelined_loop_fsm.ir.interface.binpb",
        ":simple_pipelined_loop_fsm.opt.ir.interface.binpb",
        ":simple_unsigned_nested_pipelined_loop_fsm.ir.interface.binpb",
        ":simple_unsigned_nested_pipelined_loop_fsm.opt.ir.interface.binpb",
        ":simple_unsigned_pipelined_loop_fsm.ir.interface.binpb",
        ":simple_unsigned_pipelined_loop_fsm.opt.ir.interface.binpb",
    ],
    deps = [
        "//xls/common:runfiles",
        "//xls/ir:xls_ir_interface_py_pb2",
        "@abseil-py//absl/testing:absltest",
    ],
)

py_test(
    name = "eval_proc_main_test",
    srcs = ["eval_proc_main_test.py"],
    data = [
        "//xls/contrib/xlscc/integration_tests/testdata:eval_proc_main_test.ram_rewrites.textproto",
        "//xls/contrib/xlscc/integration_tests/testdata:eval_proc_main_test.test_memory.block.ir",
        "//xls/contrib/xlscc/integration_tests/testdata:eval_proc_main_test.test_memory.ir",
        "//xls/contrib/xlscc/integration_tests/testdata:eval_proc_main_test.test_memory.opt.ir",
        "//xls/contrib/xlscc/integration_tests/testdata:eval_proc_main_test.test_memory.sig.textproto",
        "//xls/tools:eval_proc_main",
    ],
    deps = [
        "//xls/common:runfiles",
        "@abseil-py//absl/logging",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
    ],
)
