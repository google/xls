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

# Golden IR diff tests for xls/spin testdata.

load("//xls/spin:priv-defs.bzl", "spin_ir_golden_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],
)

exports_files(glob([
    "*.ir",
    "*.x",
]))

spin_ir_golden_test(
    name = "assertion_ir_golden",
    src = "assertion.x",
    dslx_top = "Asserting",
    golden = "assertion.ir",
)

spin_ir_golden_test(
    name = "blocking_receive_ir_golden",
    src = "blocking_receive.x",
    dslx_top = "Reader",
    golden = "blocking_receive.ir",
)

spin_ir_golden_test(
    name = "conditional_io_ir_golden",
    src = "conditional_io.x",
    convert_tests = True,
    dslx_top = "ConditionalIoTest",
    golden = "conditional_io.ir",
)

spin_ir_golden_test(
    name = "counter_ir_golden",
    src = "counter.x",
    convert_tests = True,
    dslx_top = "CounterTest",
    golden = "counter.ir",
)

spin_ir_golden_test(
    name = "multi_proc_ir_golden",
    src = "multi_proc.x",
    dslx_top = "ProcA",
    golden = "multi_proc.ir",
)

spin_ir_golden_test(
    name = "non_blocking_receive_ir_golden",
    src = "non_blocking_receive.x",
    dslx_top = "NbReader",
    golden = "non_blocking_receive.ir",
)

spin_ir_golden_test(
    name = "passthrough_ir_golden",
    src = "passthrough.x",
    convert_tests = True,
    dslx_top = "PassthroughTest",
    golden = "passthrough.ir",
)

spin_ir_golden_test(
    name = "predicated_receive_ir_golden",
    src = "predicated_receive.x",
    dslx_top = "CondReader",
    golden = "predicated_receive.ir",
)

spin_ir_golden_test(
    name = "predicated_send_ir_golden",
    src = "predicated_send.x",
    dslx_top = "CondWriter",
    golden = "predicated_send.ir",
)

spin_ir_golden_test(
    name = "func_ir_golden",
    src = "func.x",
    dslx_top = "myfunc",
    golden = "func.ir",
)

spin_ir_golden_test(
    name = "proc_ir_golden",
    src = "proc.x",
    dslx_top = "MyProc",
    golden = "proc.ir",
)

spin_ir_golden_test(
    name = "unconditional_send_ir_golden",
    src = "unconditional_send.x",
    dslx_top = "Writer",
    golden = "unconditional_send.ir",
)

spin_ir_golden_test(
    name = "xr_hints_ir_golden",
    src = "xr_hints.x",
    dslx_top = "RwProc",
    golden = "xr_hints.ir",
)

spin_ir_golden_test(
    name = "channel_naming_ir_golden",
    src = "channel_naming.x",
    convert_tests = True,
    dslx_top = "MultiplyBy4Test",
    golden = "channel_naming.ir",
)

spin_ir_golden_test(
    name = "deadlock_ir_golden",
    src = "deadlock.x",
    convert_tests = True,
    dslx_top = "Test",
    golden = "deadlock.ir",
)
