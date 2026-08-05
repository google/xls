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
    "xls_dslx_library",
    "xls_dslx_test",
)

package(default_visibility = ["//xls:xls_public"])

xls_dslx_library(
    name = "bounded_fifo_dslx",
    srcs = ["bounded_fifo.x"],
)

xls_dslx_test(
    name = "bounded_fifo_without_bounded_fifos_dslx_test",
    srcs = ["bounded_fifo.x"],
    dslx_test_args = {"trace_channels": "true"},
    deps = [":bounded_fifo_dslx"],
)

xls_dslx_test(
    name = "bounded_fifo_with_bounded_fifos_dslx_test",
    srcs = ["bounded_fifo.x"],
    dslx_test_args = {
        "simulate_bounded_fifos": "true",
        "trace_channels": "true",
    },
    deps = [":bounded_fifo_dslx"],
)

xls_dslx_library(
    name = "bounded_fifo_impl_dslx",
    srcs = ["bounded_fifo_impl.x"],
)

xls_dslx_test(
    name = "bounded_fifo_impl_without_bounded_fifos_dslx_test",
    srcs = ["bounded_fifo_impl.x"],
    dslx_test_args = {"trace_channels": "true"},
    deps = [":bounded_fifo_impl_dslx"],
)

xls_dslx_test(
    name = "bounded_fifo_impl_with_bounded_fifos_dslx_test",
    srcs = ["bounded_fifo_impl.x"],
    dslx_test_args = {
        "simulate_bounded_fifos": "true",
        "trace_channels": "true",
    },
    deps = [":bounded_fifo_impl_dslx"],
)

xls_dslx_library(
    name = "scheduling_bug_dslx",
    srcs = ["scheduling_bug.x"],
)

# Passes: fixed proc order hides the bug (WriterA always runs before PriorityMux).
xls_dslx_test(
    name = "scheduling_bug_without_randomization_dslx_test",
    srcs = ["scheduling_bug.x"],
    deps = [":scheduling_bug_dslx"],
)

# Fails: randomized scheduling exposes the bug (PriorityMux may run before
# WriterA, fall back to WriterB, and fire the assertion).
xls_dslx_test(
    name = "scheduling_bug_with_randomization_dslx_test",
    srcs = ["scheduling_bug.x"],
    dslx_test_args = {
        "randomize_proc_execution": "true",
        "mid_tick_yield": "true",
        "seed": "1",
    },
    tags = ["manual"],
    deps = [":scheduling_bug_dslx"],
)

xls_dslx_library(
    name = "send_atomicity_dslx",
    srcs = ["send_atomicity.x"],
)

xls_dslx_test(
    name = "send_atomicity_without_mid_tick_yield_dslx_test",
    srcs = ["send_atomicity.x"],
    deps = [":send_atomicity_dslx"],
)

xls_dslx_test(
    name = "send_atomicity_with_mid_tick_yield_dslx_test",
    srcs = ["send_atomicity.x"],
    dslx_test_args = {
        "randomize_proc_execution": "true",
        "mid_tick_yield": "true",
        "seed": "1",
    },
    tags = ["manual"],
    deps = [":send_atomicity_dslx"],
)

xls_dslx_library(
    name = "valid_data_race_dslx",
    srcs = ["valid_data_race.x"],
)

xls_dslx_test(
    name = "valid_data_race_without_randomization_dslx_test",
    srcs = ["valid_data_race.x"],
    deps = [":valid_data_race_dslx"],
)

xls_dslx_test(
    name = "valid_data_race_with_randomization_dslx_test",
    srcs = ["valid_data_race.x"],
    dslx_test_args = {
        "randomize_proc_execution": "true",
        "mid_tick_yield": "true",
        "seed": "2",
    },
    tags = ["manual"],
    deps = [":valid_data_race_dslx"],
)
