// Copyright 2020 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The default list of parameterizations of the test. Use with testing::ValuesIn
// in the INSTANTIATE_TEST_SUITE_P invocation.
inline constexpr SimulationTarget kDefaultSimulationTargets[] = {
#if !defined(ABSL_HAVE_ADDRESS_SANITIZER) && \
    !defined(ABSL_HAVE_MEMORY_SANITIZER)
    // iverilog crashes with ASAN.
    {.simulator = "iverilog", .use_system_verilog = false},
#endif
};

// Test parameterization which fixes use_system_verilog to false, and is
// only parameterized on Verilog simulator. Use with testing::ValuesIn in the
// INSTANTIATE_TEST_SUITE_P invocation.
inline constexpr SimulationTarget kVerilogOnlySimulationTargets[] = {
#if !defined(ABSL_HAVE_ADDRESS_SANITIZER) && \
    !defined(ABSL_HAVE_MEMORY_SANITIZER)
    // iverilog crashes with ASAN.
    {.simulator = "iverilog", .use_system_verilog = false},
#endif
};
