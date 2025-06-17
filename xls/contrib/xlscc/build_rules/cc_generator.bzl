# Copyright 2023 The XLS Authors
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

"""Provides fuzz test functionality for XLScc"""

def xls_ac_fuzz_binaries(name, deps, seed_start, seed_count, test_ac_fixed, test_ac_int):
    """Generate fuzz test binaries for a range of seeds and the result comparison binary

    Args:
          name: descriptive name of the fuzz test.
          deps: dependencies required by the fuzzer.
          seed_start: start seed of test to generate.
          seed_count: number of tests to generate.

    Returns:
          All of the outputs of the fuzzing process.
          Including generated source code and binaries.
    """
    cc_files = []
    test_list = []
    for x in range(seed_count):
        test_outputs = []
        seed = seed_start + x
        srcfile = "{}_{}.cc".format(name, seed)
        cc_files.append(srcfile)
        cmd = "./$(location cc_generate_test) -seed=" + str(seed) + " --cc_filepath=$(OUTS) \
          --test_ac_int=" + str(test_ac_int) + " --test_ac_fixed=" + str(test_ac_fixed)
        native.genrule(
            name = "fuzzfiles_{}_{}".format(name, seed),
            outs = [srcfile],
            cmd = cmd,
            tools = ["cc_generate_test"],
        )

        native.cc_binary(
            name = "{}_{}".format(name, seed),
            srcs = [srcfile],
            deps = [
                "@com_github_hlslibs_ac_types//:ac_int",
                "@com_github_hlslibs_ac_types//:ac_fixed",
            ],
        )
        test_outputs.append("{}_{}".format(name, seed))
        test_outputs.append(srcfile)

        native.filegroup(
            name = "{}_group_{}".format(name, seed),
            data = test_outputs,
        )

        test_outputs.extend([
            "//xls/contrib/xlscc:synth_only_headers",
            "//xls/contrib/xlscc:xlscc",
            "@com_github_hlslibs_ac_types//:ac_types_as_data",
        ])

        native.cc_test(
            name = "{}_{}_test".format(name, seed),
            testonly = 1,
            srcs = ["cc_fuzz_tester.cc"],
            data = test_outputs,
            args = [
                "--seed={}".format(seed),
                "--input_path=xls/contrib/xlscc/unit_tests/{}_".format(name),
            ],
            deps = [
                ":cc_generator",
                ":unit_test",
                "@com_google_absl//absl/container:inlined_vector",
                "@com_google_absl//absl/log:check",
                "@com_google_absl//absl/log",
                "@com_google_absl//absl/flags:flag",
                "@com_google_absl//absl/status:statusor",
                "@com_google_absl//absl/strings",
                "@com_google_absl//absl/strings:str_format",
                "//xls/codegen:combinational_generator",
                "//xls/codegen:module_signature",
                "//xls/codegen:codegen_result",
                "//xls/common:init_xls",
                "//xls/common:subprocess",
                "//xls/common/file:filesystem",
                "//xls/common/file:get_runfile_path",
                "//xls/common/file:temp_directory",
                "//xls/common/status:matchers",
                "//xls/common/status:status_macros",
                "//xls/interpreter:ir_interpreter",
                "//xls/ir",
                "//xls/ir:events",
                "//xls/ir:function_builder",
                "//xls/ir:ir_test_base",
                "//xls/ir:value",
                "//xls/passes",
                "//xls/passes:optimization_pass_pipeline",
                "//xls/simulation:module_simulator",
                "//xls/simulation:verilog_simulator",
                "//xls/simulation:verilog_simulators",
                "//xls/simulation:default_verilog_simulator",
            ] + deps,
        )
        test_list.append(":{}_{}_test".format(name, seed))
    native.test_suite(
        name = name,
        tests = test_list,
    )
    return [DefaultInfo(files = depset(cc_files))]
