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

"""BUILD helpers for using iverilog.
"""

def iverilog_compile(srcs, flags = ""):
    """Compiles the first .v files given in srcs into a .vvp file.
    Passes the flags to iverilog.
    """
    vvp_file = srcs[0] + "vp"  # Changes .v to .vvp
    native.genrule(
        name = "gen_" + vvp_file,
        srcs = srcs,
        outs = [vvp_file],
        cmd = (
            "$(location @com_icarus_iverilog//:iverilog) " +
            flags + " " +
            "-o $@ " +
            "$(location " + srcs[0] + ")"
        ),
        tools = ["@com_icarus_iverilog//:iverilog"],
    )

    # Creates a dummy test which will force the .vvp file production.
    native.sh_test(
        name = "force_on_test_build_" + vvp_file,
        srcs = ["@com_google_xls//dependency_support/com_icarus_iverilog:dummy.sh"],
        data = [vvp_file],
    )

def vpi_binary(name, srcs, **kwargs):
    """Creates a .vpi file with the given name from the given sources.
    All the extra arguments are passed directly to cc_binary.
    """
    so_name = name + ".so"
    native.cc_binary(
        name = so_name,
        srcs = srcs,
        linkshared = 1,
        **kwargs
    )

    native.genrule(
        name = "gen_" + name,
        srcs = [so_name],
        outs = [name],
        cmd = "cp $< $@",
        output_to_bindir = 1,
        executable = 1,
    )
