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

# Build extension to run yosys.
#
# Currently this file just provides a rule to run Yosys against verilog files.
#
# In the future it will provide more specific build definitions for things
# around synthesis and formal verification.

"""
 yosys_syntax_check(name, srcs)

 BUILD extension to parse (System-)Verilog files with yosys. At this point
 a mere code acceptance test that should succeed if the verilog file(s)
 presented are parseable by yosys.
"""

def yosys_syntax_check(name, srcs):
    yosys_binary = "@at_clifford_yosys//:yosys"
    native.genrule(
        name = name,
        srcs = srcs,
        outs = [srcs[0] + ".syntax_checked"],  # yosys parse output.
        tools = [yosys_binary],
        cmd = "$(location " + yosys_binary + ") -Q -T -p 'read_verilog -sv $(SRCS)' 2>&1 | tee $(OUTS)",
    )
