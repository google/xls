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

load("@rules_hdl//cocotb:cocotb.bzl", "cocotb_test")

def cocotb_xls_test(**kwargs):
    name = kwargs["name"]
    top = kwargs["hdl_toplevel"]

    if "timescale" in kwargs:
        timescale = kwargs["timescale"]
        timestamp_target = name + "-timescale"
        timestamp_verilog = name + "_timescale.v"
        native.genrule(
            name = timestamp_target,
            srcs = [],
            cmd = "echo \\`timescale {}/{} > $@".format(
                timescale["unit"],
                timescale["precission"],
            ),
            outs = [timestamp_verilog],
        )
        kwargs["verilog_sources"].insert(0, timestamp_verilog)
        kwargs.pop("timescale")

    cocotb_test(**kwargs)
