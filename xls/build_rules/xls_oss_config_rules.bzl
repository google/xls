# Copyright 2021 The XLS Authors
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

"""This module contains oss configurations for XLS build rules."""

CONFIG = {
    "xls_outs_attrs": {
        "outs": attr.string_list(
            doc = "The list of generated files.",
        ),
    },
}

DEFAULT_BENCHMARK_SYNTH_DELAY_MODEL = "asap7"
DEFAULT_BENCHMARK_SYNTH_AREA_MODEL = "asap7"

def enable_generated_file_wrapper(**kwargs):  # @unused
    """The function is a placeholder for enable_generated_file_wrapper.

    The function is intended to be empty.

    Args:
      **kwargs: Keyword arguments. Named arguments.
    """
    pass

def delay_model_to_standard_cells(delay_model):
    """Maps the delay model to the corresponding standard cells target

    Args:
      delay_model: string, the delay model's name.

    Returns:
      the target for the corresponding standard cells
    """
    if delay_model == "sky130":
        return "@com_google_skywater_pdk_sky130_fd_sc_hd//:sky130_fd_sc_hd"
    if delay_model == "asap7":
        return "@org_theopenroadproject_asap7sc7p5t_27//:asap7-sc7p5t_rev27_rvt_4x"
    if delay_model == "unit":
        # No real delay model used; default to ASAP7 for now.
        return "@org_theopenroadproject_asap7sc7p5t_27//:asap7-sc7p5t_rev27_rvt_4x"
    fail("No cells known for delay model: {}".format(delay_model))
