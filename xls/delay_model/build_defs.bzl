# Copyright 2020 Google LLC
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

"""Contains macros for creating XLS delay models."""

# genrules are used in this file

def delay_model(
        name,
        model_name,
        srcs,
        **kwargs):
    """Generates a delay model cc_library from a DelayModel protobuf.

    Args:

      name: Name of the cc_library target to generate.
      model_name: Name of the model. This is the string that is used to access
        the model when calling xls::GetDelayEstimator.
      srcs: The pbtext file containing the DelayModel proto. There should only
        be a single source file.
      **kwargs: Keyword args to pass to cc_library rule.
    """

    if len(srcs) != 1:
        fail("More than one source not currently supported.")

    native.genrule(
        name = "{}_source".format(name),
        srcs = srcs,
        outs = ["{}.cc".format(name)],
        cmd = "$(location //xls/delay_model:generate_delay_lookup) " +
              "--model_name=" + model_name + " $< " +
              " > $(OUTS)",
        exec_tools = [
            "//xls/delay_model:generate_delay_lookup",
        ],
    )
    native.cc_library(
        name = name,
        srcs = [":{}_source".format(name)],
        alwayslink = 1,
        deps = [
            "@com_google_absl//absl/memory",
            "@com_google_absl//absl/status",
            "//xls/common:module_initializer",
            "//xls/common/logging",
            "//xls/common/status:statusor",
            "//xls/delay_model:delay_estimator",
            "//xls/ir",
        ],
        **kwargs
    )
