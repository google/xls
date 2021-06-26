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
        "outs": attr.output_list(
            doc = "The list of generated files.",
        ),
    },
}

def generated_file(
        name = None,
        wrapped_target = None,
        tags = None,
        testonly = None):
    """The function is a placeholder for generated_file.

    The function is intended to be empty.

    Args:
      name: Optional name of the marker rule created by this macro.
      wrapped_target: The target to wrap.
      tags: A list of tags to set on the artifacts.
      testonly: Optional standard testonly attribute.
    """
    pass

def presubmit_generated_file(
        name = None,
        wrapped_target = None,
        tags = None,
        testonly = None):
    """The function is a placeholder for presubmit_generated_file.

    The function is intended to be empty.

    Args:
      name: Optional name of the marker rule created by this macro.
      wrapped_target: The target to wrap.
      tags: A non-empty list of string tags.
      testonly: Optional standard testonly attribute.
    """
    pass

def enable_generated_file_wrapper(
        wrapped_target = None,
        tags = None,
        testonly = None,
        enable_generated_file = True,
        enable_presubmit_generated_file = False):
    """The function is a placeholder for enable_generated_file_wrapper.

    The function is intended to be empty.

    Args:
      wrapped_target: The target to wrap.
      tags: A non-empty list of string tags.
      testonly: Optional standard testonly attribute.
      enable_generated_file: Enable generated_file.
      enable_presubmit_generated_file: Enable presubmit_generated_file.
    """
    pass
