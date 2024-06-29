#
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

"""Helper functions to verify that flags are given permissible values."""


def list_contains_only_integers(lst):
  """Returns True if the input list contains only str representations of digits.

  Args:
    lst: A list.

  Elements must be strings, and if any element is not a string representation of
  a digit, the function returns False.
  """
  if not isinstance(lst, list):
    raise TypeError(f"The given input {lst} is not a list")
  for elm in lst:
    if not isinstance(elm, str):
      raise TypeError(f"Element {elm} is not a string")
    if not elm.isdigit():
      return False
  return True


def valid_axis_scale(scale_string):
  """Returns True if the input is "linear" or "log", False otherwise.

  Args:
    scale_string: a string, must be "linear" or "log"
  """
  if not isinstance(scale_string, str):
    raise TypeError(
        f"Value for x- or y-axis scale flag {scale_string} is not a string."
    )
  if (scale_string != "linear") and (scale_string != "log"):
    raise ValueError(
        "Value for x- or y-axis scale flag is not 'linear' or 'log'."
    )
