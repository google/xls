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
"""Adapter, checks simulator is running in an appropriate environment."""


def check_simulator(simulator: str) -> None:  # pylint: disable=unused-argument
  return None


# TODO(leary): 2020-06-06 Turn this into a sample program run against the
# simulator as a capability check.
def runs_system_verilog() -> bool:
  """Returns whether the default simulator runs SystemVerilog programs."""
  return False
