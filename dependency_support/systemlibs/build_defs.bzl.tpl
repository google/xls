# -*- Python -*-

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

"""Skylark macros for system libraries."""

SYSTEM_LIBS_ENABLED = %{syslibs_enabled}

SYSTEM_LIBS_LIST = [
%{syslibs_list}
]


def if_any_system_libs(a, b=[]):
  """Conditional which evaluates to 'a' if any system libraries are configured."""
  if SYSTEM_LIBS_ENABLED:
    return a
  else:
    return b


def if_system_lib(lib, a, b=[]):
  """Conditional which evaluates to 'a' if we're using the system version of lib"""

  if SYSTEM_LIBS_ENABLED and lib in SYSTEM_LIBS_LIST:
    return a
  else:
    return b


def if_not_system_lib(lib, a, b=[]):
  """Conditional which evaluates to 'a' if we're using the system version of lib"""

  return if_system_lib(lib, b, a)
