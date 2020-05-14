# Lint as: python3
#
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

"""Helpful utility functions for use in debugging."""

import functools

from absl import logging


def debugged(f):
  """Decorates f to vlog(5) on entry/return."""

  @functools.wraps(f)
  def debugged_f(*args, **kwargs):
    args_str = ', '.join(str(a) for a in args)
    logging.vlog(5, '%s args: %s kwargs: %s <start>', f.__name__, args_str,
                 kwargs)
    r = f(*args, **kwargs)
    logging.vlog(5, '%s args: %s kwargs: %s => %s', f.__name__, args_str,
                 kwargs, r)
    return r

  return debugged_f
