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

"""Command line tool helper functions."""

import datetime
from typing import Text


def parse_duration(s: Text) -> datetime.timedelta:
  if s.endswith('h'):
    return datetime.timedelta(hours=int(s[:-1]))
  if s.endswith('m'):
    return datetime.timedelta(minutes=int(s[:-1]))
  if s.endswith('s'):
    return datetime.timedelta(seconds=int(s[:-1]))
  raise ValueError('Invalid duration text')
