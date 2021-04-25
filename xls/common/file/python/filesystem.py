# Lint as: python3
#
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
"""General filesystem utilities (above basic open/close/etc. routines)."""

from google.protobuf import message
from google.protobuf import text_format
from xls.common import gfile


def parse_text_proto_file(proto_path: str, output: message.Message) -> None:
  """Parses a text-format proto at the given path into the given message."""
  with gfile.open(proto_path, 'r') as f:
    proto_text = f.read()
  text_format.Parse(proto_text, output)
