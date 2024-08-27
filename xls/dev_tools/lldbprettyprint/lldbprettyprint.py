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

"""Pretty printers for LLDB."""

import pathlib
import sys

# Add file directory to sys.path. Compat for OSS.
sys.path.append(pathlib.PosixPath(__file__).parent)

# pylint: disable-next=g-import-not-at-top
import xlsir


# pylint: disable-next=invalid-name,unused-argument
def __lldb_init_module(debugger, internal_dict):
  xlsir.init_printers(debugger)
