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

"""Adapter for file APIs (internal cluster filesystem vs local Python API)."""

import builtins
import os
import shutil


def exists(path: str):
  return os.path.exists(path)


def open(path: str, mode: str):  # pylint: disable=redefined-builtin
  return builtins.open(path, mode)


def recursively_copy_dir(
    oldpath: str,
    newpath: str,
    preserve_file_mask: bool = True,  # pylint: disable=unused-argument
) -> None:
  return shutil.copytree(oldpath, newpath)


def remove(path: str) -> None:
  if os.path.isfile(path):
    os.remove(path)
  else:
    shutil.rmtree(path)


def make_dirs(path: str) -> None:
  if not os.path.isdir(path):
    os.makedirs(path)
