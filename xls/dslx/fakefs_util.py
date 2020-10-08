# Lint as: python3
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

"""Adapter for fakefs between internal/external versions."""

from pyfakefs import fake_filesystem as fakefs


def create_file(fs: fakefs.FakeFilesystem, *args, **kwargs) -> fakefs.FakeFile:
  # Note: avoids a warning for deprecated CreateFile, even though there's
  # version skew between internal and external versions.
  fcreate_file = (getattr(fs, 'create_file', None) or getattr(fs, 'CreateFile'))
  return fcreate_file(*args, **kwargs)
