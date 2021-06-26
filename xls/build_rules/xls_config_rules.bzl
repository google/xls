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

"""This module contains configurations for XLS build rules.

This module exposes the configuration parameters for the XLS build rules. It
**must** contain all configuration parameters for the XLS build rules.
"""

load(
    "//xls/build_rules:xls_oss_config_rules.bzl",
    _CONFIG = "CONFIG",
    _enable_generated_file_wrapper = "enable_generated_file_wrapper",
    _generated_file = "generated_file",
    _presubmit_generated_file = "presubmit_generated_file",
)

CONFIG = _CONFIG
generated_file = _generated_file
presubmit_generated_file = _presubmit_generated_file
enable_generated_file_wrapper = _enable_generated_file_wrapper
