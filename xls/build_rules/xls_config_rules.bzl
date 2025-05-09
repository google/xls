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
    _DEFAULT_BENCHMARK_SYNTH_AREA_MODEL = "DEFAULT_BENCHMARK_SYNTH_AREA_MODEL",
    _DEFAULT_BENCHMARK_SYNTH_DELAY_MODEL = "DEFAULT_BENCHMARK_SYNTH_DELAY_MODEL",
    _delay_model_to_standard_cells = "delay_model_to_standard_cells",
    _enable_generated_file_wrapper = "enable_generated_file_wrapper",
)

CONFIG = _CONFIG
DEFAULT_BENCHMARK_SYNTH_DELAY_MODEL = _DEFAULT_BENCHMARK_SYNTH_DELAY_MODEL
DEFAULT_BENCHMARK_SYNTH_AREA_MODEL = _DEFAULT_BENCHMARK_SYNTH_AREA_MODEL
delay_model_to_standard_cells = _delay_model_to_standard_cells
enable_generated_file_wrapper = _enable_generated_file_wrapper
