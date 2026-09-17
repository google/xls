# Copyright 2026 The XLS Authors
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

"""Public API for wiring up an automatic busperf ready/valid setup."""

load(
    ":busperf_macros.bzl",
    _xls_busperf_setup = "xls_busperf_setup",
    _xls_busperf_yaml = "xls_busperf_yaml",
)
load(":busperf_report_rules.bzl", _busperf_analyze = "busperf_analyze")

xls_busperf_yaml = _xls_busperf_yaml
xls_busperf_setup = _xls_busperf_setup
busperf_analyze = _busperf_analyze
