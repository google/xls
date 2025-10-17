# Copyright 2025 The XLS Authors
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

import pathlib
import os

_report_file = pathlib.Path("report_perf.log")
_report_file.parent.mkdir(exist_ok=True)

def create_report_file():
    with _report_file.open('w') as f:
        f.write("| test name | total cycles | latency [cycles] | total bytes decoded | throughput [GiB/s] |\n")
        f.write("| --------- | ------------ | ---------------- | ------------------- | ------------------ |\n")

create_report_file()

def report_test_result(test_name, duration, latency, total_decoded_bytes, gigabytes_per_second):
    with _report_file.open("a") as f:
        f.write(f"| {test_name} | {round(duration)} | {round(latency)} | {round(total_decoded_bytes)} | {gigabytes_per_second:.3f} |\n")
