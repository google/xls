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

"""Common OpenROAD warnings to suppress for different PDKs."""

SKY130_SUPPPRESSED_WARNINGS = [
    "ODB-0227",  # LEF file INFO output; excessively noisy.
    "ORD-0046",  # "-defer_connection has been deprecated."; accurate, but not relevant to the user.
    "STA-1140",  # "library ... already exists"; SKY130 has duplicate names for some reason.
    "STA-1173",  # "default_fanout_load is 0.0"; SKY130 uses this, despite the inaccuracy.
    "STA-1256",  # "table template ... not found"; SKY130 seems fine despite these warnings.
]

ASAP7_SUPPPRESSED_WARNINGS = [
    "ODB-0227",  # LEF file INFO output; excessively noisy.
    "ORD-0046",  # "-defer_connection has been deprecated."; accurate, but not relevant to the user.
    "STA-1140",  # "library ... already exists"; ASAP7 has duplicate names for some reason.
    "STA-1212",  # "timing group from output port"; ASAP7 has a lot of these warnings.
]
