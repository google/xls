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

# Tests that consume the publicly-exposed API.

package(
    default_applicable_licenses = ["//:license"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_test(
    name = "function_builder_test",
    srcs = ["function_builder_test.cc"],
    deps = [
        "//xls/public:function_builder",
        "//xls/public:ir",
        "//xls/public:status_matchers",
        "//xls/public:xls_gunit_main",
        "@com_google_googletest//:gtest",
    ],
)
