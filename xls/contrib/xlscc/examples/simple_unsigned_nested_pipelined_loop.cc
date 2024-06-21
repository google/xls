// Copyright 2023 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

class MyBlock {
 public:
  __xls_channel<unsigned int, __xls_channel_dir_In> in;
  __xls_channel<unsigned int, __xls_channel_dir_Out> out;

#pragma hls_top
  void foo() {
    int a = in.read();

#pragma hls_pipeline_init_interval 1
    for (unsigned int i = 1; i <= 4; ++i) {
      a += i;
#pragma hls_pipeline_init_interval 1
      for (unsigned int j = 1; j <= 8; ++j) {
        a += j;
      }
    }

    out.write(a);
  }
};
