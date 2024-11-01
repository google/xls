// Copyright 2024 The XLS Authors
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

// Block used to test memory in eval_proc_main.

class Test {
 public:
  int i = 0;

#pragma hls_top
  void Run() {
#pragma hls_pipeline_init_interval 1
    for (int i = 0; i < 4; ++i) {
      mem[i] = in.read();
    }
#pragma hls_pipeline_init_interval 1
    for (int i = 0; i < 4; ++i) {
      out.write(mem[i] * 3);
    }
  }

 private:
  __xls_channel<int, __xls_channel_dir_In>& in;
  __xls_channel<int, __xls_channel_dir_Out>& out;

  __xls_memory<int, 4>& mem;
};
