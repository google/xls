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

#include <stdio.h>
#include <stdint.h>

uint64_t __sample__main(void** input_ptrs, void** output_ptrs, void* tmp_buffer,
                        void* events, void* user_data,
                        int64_t continuation_point);

// Main function for driving a JIT-generated function. Should be linked against
// an object file created from JIT-generated IR.
int main (int argc, char** argv) {
  uint8_t input0 = 0xef;
  uint16_t input1 = 0x1234;
  void* inputs[2] = {&input0, &input1};

  uint32_t output0;
  void* outputs[1] = {&output0};
  __sample__main(inputs, outputs, NULL, NULL, NULL, 0);

  printf("0x%x\n", output0);

  return 0;
}
