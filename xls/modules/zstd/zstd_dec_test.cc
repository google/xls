// Copyright 2020 The XLS Authors
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
//

#include <memory>
#include <fstream>

#include "gtest/gtest.h"
#include "zstd.h"

#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/events.h"

#include "xls/interpreter/interpreter_proc_runtime.h"
#include "xls/interpreter/serial_proc_runtime.h"

namespace xls {
namespace {

class ZstdDecoderTest : public ::testing::Test {
 public:
  void SetUp() {
    XLS_ASSERT_OK_AND_ASSIGN(std::filesystem::path ir_path, xls::GetXlsRunfilePath(this->ir_file));
    XLS_ASSERT_OK_AND_ASSIGN(std::string ir_text, xls::GetFileContents(ir_path));
    XLS_ASSERT_OK_AND_ASSIGN(this->package, xls::Parser::ParsePackage(ir_text));
    XLS_ASSERT_OK_AND_ASSIGN(this->interpreter, CreateInterpreterSerialProcRuntime(this->package.get()));

    auto& queue_manager = this->interpreter->queue_manager();
    XLS_ASSERT_OK_AND_ASSIGN(this->recv_queue, queue_manager.GetQueueByName(this->recv_channel_name));
    XLS_ASSERT_OK_AND_ASSIGN(this->send_queue, queue_manager.GetQueueByName(this->send_channel_name));
  }

  void PrintTraceMessages() {
    XLS_ASSERT_OK_AND_ASSIGN(Proc *proc, this->package->GetProc(this->proc_name));
    const InterpreterEvents& events = this->interpreter->GetInterpreterEvents(proc);

    if (!events.trace_msgs.empty()) {
      for (const auto& tm : events.trace_msgs) {
        std::cout << "[TRACE] " << tm << std::endl;
      }
    }
    this->interpreter->ClearInterpreterEvents();
  }

  const char *proc_name = "__zstd_dec__ZstdDecoder_0_next";
  const char *recv_channel_name = "zstd_dec__output_s";
  const char *send_channel_name = "zstd_dec__input_r";

  const char *ir_file = "xls/modules/zstd/zstd_dec_opt_ir.opt.ir";
  const char *zstd_file = "xls/modules/zstd/data.zst";

  std::unique_ptr<Package> package;
  std::unique_ptr<SerialProcRuntime> interpreter;
  ChannelQueue *recv_queue, *send_queue;
};

/* TESTS */

TEST(ZstdLib, Version) {
  ASSERT_EQ(ZSTD_VERSION_STRING, "1.5.5");
}

TEST_F(ZstdDecoderTest, Passthough) {
  auto value = Value(UBits(0xDEADBEEFDEADBEEF, 64));
  XLS_EXPECT_OK(this->send_queue->Write(value));
  XLS_EXPECT_OK(this->interpreter->Tick());
  this->PrintTraceMessages();
  auto read_value = this->recv_queue->Read();
  EXPECT_EQ(read_value.has_value(), true);
  EXPECT_EQ(read_value.value(), value);
}

 TEST_F(ZstdDecoderTest, FileFeed) {
   std::ifstream compressed_file(zstd_file, std::ios::binary);
   ASSERT_TRUE(compressed_file.is_open());

   uint64_t data;
   int i = 0;
   while (compressed_file.read((char*) &data, sizeof(uint64_t))) {
     if (i == 10) break;

     auto value = Value(UBits(data, 64));
     XLS_EXPECT_OK(this->send_queue->Write(value));
     XLS_EXPECT_OK(this->interpreter->Tick());
     this->PrintTraceMessages();
     auto read_value = this->recv_queue->Read();
     EXPECT_EQ(read_value.has_value(), true);
     EXPECT_EQ(read_value.value(), value);
     ++i;
   }
 }

}  // namespace
}  // namespace xls
