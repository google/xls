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

#include "xls/simulation/generic/runtime_manager.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/ir_parser.h"

namespace xls::simulation::generic {
namespace {

class RuntimeManagerTest : public ::testing::Test,
                           public ::testing::WithParamInterface<bool> {
 private:
  std::vector<Value> create_increasing_input_values(int amount, int bit_count,
                                                    int start = 0,
                                                    int increment = 1) {
    std::vector<Value> data;
    for (int i = start; i < start + amount; i = i + increment) {
      data.push_back(Value(UBits(i, bit_count)));
    }
    return data;
  }

 public:
  void SetUp() {
    XLS_ASSERT_OK_AND_ASSIGN(std::filesystem::path ir_path,
                             xls::GetXlsRunfilePath(ir_file));
    XLS_ASSERT_OK_AND_ASSIGN(std::string ir_text,
                             xls::GetFileContents(ir_path));
    XLS_ASSERT_OK_AND_ASSIGN(this->package, xls::Parser::ParsePackage(ir_text));
    this->data_vector = create_increasing_input_values(10, 32);
  }

  std::unique_ptr<Package> package;
  std::unique_ptr<RuntimeManager> manager;
  std::vector<Value> data_vector;
  std::string recv_channel_name = "passthrough__data_s";
  std::string send_channel_name = "passthrough__data_r";
  std::uint8_t status_addr = 0x0;
  std::string ir_file =
      "xls/simulation/generic/testdata/pasthrough_opt_ir.opt.ir";
};

TEST_P(RuntimeManagerTest, Init) {
  XLS_EXPECT_OK(RuntimeManager::Create(package.get(), /*use_jit=*/GetParam()));
}

TEST_P(RuntimeManagerTest, DeadlockStateAfterInit) {
  XLS_ASSERT_OK_AND_ASSIGN(
      auto manager,
      RuntimeManager::Create(package.get(), /*use_jit=*/GetParam()));

  EXPECT_EQ(manager->HasDeadlock(), false);
}

TEST_P(RuntimeManagerTest, RuntimeSimulation) {
  XLS_ASSERT_OK_AND_ASSIGN(
      auto manager,
      RuntimeManager::Create(package.get(), /*use_jit=*/GetParam()));

  auto& queue_manager = manager->runtime().queue_manager();
  XLS_ASSERT_OK_AND_ASSIGN(auto recv_queue,
                           queue_manager.GetQueueByName(recv_channel_name));
  XLS_ASSERT_OK_AND_ASSIGN(auto send_queue,
                           queue_manager.GetQueueByName(send_channel_name));
  XLS_EXPECT_OK(manager->Reset());

  for (const auto& x : data_vector) {
    XLS_EXPECT_OK(send_queue->Write(x));
  }

  for (int i = 0; i < data_vector.size(); ++i) {
    XLS_EXPECT_OK(manager->Update());
    auto val = recv_queue->Read();
    EXPECT_EQ(val.has_value(), true);
    EXPECT_EQ(val.value(), data_vector[i]);
  }
}

TEST_P(RuntimeManagerTest, HasDeadlock) {
  XLS_ASSERT_OK_AND_ASSIGN(
      auto manager,
      RuntimeManager::Create(package.get(), /*use_jit=*/GetParam()));

  XLS_EXPECT_OK(manager->Reset());
  XLS_EXPECT_OK(manager->Update());
  EXPECT_EQ(manager->HasDeadlock(), false);
  EXPECT_EQ(manager->Update().code(), absl::StatusCode::kInternal);
  EXPECT_EQ(manager->HasDeadlock(), true);
}

TEST_P(RuntimeManagerTest, Reset) {
  XLS_ASSERT_OK_AND_ASSIGN(
      auto manager,
      RuntimeManager::Create(package.get(), /*use_jit=*/GetParam()));

  XLS_EXPECT_OK(manager->Reset());
  XLS_EXPECT_OK(manager->Update());
  EXPECT_EQ(manager->Update().code(), absl::StatusCode::kInternal);
  EXPECT_EQ(manager->HasDeadlock(), true);
  XLS_EXPECT_OK(manager->Reset());
  EXPECT_EQ(manager->HasDeadlock(), false);
}

TEST_P(RuntimeManagerTest, GetPayloadData) {
  XLS_ASSERT_OK_AND_ASSIGN(
      auto manager,
      RuntimeManager::Create(package.get(), /*use_jit=*/GetParam()));

  XLS_ASSERT_OK(manager->Reset());
  XLS_ASSERT_OK(manager->Update());

  auto& status = manager->status();

  XLS_ASSERT_OK_AND_ASSIGN(auto value8, status.GetPayloadData8(status_addr));
  EXPECT_EQ(value8, static_cast<uint8_t>(0x0));
  XLS_ASSERT_OK_AND_ASSIGN(auto value16, status.GetPayloadData16(status_addr));
  EXPECT_EQ(value16, static_cast<uint16_t>(0x0));
  XLS_ASSERT_OK_AND_ASSIGN(auto value32, status.GetPayloadData32(status_addr));
  EXPECT_EQ(value32, static_cast<uint32_t>(0x0));
  XLS_ASSERT_OK_AND_ASSIGN(auto value64, status.GetPayloadData64(status_addr));
  EXPECT_EQ(value64, static_cast<uint64_t>(0x0));

  EXPECT_EQ(manager->Update().code(), absl::StatusCode::kInternal);
  EXPECT_EQ(manager->HasDeadlock(), true);

  XLS_ASSERT_OK_AND_ASSIGN(value8, status.GetPayloadData8(status_addr));
  EXPECT_EQ(value8, static_cast<uint8_t>(0x1));
  XLS_ASSERT_OK_AND_ASSIGN(value16, status.GetPayloadData16(status_addr));
  EXPECT_EQ(value16, static_cast<uint16_t>(0x1));
  XLS_ASSERT_OK_AND_ASSIGN(value32, status.GetPayloadData32(status_addr));
  EXPECT_EQ(value32, static_cast<uint32_t>(0x1));
  XLS_ASSERT_OK_AND_ASSIGN(value64, status.GetPayloadData64(status_addr));
  EXPECT_EQ(value64, static_cast<uint64_t>(0x1));
}

TEST_P(RuntimeManagerTest, GetChannelWidth) {
  XLS_ASSERT_OK_AND_ASSIGN(
      auto manager,
      RuntimeManager::Create(package.get(), /*use_jit=*/GetParam()));
  auto& status = manager->status();
  EXPECT_EQ(status.GetChannelWidth(), 1);
}

INSTANTIATE_TEST_SUITE_P(RuntimeManagerTestInstantiation, RuntimeManagerTest,
                         testing::Values(true, false));

}  // namespace
}  // namespace xls::simulation::generic
