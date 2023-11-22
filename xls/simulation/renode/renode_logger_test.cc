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

#include "xls/simulation/renode/renode_logger.h"

#include "absl/status/status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/simulation/generic/iconnection.h"
#include "xls/simulation/renode/renode_protocol.h"

namespace xls::simulation::renode {
namespace {
using namespace xls::simulation::generic;

class MockConnection : public IConnection {
 public:
  MOCK_METHOD(absl::Status, Log,
              (absl::LogSeverity level, std::string_view msg), (override));
  MOCK_METHOD(IMasterPort*, GetMasterPort, (), (override));
};

class RenodeLoggerInitTest : public ::testing::Test {
 protected:
  RenodeLoggerInitTest() {}
  static void SetUpTestSuite() {}
  void TearDown() override { auto _ = RenodeLogger::UnRegisterRenodeLogger(); }
};

TEST_F(RenodeLoggerInitTest, RegisterLogger) {
  MockConnection logger{};
  XLS_EXPECT_OK(RenodeLogger::RegisterRenodeLogger(logger));
}

TEST_F(RenodeLoggerInitTest, RegisterLoggerTwice) {
  MockConnection logger{};
  XLS_EXPECT_OK(RenodeLogger::RegisterRenodeLogger(logger));
  EXPECT_THAT(
      RenodeLogger::RegisterRenodeLogger(logger),
      ::xls::status_testing::StatusIs(absl::StatusCode::kAlreadyExists,
                                      "RenodeLogger already registered"));
}

TEST_F(RenodeLoggerInitTest, RemoveNonRegisteredLogger) {
  EXPECT_THAT(RenodeLogger::UnRegisterRenodeLogger(),
              ::xls::status_testing::StatusIs(absl::StatusCode::kAlreadyExists,
                                              "RenodeLogger not registered"));
}

TEST_F(RenodeLoggerInitTest, RemoveRegisteredLogger) {
  MockConnection logger{};
  XLS_EXPECT_OK(RenodeLogger::RegisterRenodeLogger(logger));
  XLS_EXPECT_OK(RenodeLogger::UnRegisterRenodeLogger());
}

class RenodeLoggerLoggingTest : public ::testing::Test {
 protected:
  RenodeLoggerLoggingTest() {}
  static void SetUpTestSuite() {
    XLS_EXPECT_OK(RenodeLogger::RegisterRenodeLogger(logger));
  }
  static void TearDownTestSuite() {
    auto _ = RenodeLogger::UnRegisterRenodeLogger();
  }
  static MockConnection logger;
};

MockConnection RenodeLoggerLoggingTest::logger{};

TEST_F(RenodeLoggerLoggingTest, LogInfo) {
  EXPECT_CALL(logger, Log(::absl::LogSeverity::kInfo,
                          ::testing::HasSubstr("This is INFO level log test")))
      .RetiresOnSaturation();
  XLS_LOG(INFO) << "This is INFO level log test";
}

TEST_F(RenodeLoggerLoggingTest, LogWarning) {
  EXPECT_CALL(logger,
              Log(::absl::LogSeverity::kWarning,
                  ::testing::HasSubstr("This is WARNING level log test")))
      .RetiresOnSaturation();
  XLS_LOG(WARNING) << "This is WARNING level log test";
}

TEST_F(RenodeLoggerLoggingTest, LogError) {
  EXPECT_CALL(logger, Log(::absl::LogSeverity::kError,
                          ::testing::HasSubstr("This is ERROR level log test")))
      .RetiresOnSaturation();
  XLS_LOG(ERROR) << "This is ERROR level log test";
}

}  // namespace
}  // namespace xls::simulation::renode
