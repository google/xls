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

#include "xls/common/file/named_pipe.h"

#include <cstdint>
#include <filesystem>  // NOLINT
#include <optional>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/common/thread.h"

namespace xls {
namespace {

using status_testing::IsOk;
using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::Not;
using ::testing::Optional;

TEST(NamedPipeTest, SingleReadAndWrite) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory temp_dir, TempDirectory::Create());
  std::filesystem::path path = temp_dir.path() / "the_pipe";
  XLS_ASSERT_OK_AND_ASSIGN(NamedPipe pipe, NamedPipe::Create(path));

  // Two threads are required because fopen blocks until the other side of the
  // pipe has been opened.
  Thread write_thread([&pipe]() {
    FileLineWriter fw = pipe.OpenForWriting().value();
    CHECK_OK(fw.WriteLine("Hello world!"));
  });
  XLS_ASSERT_OK_AND_ASSIGN(FileLineReader fr, pipe.OpenForReading());
  write_thread.Join();

  EXPECT_THAT(fr.ReadLine(), IsOkAndHolds("Hello world!"));
}

TEST(NamedPipeTest, MultipleReadAndWrite) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory temp_dir, TempDirectory::Create());
  std::filesystem::path path = temp_dir.path() / "the_pipe";
  XLS_ASSERT_OK_AND_ASSIGN(NamedPipe pipe, NamedPipe::Create(path));

  Thread write_thread([&pipe]() {
    FileLineWriter fw = pipe.OpenForWriting().value();
    for (int64_t i = 0; i < 10; ++i) {
      CHECK_OK(fw.WriteLine(absl::StrFormat("Line #%d", i)));
    }
  });
  XLS_ASSERT_OK_AND_ASSIGN(FileLineReader fr, pipe.OpenForReading());
  write_thread.Join();

  for (int64_t i = 0; i < 10; ++i) {
    EXPECT_THAT(fr.ReadLine(),
                IsOkAndHolds(Optional(absl::StrFormat("Line #%d", i))));
  }
}

TEST(NamedPipeTest, PipeImmediatelyClosedOnWriteEnd) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory temp_dir, TempDirectory::Create());
  std::filesystem::path path = temp_dir.path() / "the_pipe";
  XLS_ASSERT_OK_AND_ASSIGN(NamedPipe pipe, NamedPipe::Create(path));

  Thread write_thread(
      [&pipe]() { FileLineWriter fw = pipe.OpenForWriting().value(); });
  XLS_ASSERT_OK_AND_ASSIGN(FileLineReader fr, pipe.OpenForReading());

  EXPECT_THAT(fr.ReadLine(), IsOkAndHolds(std::nullopt));
  write_thread.Join();
}

TEST(NamedPipeTest, FullBuffer) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory temp_dir, TempDirectory::Create());
  std::filesystem::path path = temp_dir.path() / "the_pipe";
  XLS_ASSERT_OK_AND_ASSIGN(NamedPipe pipe, NamedPipe::Create(path));
  // Write a sufficiently large number of values to fill the pipe buffer.
  constexpr int64_t kCount = 1'000'000;

  Thread write_thread([&pipe]() {
    FileLineWriter fw = pipe.OpenForWriting().value();
    for (int64_t i = 0; i < kCount; ++i) {
      CHECK_OK(fw.WriteLine(absl::StrFormat("Line #%d", i)));
    }
  });

  // Wait for some amount of time for the pipe buffer to fill.
  absl::SleepFor(absl::Seconds(5));

  XLS_ASSERT_OK_AND_ASSIGN(FileLineReader fr, pipe.OpenForReading());
  for (int64_t i = 0; i < kCount; ++i) {
    EXPECT_THAT(fr.ReadLine(),
                IsOkAndHolds(Optional(absl::StrFormat("Line #%d", i))));
  }
  write_thread.Join();
}

TEST(NamedPipeTest, EmptyLines) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory temp_dir, TempDirectory::Create());
  std::filesystem::path path = temp_dir.path() / "the_pipe";
  XLS_ASSERT_OK_AND_ASSIGN(NamedPipe pipe, NamedPipe::Create(path));

  Thread write_thread([&pipe]() {
    FileLineWriter fw = pipe.OpenForWriting().value();
    CHECK_OK(fw.WriteLine(""));
    CHECK_OK(fw.WriteLine(""));
    CHECK_OK(fw.WriteLine(""));
  });
  XLS_ASSERT_OK_AND_ASSIGN(FileLineReader fr, pipe.OpenForReading());
  write_thread.Join();

  EXPECT_THAT(fr.ReadLine(), IsOkAndHolds(""));
  EXPECT_THAT(fr.ReadLine(), IsOkAndHolds(""));
  EXPECT_THAT(fr.ReadLine(), IsOkAndHolds(""));

  // Trying to read another line should return nullopt as the writing side has
  // been closed.
  EXPECT_THAT(fr.ReadLine(), IsOkAndHolds(std::nullopt));
}

TEST(NamedPipeTest, CreationAndCleanup) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory temp_dir, TempDirectory::Create());
  std::filesystem::path path = temp_dir.path() / "the_pipe";

  EXPECT_THAT(FileExists(path), StatusIs(absl::StatusCode::kNotFound));
  {
    XLS_ASSERT_OK_AND_ASSIGN(NamedPipe pipe, NamedPipe::Create(path));
    EXPECT_EQ(path, pipe.path());
    XLS_ASSERT_OK(FileExists(path));
  }
  EXPECT_THAT(FileExists(path), StatusIs(absl::StatusCode::kNotFound));
}

TEST(NamedPipeTest, CreationAndRelease) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory temp_dir, TempDirectory::Create());
  std::filesystem::path path = temp_dir.path() / "the_pipe";

  EXPECT_THAT(FileExists(path), StatusIs(absl::StatusCode::kNotFound));
  {
    XLS_ASSERT_OK_AND_ASSIGN(NamedPipe pipe, NamedPipe::Create(path));
    EXPECT_EQ(path, pipe.path());
    XLS_ASSERT_OK(FileExists(path));
    // This should prevent removal of the named_pipe.
    (std::move(pipe)).Release();
  }
  XLS_ASSERT_OK(FileExists(path));
}

TEST(NamedPipeTest, NonexistentBadPath) {
  EXPECT_THAT(NamedPipe::Create("/this/is/a/bad/path"), Not(IsOk()));
}

}  // namespace
}  // namespace xls
