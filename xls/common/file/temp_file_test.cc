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

#include "xls/common/file/temp_file.h"

#include <stdlib.h>  // NOLINT (needed for mkdtemp())
#include <unistd.h>

#include <filesystem>
#include <string>
#include <system_error>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/match.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

TEST(TempFile, CreateCreatesEmptyFile) {
  std::error_code ec;
  auto file = TempFile::Create();

  XLS_ASSERT_OK(file);
  EXPECT_THAT(file->path().string(), Not(IsEmpty()));
  EXPECT_TRUE(std::filesystem::exists(file->path().string(), ec));
  EXPECT_TRUE(std::filesystem::is_empty(file->path().string(), ec));
}

TEST(TempFile, CreateCreatesFileWithSuffix) {
  std::error_code ec;
  auto file = TempFile::Create(".foobar");

  XLS_ASSERT_OK(file);
  EXPECT_THAT(file->path().string(), Not(IsEmpty()));
  EXPECT_TRUE(absl::EndsWith(file->path().string(), ".foobar"));
  EXPECT_TRUE(std::filesystem::exists(file->path().string(), ec));
  EXPECT_TRUE(std::filesystem::is_empty(file->path().string(), ec));
}

TEST(TempFile, CreateInNonexistingDirectoryFails) {
  std::error_code ec;
  auto file =
      TempFile::CreateInDirectory(/*directory=*/"nonexisting_path_name___");

  EXPECT_THAT(file.status(),
              StatusIs(absl::StatusCode::kUnavailable,
                       HasSubstr("Failed to create temporary file")));
}

TEST(TempFile, CreateWithContentCreatesEmptyFileInSpecifiedDirectory) {
  std::error_code ec;
  auto global_temp_dir = std::filesystem::temp_directory_path(ec);
  ASSERT_FALSE(ec);
  std::string temp_dir = (global_temp_dir / "temp_file_test_XXXXXX").string();
  ASSERT_NE(mkdtemp(temp_dir.data()), nullptr);

  {
    auto file = TempFile::CreateWithContentInDirectory("hey there", temp_dir);

    XLS_EXPECT_OK(file);
    if (file.ok()) {
      EXPECT_THAT(file->path().string(), Not(IsEmpty()));
      EXPECT_TRUE(std::filesystem::exists(file->path().string(), ec));
      EXPECT_FALSE(std::filesystem::is_empty(file->path().string(), ec));
      EXPECT_EQ(file->path().parent_path(), temp_dir);
    }
  }
  ASSERT_EQ(rmdir(temp_dir.c_str()), 0);
}

TEST(TempFile, CreateWithContentOfEmptyStringCreatesEmptyFile) {
  std::error_code ec;
  auto file = TempFile::CreateWithContent("");

  XLS_ASSERT_OK(file);
  EXPECT_THAT(file->path().string(), Not(IsEmpty()));
  EXPECT_TRUE(std::filesystem::exists(file->path().string(), ec));
  EXPECT_TRUE(std::filesystem::is_empty(file->path().string(), ec));
}

TEST(TempFile, CreateWithContentCreatesNonemptyFile) {
  std::error_code ec;
  auto file = TempFile::CreateWithContent("test");

  XLS_ASSERT_OK(file);
  EXPECT_THAT(file->path().string(), Not(IsEmpty()));
  EXPECT_TRUE(std::filesystem::exists(file->path().string(), ec));

  auto contents = GetFileContents(file->path().string());
  XLS_ASSERT_OK(contents);
  EXPECT_EQ(*contents, "test");
}

TEST(TempFile, DestructorDeletesFile) {
  std::error_code ec;

  std::filesystem::path path;
  {
    auto file = TempFile::Create();
    XLS_ASSERT_OK(file);
    path = file->path();
    EXPECT_TRUE(std::filesystem::exists(path.string(), ec));
  }
  EXPECT_FALSE(std::filesystem::exists(path.string(), ec));
}

TEST(TempFile, MoveConstructedObjectIsDeletedEventually) {
  std::error_code ec;

  std::filesystem::path path;
  {
    auto file = TempFile::Create();
    XLS_ASSERT_OK(file);
    path = file->path();
    auto moved_file = TempFile(std::move(*file));
    EXPECT_TRUE(std::filesystem::exists(path.string(), ec));
  }
  EXPECT_FALSE(std::filesystem::exists(path.string(), ec));
}

TEST(TempFile, MoveAssignedObjectIsDeletedEventually) {
  std::error_code ec;

  std::filesystem::path path;
  std::filesystem::path other_path;
  {
    auto file = TempFile::Create();
    auto other_file = TempFile::Create();
    XLS_ASSERT_OK(file);
    XLS_ASSERT_OK(other_file);
    path = file->path();
    other_path = other_file->path();
    other_file = std::move(*file);
    EXPECT_TRUE(std::filesystem::exists(path.string(), ec));
    EXPECT_FALSE(std::filesystem::exists(other_path.string(), ec));
  }
  EXPECT_FALSE(std::filesystem::exists(path.string(), ec));
}

TEST(TempFile, ReleaseCausesDestructorToNotDeleteFile) {
  std::error_code ec;

  std::filesystem::path path;
  std::filesystem::path released_path;
  {
    auto file = TempFile::Create();
    XLS_ASSERT_OK(file);
    path = file->path();
    released_path = std::move(*file).Release();
    EXPECT_TRUE(std::filesystem::exists(path.string(), ec));
  }
  EXPECT_EQ(released_path, path);
  EXPECT_TRUE(std::filesystem::exists(path.string(), ec));
}

}  // namespace
}  // namespace xls
