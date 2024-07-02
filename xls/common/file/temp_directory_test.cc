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

#include "xls/common/file/temp_directory.h"

#include <filesystem>  // NOLINT
#include <system_error>
#include <utility>

#include "gtest/gtest.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace {

TEST(TempDirectory, CreateCreatesATemporaryDirectory) {
  std::error_code ec;

  auto temp_dir = TempDirectory::Create();
  XLS_ASSERT_OK(temp_dir);

  EXPECT_TRUE(std::filesystem::exists(temp_dir->path(), ec));
  EXPECT_TRUE(std::filesystem::is_directory(temp_dir->path(), ec));
}

TEST(TempDirectory, DestructorDeletesTheTemporaryDirectory) {
  std::error_code ec;

  std::filesystem::path path;
  {
    auto temp_dir = TempDirectory::Create();
    XLS_ASSERT_OK(temp_dir);
    path = temp_dir->path();
    EXPECT_TRUE(std::filesystem::exists(path, ec));
  }
  EXPECT_FALSE(std::filesystem::exists(path, ec));
  EXPECT_FALSE(ec);  // The exists call should not have failed.
}

TEST(TempDirectory, MoveAssignmentDeletesTheTemporaryDirectory) {
  std::error_code ec;

  std::filesystem::path path;
  auto temp_dir = TempDirectory::Create();
  XLS_ASSERT_OK(temp_dir);
  path = temp_dir->path();
  EXPECT_TRUE(std::filesystem::exists(path, ec));
  temp_dir = TempDirectory::Create();  // Move assign.

  XLS_ASSERT_OK(temp_dir);
  EXPECT_FALSE(std::filesystem::exists(path, ec));
  EXPECT_FALSE(ec);  // The exists call should not have failed.
}

TEST(TempDirectory, ReleaseCausesTheDirectoryToNotBeDeleted) {
  std::error_code ec;

  std::filesystem::path path;
  std::filesystem::path released_path;
  {
    auto temp_dir = TempDirectory::Create();
    XLS_ASSERT_OK(temp_dir);
    path = temp_dir->path();
    released_path = std::move(*temp_dir).Release();
  }

  EXPECT_EQ(released_path, path);
  EXPECT_TRUE(std::filesystem::exists(path, ec));

  // Clean up
  std::filesystem::remove_all(path, ec);
  EXPECT_FALSE(ec);  // The remove_all call should not have failed.
}

TEST(TempDirectory, CleanupDeletesTheTemporaryDirectory) {
  std::error_code ec;

  std::filesystem::path path;
  auto temp_dir = TempDirectory::Create();
  XLS_ASSERT_OK(temp_dir);
  path = temp_dir->path();
  EXPECT_TRUE(std::filesystem::exists(path, ec));
  XLS_EXPECT_OK(std::move(*temp_dir).Cleanup());

  EXPECT_FALSE(std::filesystem::exists(path, ec));
  EXPECT_FALSE(ec);  // The exists call should not have failed.
}

TEST(TempDirectory, DirectoriesAreCleanedUpRecursively) {
  std::error_code ec;

  std::filesystem::path path;
  auto temp_dir = TempDirectory::Create();
  XLS_ASSERT_OK(temp_dir);
  path = temp_dir->path();
  auto subdir = path / "dir";
  auto file_in_subdir = subdir / "file";
  std::filesystem::create_directories(subdir, ec);
  EXPECT_FALSE(ec);
  XLS_EXPECT_OK(SetFileContents(file_in_subdir.string(), "Test"));
  XLS_EXPECT_OK(std::move(*temp_dir).Cleanup());

  EXPECT_FALSE(std::filesystem::exists(path, ec));
  EXPECT_FALSE(ec);  // The exists call should not have failed.
  EXPECT_FALSE(std::filesystem::exists(subdir, ec));
  EXPECT_FALSE(ec);  // The exists call should not have failed.
  EXPECT_FALSE(std::filesystem::exists(file_in_subdir, ec));
  EXPECT_FALSE(ec);  // The exists call should not have failed.
}

}  // namespace
}  // namespace xls
