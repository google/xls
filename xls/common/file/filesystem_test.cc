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

#include "xls/common/file/filesystem.h"

#include <unistd.h>

#include <filesystem>  // NOLINT
#include <fstream>
#include <ios>
#include <string>
#include <system_error>  // NOLINT(build/c++11)

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/file/filesystem_test.pb.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::AnyOf;
using ::testing::Contains;
using ::testing::Eq;
using ::testing::HasSubstr;

TEST(FilesystemTest, FileExistsReturnsTrueIfTheFileExists) {
  absl::StatusOr<TempDirectory> temp_dir = TempDirectory::Create();
  XLS_ASSERT_OK(temp_dir);
  absl::StatusOr<TempFile> temp_file = TempFile::Create();
  XLS_ASSERT_OK(temp_file);

  XLS_EXPECT_OK(FileExists(temp_file->path()));
  EXPECT_THAT(FileExists(temp_dir->path() / "nonexisting"),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(FileExists(temp_file->path() / "subdirectory_of_file"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(FilesystemTest, FileIsExecutable) {
  absl::StatusOr<TempFile> temp_file = TempFile::Create();
  XLS_ASSERT_OK(temp_file);

  ASSERT_THAT(FileIsExecutable(temp_file->path()), IsOkAndHolds(false));
  ASSERT_THAT(FileIsExecutable(temp_file->path().parent_path()),
              IsOkAndHolds(true));

  // Set the owner and group execute bit.
  std::filesystem::permissions(temp_file->path(),
                               std::filesystem::perms::owner_read |
                                   std::filesystem::perms::owner_write |
                                   std::filesystem::perms::owner_exec |
                                   std::filesystem::perms::group_exec);
  ASSERT_THAT(FileIsExecutable(temp_file->path()), IsOkAndHolds(true));

  // Unset the group execute bit but leave the owner execute bit.
  std::filesystem::permissions(temp_file->path(),
                               std::filesystem::perms::owner_read |
                                   std::filesystem::perms::owner_write |
                                   std::filesystem::perms::owner_exec);
  ASSERT_THAT(FileIsExecutable(temp_file->path()), IsOkAndHolds(true));

  // Unset the owner execute bit.
  std::filesystem::permissions(
      temp_file->path(),
      std::filesystem::perms::owner_read | std::filesystem::perms::owner_write);
  ASSERT_THAT(FileIsExecutable(temp_file->path()), IsOkAndHolds(false));

  EXPECT_THAT(FileIsExecutable("/this_is_not_a_real_directory/garbage"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(FilesystemTest, RecursivelyCreateDirCreatesDirectory) {
  absl::StatusOr<TempDirectory> temp_dir = TempDirectory::Create();
  XLS_ASSERT_OK(temp_dir);
  auto path = temp_dir->path() / "my_dir";

  XLS_EXPECT_OK(RecursivelyCreateDir(path));

  std::error_code ec;
  EXPECT_TRUE(std::filesystem::exists(path, ec));
  EXPECT_TRUE(std::filesystem::is_directory(path, ec));
}

TEST(FilesystemTest, RecursivelyCreateExistingDirSucceedsAndDoesNothing) {
  absl::StatusOr<TempDirectory> temp_dir = TempDirectory::Create();
  XLS_ASSERT_OK(temp_dir);

  XLS_EXPECT_OK(RecursivelyCreateDir(temp_dir->path()));
}

TEST(FilesystemTest, RecursivelyCreateDirWhereAFileIsFails) {
  absl::StatusOr<TempFile> temp_file = TempFile::Create();
  XLS_ASSERT_OK(temp_file);

  absl::Status status = RecursivelyCreateDir(temp_file->path());

  EXPECT_THAT(status, AnyOf(StatusIs(absl::StatusCode::kAlreadyExists),
                            StatusIs(absl::StatusCode::kFailedPrecondition)));
}

TEST(FilesystemTest, RecursivelyDeleteDirectory) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory temp_dir, TempDirectory::Create());
  auto path = temp_dir.path() / "my_dir";
  XLS_ASSERT_OK(RecursivelyCreateDir(path));
  XLS_ASSERT_OK(SetFileContents(path / "foo.txt", "foo"));
  XLS_ASSERT_OK(SetFileContents(path / "bar.txt", "bar"));
  XLS_ASSERT_OK(RecursivelyCreateDir(path / "blah" / "something" / "qux"));

  std::error_code ec;
  EXPECT_TRUE(std::filesystem::exists(path, ec));

  XLS_EXPECT_OK(RecursivelyDeletePath(path));
  EXPECT_FALSE(std::filesystem::exists(path, ec));
}

TEST(FilesystemTest, RecursivelyDeleteFile) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory temp_dir, TempDirectory::Create());
  auto dir_path = temp_dir.path() / "my_dir";
  XLS_ASSERT_OK(RecursivelyCreateDir(dir_path));
  auto file_path = dir_path / "foo.txt";
  XLS_ASSERT_OK(SetFileContents(file_path, "foo"));

  std::error_code ec;
  EXPECT_TRUE(std::filesystem::exists(dir_path, ec));
  EXPECT_TRUE(std::filesystem::exists(file_path, ec));

  XLS_EXPECT_OK(RecursivelyDeletePath(file_path));
  EXPECT_TRUE(std::filesystem::exists(dir_path, ec));
  EXPECT_FALSE(std::filesystem::exists(file_path, ec));
}

TEST(FilesystemTest, RecursivelyDeleteNonexistentFile) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory temp_dir, TempDirectory::Create());
  auto dir_path = temp_dir.path() / "my_dir";
  XLS_ASSERT_OK(RecursivelyCreateDir(dir_path));
  auto file_path = dir_path / "does_not_exist";

  EXPECT_THAT(RecursivelyDeletePath(file_path),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("File or directory does not exist")));
}

TEST(FilesystemTest, GetFileContentsReadsFile) {
  static constexpr char kContents[] = "h\ne\0y!";
  // Make sure to include the \0 in the string, to verify that binary data can
  // be read.
  std::string contents(kContents, sizeof(kContents));

  absl::StatusOr<TempFile> temp_file = TempFile::CreateWithContent(contents);
  XLS_ASSERT_OK(temp_file);

  absl::StatusOr<std::string> read_contents =
      GetFileContents(temp_file->path());

  XLS_ASSERT_OK(read_contents);
  EXPECT_EQ(*read_contents, contents);
}

TEST(FilesystemTest, GetFileContentsOfDirectoryFails) {
  absl::StatusOr<TempDirectory> temp_dir = TempDirectory::Create();
  XLS_ASSERT_OK(temp_dir);

  absl::StatusOr<std::string> contents = GetFileContents(temp_dir->path());

  EXPECT_THAT(contents, StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(FilesystemTest, SetFileContentsCreatesFileWhenMissing) {
  absl::StatusOr<TempDirectory> temp_dir = TempDirectory::Create();
  XLS_ASSERT_OK(temp_dir);

  XLS_EXPECT_OK(SetFileContents(temp_dir->path() / "file", "hello"));

  absl::StatusOr<std::string> read_contents =
      GetFileContents(temp_dir->path() / "file");
  XLS_ASSERT_OK(read_contents);
  EXPECT_EQ(*read_contents, "hello");
}

TEST(FilesystemTest, SetFileContentsOverwritesFile) {
  static constexpr char kContents[] = "h\ne\0y!";
  // Make sure to include the \0 in the string, to verify that binary data can
  // be written.
  std::string contents(kContents, sizeof(kContents));

  absl::StatusOr<TempFile> temp_file = TempFile::CreateWithContent("abc");
  XLS_ASSERT_OK(temp_file);

  XLS_EXPECT_OK(SetFileContents(temp_file->path(), contents));

  absl::StatusOr<std::string> read_contents =
      GetFileContents(temp_file->path());
  XLS_ASSERT_OK(read_contents);
  EXPECT_EQ(*read_contents, contents);
}

TEST(FilesystemTest, SetFileContentsWithSmallerContent) {
  XLS_ASSERT_OK_AND_ASSIGN(TempFile temp_file,
                           TempFile::CreateWithContent("abcdefghi"));
  EXPECT_THAT(GetFileContents(temp_file.path()), IsOkAndHolds("abcdefghi"));

  XLS_ASSERT_OK(SetFileContents(temp_file.path(), "123"));

  EXPECT_THAT(GetFileContents(temp_file.path()), IsOkAndHolds("123"));
}

TEST(FilesystemTest, VerifyPermissionsOfTempFile) {
  XLS_ASSERT_OK_AND_ASSIGN(TempFile temp_file,
                           TempFile::CreateWithContent("abcdefghi"));
  std::filesystem::file_status status =
      std::filesystem::status(temp_file.path());
  // File should be readable and writable by owner (at least) and not excutable
  // by anyone.
  EXPECT_NE(std::filesystem::perms::none,
            status.permissions() & std::filesystem::perms::owner_read);
  EXPECT_NE(std::filesystem::perms::none,
            status.permissions() & std::filesystem::perms::owner_write);
  EXPECT_EQ(std::filesystem::perms::none,
            status.permissions() & std::filesystem::perms::owner_exec);
  EXPECT_EQ(std::filesystem::perms::none,
            status.permissions() & std::filesystem::perms::group_exec);
  EXPECT_EQ(std::filesystem::perms::none,
            status.permissions() & std::filesystem::perms::others_exec);
}

TEST(FilesystemTest, VerifyPermissionsOfSetFileContentsFile) {
  std::filesystem::path file_path =
      std::filesystem::path(::testing::TempDir()) / "file";
  XLS_EXPECT_OK(SetFileContents(file_path, "hello"));
  std::filesystem::file_status status = std::filesystem::status(file_path);
  // File should be readable and writable by owner (at least) and not excutable
  // by anyone.
  EXPECT_NE(std::filesystem::perms::none,
            status.permissions() & std::filesystem::perms::owner_read);
  EXPECT_NE(std::filesystem::perms::none,
            status.permissions() & std::filesystem::perms::owner_read);
  EXPECT_EQ(std::filesystem::perms::none,
            status.permissions() & std::filesystem::perms::owner_exec);
  EXPECT_EQ(std::filesystem::perms::none,
            status.permissions() & std::filesystem::perms::group_exec);
  EXPECT_EQ(std::filesystem::perms::none,
            status.permissions() & std::filesystem::perms::others_exec);
}

TEST(FilesystemTest, SetFileContentsOfDirectoryFails) {
  absl::StatusOr<TempDirectory> temp_dir = TempDirectory::Create();
  XLS_ASSERT_OK(temp_dir);

  absl::StatusOr<std::string> contents = SetFileContents(temp_dir->path(), ".");

  EXPECT_THAT(contents, StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(FilesystemTest, AppendStringToFileCreatesFileWhenMissing) {
  absl::StatusOr<TempDirectory> temp_dir = TempDirectory::Create();
  XLS_ASSERT_OK(temp_dir);

  XLS_EXPECT_OK(AppendStringToFile(temp_dir->path() / "file", "hello"));

  absl::StatusOr<std::string> read_contents =
      GetFileContents(temp_dir->path() / "file");
  EXPECT_THAT(read_contents, IsOkAndHolds(Eq("hello")));
}

TEST(FilesystemTest, AppendStringToFileAppendsToFile) {
  absl::StatusOr<TempDirectory> temp_dir = TempDirectory::Create();
  XLS_ASSERT_OK(temp_dir);
  XLS_EXPECT_OK(SetFileContents(temp_dir->path() / "file", "hello "));

  XLS_EXPECT_OK(AppendStringToFile(temp_dir->path() / "file", "there"));

  absl::StatusOr<std::string> read_contents =
      GetFileContents(temp_dir->path() / "file");
  EXPECT_THAT(read_contents, IsOkAndHolds(Eq("hello there")));
}

TEST(FilesystemTest, ParseTextProtoFileOfNonexistingFileFails) {
  absl::StatusOr<TempDirectory> temp_dir = TempDirectory::Create();
  XLS_ASSERT_OK(temp_dir);

  absl::StatusOr<FilesystemTest> contents =
      ParseTextProtoFile<FilesystemTest>(temp_dir->path() / "abc");

  EXPECT_THAT(contents, StatusIs(absl::StatusCode::kNotFound));
}

TEST(FilesystemTest, ParseTextProtoFileOfFileWithInvalidSyntaxFails) {
  absl::StatusOr<TempFile> temp_file = TempFile::CreateWithContent("abc");
  XLS_ASSERT_OK(temp_file);

  absl::StatusOr<FilesystemTest> contents =
      ParseTextProtoFile<FilesystemTest>(temp_file->path());

  EXPECT_THAT(contents, StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(FilesystemTest, ParseTextProtoFileParsesTextProto) {
  absl::StatusOr<TempFile> temp_file =
      TempFile::CreateWithContent("field: \"hi\"");
  XLS_ASSERT_OK(temp_file);

  absl::StatusOr<FilesystemTest> contents =
      ParseTextProtoFile<FilesystemTest>(temp_file->path());

  XLS_ASSERT_OK(contents);
  EXPECT_EQ(contents->field(), "hi");
}

TEST(FilesystemTest, SetProtobinFileWritesAFile) {
  absl::StatusOr<TempFile> temp_file = TempFile::Create();
  XLS_ASSERT_OK(temp_file);
  FilesystemTest test;
  test.set_field("hi");

  XLS_EXPECT_OK(SetProtobinFile(temp_file->path(), test));

  FilesystemTest content;
  XLS_ASSERT_OK(ParseProtobinFile(temp_file->path(), &content));
  EXPECT_EQ(content.field(), "hi");
}

TEST(FilesystemTest, SetTextProtoFileWritesAFile) {
  absl::StatusOr<TempFile> temp_file = TempFile::Create();
  XLS_ASSERT_OK(temp_file);
  FilesystemTest test;
  test.set_field("hi");

  XLS_EXPECT_OK(SetTextProtoFile(temp_file->path(), test));

  absl::StatusOr<std::string> contents = GetFileContents(temp_file->path());
  EXPECT_THAT(contents,
              IsOkAndHolds(HasSubstr("# proto-message: FilesystemTest")));
  EXPECT_THAT(contents, IsOkAndHolds(HasSubstr("field: \"hi\"\n")));
}

TEST(FilesystemTest, SetTextProtoFileFailsWhenRequiredFieldIsMissing) {
  absl::StatusOr<TempFile> temp_file = TempFile::Create();
  XLS_ASSERT_OK(temp_file);
  FilesystemTest test;

  absl::Status status = SetTextProtoFile(temp_file->path(), test);

  EXPECT_THAT(status, StatusIs(absl::StatusCode::kFailedPrecondition,
                               HasSubstr("missing required field")));
}

TEST(FilesystemTest, GetCurrentDirectoryReturnsCurrentDirectory) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory temp_dir, TempDirectory::Create());
  XLS_ASSERT_OK_AND_ASSIGN(std::filesystem::path initial_cwd,
                           GetCurrentDirectory());
  ASSERT_EQ(0, chdir(temp_dir.path().c_str()));

  XLS_ASSERT_OK_AND_ASSIGN(std::filesystem::path new_cwd,
                           GetCurrentDirectory());
  EXPECT_TRUE(std::filesystem::equivalent(temp_dir.path(), new_cwd));

  ASSERT_EQ(0,
            chdir(initial_cwd.c_str()));  // Change back to the original path.
}

TEST(FilesystemTest, GetDirectoryEntriesFailsWhenPathDoesNotExist) {
  absl::StatusOr<TempDirectory> temp_dir = TempDirectory::Create();
  XLS_ASSERT_OK(temp_dir);

  auto entries = GetDirectoryEntries(temp_dir->path() / "nonexisting");

  EXPECT_THAT(entries, StatusIs(absl::StatusCode::kNotFound));
}

TEST(FilesystemTest, GetDirectoryEntriesFailsWhenPathIsFile) {
  absl::StatusOr<TempFile> temp_file = TempFile::Create();
  XLS_ASSERT_OK(temp_file);

  auto entries = GetDirectoryEntries(temp_file->path() / "nonexisting");

  EXPECT_THAT(entries, StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(FilesystemTest, GetDirectoryEntriesGivesAbsolutePathsWhenPathIsAbsolute) {
  absl::StatusOr<TempDirectory> temp_dir = TempDirectory::Create();
  XLS_ASSERT_OK(temp_dir);
  XLS_EXPECT_OK(SetFileContents(temp_dir->path() / "a.txt", "hello"));
  XLS_EXPECT_OK(SetFileContents(temp_dir->path() / "b.txt", "hello"));
  EXPECT_TRUE(temp_dir->path().is_absolute());

  auto entries = GetDirectoryEntries(temp_dir->path());

  EXPECT_THAT(entries, IsOkAndHolds(Contains(temp_dir->path() / "a.txt")));
  EXPECT_THAT(entries, IsOkAndHolds(Contains(temp_dir->path() / "b.txt")));
}

TEST(FilesystemTest, GetDirectoryEntriesGivesRelativePathsWhenPathIsRelative) {
  absl::StatusOr<std::filesystem::path> initial_cwd = GetCurrentDirectory();
  XLS_ASSERT_OK(initial_cwd);
  absl::StatusOr<TempDirectory> temp_dir = TempDirectory::Create();
  XLS_ASSERT_OK(temp_dir);
  XLS_EXPECT_OK(SetFileContents(temp_dir->path() / "a.txt", "hello"));
  XLS_EXPECT_OK(SetFileContents(temp_dir->path() / "b.txt", "hello"));
  ASSERT_EQ(0, chdir(temp_dir->path().c_str()));
  std::filesystem::path relative_path = ".";

  auto entries = GetDirectoryEntries(relative_path);

  EXPECT_THAT(entries, IsOkAndHolds(Contains(relative_path / "a.txt")));
  EXPECT_THAT(entries, IsOkAndHolds(Contains(relative_path / "b.txt")));

  ASSERT_EQ(0,
            chdir(initial_cwd->c_str()));  // Change back to the original path.
}

TEST(FilesystemTest, GetRealPath) {
#ifndef __linux__
  GTEST_SKIP() << "Skipping as procfs only available on linux";
#endif  /* __linux__ */
  XLS_ASSERT_OK_AND_ASSIGN(std::filesystem::path link_path,
                           GetRealPath("/proc/self/exe"));
  XLS_ASSERT_OK_AND_ASSIGN(std::filesystem::path real_path,
                           GetRealPath(link_path));
  ASSERT_EQ(link_path, real_path);
}

TEST(FilesystemTest, ParseProtobinFile) {
  absl::StatusOr<TempDirectory> temp_dir = TempDirectory::Create();
  const std::filesystem::path& temp_file = temp_dir->path() / "a.txt";
  FilesystemTest content_golden;
  content_golden.set_field("hi");
  std::fstream output_file(temp_file.string(),
                           std::ios::out | std::ios::trunc | std::ios::binary);
  EXPECT_TRUE(content_golden.SerializeToOstream(&output_file));
  output_file.close();
  FilesystemTest content;
  XLS_ASSERT_OK(ParseProtobinFile(temp_dir->path() / "a.txt", &content));
  EXPECT_EQ(content.field(), "hi");
}

}  // namespace
}  // namespace xls
