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

#ifndef XLS_DSLX_VIRTUALIZABLE_FILE_SYSTEM_H_
#define XLS_DSLX_VIRTUALIZABLE_FILE_SYSTEM_H_

#include <filesystem>  // NOLINT
#include <optional>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"

namespace xls::dslx {

// A virtualizable filesystem seam so that we can perform imports using either
// the real filesystem or a virtual filesystem.
//
// This is useful for use-cases like language servers where a mix of real
// filesystem and virtualized filesystem contents (i.e. with temporary edits in
// working buffers) are used.
class VirtualizableFilesystem {
 public:
  virtual ~VirtualizableFilesystem() = default;

  virtual absl::Status FileExists(const std::filesystem::path& path) = 0;

  virtual absl::StatusOr<std::string> GetFileContents(
      const std::filesystem::path& path) = 0;

  virtual absl::StatusOr<std::filesystem::path> GetCurrentDirectory() = 0;
};

// A simple (and often "default") implementation of the virtualizable file
// system that just calls into the XLS "real filesystem" routines that talk to
// the running operating system.
class RealFilesystem : public VirtualizableFilesystem {
 public:
  ~RealFilesystem() override = default;
  absl::Status FileExists(const std::filesystem::path& path) override;
  absl::StatusOr<std::string> GetFileContents(
      const std::filesystem::path& path) override;
  absl::StatusOr<std::filesystem::path> GetCurrentDirectory() override;
};

// A fake filesystem that gives back the same file content for all requested
// paths, useful in testing.
//
// If `expect_path` is given in addition to the file content, then we will error
// out if the GetFileContents has any other path.
class UniformContentFilesystem : public VirtualizableFilesystem {
 public:
  explicit UniformContentFilesystem(
      std::string_view file_content,
      std::optional<std::string_view> expect_path = std::nullopt)
      : file_content_(file_content), expect_path_(expect_path) {}

  ~UniformContentFilesystem() override = default;

  absl::Status FileExists(const std::filesystem::path& path) override {
    return absl::NotFoundError("UniformContentFilesystem");
  }
  absl::StatusOr<std::string> GetFileContents(
      const std::filesystem::path& path) override {
    if (expect_path_.has_value() && path != expect_path_.value()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "UniformContentFilesystem expected path: `%s` but got: `%s`",
          expect_path_.value(), path.string()));
    }
    return file_content_;
  }
  absl::StatusOr<std::filesystem::path> GetCurrentDirectory() override {
    return absl::NotFoundError("UniformContentFilesystem");
  }

 private:
  std::string file_content_;
  std::optional<std::string_view> expect_path_;
};

// Fake filesystem for use in tests that just resolves paths in directly against
// a given map.
//
// `cwd` is used to resolve relative paths to absolute paths before lookup in
// the `files` map.
class FakeFilesystem : public VirtualizableFilesystem {
 public:
  FakeFilesystem(absl::flat_hash_map<std::filesystem::path, std::string> files,
                 std::filesystem::path cwd);

  ~FakeFilesystem() override = default;

  absl::Status FileExists(const std::filesystem::path& path) override;

  absl::StatusOr<std::string> GetFileContents(
      const std::filesystem::path& path) override;

  absl::StatusOr<std::filesystem::path> GetCurrentDirectory() override {
    return cwd_;
  }

 private:
  absl::flat_hash_map<std::filesystem::path, std::string> files_;
  std::filesystem::path cwd_;
};

// A fake filesystem that always returns errors, useful in testing when we don't
// expect any virtual filesystem operations to be performed.
class AllErrorsFilesystem : public VirtualizableFilesystem {
 public:
  ~AllErrorsFilesystem() override = default;
  absl::Status FileExists(const std::filesystem::path& path) override {
    return absl::InvalidArgumentError("AllErrorsFilesystem");
  }
  absl::StatusOr<std::string> GetFileContents(
      const std::filesystem::path& path) override {
    return absl::InvalidArgumentError("AllErrorsFilesystem");
  }
  absl::StatusOr<std::filesystem::path> GetCurrentDirectory() override {
    return absl::InvalidArgumentError("AllErrorsFilesystem");
  }
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_VIRTUALIZABLE_FILE_SYSTEM_H_
