#include "xls/dslx/virtualizable_file_system.h"

#include <filesystem>  // NOLINT
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/file/filesystem.h"

namespace xls::dslx {

absl::Status RealFilesystem::FileExists(const std::filesystem::path& path) {
  return xls::FileExists(path);
}

absl::StatusOr<std::string> RealFilesystem::GetFileContents(
    const std::filesystem::path& path) {
  return xls::GetFileContents(path);
}

absl::StatusOr<std::filesystem::path> RealFilesystem::GetCurrentDirectory() {
  return xls::GetCurrentDirectory();
}

}  // namespace xls::dslx
