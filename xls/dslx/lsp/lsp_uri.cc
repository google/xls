#include "xls/dslx/lsp/lsp_uri.h"

#include <filesystem>  // NOLINT
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"

namespace xls::dslx {

LspUri::LspUri(std::string uri) : uri_(std::move(uri)) {
  CHECK(absl::StartsWith(uri_, kFileUriPrefix))
      << "URI must start with `" << kFileUriPrefix << "`: " << uri_;
}

/* static */ LspUri LspUri::FromFilesystemPath(
    const std::filesystem::path& path) {
  return LspUri(absl::StrCat(kFileUriPrefix, path.string()));
}

std::filesystem::path LspUri::GetFilesystemPath() const {
  CHECK(absl::StartsWith(uri_, kFileUriPrefix));
  return std::filesystem::path(uri_.substr(kFileUriPrefix.size()));
}

}  // namespace xls::dslx
