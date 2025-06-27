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

#include "xls/dslx/lsp/lsp_uri.h"

#include <filesystem>
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
