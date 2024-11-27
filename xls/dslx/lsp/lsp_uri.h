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

#ifndef XLS_DSLX_LSP_LSP_URI_H_
#define XLS_DSLX_LSP_LSP_URI_H_

#include <filesystem>  // NOLINT
#include <ostream>
#include <string>
#include <string_view>

#include "absl/strings/str_format.h"

namespace xls::dslx {

// Just a wrapper type (around a URI string) that indicates the contents are
// validated to be in URI format -- this helps us use types to indicate
// validation invariants and track the differences between strings, URIs, and
// filesystem paths in the language server implementation.
class LspUri {
 public:
  static inline constexpr std::string_view kFileUriPrefix = "file://";

  static LspUri FromFilesystemPath(const std::filesystem::path& path);

  // Precondition: `uri` must be known to be a URI (i.e. start with
  // kFileUriPrefix), or this constructor will CHECK-fail. This precondition
  // allows us to assume the contents (given by `GetStringView()`) are always a
  // proper URI, and can be converted to a filesystem path via
  // `GetFilesystemPath()`.
  explicit LspUri(std::string uri);

  LspUri(const LspUri&) = default;
  LspUri(LspUri&&) = default;
  LspUri& operator=(const LspUri&) = default;
  LspUri& operator=(LspUri&&) = default;
  LspUri() = default;

  template <typename H>
  friend H AbslHashValue(H h, const LspUri& self) {
    return H::combine(std::move(h), self.uri_);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const LspUri& uri) {
    absl::Format(&sink, "%s", uri.uri_);
  }

  bool operator<(const LspUri& other) const { return uri_ < other.uri_; }
  bool operator==(const LspUri& other) const = default;

  // Returns the underlying string_view contents for compatibility with APIs
  // that require the string_view.
  //
  // Note: we should try to keep the LspUri invariant as far as we can, i.e. use
  // the types to help indicate that the URI form is maintained where possible.
  std::string_view GetStringView() const { return uri_; }

  // Strips the URI prefix off of the underlying string and returns the result
  // as a filesystem path.
  std::filesystem::path GetFilesystemPath() const;

 private:
  std::string uri_;
};

inline std::ostream& operator<<(std::ostream& os, const LspUri& uri) {
  os << uri.GetStringView();
  return os;
}

}  // namespace xls::dslx

#endif  // XLS_DSLX_LSP_LSP_URI_H_
