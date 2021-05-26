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

#ifndef XLS_TOOLS_PROTO_TO_DSLX_H_
#define XLS_TOOLS_PROTO_TO_DSLX_H_

#include <filesystem>
#include <memory>

#include "google/protobuf/descriptor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/variant.h"
#include "xls/dslx/ast.h"

namespace xls {

// ProtoToDslx accepts a proto schema and textproto instantiating such, and
// converts those definitions into a cooresponding DSLX file.
// Args:
//   source_root: The path to the root directory containing the input schema
//       _as_well_as_ any .proto files referenced therein (e.g. that are
//       imported).
//   proto_schema_path: The .proto file containing the declaration of the
//       schema to translate.
//   message_name: The name of the message inside the top-level proto file to
//       emit.
//   text_proto: The text of the message definition to translate.
//   binding_name: The name to assign to the resulting DSLX constant.
absl::StatusOr<std::unique_ptr<dslx::Module>> ProtoToDslx(
    const std::filesystem::path& source_root,
    const std::filesystem::path& proto_schema_path,
    absl::string_view message_name, absl::string_view text_proto,
    absl::string_view binding_name);

// As above, but doesn't refer directly to the filesystem for resolution.
//
// Args:
//  proto_def: Contents of the proto schema file (i.e. `.proto` file).
//  ..rest: as above
absl::StatusOr<std::unique_ptr<dslx::Module>> ProtoToDslxViaText(
    absl::string_view proto_def, absl::string_view message_name,
    absl::string_view text_proto, absl::string_view binding_name);

}  // namespace xls

#endif  // XLS_TOOLS_PROTO_TO_DSLX_H_
