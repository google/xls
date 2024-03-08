// Copyright 2021 The XLS Authors
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

#ifndef XLS_DSLX_TYPE_SYSTEM_TYPE_INFO_TO_PROTO_H_
#define XLS_DSLX_TYPE_SYSTEM_TYPE_INFO_TO_PROTO_H_

#include <string>

#include "absl/status/statusor.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/type_info.pb.h"

namespace xls::dslx {

// Converts the given type information object to protobuf form for
// serialization.
absl::StatusOr<TypeInfoProto> TypeInfoToProto(const TypeInfo& type_info);

// Converts the given protobuf representation of an AST node in module "m" into
// a human readable string suitable for debugging and convenient testing.
absl::StatusOr<std::string> ToHumanString(const AstNodeTypeInfoProto& antip,
                                          const ImportData& import_data);

// As above, but puts every node in the TypeInfoProto on its own line.
absl::StatusOr<std::string> ToHumanString(const TypeInfoProto& tip,
                                          const ImportData& import_data);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_TYPE_INFO_TO_PROTO_H_
