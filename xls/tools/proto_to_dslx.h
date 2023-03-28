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

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/variant.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/dynamic_message.h"
#include "google/protobuf/message.h"
#include "xls/dslx/frontend/ast.h"

namespace xls {
namespace internal {

struct MessageRecord;

// Holds the information needed to translate a proto element into DSLX - its
// name, value type (struct or int), and child elements, if applicable.
using NameToRecord =
    absl::flat_hash_map<std::string, std::unique_ptr<MessageRecord>>;

}  // namespace internal

// Provides functionality to construct DSLX definitions and constant
// instantiations from a proto schema and messages.
class ProtoToDslxManager {
 public:
  // Constructs an instance that will add DSLX definitions to the
  // previously constructed DSLX module.
  explicit ProtoToDslxManager(dslx::Module* module);

  ~ProtoToDslxManager();

  // AddProtoInstantiationToDslxModule accepts a proto message and
  // creates a DSLX struct corresponding to the type of message (once),
  // along with a DSLX constant corresponding to that message.
  //
  // Args:
  //   binding_name: The name to assign to the resulting DSLX constant.
  absl::Status AddProtoInstantiationToDslxModule(
      std::string_view binding_name, const google::protobuf::Message& message);

 private:
  // AddProtoTypeToDslxModule accepts a proto message and adds its type
  // definition into a corresponding DSLX module as a DSLX struct.
  //
  // Args:
  //   message: message to create DSLX definition for.
  absl::Status AddProtoTypeToDslxModule(const google::protobuf::Message& message);

  dslx::Module* module_ = nullptr;
  absl::flat_hash_map<const google::protobuf::Descriptor*, internal::NameToRecord>
      name_to_records_;
};

// ProtoToDslx accepts a proto schema and textproto instantiating such, and
// converts those definitions into a corresponding DSLX file.
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
    std::string_view message_name, std::string_view text_proto,
    std::string_view binding_name);

// As above, but doesn't refer directly to the filesystem for resolution.
//
// Args:
//  proto_def: Contents of the proto schema file (i.e. `.proto` file).
//  ..rest: as above
absl::StatusOr<std::unique_ptr<dslx::Module>> ProtoToDslxViaText(
    std::string_view proto_def, std::string_view message_name,
    std::string_view text_proto, std::string_view binding_name);

// Compiles the specified proto schema into a "Descriptor" (contained in the
// returned pool), potentially loading dependent schema files along the way.
// Args:
//  proto_def: Contents of the proto schema file (i.e. `.proto` file).
absl::StatusOr<std::unique_ptr<google::protobuf::DescriptorPool>>
ProcessStringProtoSchema(std::string_view proto_def);

// Helper function to create a Proto Message from a string.
// Args:
//   text_proto: The text of the message definition.
//   message_name: The name of the message.
//   descriptor_pool: Pool containing the definition of the message
//                    given by message_name.
//   factory: Factory to be used to create the message.
absl::StatusOr<std::unique_ptr<google::protobuf::Message>> ConstructProtoViaText(
    std::string_view text_proto, std::string_view message_name,
    google::protobuf::DescriptorPool* descriptor_pool,
    google::protobuf::DynamicMessageFactory* factory);

}  // namespace xls

#endif  // XLS_TOOLS_PROTO_TO_DSLX_H_
