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

// Converts a protobuf schema and instantiating message into DSLX structs and
// constant data.

// We currently only support flat structures (i.e., no submessages) and integral
// types. Supporting the former should just be a matter of engineering; we just
// haven't yet observed the need for it.

// The Emit.* functions are pretty messy and hard to grok. Should updates be
// needed in the future, it's probably worth considering changing them to
// instead build up a DSLX AST and call .Format() on the result.

#include "google/protobuf/compiler/importer.h"
#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/dynamic_message.h"
#include "google/protobuf/text_format.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/variant.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"

ABSL_FLAG(std::string, proto_def_path, "",
          "Path to the [structure] definition of the proto "
          "(i.e., the '.proto' file) to parse.");
ABSL_FLAG(std::string, output_path, "", "Path to which to write the output.");
ABSL_FLAG(std::string, proto_name, "",
          "Fully-qualified name of the proto message (i.e., the schema) "
          "to parse.");
ABSL_FLAG(std::string, source_root_path, ".",
          "Path to the root of the source tree, i.e., the directory in which "
          "xls can be found. Defaults to ${CWD}. "
          "(Needed for locating transitive proto dependencies.)");
ABSL_FLAG(std::string, textproto_path, "",
          "Path to the textproto to translate into DSLX.");
ABSL_FLAG(std::string, var_name, "",
          "The name of the DSLX variable to instantiate.");

namespace xls {

using google::protobuf::Descriptor;
using google::protobuf::DescriptorPool;
using google::protobuf::DynamicMessageFactory;
using google::protobuf::FieldDescriptor;
using google::protobuf::FileDescriptorProto;
using google::protobuf::Message;
using google::protobuf::Reflection;
using google::protobuf::compiler::DiskSourceTree;
using google::protobuf::compiler::SourceTreeDescriptorDatabase;

struct ParsedMessage;
using MessageMap =
    absl::flat_hash_map<std::string, std::unique_ptr<ParsedMessage>>;

// Simple output logger for any errors coming from a
// SourceTreeDescriptorDatabase.
class DbErrorCollector : public google::protobuf::compiler::MultiFileErrorCollector {
 public:
  void AddError(const std::string& filename, int line, int column,
                const std::string& message) override {
    XLS_LOG(ERROR) << message;
  }
  void AddWarning(const std::string& filename, int line, int column,
                  const std::string& message) override {
    XLS_LOG(WARNING) << message;
  }
};

// Simple output logger for any errors coming from a DescriptorPool.
class PoolErrorCollector : public DescriptorPool::ErrorCollector {
 public:
  void AddError(const std::string& filename, const std::string& element_name,
                const Message* descriptor, ErrorLocation location,
                const std::string& message) override {
    XLS_LOG(ERROR) << message;
  }

  void AddWarning(const std::string& filename, const std::string& element_name,
                  const Message* descriptor, ErrorLocation location,
                  const std::string& message) override {
    XLS_LOG(WARNING) << message;
  }
};

// Takes the input .proto file and creates a Descriptor pool (set of Message
// Descriptors) we can use to query its structure.
absl::StatusOr<std::unique_ptr<DescriptorPool>> ProcessProtoDef(
    const std::string& source_root_path, const std::string& proto_def_path,
    const std::string& textproto_path) {
  DiskSourceTree source_tree;

  // Our proto might have other dependencies, so we have to let the proto
  // compiler know about the layout of our source tree.
  source_tree.MapPath("/", "/");
  source_tree.MapPath("", "./");

  SourceTreeDescriptorDatabase db(&source_tree);
  FileDescriptorProto descriptor_proto;
  DbErrorCollector db_collector;
  db.RecordErrorsTo(&db_collector);
  XLS_RET_CHECK(db.FindFileByName(proto_def_path, &descriptor_proto));

  auto pool = std::make_unique<DescriptorPool>();
  PoolErrorCollector pool_collector;
  for (const auto& dependency : descriptor_proto.dependency()) {
    FileDescriptorProto dep_desc;
    XLS_RET_CHECK(db.FindFileByName(dependency, &dep_desc));
    XLS_RET_CHECK(pool->BuildFileCollectingErrors(dep_desc, &pool_collector) !=
                  nullptr)
        << "Error building dependency proto " << dependency;
  }

  pool->BuildFileCollectingErrors(descriptor_proto, &pool_collector);
  return pool;
}

bool FieldIsIntegral(FieldDescriptor::Type type) {
  switch (type) {
    case FieldDescriptor::Type::TYPE_BOOL:
    case FieldDescriptor::Type::TYPE_FIXED32:
    case FieldDescriptor::Type::TYPE_FIXED64:
    case FieldDescriptor::Type::TYPE_INT32:
    case FieldDescriptor::Type::TYPE_INT64:
    case FieldDescriptor::Type::TYPE_SFIXED32:
    case FieldDescriptor::Type::TYPE_SFIXED64:
    case FieldDescriptor::Type::TYPE_SINT32:
    case FieldDescriptor::Type::TYPE_SINT64:
    case FieldDescriptor::Type::TYPE_UINT32:
    case FieldDescriptor::Type::TYPE_UINT64:
      return true;
    default:
      return false;
  }
}

int GetFieldWidth(FieldDescriptor::Type type) {
  switch (type) {
    case FieldDescriptor::Type::TYPE_BOOL:
      return 1;
    case FieldDescriptor::Type::TYPE_FIXED32:
    case FieldDescriptor::Type::TYPE_INT32:
    case FieldDescriptor::Type::TYPE_SFIXED32:
    case FieldDescriptor::Type::TYPE_SINT32:
    case FieldDescriptor::Type::TYPE_UINT32:
      return 32;
    case FieldDescriptor::Type::TYPE_FIXED64:
    case FieldDescriptor::Type::TYPE_INT64:
    case FieldDescriptor::Type::TYPE_SFIXED64:
    case FieldDescriptor::Type::TYPE_SINT64:
    case FieldDescriptor::Type::TYPE_UINT64:
      return 64;
    default:
      XLS_LOG(FATAL) << "Should not get here!";
  }
}

// Returns the [integral] value contained in the specified field (...of the
// specified message, etc.). If "index" is set, then the field is treated as
// "repeated".
uint64 GetFieldValue(const Message& message, const Reflection& reflection,
                     const FieldDescriptor& fd,
                     absl::optional<int> index = absl::nullopt) {
  switch (fd.type()) {
    case FieldDescriptor::Type::TYPE_BOOL:
      if (index) {
        return reflection.GetRepeatedBool(message, &fd, *index);
      }
      return reflection.GetBool(message, &fd);
    case FieldDescriptor::Type::TYPE_INT32:
    case FieldDescriptor::Type::TYPE_SFIXED32:
    case FieldDescriptor::Type::TYPE_SINT32:
      if (index) {
        return reflection.GetRepeatedInt32(message, &fd, *index);
      }
      return reflection.GetInt32(message, &fd);
    case FieldDescriptor::Type::TYPE_FIXED32:
    case FieldDescriptor::Type::TYPE_UINT32:
      if (index) {
        return reflection.GetRepeatedUInt32(message, &fd, *index);
      }
      return reflection.GetUInt32(message, &fd);

    case FieldDescriptor::Type::TYPE_FIXED64:
    case FieldDescriptor::Type::TYPE_INT64:
    case FieldDescriptor::Type::TYPE_SFIXED64:
    case FieldDescriptor::Type::TYPE_SINT64:
    case FieldDescriptor::Type::TYPE_UINT64:
      if (index) {
        return reflection.GetRepeatedInt64(message, &fd, *index);
      }
      return reflection.GetInt64(message, &fd);
    default:
      XLS_LOG(FATAL) << "Should not get here!";
  }
}

// Since a given proto Message could be present multiple times in any given
// hierarchy, we need a global "ParsedMessage" repository that tracks, for every
// Message type, the maximum number of entries encountered for its repeated
// fields, because we ultimately need to turn it into a fixed-width array in
// DSLX.
struct ParsedMessage {
  struct Element {
    // Message name or bit width.
    absl::variant<std::string, int> type;

    // The greatest number of repeated entries seen (for any single instance)
    // across all instances.
    int count;
  };

  // The name of this message type.
  std::string name;

  // Field name -> type & count.
  absl::flat_hash_map<std::string, Element> children;

  const Descriptor* descriptor;
};

// Collect each sub-Message definition DSLX structs based on the given message.
// One wrinkle: repeated fields. Since each message instance could have a
// different number of elements in any repeated field, and since DSLX doesn't
// support variable-length arrays, we need to emit structures that are sized to
// the maximum count observed for each repeated field.
absl::Status CollectStructureDefs(const Descriptor& descriptor,
                                  const Message& message,
                                  MessageMap* messages) {
  std::string name = descriptor.name();
  if (!messages->contains(name)) {
    messages->insert({name, std::make_unique<ParsedMessage>()});
    messages->at(name)->name = name;
    messages->at(name)->descriptor = &descriptor;
  }
  ParsedMessage* parsed_msg = messages->at(descriptor.name()).get();

  const Reflection* reflection = message.GetReflection();
  for (int field_idx = 0; field_idx < descriptor.field_count(); field_idx++) {
    const FieldDescriptor* fd = descriptor.field(field_idx);
    if (fd->type() == FieldDescriptor::Type::TYPE_MESSAGE) {
      std::string subtype_name = fd->message_type()->name();
      if (fd->is_repeated()) {
        for (int sub_msg_idx = 0;
             sub_msg_idx < reflection->FieldSize(message, fd); sub_msg_idx++) {
          const Message& sub_message =
              reflection->GetRepeatedMessage(message, fd, sub_msg_idx);
          XLS_RETURN_IF_ERROR(
              CollectStructureDefs(*fd->message_type(), sub_message, messages));
        }

        parsed_msg->children[fd->name()].type = subtype_name;
        parsed_msg->children[fd->name()].count =
            std::max(parsed_msg->children[fd->name()].count,
                     reflection->FieldSize(message, fd));
      } else {
        const Message& sub_message = reflection->GetMessage(message, fd);
        XLS_RETURN_IF_ERROR(
            CollectStructureDefs(*fd->message_type(), sub_message, messages));
        parsed_msg->children[fd->name()].type = subtype_name;
        parsed_msg->children[fd->name()].count = 1;
      }
    } else {  // If not a Message, then its an integral type.
      XLS_RET_CHECK(FieldIsIntegral(fd->type()));
      if (fd->is_repeated()) {
        parsed_msg->children[fd->name()].count =
            std::max(parsed_msg->children[fd->name()].count,
                     reflection->FieldSize(message, fd));
      } else {
        parsed_msg->children[fd->name()].count = 1;
      }
      parsed_msg->children[fd->name()].type = GetFieldWidth(fd->type());
    }
  }

  return absl::OkStatus();
}

// Takes a collected structure/message definition (from above) and emits it as
// DSLX.
std::string EmitStruct(const ParsedMessage& message,
                       const MessageMap& messages) {
  std::vector<std::string> fields;
  // Need to iterate in message-def order.
  for (int i = 0; i < message.descriptor->field_count(); i++) {
    const FieldDescriptor* fd = message.descriptor->field(i);
    std::string field_name = fd->name();
    ParsedMessage::Element element = message.children.at(fd->name());

    std::string field_type;
    if (absl::holds_alternative<std::string>(element.type)) {
      field_type = absl::get<std::string>(element.type);
    } else {
      field_type = absl::StrCat("bits[", absl::get<int>(element.type), "]");
    }

    if (element.count == 1) {
      fields.push_back(absl::StrFormat("  %s: %s,", field_name, field_type));
    } else {
      fields.push_back(absl::StrFormat("  %s: %s[%d],", field_name, field_type,
                                       element.count));
      // u32 is the default "count of populated elements" type.
      fields.push_back(absl::StrFormat("  %s_count: u32,", field_name));
    }
  }

  return absl::StrFormat("struct %s {\n%s\n}\n", message.name,
                         absl::StrJoin(fields, "\n"));
}

// Basically a toposort of message decls.
std::vector<std::string> EmitStructs(const MessageMap& messages) {
  // Map of ParsedMessage to the messages it depends on (but that have not yet
  // been emitted).
  using BlockingSet = absl::flat_hash_set<const ParsedMessage*>;
  absl::flat_hash_map<const ParsedMessage*, BlockingSet> blockers;
  for (const auto& [name, message] : messages) {
    blockers[message.get()] = BlockingSet();
    for (const auto& [field_name, element] : message->children) {
      if (absl::holds_alternative<std::string>(element.type)) {
        std::string message_name = absl::get<std::string>(element.type);
        blockers[message.get()].insert(messages.at(message_name).get());
      }
    }
  }

  // Now iterate through the structs, emitting any that aren't blocked on prior
  // definitions. Once emitted, remove a struct from the dependees of the
  // remaining ones.
  std::vector<std::string> structs;
  while (!blockers.empty()) {
    absl::flat_hash_set<const ParsedMessage*> newly_done;
    bool progress = false;
    for (const auto& [message, dependencies] : blockers) {
      if (!dependencies.empty()) {
        continue;
      }

      progress = true;
      structs.push_back(EmitStruct(*message, messages));
      newly_done.insert(message);
    }

    // Clean up anyone who's done from others' dependees.
    absl::flat_hash_set<const ParsedMessage*> to_erase;
    for (const ParsedMessage* message : newly_done) {
      for (auto& [_, dependencies] : blockers) {
        dependencies.erase(message);
      }

      blockers.erase(message);
    }

    if (!progress) {
      XLS_LOG(QFATAL) << "Infinite loop trying to emit struct defs.";
    }
  }
  return structs;
}

// Instantiate a message as a DSLX constant.
std::string EmitData(const Message& message, const Descriptor& descriptor,
                     const MessageMap& parsed_msgs, int indent_level = 0) {
  std::string prefix(indent_level * 2, ' ');
  std::string sub_prefix((indent_level + 1) * 2, ' ');
  const Reflection* reflection = message.GetReflection();
  const ParsedMessage& parsed_msg = *parsed_msgs.at(descriptor.name());

  std::vector<std::string> fields;
  for (int field_idx = 0; field_idx < descriptor.field_count(); field_idx++) {
    const FieldDescriptor* fd = descriptor.field(field_idx);
    std::string field_name = fd->name();

    if (fd->type() == FieldDescriptor::Type::TYPE_MESSAGE) {
      std::string type_name = fd->message_type()->name();
      if (fd->is_repeated()) {
        int total_submsgs = parsed_msg.children.at(fd->name()).count;
        int num_submsgs = reflection->FieldSize(message, fd);
        std::vector<std::string> values;
        for (int submsg_idx = 0; submsg_idx < num_submsgs; submsg_idx++) {
          const Message& sub_message =
              reflection->GetRepeatedMessage(message, fd, submsg_idx);
          values.push_back(EmitData(sub_message, *sub_message.GetDescriptor(),
                                    parsed_msgs, indent_level + 2));
        }

        if (num_submsgs != total_submsgs) {
          values.push_back("...");
        }
        fields.push_back(absl::StrFormat(
            "%s%s: %s[%d]:[\n%s\n%s]", sub_prefix, field_name, type_name,
            total_submsgs, absl::StrJoin(values, ",\n"), sub_prefix));
        // u32 is the default "count of populated elements" type.
        fields.push_back(absl::StrFormat("%s%s_count: u32:0x%x", sub_prefix,
                                         field_name, num_submsgs));
      } else {
        const Message& sub_message = reflection->GetMessage(message, fd);
        fields.push_back(
            absl::StrFormat("%s%s:%s", sub_prefix, field_name,
                            EmitData(sub_message, *sub_message.GetDescriptor(),
                                     parsed_msgs, indent_level + 1)));
      }
    } else {  // If not a Message, than it's an integral type.
      int bit_width = absl::get<int>(parsed_msg.children.at(fd->name()).type);
      if (fd->is_repeated()) {
        int total_submsgs = parsed_msg.children.at(fd->name()).count;
        int num_submsgs = reflection->FieldSize(message, fd);
        std::vector<std::string> values;
        for (int submsg_idx = 0; submsg_idx < num_submsgs; submsg_idx++) {
          uint64 value = GetFieldValue(message, *reflection, *fd, submsg_idx);
          values.push_back(absl::StrCat("bits[", bit_width, "]:", value));
        }
        if (num_submsgs != total_submsgs) {
          values.push_back("...");
        }
        fields.push_back(absl::StrFormat("%s%s: bits[%d][%d]:[%s]", sub_prefix,
                                         field_name, bit_width, total_submsgs,
                                         absl::StrJoin(values, ", ")));
        fields.push_back(absl::StrFormat("%s%s_count: bits[32]:0x%x",
                                         sub_prefix, field_name, num_submsgs));
      } else {
        uint64 value = GetFieldValue(message, *reflection, *fd);
        fields.push_back(absl::StrFormat("%s%s: bits[%d]: 0x%x", sub_prefix,
                                         field_name, bit_width, value));
      }
    }
  }

  return absl::StrFormat("%s%s {\n%s\n%s}", prefix, descriptor.name(),
                         absl::StrJoin(fields, ",\n"), prefix);
}

absl::Status RealMain(const std::string& source_root_path,
                      const std::string& proto_def_path,
                      const std::string& proto_name,
                      const std::string& textproto_path,
                      const std::string& var_name,
                      const std::string& output_path) {
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<DescriptorPool> pool,
      ProcessProtoDef(source_root_path, proto_def_path, textproto_path));
  const Descriptor* descriptor = pool->FindMessageTypeByName(proto_name);
  XLS_RET_CHECK(descriptor != nullptr);

  DynamicMessageFactory factory;
  const Message* message = factory.GetPrototype(descriptor);
  XLS_RET_CHECK(message != nullptr);
  std::unique_ptr<Message> new_message(message->New());

  XLS_ASSIGN_OR_RETURN(std::string textproto, GetFileContents(textproto_path));
  google::protobuf::TextFormat::ParseFromString(textproto, new_message.get());

  absl::flat_hash_map<std::string, std::unique_ptr<ParsedMessage>>
      parsed_messages;
  XLS_RETURN_IF_ERROR(
      CollectStructureDefs(*descriptor, *new_message, &parsed_messages));

  std::vector<std::string> output = EmitStructs(parsed_messages);

  // Until we can export constant defs, delcare the result as a local var.
  output.push_back(
      absl::StrFormat("pub fn %s() -> %s {", var_name, descriptor->name()));
  output.push_back(
      absl::StrFormat("  let tmp: %s = %s;", descriptor->name(),
                      EmitData(*new_message, *descriptor, parsed_messages, 1)));
  output.push_back("  tmp");
  output.push_back("}\n");

  return SetFileContents(output_path, absl::StrJoin(output, "\n"));
}

}  // namespace xls

int main(int argc, char* argv[]) {
  xls::InitXls(argv[0], argc, argv);

  std::string proto_def_path = absl::GetFlag(FLAGS_proto_def_path);
  XLS_QCHECK(!proto_def_path.empty()) << "--proto_def_path must be specified.";

  std::string source_root_path = absl::GetFlag(FLAGS_source_root_path);
  XLS_QCHECK(!source_root_path.empty())
      << "--source_root_path must be specified.";

  std::string output_path = absl::GetFlag(FLAGS_output_path);
  XLS_QCHECK(!output_path.empty()) << "--output_path must be specified.";

  std::string proto_name = absl::GetFlag(FLAGS_proto_name);
  XLS_QCHECK(!proto_name.empty()) << "--proto_name must be specified.";

  std::string textproto_path = absl::GetFlag(FLAGS_textproto_path);
  XLS_QCHECK(!textproto_path.empty()) << "--textproto_path must be specified.";

  std::string var_name = absl::GetFlag(FLAGS_var_name);
  XLS_QCHECK(!var_name.empty()) << "--var_name must be specified.";
  XLS_QCHECK_OK(xls::RealMain(source_root_path, proto_def_path, proto_name,
                              textproto_path, var_name, output_path));

  return 0;
}
