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
#include "xls/dslx/cpp_ast.h"

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
absl::Status EmitStruct(
    dslx::Module* module,
    absl::flat_hash_map<std::string, dslx::TypeDefinition>* type_defs,
    const ParsedMessage& message, const MessageMap& messages) {
  // Need to iterate in message-def order.
  dslx::Span span(dslx::Pos{}, dslx::Pos{});
  std::vector<std::pair<dslx::NameDef*, dslx::TypeAnnotation*>> members;
  for (int i = 0; i < message.descriptor->field_count(); i++) {
    const FieldDescriptor* fd = message.descriptor->field(i);
    auto* name_def = module->Make<dslx::NameDef>(span, fd->name(), nullptr);
    ParsedMessage::Element element = message.children.at(fd->name());

    dslx::TypeAnnotation* type_annot;
    if (absl::holds_alternative<std::string>(element.type)) {
      std::string type_name = absl::get<std::string>(element.type);
      auto* type_ref = module->Make<dslx::TypeRef>(span, type_name,
                                                   type_defs->at(type_name));
      type_annot = module->Make<dslx::TypeRefTypeAnnotation>(
          span, type_ref, std::vector<dslx::Expr*>());
    } else {
      auto* bits_type = module->Make<dslx::BuiltinTypeAnnotation>(
          span, dslx::BuiltinType::kBits);
      auto* array_size = module->Make<dslx::Number>(
          span, absl::StrCat(absl::get<int>(element.type)),
          dslx::NumberKind::kOther, /*type=*/nullptr);
      type_annot =
          module->Make<dslx::ArrayTypeAnnotation>(span, bits_type, array_size);
    }

    if (element.count == 1) {
      members.push_back(std::make_pair(name_def, type_annot));
    } else {
      auto* array_size = module->Make<dslx::Number>(
          span, absl::StrCat(element.count), dslx::NumberKind::kOther,
          /*type=*/nullptr);
      type_annot =
          module->Make<dslx::ArrayTypeAnnotation>(span, type_annot, array_size);
      members.push_back(std::make_pair(name_def, type_annot));

      auto* name_def = module->Make<dslx::NameDef>(
          span, absl::StrCat(fd->name(), "_count"), nullptr);
      auto* u32_annot = module->Make<dslx::BuiltinTypeAnnotation>(
          span, dslx::BuiltinType::kU32);
      members.push_back({name_def, u32_annot});
    }
  }

  auto* name_def = module->Make<dslx::NameDef>(span, message.name, nullptr);
  auto* struct_def = module->Make<dslx::StructDef>(
      span, name_def, std::vector<dslx::ParametricBinding*>(), members,
      /*is_public=*/true);
  name_def->set_definer(struct_def);
  module->AddTop(struct_def);
  type_defs->insert({message.name, struct_def});

  return absl::OkStatus();
}

// Basically a toposort of message decls.
absl::Status EmitStructs(
    dslx::Module* module,
    absl::flat_hash_map<std::string, dslx::TypeDefinition>* type_defs,
    const MessageMap& messages) {
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
      XLS_RETURN_IF_ERROR(EmitStruct(module, type_defs, *message, messages));
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
      return absl::InternalError("Infinite loop trying to emit struct defs.");
    }
  }
  return absl::OkStatus();
}

// Creates a zero-valued element of the described type.
absl::StatusOr<dslx::Expr*> MakeZeroValuedElement(
    dslx::Module* module, dslx::TypeAnnotation* type_annot) {
  dslx::Span span(dslx::Pos{}, dslx::Pos{});
  if (dslx::TypeRefTypeAnnotation* typeref_type =
          dynamic_cast<dslx::TypeRefTypeAnnotation*>(type_annot)) {
    // TODO(rspringer): Could be enumdef or structdef!
    dslx::StructDef* struct_def = absl::get<dslx::StructDef*>(
        typeref_type->type_ref()->type_definition());
    std::vector<std::pair<std::string, dslx::Expr*>> members;
    for (const auto& child : struct_def->members()) {
      XLS_ASSIGN_OR_RETURN(dslx::Expr * expr,
                           MakeZeroValuedElement(module, child.second));
      members.push_back({child.first->identifier(), expr});
    }
    return module->Make<dslx::StructInstance>(span, struct_def, members);
  } else if (dslx::ArrayTypeAnnotation* array_type =
                 dynamic_cast<dslx::ArrayTypeAnnotation*>(type_annot)) {
    // Special case: when it's an array of bits, then we should really just
    // return a number.
    dslx::TypeAnnotation* element_type = array_type->element_type();
    dslx::BuiltinTypeAnnotation* element_as_builtin =
        dynamic_cast<dslx::BuiltinTypeAnnotation*>(element_type);
    if (element_as_builtin->builtin_type() == dslx::BuiltinType::kBits) {
      return module->Make<dslx::Number>(span, "0", dslx::NumberKind::kOther,
                                        array_type);
    }

    XLS_ASSIGN_OR_RETURN(
        dslx::Expr * member,
        MakeZeroValuedElement(module, array_type->element_type()));
    // Currently, the array size has to be a Number - think about how values
    // must be specified in proto definitions.
    auto* array_size = dynamic_cast<dslx::Number*>(array_type->dim());
    XLS_RET_CHECK(array_size) << "Array size must be a simple number.";
    XLS_ASSIGN_OR_RETURN(uint64 real_size, array_size->GetAsUint64());
    return module->Make<dslx::ConstantArray>(
        span, std::vector<dslx::Expr*>(real_size, member),
        /*has_ellipsis=*/false);
  } else {
    dslx::BuiltinTypeAnnotation* builtin_type =
        dynamic_cast<dslx::BuiltinTypeAnnotation*>(type_annot);
    XLS_RET_CHECK(builtin_type);
    return module->Make<dslx::Number>(span, "0", dslx::NumberKind::kOther,
                                      builtin_type);
  }
}

// Instantiate a message as a DSLX constant.
absl::StatusOr<dslx::Expr*> EmitData(
    dslx::Module* module,
    absl::flat_hash_map<std::string, dslx::TypeDefinition>* type_defs,
    const Message& message, const Descriptor& descriptor,
    const MessageMap& parsed_msgs) {
  const Reflection* reflection = message.GetReflection();
  const ParsedMessage& parsed_msg = *parsed_msgs.at(descriptor.name());

  dslx::Span span(dslx::Pos{}, dslx::Pos{});
  auto* struct_def =
      absl::get<dslx::StructDef*>(type_defs->at(descriptor.name()));
  std::vector<std::pair<std::string, dslx::Expr*>> members;
  for (int field_idx = 0; field_idx < descriptor.field_count(); field_idx++) {
    const FieldDescriptor* fd = descriptor.field(field_idx);
    std::string field_name = fd->name();

    if (fd->type() == FieldDescriptor::Type::TYPE_MESSAGE) {
      if (fd->is_repeated()) {
        int total_submsgs = parsed_msg.children.at(fd->name()).count;
        int num_submsgs = reflection->FieldSize(message, fd);
        std::vector<dslx::Expr*> array_members;

        for (int submsg_idx = 0; submsg_idx < num_submsgs; submsg_idx++) {
          const Message& sub_message =
              reflection->GetRepeatedMessage(message, fd, submsg_idx);
          // Going to be either a struct instance or an integral type
          XLS_ASSIGN_OR_RETURN(
              dslx::Expr * expr,
              EmitData(module, type_defs, sub_message,
                       *sub_message.GetDescriptor(), parsed_msgs));
          array_members.push_back(expr);
        }

        bool has_ellipsis = false;
        if (num_submsgs != total_submsgs) {
          has_ellipsis = true;
          // TODO(https://github.com/google/xls/issues/249): Marking an array
          // as "has_ellipsis" seems to still require that we specify all
          // members. Until resolved (?), we'll create fake, zero-valued,
          // members.
          // Fortunately, we have the _count member to indicate which are valid.
          std::string type_name = fd->message_type()->name();
          auto* type_ref = module->Make<dslx::TypeRef>(
              span, type_name, type_defs->at(type_name));
          auto* typeref_type = module->Make<dslx::TypeRefTypeAnnotation>(
              span, type_ref, std::vector<dslx::Expr*>());
          for (int i = 0; i < total_submsgs - num_submsgs; i++) {
            XLS_ASSIGN_OR_RETURN(dslx::Expr * element,
                                 MakeZeroValuedElement(module, typeref_type));
            array_members.push_back(element);
          }
        }

        auto* array = module->Make<dslx::ConstantArray>(span, array_members,
                                                        has_ellipsis);
        members.push_back(std::make_pair(field_name, array));

        auto* u32_type = module->Make<dslx::BuiltinTypeAnnotation>(
            span, dslx::BuiltinType::kU32);
        auto* num_array_members =
            module->Make<dslx::Number>(span, absl::StrCat(num_submsgs),
                                       dslx::NumberKind::kOther, u32_type);
        members.push_back(std::make_pair(absl::StrCat(field_name, "_count"),
                                         num_array_members));
      } else {
        const Message& sub_message = reflection->GetMessage(message, fd);
        XLS_ASSIGN_OR_RETURN(
            dslx::Expr * expr,
            EmitData(module, type_defs, sub_message,
                     *sub_message.GetDescriptor(), parsed_msgs));
        members.push_back(std::make_pair(field_name, expr));
      }
    } else {  // If not a Message, than it's an integral type.
      int bit_width = absl::get<int>(parsed_msg.children.at(fd->name()).type);
      auto* bits_type = module->Make<dslx::BuiltinTypeAnnotation>(
          span, dslx::BuiltinType::kBits);
      auto* array_dim = module->Make<dslx::Number>(
          span, absl::StrCat(bit_width), dslx::NumberKind::kOther,
          /*type=*/nullptr);
      auto* array_elem_type =
          module->Make<dslx::ArrayTypeAnnotation>(span, bits_type, array_dim);

      if (fd->is_repeated()) {
        int total_submsgs = parsed_msg.children.at(fd->name()).count;
        int num_submsgs = reflection->FieldSize(message, fd);
        std::vector<dslx::Expr*> array_members;
        for (int submsg_idx = 0; submsg_idx < num_submsgs; submsg_idx++) {
          uint64 value = GetFieldValue(message, *reflection, *fd, submsg_idx);
          array_members.push_back(module->Make<dslx::Number>(
              span, absl::StrCat(value), dslx::NumberKind::kOther,
              array_elem_type));
        }

        bool has_ellipsis = false;
        if (num_submsgs != total_submsgs) {
          has_ellipsis = true;
          // TODO(https://github.com/google/xls/issues/249): Marking an array
          // as "has_ellipsis" seems to still require that we specify all
          // members. Until resolved (?), we'll create fake, zero-valued,
          // members.
          // Fortunately, we have the _count member to indicate which are valid.
          for (int i = 0; i < total_submsgs - num_submsgs; i++) {
            array_members.push_back(module->Make<dslx::Number>(
                span, "0", dslx::NumberKind::kOther, array_elem_type));
          }
        }
        auto* array = module->Make<dslx::ConstantArray>(span, array_members,
                                                        has_ellipsis);
        members.push_back(std::make_pair(field_name, array));

        auto* u32_type = module->Make<dslx::BuiltinTypeAnnotation>(
            span, dslx::BuiltinType::kU32);
        auto* num_array_members =
            module->Make<dslx::Number>(span, absl::StrCat(num_submsgs),
                                       dslx::NumberKind::kOther, u32_type);
        members.push_back(std::make_pair(absl::StrCat(field_name, "_count"),
                                         num_array_members));
      } else {
        uint64 value = GetFieldValue(message, *reflection, *fd);
        dslx::Number* number = module->Make<dslx::Number>(
            span, absl::StrCat(value), dslx::NumberKind::kOther,
            array_elem_type);
        members.push_back(std::make_pair(field_name, number));
      }
    }
  }

  // The caller assigns the name (decl'ed differently if top-level or member)
  return module->Make<dslx::StructInstance>(span, struct_def, members);
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

  dslx::Module module(var_name);
  absl::flat_hash_map<std::string, dslx::TypeDefinition> type_defs;
  XLS_RETURN_IF_ERROR(EmitStructs(&module, &type_defs, parsed_messages));
  XLS_ASSIGN_OR_RETURN(dslx::Expr * expr,
                       EmitData(&module, &type_defs, *new_message, *descriptor,
                                parsed_messages));
  dslx::Span span{dslx::Pos{}, dslx::Pos{}};
  auto* name_def =
      module.Make<dslx::NameDef>(span, var_name, /*definer=*/nullptr);
  auto* constant_def =
      module.Make<dslx::ConstantDef>(span, name_def, expr, /*is_public=*/true);
  name_def->set_definer(constant_def);
  module.AddTop(constant_def);

  return SetFileContents(output_path, module.ToString());
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
