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

#include "xls/tools/proto_to_dslx.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>  // NOLINT
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/types/span.h"
#include "google/protobuf/compiler/importer.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/dynamic_message.h"
#include "google/protobuf/text_format.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/math_util.h"
#include "xls/common/proto_adaptor_utils.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"

namespace xls {
namespace internal {

struct MessageRecord {
  struct ChildElement {
    // Message name (struct or enum) or bit width (integer).
    std::variant<std::string, google::protobuf::FieldDescriptor::Type> type;

    // The greatest number of repeated entries seen in any single instance,
    // across all instances of this message.
    int64_t count;

    // True if this element is of an unsupported type, such as "string".
    bool unsupported;
  };

  // The name of this message type.
  std::string name;

  // Field name -> type & count.
  absl::flat_hash_map<std::string, ChildElement> children;

  // The [proto] descriptor for this message/struct, if applicable.
  std::variant<const google::protobuf::Descriptor*, const google::protobuf::EnumDescriptor*>
      descriptor;

  // The typedef associated with this message, if it describes a struct.
  dslx::TypeAnnotation* dslx_type = nullptr;
  dslx::EnumDef* enum_def = nullptr;
  dslx::StructDef* struct_def = nullptr;
};

}  // namespace internal

namespace {

using ::google::protobuf::Descriptor;
using ::google::protobuf::DescriptorPool;
using ::google::protobuf::EnumDescriptor;
using ::google::protobuf::FieldDescriptor;
using ::google::protobuf::FileDescriptorProto;
using ::google::protobuf::Message;
using ::google::protobuf::Reflection;
using ::google::protobuf::compiler::DiskSourceTree;
using ::google::protobuf::compiler::SourceTreeDescriptorDatabase;

using ::xls::internal::MessageRecord;
using ::xls::internal::NameToRecord;

// Returns true if the provided field type is integral.
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

// Returns the width, in bits, of the provided integral proto type.
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
      break;
  }
  LOG(FATAL) << "Should not get here!";
}

bool IsFieldSigned(FieldDescriptor::Type type) {
  switch (type) {
    case FieldDescriptor::Type::TYPE_BOOL:
    case FieldDescriptor::Type::TYPE_FIXED32:
    case FieldDescriptor::Type::TYPE_UINT32:
    case FieldDescriptor::Type::TYPE_FIXED64:
    case FieldDescriptor::Type::TYPE_UINT64:
      return false;
    case FieldDescriptor::Type::TYPE_INT32:
    case FieldDescriptor::Type::TYPE_SFIXED32:
    case FieldDescriptor::Type::TYPE_SINT32:
    case FieldDescriptor::Type::TYPE_INT64:
    case FieldDescriptor::Type::TYPE_SFIXED64:
    case FieldDescriptor::Type::TYPE_SINT64:
      return true;
    default:
      break;
  }
  LOG(FATAL) << "Should not get here!";
}

// Returns the [integral] value contained in the specified field (...of the
// specified message, etc.). If "index" is set, then the field is treated as
// "repeated".
uint64_t GetFieldValue(const Message& message, const Reflection& reflection,
                       const FieldDescriptor& fd,
                       std::optional<int> index = std::nullopt) {
  switch (fd.type()) {
    case FieldDescriptor::Type::TYPE_BOOL:
      if (index) {
        return static_cast<uint64_t>(
            reflection.GetRepeatedBool(message, &fd, *index));
      }
      return static_cast<uint64_t>(reflection.GetBool(message, &fd));
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
      if (index) {
        return reflection.GetRepeatedInt64(message, &fd, *index);
      }
      return reflection.GetInt64(message, &fd);
    case FieldDescriptor::Type::TYPE_UINT64:
      if (index) {
        return reflection.GetRepeatedUInt64(message, &fd, *index);
      }
      return reflection.GetUInt64(message, &fd);
    default:
      break;
  }
  LOG(FATAL) << "Should not get here!";
}

// Returns the name of the described type with any parent elements prepended,
// e.g., "parent_child_grandchild".
template <typename DescriptorT>
std::string GetParentPrefixedName(std::string_view top_package,
                                  const DescriptorT* descriptor) {
  std::string_view package = descriptor->file()->package();
  // Generic lambda!
  auto get_msg_name = [package, top_package](const auto* descriptor) {
    if (package == top_package) {
      return std::string(descriptor->name());
    }
    return absl::StrReplaceAll(descriptor->full_name(), {{".", "_"}});
  };

  std::deque<std::string> types({get_msg_name(descriptor)});
  const Descriptor* parent = descriptor->containing_type();
  while (parent != nullptr) {
    types.push_front(get_msg_name(parent));
    parent = parent->containing_type();
  }

  return absl::StrJoin(types.begin(), types.end(), "__");
}

// Simple output logger for any errors coming from a
// SourceTreeDescriptorDatabase.
class DbErrorCollector : public google::protobuf::compiler::MultiFileErrorCollector {
 public:
  void RecordError(std::string_view filename, int line, int column,
                   std::string_view message) final {
    LOG(ERROR) << message;
  }

  void RecordWarning(std::string_view filename, int line, int column,
                     std::string_view message) final {
    LOG(WARNING) << message;
  }
};

// Simple output logger for any errors coming from a DescriptorPool.
class PoolErrorCollector : public DescriptorPool::ErrorCollector {
 public:
  void RecordError(std::string_view filename, std::string_view element_name,
                   const Message* descriptor, ErrorLocation location,
                   std::string_view message) final {
    LOG(ERROR) << message;
  }

  void RecordWarning(std::string_view filename, std::string_view element_name,
                     const Message* descriptor, ErrorLocation location,
                     std::string_view message) final {
    LOG(WARNING) << message;
  }
};

// Compiles the specified proto schema into a "Descriptor" (contained in the
// returned pool), potentially loading dependent schema files along the way.
absl::StatusOr<std::unique_ptr<DescriptorPool>> ProcessProtoSchema(
    const std::filesystem::path& source_root,
    const std::filesystem::path& proto_schema_path) {
  DiskSourceTree source_tree;

  // Our proto might have other dependencies, so we have to let the proto
  // compiler know about the layout of our source tree.
  source_tree.MapPath("/", "/");
  source_tree.MapPath("", source_root.string());

  SourceTreeDescriptorDatabase db(&source_tree);
  FileDescriptorProto descriptor_proto;
  DbErrorCollector db_collector;
  db.RecordErrorsTo(&db_collector);
  XLS_RET_CHECK(
      db.FindFileByName(std::string(proto_schema_path), &descriptor_proto));

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

// Creates a zero-valued element of the described type.
absl::StatusOr<dslx::Expr*> MakeZeroValuedElement(
    dslx::Module* module, dslx::TypeAnnotation* type_annot) {
  dslx::Span span(dslx::Pos{}, dslx::Pos{});
  if (dslx::TypeRefTypeAnnotation* typeref_type =
          dynamic_cast<dslx::TypeRefTypeAnnotation*>(type_annot)) {
    // TODO(rspringer): Could be enumdef or structdef!
    XLS_ASSIGN_OR_RETURN(
        dslx::StructDef * struct_def,
        ResolveLocalStructDef(typeref_type->type_ref()->type_definition()));
    std::vector<std::pair<std::string, dslx::Expr*>> members;
    for (const auto& child : struct_def->members()) {
      XLS_ASSIGN_OR_RETURN(dslx::Expr * expr,
                           MakeZeroValuedElement(module, child.type));
      members.push_back({child.name, expr});
    }
    return module->Make<dslx::StructInstance>(span, typeref_type, members);
  }
  if (dslx::ArrayTypeAnnotation* array_type =
          dynamic_cast<dslx::ArrayTypeAnnotation*>(type_annot)) {
    // Special case: when it's an array of bits, then we should really just
    // return a number.
    dslx::TypeAnnotation* element_type = array_type->element_type();
    dslx::BuiltinTypeAnnotation* element_as_builtin =
        dynamic_cast<dslx::BuiltinTypeAnnotation*>(element_type);
    if (element_as_builtin->builtin_type() == dslx::BuiltinType::kSN ||
        element_as_builtin->builtin_type() == dslx::BuiltinType::kUN) {
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
    const dslx::FileTable& file_table = *module->file_table();
    XLS_ASSIGN_OR_RETURN(uint64_t real_size,
                         array_size->GetAsUint64(file_table));
    return module->Make<dslx::ConstantArray>(
        span, std::vector<dslx::Expr*>(real_size, member),
        /*has_ellipsis=*/false);
  }
  dslx::BuiltinTypeAnnotation* builtin_type =
      dynamic_cast<dslx::BuiltinTypeAnnotation*>(type_annot);
  XLS_RET_CHECK(builtin_type);
  return module->Make<dslx::Number>(span, "0", dslx::NumberKind::kOther,
                                    builtin_type);
}

// Adds enum structural information to the MessageMap.
absl::Status CollectEnumDef(std::string_view top_package,
                            const EnumDescriptor* ed,
                            NameToRecord* name_to_record) {
  std::string name = GetParentPrefixedName(top_package, ed);
  if (!name_to_record->contains(name)) {
    name_to_record->insert({name, std::make_unique<MessageRecord>()});
    name_to_record->at(name)->name = name;
    name_to_record->at(name)->descriptor = ed;
  }
  return absl::OkStatus();
}

// Walks the provided message and creates a corresponding MessageRecord, which
// contains all data necessary (including child element descriptions) to
// translate it into DSLX.
absl::Status CollectMessageLayout(std::string_view top_package,
                                  const Descriptor& descriptor,
                                  NameToRecord* name_to_record) {
  std::string name = GetParentPrefixedName(top_package, &descriptor);
  if (!name_to_record->contains(name)) {
    name_to_record->insert({name, std::make_unique<MessageRecord>()});
    name_to_record->at(name)->name = name;
    name_to_record->at(name)->descriptor = &descriptor;
  }
  MessageRecord* message_record = name_to_record->at(name).get();

  for (int field_idx = 0; field_idx < descriptor.field_count(); field_idx++) {
    const FieldDescriptor* fd = descriptor.field(field_idx);
    std::string_view field_name = fd->name();

    MessageRecord::ChildElement child_element{/*type=*/"", /*count=*/0,
                                              /*unsupported=*/false};
    if (fd->type() == FieldDescriptor::Type::TYPE_MESSAGE) {
      const Descriptor* sub_desc = fd->message_type();
      XLS_RETURN_IF_ERROR(
          CollectMessageLayout(top_package, *sub_desc, name_to_record));
      child_element.type = GetParentPrefixedName(top_package, sub_desc);
    } else if (fd->type() == FieldDescriptor::Type::TYPE_ENUM) {
      const EnumDescriptor* ed = fd->enum_type();
      XLS_RETURN_IF_ERROR(CollectEnumDef(top_package, ed, name_to_record));
      child_element.type = GetParentPrefixedName(top_package, ed);
    } else if (FieldIsIntegral(fd->type())) {
      child_element.type = fd->type();
    } else {
      child_element.unsupported = true;
    }
    message_record->children[field_name] = child_element;
  }
  return absl::OkStatus();
}

// [Forward decl]: Dispatcher for collecting the counts of elements in a message
// (submessages, enums, integral elements).
absl::Status CollectElementCounts(std::string_view top_package,
                                  const Message& message,
                                  NameToRecord* name_to_record);

// Collects the number of entries in a "message" field, and recurses to collect
// its child counts.
absl::StatusOr<int64_t> CollectMessageCounts(std::string_view top_package,
                                             const Message& message,
                                             const FieldDescriptor* fd,
                                             MessageRecord* message_record,
                                             NameToRecord* name_to_record) {
  const Reflection* reflection = message.GetReflection();
  if (fd->is_repeated()) {
    for (int i = 0; i < reflection->FieldSize(message, fd); i++) {
      const Message& sub_message =
          reflection->GetRepeatedMessage(message, fd, i);
      XLS_RETURN_IF_ERROR(
          CollectElementCounts(top_package, sub_message, name_to_record));
    }

    return reflection->FieldSize(message, fd);
  }

  const Message& sub_message = reflection->GetMessage(message, fd);
  XLS_RETURN_IF_ERROR(
      CollectElementCounts(top_package, sub_message, name_to_record));
  return 1;
}

// Collects the number of entries in an enum or integral field.
absl::StatusOr<int64_t> CollectEnumOrIntegralCount(
    const Message& message, const FieldDescriptor* fd,
    MessageRecord* message_record) {
  if (fd->is_repeated()) {
    const Reflection* reflection = message.GetReflection();
    return std::max(message_record->children[fd->name()].count,
                    static_cast<int64_t>(reflection->FieldSize(message, fd)));
  }

  return 1;
}

// Walks the fields of the passed message and collects the counts of all present
// elements and subelements.
absl::Status CollectElementCounts(std::string_view top_package,
                                  const Message& message,
                                  NameToRecord* name_to_record) {
  const Descriptor* descriptor = message.GetDescriptor();
  std::string message_name = GetParentPrefixedName(top_package, descriptor);
  MessageRecord* message_record = name_to_record->at(message_name).get();
  for (int i = 0; i < descriptor->field_count(); i++) {
    const FieldDescriptor* fd = descriptor->field(i);
    std::string_view field_name = fd->name();
    if (fd->type() == FieldDescriptor::Type::TYPE_MESSAGE) {
      XLS_ASSIGN_OR_RETURN(
          int64_t count, CollectMessageCounts(top_package, message, fd,
                                              message_record, name_to_record));
      message_record->children[field_name].count =
          std::max(message_record->children[field_name].count, count);
    } else if (fd->type() == FieldDescriptor::Type::TYPE_ENUM ||
               FieldIsIntegral(fd->type())) {
      XLS_ASSIGN_OR_RETURN(int64_t count, CollectEnumOrIntegralCount(
                                              message, fd, message_record));
      message_record->children[field_name].count = count;
    } else {
      VLOG(1) << "Unsupported field type: " << fd->type() << " : "
              << field_name;
      message_record->children[field_name].count = 0;
    }
  }
  return absl::OkStatus();
}

// Emits an enum definition from the parsed protobuf schema into the passed
// Module.
absl::Status EmitEnumDef(dslx::Module* module, MessageRecord* message_record) {
  const EnumDescriptor* descriptor =
      std::get<const EnumDescriptor*>(message_record->descriptor);
  std::vector<dslx::EnumMember> members;
  dslx::Span span(dslx::Pos{}, dslx::Pos{});
  int num_values = descriptor->value_count();
  members.reserve(num_values);
  int32_t max_value = 0;
  for (int i = 0; i < num_values; i++) {
    const google::protobuf::EnumValueDescriptor* value = descriptor->value(i);
    auto* name_def = module->Make<dslx::NameDef>(
        span, std::string(value->name()), /*definer=*/nullptr);
    auto* number =
        module->Make<dslx::Number>(span, absl::StrCat(value->number()),
                                   dslx::NumberKind::kOther, /*type=*/nullptr);
    max_value = std::max(max_value, value->number());
    members.push_back(dslx::EnumMember{name_def, number});
  }

  auto* name_def = module->Make<dslx::NameDef>(span, message_record->name,
                                               /*definer=*/nullptr);
  int width = CeilOfLog2(max_value) + 1;
  auto* bits_type = module->Make<dslx::BuiltinTypeAnnotation>(
      span, dslx::BuiltinType::kBits,
      module->GetOrCreateBuiltinNameDef(dslx::BuiltinType::kBits));
  auto* bit_count = module->Make<dslx::Number>(
      span, absl::StrCat(width), dslx::NumberKind::kOther, /*type=*/nullptr);
  auto* type =
      module->Make<dslx::ArrayTypeAnnotation>(span, bits_type, bit_count);
  auto* enum_def = module->Make<dslx::EnumDef>(span, name_def, type, members,
                                               /*is_public=*/true);
  name_def->set_definer(enum_def);
  XLS_RETURN_IF_ERROR(
      module->AddTop(enum_def, /*make_collision_error=*/nullptr));

  // Make a type annotation by which we can refer to this type.
  auto* type_ref = module->Make<dslx::TypeRef>(span, enum_def);
  message_record->dslx_type = module->Make<dslx::TypeRefTypeAnnotation>(
      span, type_ref, /*parametrics=*/std::vector<dslx::ExprOrType>{});
  message_record->enum_def = enum_def;

  return absl::OkStatus();
}

// Emits an enum definition from the parsed protobuf schema into the passed
// Module.
absl::Status EmitStructDef(dslx::Module* module, MessageRecord* message_record,
                           NameToRecord* name_to_record) {
  dslx::Span span(dslx::Pos{}, dslx::Pos{});
  std::vector<dslx::StructMember> elements;

  const Descriptor* descriptor =
      std::get<const Descriptor*>(message_record->descriptor);
  for (int i = 0; i < descriptor->field_count(); i++) {
    const FieldDescriptor* fd = descriptor->field(i);
    const std::string_view name = fd->name();
    MessageRecord::ChildElement element =
        message_record->children.at(fd->name());
    if (element.unsupported) {
      continue;
    }

    dslx::TypeAnnotation* type_annot;
    if (std::holds_alternative<std::string>(element.type)) {
      // Message/struct or enum.
      std::string type_name = std::get<std::string>(element.type);
      type_annot = name_to_record->at(type_name)->dslx_type;
    } else {
      // Anything else that's supported, i.e., a number.
      FieldDescriptor::Type field_type =
          std::get<FieldDescriptor::Type>(element.type);
      dslx::BuiltinTypeAnnotation* bits_type;
      if (IsFieldSigned(field_type)) {
        bits_type = module->Make<dslx::BuiltinTypeAnnotation>(
            span, dslx::BuiltinType::kSN,
            module->GetOrCreateBuiltinNameDef(dslx::BuiltinType::kSN));
      } else {
        bits_type = module->Make<dslx::BuiltinTypeAnnotation>(
            span, dslx::BuiltinType::kUN,
            module->GetOrCreateBuiltinNameDef(dslx::BuiltinType::kUN));
      }
      auto* array_size = module->Make<dslx::Number>(
          span, absl::StrCat(GetFieldWidth(field_type)),
          dslx::NumberKind::kOther, /*type=*/nullptr);

      type_annot =
          module->Make<dslx::ArrayTypeAnnotation>(span, bits_type, array_size);
    }

    // Zero-count elements are repeated fields that have no instances,
    // which results in empty arrays at instantiation time, which isn't
    // supported.
    if (element.count == 0) {
      continue;
    }
    if (!fd->is_repeated()) {
      elements.push_back(
          dslx::StructMember{span, std::string(name), type_annot});
    } else {
      auto* array_size = module->Make<dslx::Number>(
          span, absl::StrCat(element.count), dslx::NumberKind::kOther,
          /*type=*/nullptr);
      type_annot =
          module->Make<dslx::ArrayTypeAnnotation>(span, type_annot, array_size);
      elements.push_back(
          dslx::StructMember{span, std::string(name), type_annot});

      std::string name_with_count = absl::StrCat(fd->name(), "_count");
      auto* u32_annot = module->Make<dslx::BuiltinTypeAnnotation>(
          span, dslx::BuiltinType::kU32,
          module->GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU32));
      elements.push_back(dslx::StructMember{span, name_with_count, u32_annot});
    }
  }

  auto* name_def =
      module->Make<dslx::NameDef>(span, message_record->name, nullptr);
  auto* struct_def = module->Make<dslx::StructDef>(
      span, name_def, std::vector<dslx::ParametricBinding*>(), elements,
      /*is_public=*/true);
  name_def->set_definer(struct_def);
  XLS_RETURN_IF_ERROR(
      module->AddTop(struct_def, /*make_collision_error=*/nullptr));

  auto* type_ref = module->Make<dslx::TypeRef>(span, struct_def);
  auto* type_ref_type_annotation = module->Make<dslx::TypeRefTypeAnnotation>(
      span, type_ref, /*parametrics=*/std::vector<dslx::ExprOrType>{});
  message_record->dslx_type = type_ref_type_annotation;
  message_record->struct_def = struct_def;

  return absl::OkStatus();
}

// Less-than functor for ParsedMessages. Used below for sorting a btree_map.
struct MessageRecordLess {
  bool operator()(const MessageRecord* lhs, const MessageRecord* rhs) const {
    return strcmp(lhs->name.c_str(), rhs->name.c_str()) < 0;
  }
};

// Basically a toposort of message decls.
absl::Status EmitTypeDefs(dslx::Module* module, NameToRecord* name_to_record) {
  // Map of ParsedMessage to the messages it depends on (but that have not yet
  // been emitted).
  using BlockingSet = absl::flat_hash_set<const MessageRecord*>;
  // Use a sorted container - sorted by message name - so we have a consistent
  // output order.
  absl::btree_map<MessageRecord*, BlockingSet, MessageRecordLess> blockers;
  for (const auto& [name, message_record] : *name_to_record) {
    blockers[message_record.get()] = BlockingSet();
    for (const auto& [field_name, element] : message_record->children) {
      if (!element.unsupported &&
          std::holds_alternative<std::string>(element.type)) {
        std::string message_name = std::get<std::string>(element.type);
        blockers[message_record.get()].insert(
            name_to_record->at(message_name).get());
      }
    }
  }

  // Now iterate through the structs, emitting any that aren't blocked on prior
  // definitions. Once emitted, remove a struct from the dependees of the
  // remaining ones.
  std::vector<std::string> structs;
  while (!blockers.empty()) {
    absl::flat_hash_set<MessageRecord*> newly_done;
    bool progress = false;
    for (auto& [message_record, dependencies] : blockers) {
      if (!dependencies.empty()) {
        continue;
      }

      progress = true;
      if (std::holds_alternative<const Descriptor*>(
              message_record->descriptor)) {
        XLS_RETURN_IF_ERROR(
            EmitStructDef(module, message_record, name_to_record));
      } else {
        XLS_RETURN_IF_ERROR(EmitEnumDef(module, message_record));
      }
      newly_done.insert(message_record);
    }

    // Clean up anyone who's done from others' dependees.
    for (MessageRecord* message_record : newly_done) {
      for (auto& [_, dependencies] : blockers) {
        dependencies.erase(message_record);
      }

      blockers.erase(message_record);
    }

    XLS_RET_CHECK(progress) << "Infinite loop trying to emit struct defs.";
  }
  return absl::OkStatus();
}

// Common code for emitting an array of proto structs or enums to a DSLX array.
// "make_zero_valued_element" produces one empty element of the caller's type
// per call, for padding out under-specified arrays (where the maximum size is
// less than the current message's size).
absl::Status EmitArray(
    dslx::Module* module, const Message& message, const FieldDescriptor* fd,
    const Reflection* reflection, const MessageRecord& message_record,
    std::vector<dslx::Expr*>* array_elements,
    const std::function<absl::StatusOr<dslx::Expr*>()>&
        make_zero_valued_element,
    std::vector<std::pair<std::string, dslx::Expr*>>* elements) {
  dslx::Span span(dslx::Pos{}, dslx::Pos{});
  std::string_view field_name = fd->name();
  int64_t total_submsgs = message_record.children.at(field_name).count;
  int num_submsgs = reflection->FieldSize(message, fd);
  bool has_ellipsis = false;
  if (num_submsgs != total_submsgs) {
    // TODO(https://github.com/google/xls/issues/249): Marking an array
    // as "has_ellipsis" seems to still require that we specify all
    // members. Until resolved (?), we'll create fake, zero-valued,
    // members.
    // Fortunately, we have the _count member to indicate which are valid.
    has_ellipsis = false;
    for (int i = 0; i < total_submsgs - num_submsgs; i++) {
      XLS_ASSIGN_OR_RETURN(dslx::Expr * element, make_zero_valued_element());
      array_elements->push_back(element);
    }
  }

  auto* array =
      module->Make<dslx::ConstantArray>(span, *array_elements, has_ellipsis);
  elements->push_back(std::make_pair(std::string(field_name), array));

  auto* u32_type = module->Make<dslx::BuiltinTypeAnnotation>(
      span, dslx::BuiltinType::kU32,
      module->GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU32));
  auto* num_array_members = module->Make<dslx::Number>(
      span, absl::StrCat(num_submsgs), dslx::NumberKind::kOther, u32_type);
  elements->push_back(
      std::make_pair(absl::StrCat(field_name, "_count"), num_array_members));
  return absl::OkStatus();
}

// Forward decl of the overall data-emission driver function.
absl::StatusOr<dslx::Expr*> EmitData(std::string_view top_package,
                                     dslx::Module* module,
                                     const Message& message,
                                     const NameToRecord& name_to_record);

// Creates the DSLX elements for a struct instance.
absl::Status EmitStructData(
    std::string_view top_package, dslx::Module* module,
    const Message& message, const FieldDescriptor* fd,
    const Reflection* reflection, const MessageRecord& message_record,
    const NameToRecord& name_to_record,
    std::vector<std::pair<std::string, dslx::Expr*>>* elements) {
  dslx::Span span(dslx::Pos{}, dslx::Pos{});
  std::string_view field_name = fd->name();

  if (fd->is_repeated()) {
    int64_t total_submsgs = message_record.children.at(field_name).count;
    int num_submsgs = reflection->FieldSize(message, fd);
    if (total_submsgs == 0) {
      return absl::OkStatus();
    }

    std::vector<dslx::Expr*> array_elements;
    for (int submsg_idx = 0; submsg_idx < num_submsgs; submsg_idx++) {
      const Message& sub_message =
          reflection->GetRepeatedMessage(message, fd, submsg_idx);
      XLS_ASSIGN_OR_RETURN(
          dslx::Expr * expr,
          EmitData(top_package, module, sub_message, name_to_record));
      array_elements.push_back(expr);
    }

    std::string type_name =
        GetParentPrefixedName(top_package, fd->message_type());
    auto* typeref_type = name_to_record.at(type_name)->dslx_type;
    return EmitArray(
        module, message, fd, reflection, message_record, &array_elements,
        [module, typeref_type]() {
          return MakeZeroValuedElement(module, typeref_type);
        },
        elements);
  }

  const Message& sub_message = reflection->GetMessage(message, fd);
  XLS_ASSIGN_OR_RETURN(
      dslx::Expr * expr,
      EmitData(top_package, module, sub_message, name_to_record));
  elements->push_back(std::make_pair(std::string(field_name), expr));
  return absl::OkStatus();
}

// Emits the DSLX for an enum instance.
absl::Status EmitEnumData(
    std::string_view top_package, dslx::Module* module,
    const Message& message, const FieldDescriptor* fd,
    const Reflection* reflection, const MessageRecord& message_record,
    const NameToRecord& name_to_record,
    std::vector<std::pair<std::string, dslx::Expr*>>* elements) {
  dslx::Span span(dslx::Pos{}, dslx::Pos{});
  std::string_view field_name = fd->name();
  const EnumDescriptor* ed = fd->enum_type();

  if (fd->is_repeated()) {
    int64_t total_submsgs = message_record.children.at(fd->name()).count;
    int num_submsgs = reflection->FieldSize(message, fd);
    if (total_submsgs == 0) {
      return absl::OkStatus();
    }
    std::vector<dslx::Expr*> array_elements;

    for (int submsg_idx = 0; submsg_idx < num_submsgs; submsg_idx++) {
      const google::protobuf::EnumValueDescriptor* evd =
          reflection->GetRepeatedEnum(message, fd, submsg_idx);
      std::string type_name = GetParentPrefixedName(top_package, evd->type());
      auto* enum_def = name_to_record.at(type_name)->enum_def;
      auto* name_ref =
          module->Make<dslx::NameRef>(span, type_name, enum_def->name_def());
      array_elements.push_back(module->Make<dslx::ColonRef>(
          span, name_ref, std::string(evd->name())));
    }

    std::string type_name = GetParentPrefixedName(top_package, ed);
    auto* enum_def = name_to_record.at(type_name)->enum_def;
    auto* name_ref =
        module->Make<dslx::NameRef>(span, type_name, enum_def->name_def());
    return EmitArray(
        module, message, fd, reflection, message_record, &array_elements,
        [module, ed, name_ref]() {
          dslx::Span span(dslx::Pos{}, dslx::Pos{});
          return module->Make<dslx::ColonRef>(
              span, name_ref, std::string(ed->value(0)->name()));
        },
        elements);
  }

  const google::protobuf::EnumValueDescriptor* evd = reflection->GetEnum(message, fd);
  std::string type_name = GetParentPrefixedName(top_package, evd->type());
  auto* enum_def = name_to_record.at(type_name)->enum_def;
  auto* name_ref =
      module->Make<dslx::NameRef>(span, type_name, enum_def->name_def());
  auto* colon_ref =
      module->Make<dslx::ColonRef>(span, name_ref, std::string(evd->name()));
  elements->push_back(std::make_pair(std::string(field_name), colon_ref));
  return absl::OkStatus();
}

// Emits the DSLX for a number within a proto message instance.
absl::Status EmitIntegralData(
    dslx::Module* module, const Message& message, const FieldDescriptor* fd,
    const Reflection* reflection, const MessageRecord& message_record,
    std::vector<std::pair<std::string, dslx::Expr*>>* elements) {
  dslx::Span span(dslx::Pos{}, dslx::Pos{});
  std::string_view field_name = fd->name();
  FieldDescriptor::Type field_type = std::get<FieldDescriptor::Type>(
      message_record.children.at(fd->name()).type);
  dslx::BuiltinTypeAnnotation* bits_type;
  if (IsFieldSigned(field_type)) {
    bits_type = module->Make<dslx::BuiltinTypeAnnotation>(
        span, dslx::BuiltinType::kSN,
        module->GetOrCreateBuiltinNameDef(dslx::BuiltinType::kSN));
  } else {
    bits_type = module->Make<dslx::BuiltinTypeAnnotation>(
        span, dslx::BuiltinType::kUN,
        module->GetOrCreateBuiltinNameDef(dslx::BuiltinType::kUN));
  }
  int bit_width = GetFieldWidth(field_type);
  auto* array_dim = module->Make<dslx::Number>(span, absl::StrCat(bit_width),
                                               dslx::NumberKind::kOther,
                                               /*type=*/nullptr);
  auto* array_elem_type =
      module->Make<dslx::ArrayTypeAnnotation>(span, bits_type, array_dim);

  if (fd->is_repeated()) {
    int total_submsgs = message_record.children.at(fd->name()).count;
    int num_submsgs = reflection->FieldSize(message, fd);
    if (total_submsgs == 0) {
      return absl::OkStatus();
    }

    std::vector<dslx::Expr*> array_elements;
    for (int submsg_idx = 0; submsg_idx < num_submsgs; submsg_idx++) {
      if (IsFieldSigned(field_type)) {
        int64_t value = static_cast<int64_t>(
            GetFieldValue(message, *reflection, *fd, submsg_idx));
        array_elements.push_back(module->Make<dslx::Number>(
            span, absl::StrCat(value), dslx::NumberKind::kOther,
            array_elem_type));

      } else {
        uint64_t value = GetFieldValue(message, *reflection, *fd, submsg_idx);
        array_elements.push_back(module->Make<dslx::Number>(
            span, absl::StrCat(value), dslx::NumberKind::kOther,
            array_elem_type));
      }
    }

    return EmitArray(
        module, message, fd, reflection, message_record, &array_elements,
        [module, array_elem_type]() {
          dslx::Span span(dslx::Pos{}, dslx::Pos{});
          return module->Make<dslx::Number>(span, "0", dslx::NumberKind::kOther,
                                            array_elem_type);
        },
        elements);
  }

  uint64_t value = GetFieldValue(message, *reflection, *fd);
  dslx::Number* number = module->Make<dslx::Number>(
      span, absl::StrCat(value), dslx::NumberKind::kOther, array_elem_type);
  elements->push_back(std::make_pair(std::string(field_name), number));

  return absl::OkStatus();
}

// Instantiates a message as a DSLX constant.
absl::StatusOr<dslx::Expr*> EmitData(std::string_view top_package,
                                     dslx::Module* module,
                                     const Message& message,
                                     const NameToRecord& name_to_record) {
  const Descriptor* descriptor = message.GetDescriptor();
  std::string type_name = GetParentPrefixedName(top_package, descriptor);
  const Reflection* reflection = message.GetReflection();
  // const MessageRecord& message_record =
  // *name_to_record.at(descriptor->name());
  const MessageRecord& message_record = *name_to_record.at(type_name);

  dslx::Span span(dslx::Pos{}, dslx::Pos{});
  dslx::TypeAnnotation* struct_def = message_record.dslx_type;
  std::vector<std::pair<std::string, dslx::Expr*>> elements;
  for (int field_idx = 0; field_idx < descriptor->field_count(); field_idx++) {
    const FieldDescriptor* fd = descriptor->field(field_idx);
    std::string_view field_name = fd->name();
    MessageRecord::ChildElement element =
        message_record.children.at(field_name);
    if (element.unsupported) {
      continue;
    }

    switch (fd->type()) {
      case FieldDescriptor::Type::TYPE_MESSAGE:
        XLS_RETURN_IF_ERROR(EmitStructData(top_package, module, message, fd,
                                           reflection, message_record,
                                           name_to_record, &elements));
        break;
      case FieldDescriptor::Type::TYPE_ENUM:
        XLS_RETURN_IF_ERROR(EmitEnumData(top_package, module, message, fd,
                                         reflection, message_record,
                                         name_to_record, &elements));
        break;
      default:
        XLS_RETURN_IF_ERROR(EmitIntegralData(module, message, fd, reflection,
                                             message_record, &elements));
    }
  }

  return module->Make<dslx::StructInstance>(span, struct_def, elements);
}

absl::StatusOr<std::unique_ptr<dslx::Module>> ProtoToDslxWithDescriptorPool(
    std::string_view message_name, std::string_view text_proto,
    std::string_view binding_name, DescriptorPool* descriptor_pool,
    dslx::FileTable& file_table) {
  XLS_RET_CHECK(descriptor_pool != nullptr);

  google::protobuf::DynamicMessageFactory factory;

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Message> new_message,
                       ConstructProtoViaText(text_proto, message_name,
                                             descriptor_pool, &factory));

  auto module = std::make_unique<dslx::Module>(
      "the_module", /*fs_path=*/std::nullopt, file_table);

  ProtoToDslxManager proto_to_dslx(module.get());
  XLS_RETURN_IF_ERROR(proto_to_dslx.AddProtoInstantiationToDslxModule(
      binding_name, *new_message));

  return module;
}

}  // namespace

ProtoToDslxManager::ProtoToDslxManager(dslx::Module* module)
    : module_(module) {}

ProtoToDslxManager::~ProtoToDslxManager() = default;

absl::Status ProtoToDslxManager::AddProtoInstantiationToDslxModule(
    std::string_view binding_name, const Message& message) {
  XLS_RET_CHECK(module_ != nullptr);

  const Descriptor* descriptor = message.GetDescriptor();

  if (!name_to_records_.contains(descriptor)) {
    XLS_RETURN_IF_ERROR(AddProtoTypeToDslxModule(message));
  }
  NameToRecord& name_to_record = name_to_records_[descriptor];
  std::string_view top_package = descriptor->file()->package();

  XLS_ASSIGN_OR_RETURN(dslx::Expr * expr,
                       EmitData(top_package, module_, message, name_to_record));
  dslx::Span span{dslx::Pos{}, dslx::Pos{}};
  auto* name_def = module_->Make<dslx::NameDef>(span, std::string(binding_name),
                                                /*definer=*/nullptr);
  auto* constant_def = module_->Make<dslx::ConstantDef>(
      span, name_def, /*type_annotation=*/nullptr, expr,
      /*is_public=*/true);
  name_def->set_definer(constant_def);
  XLS_RETURN_IF_ERROR(
      module_->AddTop(constant_def, /*make_collision_error=*/nullptr));

  return absl::OkStatus();
}

absl::Status ProtoToDslxManager::AddProtoTypeToDslxModule(
    const Message& message) {
  XLS_RET_CHECK(module_ != nullptr);

  const Descriptor* descriptor = message.GetDescriptor();
  XLS_RET_CHECK(descriptor != nullptr);

  if (name_to_records_.contains(descriptor)) {
    return absl::InternalError(absl::StrFormat(
        "AddProtoTypeToDslxModule(%s) should only be called once"
        "per message type.",
        descriptor->full_name()));
  }
  NameToRecord& name_to_record = name_to_records_[descriptor];

  std::string_view top_package = descriptor->file()->package();
  XLS_RETURN_IF_ERROR(
      CollectMessageLayout(top_package, *descriptor, &name_to_record));
  XLS_RETURN_IF_ERROR(
      CollectElementCounts(top_package, message, &name_to_record));
  XLS_RETURN_IF_ERROR(EmitTypeDefs(module_, &name_to_record));

  return absl::OkStatus();
}

// As ProcessProtoSchema, but takes the proto schema definition as a string.
//
// TODO(leary): 2021-05-26 There should be a more suitable API for doing this
// when the schema is in string form -- as a temporary kluge this places the
// schema in a temporary file and loads it from that path, but really we should
// find the better API -- this also checks that there are no dependencies
// (imports) unlike the above.
absl::StatusOr<std::unique_ptr<DescriptorPool>> ProcessStringProtoSchema(
    std::string_view proto_def) {
  DiskSourceTree source_tree;

  XLS_ASSIGN_OR_RETURN(auto tempdir, TempDirectory::Create());

  XLS_RETURN_IF_ERROR(
      SetFileContents(tempdir.path() / "schema.proto", proto_def));

  // Our proto might have other dependencies, so we have to let the proto
  // compiler know about the layout of our source tree.
  source_tree.MapPath("", tempdir.path().string());

  SourceTreeDescriptorDatabase db(&source_tree);
  FileDescriptorProto descriptor_proto;
  DbErrorCollector db_collector;
  db.RecordErrorsTo(&db_collector);
  XLS_RET_CHECK(db.FindFileByName("schema.proto", &descriptor_proto));
  XLS_RET_CHECK(descriptor_proto.dependency().empty());

  auto pool = std::make_unique<DescriptorPool>();
  PoolErrorCollector pool_collector;
  pool->BuildFileCollectingErrors(descriptor_proto, &pool_collector);
  return pool;
}

absl::StatusOr<std::unique_ptr<dslx::Module>> ProtoToDslx(
    const std::filesystem::path& source_root,
    const std::filesystem::path& proto_schema_path,
    std::string_view message_name, std::string_view text_proto,
    std::string_view binding_name, dslx::FileTable& file_table) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<DescriptorPool> descriptor_pool,
                       ProcessProtoSchema(source_root, proto_schema_path));
  return ProtoToDslxWithDescriptorPool(message_name, text_proto, binding_name,
                                       descriptor_pool.get(), file_table);
}

absl::StatusOr<std::unique_ptr<dslx::Module>> ProtoToDslxViaText(
    std::string_view proto_def, std::string_view message_name,
    std::string_view text_proto, std::string_view binding_name,
    dslx::FileTable& file_table) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<DescriptorPool> descriptor_pool,
                       ProcessStringProtoSchema(proto_def));
  return ProtoToDslxWithDescriptorPool(message_name, text_proto, binding_name,
                                       descriptor_pool.get(), file_table);
}

absl::StatusOr<std::unique_ptr<Message>> ConstructProtoViaText(
    std::string_view text_proto, std::string_view message_name,
    DescriptorPool* descriptor_pool, google::protobuf::DynamicMessageFactory* factory) {
  XLS_RET_CHECK(descriptor_pool != nullptr);
  XLS_RET_CHECK(factory != nullptr);

  const Descriptor* descriptor =
      descriptor_pool->FindMessageTypeByName(ToProtoString(message_name));
  XLS_RET_CHECK_NE(descriptor, nullptr);

  const Message* message = factory->GetPrototype(descriptor);
  XLS_RET_CHECK(message != nullptr);
  std::unique_ptr<Message> new_message(message->New());

  google::protobuf::TextFormat::ParseFromString(ToProtoString(text_proto),
                                      new_message.get());

  return new_message;
}

absl::StatusOr<std::unique_ptr<dslx::Module>> CreateDslxFromParams(
    std::string_view module_name,
    absl::Span<const std::pair<std::string_view, const google::protobuf::Message*>>
        params,
    dslx::FileTable& file_table) {
  auto module =
      std::make_unique<dslx::Module>(std::string{module_name},
                                     /*fs_path=*/std::nullopt, file_table);

  ProtoToDslxManager proto_to_dslx(module.get());

  for (std::pair<std::string_view, const google::protobuf::Message*> p : params) {
    std::string_view binding_name = p.first;
    const google::protobuf::Message* proto_param = p.second;

    XLS_RET_CHECK(proto_param != nullptr);

    XLS_RET_CHECK_OK(proto_to_dslx.AddProtoInstantiationToDslxModule(
        binding_name, *proto_param));
  }

  return module;
}

}  // namespace xls
