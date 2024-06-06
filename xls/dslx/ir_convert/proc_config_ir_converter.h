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
#ifndef XLS_DSLX_IR_CONVERT_PROC_CONFIG_IR_CONVERTER_H_
#define XLS_DSLX_IR_CONVERT_PROC_CONFIG_IR_CONVERTER_H_

#include <string>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/ir_convert/conversion_info.h"
#include "xls/dslx/ir_convert/extract_conversion_order.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/ir/channel.h"
#include "xls/ir/name_uniquer.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls::dslx {

using ProcConfigValue = std::variant<Value, Channel*>;
using MemberNameToValue = absl::flat_hash_map<std::string, ProcConfigValue>;

// ProcConversionData holds various information about individual proc instances
// needed throughout the conversion process, packaged together to avoid
// overcomplicating fn signatures.
struct ProcConversionData {
  // Maps a proc instance ID to the set of config fn args, either IR Values or
  // Channels.
  absl::flat_hash_map<ProcId, std::vector<ProcConfigValue>> id_to_config_args;

  // Maps a proc instance ID to the IR Value to use for the first execution
  // of the `next` function.
  absl::flat_hash_map<ProcId, Value> id_to_initial_value;

  // Maps a proc instance ID to its set of members and values.
  absl::flat_hash_map<ProcId, MemberNameToValue> id_to_members;
};

// ProcConfigIrConverter is specialized for converting - you guessed it! - Proc
// Config functions into IR.  Config functions don't _actually_ lower to IR:
// instead they define constants and bind channels to Proc members.
class ProcConfigIrConverter : public AstNodeVisitorWithDefault {
 public:
  ProcConfigIrConverter(PackageConversionData* conversion_info, Function* f,
                        TypeInfo* type_info, ImportData* import_data,
                        ProcConversionData* proc_data,
                        const ParametricEnv& bindings, const ProcId& proc_id);

  absl::Status HandleBlock(const Block* node) override;
  absl::Status HandleStatement(const Statement* node) override;
  absl::Status HandleChannelDecl(const ChannelDecl* node) override;
  absl::Status HandleColonRef(const ColonRef* node) override;
  absl::Status HandleFunction(const Function* node) override;
  absl::Status HandleInvocation(const Invocation* node) override;
  absl::Status HandleLet(const Let* node) override;
  absl::Status HandleNameRef(const NameRef* node) override;
  absl::Status HandleNumber(const Number* node) override;
  absl::Status HandleParam(const Param* node) override;
  absl::Status HandleSpawn(const Spawn* node) override;
  absl::Status HandleStructInstance(const StructInstance* node) override;
  absl::Status HandleXlsTuple(const XlsTuple* node) override;

  // Sets the mapping from the elements in the config-ending tuple to the
  // corresponding Proc members.
  absl::Status Finalize();

 private:
  PackageConversionData* conversion_info_;
  Function* f_;
  TypeInfo* type_info_;
  ImportData* import_data_;
  NameUniquer channel_name_uniquer_;

  ProcConversionData* proc_data_;
  absl::flat_hash_map<std::vector<Proc*>, int> instances_;

  const ParametricEnv& bindings_;
  ProcId proc_id_;

  absl::flat_hash_map<const AstNode*, ProcConfigValue> node_to_ir_;

  // Stores the last tuple created in this Function. Used to destructure any
  // output to match with Proc members.
  const XlsTuple* final_tuple_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_IR_CONVERT_PROC_CONFIG_IR_CONVERTER_H_
