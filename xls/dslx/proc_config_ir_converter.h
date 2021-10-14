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
#ifndef XLS_DSLX_PROC_CONFIG_IR_CONVERTER_H_
#define XLS_DSLX_PROC_CONFIG_IR_CONVERTER_H_

#include "absl/container/flat_hash_map.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/extract_conversion_order.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_info.h"
#include "xls/ir/channel.h"
#include "xls/ir/package.h"

namespace xls::dslx {

using ProcConfigValue = absl::variant<Value, Channel*>;
using MemberNameToValue = absl::flat_hash_map<std::string, ProcConfigValue>;

// ProcConfigIrConverter is specialized for converting - you guessed it! - Proc
// Config functions into IR.  Config functions don't _actually_ lower to IR:
// instead they define constants and bind channels to Proc members.
class ProcConfigIrConverter : public AstNodeVisitorWithDefault {
 public:
  ProcConfigIrConverter(
      Package* package, Function* f, TypeInfo* type_info,
      ImportData* import_data,
      absl::flat_hash_map<ProcId, std::vector<ProcConfigValue>>*
          proc_id_to_args,
      absl::flat_hash_map<ProcId, MemberNameToValue>* proc_id_to_members,
      const SymbolicBindings& bindings, const ProcId& proc_id);

  absl::Status HandleChannelDecl(ChannelDecl* node);
  absl::Status HandleFunction(Function* node);
  absl::Status HandleLet(Let* node);
  absl::Status HandleNameRef(NameRef* node);
  absl::Status HandleNumber(Number* node);
  absl::Status HandleParam(Param* node);
  absl::Status HandleSpawn(Spawn* node);
  absl::Status HandleXlsTuple(XlsTuple* node);

  // Sets the mapping from the elements in the config-ending tuple to the
  // corresponding Proc members.
  absl::Status Finalize();

 private:
  Package* package_;
  Function* f_;
  TypeInfo* type_info_;
  ImportData* import_data_;

  absl::flat_hash_map<ProcId, std::vector<ProcConfigValue>>* proc_id_to_args_;
  absl::flat_hash_map<ProcId, MemberNameToValue>* proc_id_to_members_;
  absl::flat_hash_map<std::vector<Proc*>, int> instances_;

  const SymbolicBindings& bindings_;
  ProcId proc_id_;

  absl::flat_hash_map<AstNode*, ProcConfigValue> node_to_ir_;

  // Stores the last tuple created in this Function. Used to destructure any
  // output to match with Proc members.
  XlsTuple* final_tuple_;
};

// Utility functions exposed for testing.
// Finds the Proc identified by the given node (either NameRef or ColonRef),
// using the associated ImportData for import Module lookup.
absl::StatusOr<Proc*> ResolveProc(Expr* node, ImportData* import_data);

}  // namespace xls::dslx

#endif  // XLS_DSLX_PROC_CONFIG_IR_CONVERTER_H_
