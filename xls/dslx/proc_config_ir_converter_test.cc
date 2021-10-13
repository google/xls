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
#include "xls/dslx/proc_config_ir_converter.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/pos.h"
#include "xls/ir/package.h"

namespace xls::dslx {
namespace {

using status_testing::StatusIs;
using testing::HasSubstr;

TEST(ProcConfigIrConverterTest, ResolveProcNameRef) {
  auto import_data = ImportData::CreateForTest();
  Module module("test_module");
  NameDef* name_def = module.Make<NameDef>(Span::Fake(), "proc_name", nullptr);
  NameDef* config_name_def =
      module.Make<NameDef>(Span::Fake(), "config_name", nullptr);
  NameDef* next_name_def =
      module.Make<NameDef>(Span::Fake(), "next_name", nullptr);
  Function* config = nullptr;
  Function* next = nullptr;
  std::vector<Param*> members;
  std::vector<ParametricBinding*> bindings;
  Proc* original_proc =
      module.Make<Proc>(Span::Fake(), name_def, config_name_def, next_name_def,
                        bindings, members, config, next, /*is_public=*/true);
  module.AddTop(original_proc);
  name_def->set_definer(original_proc);

  NameRef* name_ref = module.Make<NameRef>(Span::Fake(), "proc_name", name_def);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * p, ResolveProc(name_ref, &import_data));
  EXPECT_EQ(p, original_proc);
}

TEST(ProcConfigIrConverterTest, ResolveProcColonRef) {
  std::vector<std::string> import_tokens{"robs", "dslx", "import_module"};
  auto import_data = ImportData::CreateForTest();
  ImportTokens subject(import_tokens);
  ModuleInfo module_info;
  module_info.module = std::make_unique<Module>("import_module");
  module_info.type_info = nullptr;
  Module* import_module = module_info.module.get();

  NameDef* name_def =
      import_module->Make<NameDef>(Span::Fake(), "proc_name", nullptr);
  NameDef* config_name_def =
      import_module->Make<NameDef>(Span::Fake(), "config_name", nullptr);
  NameDef* next_name_def =
      import_module->Make<NameDef>(Span::Fake(), "next_name", nullptr);
  Function* config = nullptr;
  Function* next = nullptr;
  std::vector<Param*> members;
  std::vector<ParametricBinding*> bindings;
  Proc* original_proc = import_module->Make<Proc>(
      Span::Fake(), name_def, config_name_def, next_name_def, bindings, members,
      config, next, /*is_public=*/true);
  import_module->AddTop(original_proc);
  name_def->set_definer(original_proc);

  XLS_ASSERT_OK(import_data.Put(subject, std::move(module_info)));

  Module module("test_module");
  NameDef* module_def =
      module.Make<NameDef>(Span::Fake(), "import_module", nullptr);
  Import* import = module.Make<Import>(Span::Fake(), import_tokens, module_def,
                                       absl::nullopt);
  module_def->set_definer(import);
  NameRef* module_ref =
      module.Make<NameRef>(Span::Fake(), "import_module", module_def);
  ColonRef* colon_ref =
      module.Make<ColonRef>(Span::Fake(), module_ref, "proc_name");
  XLS_ASSERT_OK_AND_ASSIGN(Proc * p, ResolveProc(colon_ref, &import_data));
  EXPECT_EQ(p, original_proc);
}

TEST(ProcConfigIrConverterTest, BasicConversion) {
  constexpr absl::string_view kModule = R"(
proc test_proc {
  config(c_input: chan in u32) {
    let c = c_input;
    ()
  }
  next() {
    ()
  }

  c: chan in u32;
}

proc main {
  config() {
    let (p, c) = chan u32;
    spawn test_proc(c)();
    let c = p;
    ()
  }
  next() {
    ()
  }

  c: chan out u32;
}
)";

  auto import_data = ImportData::CreateForTest();

  absl::flat_hash_map<ProcId, std::vector<ProcConfigValue>> proc_id_to_args;
  absl::flat_hash_map<ProcId, MemberNameToValue> proc_id_to_members;
  SymbolicBindings bindings;
  ProcId proc_id{/*proc_stack=*/{}, /*instance=*/0};

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kModule, "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetFunctionOrError("test_proc_config"));

  Package package("the_package");
  ChannelMetadataProto metadata;
  StreamingChannel channel("the_channel", /*id=*/0, ChannelOps::kSendReceive,
                           package.GetBitsType(32), {}, FlowControl::kNone,
                           metadata);

  proc_id_to_args[proc_id] = {&channel};

  ProcConfigIrConverter converter(&package, f, tm.type_info, &import_data,
                                  &proc_id_to_args, &proc_id_to_members,
                                  bindings, proc_id);
  XLS_EXPECT_OK(f->Accept(&converter));
}

TEST(ProcConfigIrConverterTest, CatchesMissingArgMap) {
  constexpr absl::string_view kModule = R"(
proc test_proc {
  config(c_input: chan in u32) {
    let c = c_input;
    ()
  }
  next() {
    ()
  }

  c: chan in u32;
}

proc main {
  config() {
    let (p, c) = chan u32;
    spawn test_proc(c)();
    let c = p;
    ()
  }
  next() {
    ()
  }

  c: chan out u32;
}
)";

  auto import_data = ImportData::CreateForTest();

  absl::flat_hash_map<ProcId, std::vector<ProcConfigValue>> proc_id_to_args;
  absl::flat_hash_map<ProcId, MemberNameToValue> proc_id_to_members;
  SymbolicBindings bindings;
  ProcId proc_id{/*proc_stack=*/{}, /*instance=*/0};

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kModule, "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetFunctionOrError("test_proc_config"));

  Package package("the_package");
  ProcConfigIrConverter converter(&package, f, tm.type_info, &import_data,
                                  &proc_id_to_args, &proc_id_to_members,
                                  bindings, proc_id);
  EXPECT_THAT(f->Accept(&converter),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("not found in arg mapping")));
}

}  // namespace
}  // namespace xls::dslx
