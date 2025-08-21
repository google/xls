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

#ifndef XLS_CONTRIB_XLSCC_TRANSLATOR_H_
#define XLS_CONTRIB_XLSCC_TRANSLATOR_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "clang/include/clang/AST/ASTContext.h"
#include "clang/include/clang/AST/Attr.h"
#include "clang/include/clang/AST/Attrs.inc"
#include "clang/include/clang/AST/ComputeDependence.h"
#include "clang/include/clang/AST/Decl.h"
#include "clang/include/clang/AST/Expr.h"
#include "clang/include/clang/AST/Mangle.h"
#include "clang/include/clang/AST/OperationKinds.h"
#include "clang/include/clang/AST/Stmt.h"
#include "clang/include/clang/AST/Type.h"
#include "clang/include/clang/Basic/LLVM.h"
#include "clang/include/clang/Basic/SourceLocation.h"
#include "xls/contrib/xlscc/cc_parser.h"
#include "xls/contrib/xlscc/generate_fsm.h"
#include "xls/contrib/xlscc/hls_block.pb.h"
#include "xls/contrib/xlscc/metadata_output.pb.h"
#include "xls/contrib/xlscc/tracked_bvalue.h"
#include "xls/contrib/xlscc/translator_types.h"
#include "xls/ir/bits.h"
#include "xls/ir/caret.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/fileno.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/state_element.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/solvers/z3_ir_translator.h"
#include "z3/src/api/z3_api.h"

namespace xlscc {

class Translator;

class NewFSMGenerator;

std::string Debug_GenerateReadableTypeName(xls::Type* type);
std::string Debug_GenerateSliceGraph(const GeneratedFunction& func);

struct TranslationContext;

struct FunctionInProgress {
  bool add_this_return;
  // Destroy the builder last to avoid TrackedBValue errors
  std::unique_ptr<TrackedFunctionBuilder> builder;
  std::vector<const clang::NamedDecl*> ref_returns;
  std::unique_ptr<TranslationContext> translation_context;
  std::unique_ptr<GeneratedFunction> generated_function;
};

// Encapsulates a context for translating Clang AST to XLS IR.
// This is roughly equivalent to a "scope" in C++. There will typically
//  be at least one context pushed into the context stack for each C++ scope.
// The Translator::PopContext() function will propagate certain values, such
//  as new CValues for assignments to variables declared outside the scope,
//  up to the next context / outer scope.
struct TranslationContext {
  TrackedBValue not_full_condition_bval(const xls::SourceInfo& loc) const {
    if (!full_condition.valid()) {
      return fb->Literal(xls::UBits(0, 1), loc);
    }
    return fb->Not(full_condition, loc);
  }

  TrackedBValue full_condition_bval(const xls::SourceInfo& loc) const {
    if (!full_condition.valid()) {
      return fb->Literal(xls::UBits(1, 1), loc);
    }
    return full_condition;
  }

  TrackedBValue not_relative_condition_bval(const xls::SourceInfo& loc) const {
    if (!relative_condition.valid()) {
      return fb->Literal(xls::UBits(0, 1), loc);
    }
    return fb->Not(relative_condition, loc);
  }

  TrackedBValue relative_condition_bval(const xls::SourceInfo& loc) const {
    if (!relative_condition.valid()) {
      return fb->Literal(xls::UBits(1, 1), loc);
    }
    return relative_condition;
  }

  void and_condition_util(TrackedBValue and_condition,
                          TrackedBValue& mod_condition,
                          const xls::SourceInfo& loc) const {
    if (!mod_condition.valid()) {
      mod_condition = and_condition;
    } else {
      mod_condition = fb->And(mod_condition, and_condition, loc);
    }
  }

  void or_condition_util(TrackedBValue or_condition,
                         TrackedBValue& mod_condition,
                         const xls::SourceInfo& loc) const {
    if (!mod_condition.valid()) {
      mod_condition = or_condition;
    } else {
      mod_condition = fb->Or(mod_condition, or_condition, loc);
    }
  }

  void print_vars() const {
    std::cerr << "Context {" << std::endl;
    std::cerr << "  vars:" << std::endl;
    for (const auto& var : variables) {
      std::cerr << "  -- " << var.first->getNameAsString() << ": "
                << var.second.rvalue().ToString() << std::endl;
    }
    std::cerr << "}" << std::endl;
  }

  void print_vars_infix() const {
    std::cerr << "Context {" << std::endl;
    std::cerr << "  vars:" << std::endl;
    for (const auto& var : variables) {
      std::cerr << "  -- " << var.first->getNameAsString() << ": "
                << Debug_NodeToInfix(var.second.rvalue()) << std::endl;
    }
    std::cerr << "}" << std::endl;
  }

  std::shared_ptr<CType> return_type;
  xls::BuilderBase* fb = nullptr;
  clang::ASTContext* ast_context = nullptr;

  // Information being gathered about function currently being processed
  GeneratedFunction* sf = nullptr;

  // "this" uses the key of the clang::NamedDecl* of the method
  CValueMap<const clang::NamedDecl*> variables;

  const clang::NamedDecl* override_this_decl_ = nullptr;

  CValue return_cval;

  TrackedBValue last_return_condition;
  // For "control flow": assignments after a return are conditional on this
  TrackedBValue have_returned_condition;

  // Condition for assignments
  TrackedBValue full_condition;
  TrackedBValue full_condition_on_enter_block;
  TrackedBValue relative_condition;

  // These flags control the behavior of break and continue statements
  bool in_for_body = false;
  bool in_switch_body = false;

  // Used in translating for loops
  // invalid indicates no break/continue
  TrackedBValue relative_break_condition;
  TrackedBValue relative_continue_condition;

  // Switch stuff
  // hit_break is set when a break is encountered inside of a switch body.
  // This signals from GenerateIR_Stmt() to GenerateIR_Switch().
  bool hit_break = false;
  // For checking for conditional breaks. If a break occurs in a context
  //  with a condition that's not equal to the enclosing "switch condition",
  //  ie that specified by the enclosing case or default, then a conditional
  //  break is detected, which is unsupported and an error.
  TrackedBValue full_switch_cond;

  // Ignore pointer qualifiers in type translation. Normally pointers
  //  cause an unsupported error, but when this flag is true,
  //  "Foo*" is translated as "Foo".
  // This mode was created to handle the "this" pointer, which is mandatory
  //  to handle classes.
  bool ignore_pointers = false;

  // Assume for loops without pragmas are unrolled
  bool for_loops_default_unroll = false;

  // Flag set in pipelined for body
  // TODO(seanhaskell): Remove once all features are supported
  bool in_pipelined_for_body = false;
  int64_t outer_pipelined_loop_init_interval = 0;

  // When set to true, the expression is evaluated as an lvalue, for pointer
  // assignments
  bool lvalue_mode = false;

  // These flags control the behavior when the context / scope is exited
  bool propagate_up = true;
  bool propagate_break_up = true;
  bool propagate_continue_up = true;
  bool propagate_declarations = false;

  bool mask_assignments = false;

  // Don't create side-effects when exploring.
  bool mask_side_effects = false;
  bool any_side_effects_requested = false;
  bool any_writes_generated = false;
  bool any_io_ops_requested = false;

  bool mask_io_other_than_memory_writes = false;
  bool mask_memory_writes = false;

  bool allow_default_pad = false;

  // Number of times a variable is accessed
  // Always propagates up
  absl::flat_hash_map<const clang::NamedDecl*, int64_t> variables_accessed;
  absl::flat_hash_set<const clang::NamedDecl*> variables_masked_by_assignment;
};

std::string Debug_VariablesChangedBetween(const TranslationContext& before,
                                          const TranslationContext& after);

std::optional<std::list<const xls::Node*>> Debug_DeeplyCheckOperandsFromPrev(
    const xls::Node* node,
    const absl::flat_hash_set<const xls::Node*>& prev_state_io_nodes);

enum IOOpOrdering {
  kNone = 0,
  kChannelWise = 1,
  kLexical = 2,
};

struct ChannelOptions {
  xls::ChannelStrictness default_strictness =
      xls::ChannelStrictness::kProvenMutuallyExclusive;
  absl::flat_hash_map<std::string, xls::ChannelStrictness> strictness_map;
};

class Translator final : public GeneratorBase,
                         public TranslatorTypeInterface,
                         public TranslatorIOInterface {
  void debug_prints(const TranslationContext& context);

 public:
  // Make unrolling configurable from main
  explicit Translator(
      bool error_on_init_interval = false, bool error_on_uninitialized = false,
      bool generate_new_fsm = false, bool merge_states = false,
      bool split_states_on_channel_ops = false,
      DebugIrTraceFlags debug_ir_trace_flags = DebugIrTraceFlags_None,
      int64_t max_unroll_iters = 1000, int64_t warn_unroll_iters = 100,
      int64_t z3_rlimit = -1, IOOpOrdering op_ordering = IOOpOrdering::kNone,
      std::unique_ptr<CCParser> existing_parser = std::unique_ptr<CCParser>());
  ~Translator() final;

  // This function uses Clang to parse a source file and then walks its
  //  AST to discover global constructs. It will also scan the file
  //  and includes, recursively, for #pragma statements.
  //
  // Among these are functions, which can be used as entry points
  //  for translation to IR.
  //
  // source_filename must be .cc
  // Retains references to the TU until ~Translator()
  absl::Status ScanFile(std::string_view source_filename,
                        absl::Span<std::string_view> command_line_args);

  // Call after ScanFile, as the top function may be specified by #pragma
  // If none was found, an error is returned
  absl::StatusOr<std::string> GetEntryFunctionName() const;

  // See CCParser::SelectTop()
  absl::Status SelectTop(std::string_view top_function_name,
                         std::string_view top_class_name = "");

  // Generates IR as an XLS function, that is, a pure function without
  //  IO / state / side effects.
  // If top_function is 0 or "" then top must be specified via pragma
  // force_static=true Means the function is not generated with a "this"
  //  parameter & output. It is generated as if static was specified in the
  //  method prototype.
  absl::StatusOr<GeneratedFunction*> GenerateIR_Top_Function(
      xls::Package* package,
      const absl::flat_hash_map<const clang::NamedDecl*, ChannelBundle>&
          top_channel_injections,
      bool force_static = false, bool member_references_become_channels = false,
      int default_init_interval = 0,
      const clang::FunctionDecl* top_function_override = nullptr,
      std::string_view name_postfix = "");

  // Generates IR as an HLS block / XLS proc.
  absl::StatusOr<xls::Proc*> GenerateIR_Block(
      xls::Package* package, const HLSBlock& block,
      int top_level_init_interval = 0,
      const ChannelOptions& channel_options = {});

  // Generates IR as an HLS block / XLS proc.
  // Top is a method, block specification is extracted from the class.
  absl::StatusOr<xls::Proc*> GenerateIR_BlockFromClass(
      xls::Package* package, HLSBlock* block_spec_out,
      int top_level_init_interval = 0,
      const ChannelOptions& channel_options = {});

  // Generates the stub function that invokes a sub-block via IO ops.
  absl::Status GenerateIR_SubBlockStub(GeneratedFunction& sf,
                                       const clang::FunctionDecl* funcdecl,
                                       const FunctionInProgress& header);

  // Generates the proc for a sub-block which is invoked via IO ops.
  absl::Status GenerateIR_SubBlock(GeneratedFunction* gen_func,
                                   xls::Package* package,
                                   int top_level_init_interval);

  // Generate some useful metadata after either GenerateIR_Top_Function() or
  //  GenerateIR_Block() has run.
  absl::StatusOr<xlscc_metadata::MetadataOutput> GenerateMetadata();
  absl::Status GenerateFunctionMetadata(
      const clang::FunctionDecl* func,
      xlscc_metadata::FunctionPrototype* output,
      absl::flat_hash_set<const clang::NamedDecl*>& aliases_used);

  void AddSourceInfoToPackage(xls::Package& package);

  inline void SetIOTestMode() { io_test_mode_ = true; }

  absl::StatusOr<const clang::FunctionDecl*> GetTopFunction() const {
    CHECK_NE(parser_, nullptr);
    return parser_->GetTopFunction();
  }

  const GeneratedFunction* GetGeneratedFunction(
      const clang::FunctionDecl* decl) const {
    return inst_functions_.at(decl).get();
  }

 private:
  friend class CInstantiableTypeAlias;
  friend class CStructType;
  friend class CInternalTuple;
  friend class NewFSMGenerator;

  std::function<std::optional<std::string>(xls::Fileno)> LookUpInPackage();

  void AppendMessageTraces(std::string* result,
                           const xls::SourceInfo& loc) override;

  template <typename... Args>
  std::string WarningMessage(const xls::SourceInfo& loc,
                             const absl::FormatSpec<Args...>& format,
                             const Args&... args) {
    std::string result = absl::StrFormat(format, args...);

    absl::StrAppend(
        &result, absl::StrJoin(
                     loc.locations, "\n",
                     [this](std::string* out, const xls::SourceLocation& loc) {
                       absl::StrAppend(out, PrintCaret(LookUpInPackage(), loc));
                     }));

    return result;
  }

  // This object is used to push a new translation context onto the stack
  //  and then to pop it via RAII. This guard provides options for which bits of
  //  context to propagate up when popping it from the stack.
  struct PushContextGuard {
    PushContextGuard(Translator& translator, const xls::SourceInfo& loc)
        : translator(translator), loc(loc) {
      translator.PushContext();
    }
    PushContextGuard(Translator& translator, TrackedBValue and_condition,
                     const xls::SourceInfo& loc)
        : translator(translator), loc(loc) {
      translator.PushContext();
      absl::Status status = translator.and_condition(and_condition, loc);
      if (!status.ok()) {
        LOG(ERROR) << status.message();
      }
    }
    PushContextGuard(Translator& translator,
                     const TranslationContext& raw_context,
                     const xls::SourceInfo& loc)
        : translator(translator), loc(loc) {
      translator.PushContext();
      translator.context() = raw_context;
    }
    ~PushContextGuard() {
      absl::Status status = translator.PopContext(loc);
      if (!status.ok()) {
        LOG(ERROR) << status.message();
      }
    }
    Translator& translator;
    xls::SourceInfo loc;
  };

  // This guard makes pointers translate, instead of generating errors, for a
  //  period determined by RAII.
  struct IgnorePointersGuard {
    explicit IgnorePointersGuard(Translator& translator)
        : translator(translator), enabled(false) {}
    ~IgnorePointersGuard() {
      if (enabled) {
        translator.context().ignore_pointers = prev_val;
      }
    }

    void enable() {
      enabled = true;
      prev_val = translator.context().ignore_pointers;
      translator.context().ignore_pointers = true;
    }

    Translator& translator;
    bool enabled;
    bool prev_val;
  };

  // This guard makes assignments no-ops, for a period determined by RAII.
  struct MaskAssignmentsGuard {
    explicit MaskAssignmentsGuard(Translator& translator, bool engage = true)
        : translator(translator),
          prev_val(translator.context().mask_assignments) {
      if (engage) {
        translator.context().mask_assignments = true;
      }
    }
    ~MaskAssignmentsGuard() {
      translator.context().mask_assignments = prev_val;
    }

    Translator& translator;
    bool prev_val;
  };

  // This guard makes all side effects, including assignments, no-ops, for a
  // period determined by RAII.
  struct MaskSideEffectsGuard {
    explicit MaskSideEffectsGuard(Translator& translator, bool engage = true)
        : translator(translator),
          prev_val(translator.context().mask_side_effects) {
      if (engage) {
        translator.context().mask_side_effects = true;
      }
    }
    ~MaskSideEffectsGuard() {
      translator.context().mask_side_effects = prev_val;
    }

    Translator& translator;
    bool prev_val;
  };

  struct UnmaskAndIgnoreSideEffectsGuard {
    explicit UnmaskAndIgnoreSideEffectsGuard(Translator& translator)
        : translator(translator),
          prev_val(translator.context().mask_side_effects),
          prev_requested(translator.context().any_side_effects_requested) {
      translator.context().mask_side_effects = false;
    }
    ~UnmaskAndIgnoreSideEffectsGuard() {
      translator.context().mask_side_effects = prev_val;
      translator.context().any_side_effects_requested = prev_requested;
    }

    Translator& translator;
    bool prev_val;
    bool prev_requested;
  };

  struct MaskIOOtherThanMemoryWritesGuard {
    explicit MaskIOOtherThanMemoryWritesGuard(Translator& translator,
                                              bool engage = true)
        : translator(translator),
          prev_val(translator.context().mask_io_other_than_memory_writes) {
      if (engage) {
        translator.context().mask_io_other_than_memory_writes = true;
      }
    }
    ~MaskIOOtherThanMemoryWritesGuard() {
      translator.context().mask_io_other_than_memory_writes = prev_val;
    }

    Translator& translator;
    bool prev_val;
  };

  struct MaskMemoryWritesGuard {
    explicit MaskMemoryWritesGuard(Translator& translator, bool engage = true)
        : translator(translator),
          prev_val(translator.context().mask_memory_writes) {
      if (engage) {
        translator.context().mask_memory_writes = true;
      }
    }
    ~MaskMemoryWritesGuard() {
      translator.context().mask_memory_writes = prev_val;
    }

    Translator& translator;
    bool prev_val;
  };

  // This guard evaluates pointer expressions as lvalues, for a period
  // determined by RAII.
  struct LValueModeGuard {
    explicit LValueModeGuard(Translator& translator)
        : translator(translator), prev_val(translator.context().lvalue_mode) {
      translator.context().lvalue_mode = true;
    }
    ~LValueModeGuard() { translator.context().lvalue_mode = prev_val; }

    Translator& translator;
    bool prev_val;
  };

  struct OverrideThisDeclGuard {
    explicit OverrideThisDeclGuard(Translator& translator,
                                   const clang::NamedDecl* this_decl,
                                   bool activate_now = true)
        : translator_(translator), this_decl_(this_decl) {
      if (activate_now) {
        activate();
      }
    }
    ~OverrideThisDeclGuard() {
      if (prev_this_ != nullptr) {
        translator_.context().override_this_decl_ = prev_this_;
      }
    }
    void activate() {
      prev_this_ = translator_.context().override_this_decl_;
      translator_.context().override_this_decl_ = this_decl_;
    }

    Translator& translator_;
    const clang::NamedDecl* this_decl_;
    const clang::NamedDecl* prev_this_ = nullptr;
  };

  // The maximum number of iterations before loop unrolling fails.
  const int64_t max_unroll_iters_;
  // The maximum number of iterations before loop unrolling prints a warning.
  const int64_t warn_unroll_iters_;
  // The rlimit to set for z3 when unrolling loops
  const int64_t z3_rlimit_;

  // Generate an error when an init interval > supported is requested?
  const bool error_on_init_interval_;

  // Generate an error when a variable is uninitialized, or has the wrong number
  // of initializers.
  const bool error_on_uninitialized_;

  // Generates an FSM to implement pipelined loops.
  const bool generate_new_fsm_;

  // Merge states in FSM for pipelined loops.
  const bool merge_states_;

  // Split states so that IO ops on the same channel are never in the same state
  const bool split_states_on_channel_ops_;

  // Bitfield indicating which debug traces to insert into the IR.
  const DebugIrTraceFlags debug_ir_trace_flags_;

  // How to generate the token dependencies for IO Ops
  const IOOpOrdering op_ordering_;

  // Makes translation of external channel parameters optional,
  // so that IO operations can be generated without calling GenerateIR_Block()
  bool io_test_mode_ = false;

  const int64_t kNumSubBlockModeBits = 8;

  struct InstTypeHash {
    size_t operator()(
        const std::shared_ptr<CInstantiableTypeAlias>& value) const {
      const std::size_t hash =
          std::hash<const clang::NamedDecl*>()(value->base());
      return size_t(hash);
    }
  };

  struct InstTypeEq {
    bool operator()(const std::shared_ptr<CInstantiableTypeAlias>& x,
                    const std::shared_ptr<CInstantiableTypeAlias>& y) const {
      return *x == *y;
    }
  };

  absl::flat_hash_map<std::shared_ptr<CInstantiableTypeAlias>,
                      std::shared_ptr<CType>, InstTypeHash, InstTypeEq>
      inst_types_;
  absl::flat_hash_map<const clang::NamedDecl*,
                      std::unique_ptr<GeneratedFunction>>
      inst_functions_;
  // Functions are put into this map between GenerateIR_Function_Header
  //  and GenerateIR_Function_Body
  absl::flat_hash_map<const clang::NamedDecl*,
                      std::unique_ptr<FunctionInProgress>>
      functions_in_progress_;
  absl::flat_hash_set<const clang::NamedDecl*> functions_in_call_stack_;

  void print_types() {
    std::cerr << "Types {" << std::endl;
    for (const auto& var : inst_types_) {
      std::cerr << "  -- " << std::string(*var.first) << ": "
                << std::string(*var.second) << std::endl;
    }
    std::cerr << "}" << std::endl;
  }

  std::shared_ptr<CType> GetCTypeForAlias(
      const std::shared_ptr<CInstantiableTypeAlias>& alias) override;

  // The translator assumes NamedDecls are unique. This set is used to
  //  generate an error if that assumption is violated.
  absl::flat_hash_set<const clang::NamedDecl*> unique_decl_ids_;

  // Scans for top-level function candidates
  absl::Status VisitFunction(const clang::FunctionDecl* funcdecl);

  int next_asm_number_ = 1;
  int next_for_number_ = 1;
  int64_t next_channel_number_ = 1;
  int64_t next_masked_op_param_number_ = 1;

  absl::flat_hash_set<std::string> used_channel_names_;
  std::string GetUniqueChannelName(const std::string& name);

  mutable std::unique_ptr<clang::MangleContext> mangler_;

  TranslationContext& PushContext();
  absl::Status PopContext(const xls::SourceInfo& loc);
  absl::Status PropagateVariables(TranslationContext& from,
                                  TranslationContext& to,
                                  const xls::SourceInfo& loc);

  xls::Package* package_ = nullptr;
  int default_init_interval_ = 0;

  absl::flat_hash_map<const clang::FunctionDecl*, std::string>
      xls_names_for_functions_generated_;

  const clang::FunctionDecl* currently_generating_top_function_ = nullptr;

  // Initially contains keys for the channels of the top function,
  // then subroutine parameters are added as their headers are translated.
  // TODO(seanhaskell): Remove with old FSM
  absl::btree_multimap<const IOChannel*, ChannelBundle>
      external_channels_by_internal_channel_;

  // Stub functions generated by the block currently being generated.
  std::list<GeneratedFunction*> sub_block_stub_functions_;

  static bool ContainsKeyValuePair(
      const absl::btree_multimap<const IOChannel*, ChannelBundle>& map,
      const std::pair<const IOChannel*, ChannelBundle>& pair);

  // Kept ordered for determinism
  std::list<std::tuple<xls::Channel*, bool>> unused_xls_channel_ops_;

  // Used as a stack, but need to peek 2nd to top
  std::list<TranslationContext> context_stack_;

  TranslationContext& context();

  absl::Status and_condition(TrackedBValue and_condition,
                             const xls::SourceInfo& loc);

  absl::StatusOr<CValue> Generate_UnaryOp(const clang::UnaryOperator* uop,
                                          const xls::SourceInfo& loc);
  absl::StatusOr<CValue> Generate_Synthetic_ByOne(
      xls::Op xls_op, bool is_pre, CValue sub_value,
      const clang::Expr* sub_expr,  // For assignment
      const xls::SourceInfo& loc);
  absl::StatusOr<CValue> Generate_BinaryOp(
      clang::BinaryOperator::Opcode clang_op, bool is_assignment,
      std::shared_ptr<CType> result_type, const clang::Expr* lhs,
      const clang::Expr* rhs, const xls::SourceInfo& loc);
  absl::Status MinSizeArraySlices(CValue& true_cv, CValue& false_cv,
                                  std::shared_ptr<CType>& result_type,
                                  const xls::SourceInfo& loc);
  absl::StatusOr<CValue> Generate_TernaryOp(TrackedBValue cond, CValue true_cv,
                                            CValue false_cv,
                                            std::shared_ptr<CType> result_type,
                                            const xls::SourceInfo& loc);
  absl::StatusOr<CValue> Generate_TernaryOp(std::shared_ptr<CType> result_type,
                                            const clang::Expr* cond_expr,
                                            const clang::Expr* true_expr,
                                            const clang::Expr* false_expr,
                                            const xls::SourceInfo& loc);
  absl::StatusOr<CValue> GenerateIR_Expr(const clang::Expr* expr,
                                         const xls::SourceInfo& loc);
  absl::StatusOr<CValue> GenerateIR_Expr(std::shared_ptr<LValue> expr,
                                         const xls::SourceInfo& loc);
  absl::StatusOr<std::optional<CValue>> EvaluateNumericConstExpr(
      const clang::Expr* expr, const xls::SourceInfo& loc);
  absl::StatusOr<CValue> GenerateIR_MemberExpr(const clang::MemberExpr* expr,
                                               const xls::SourceInfo& loc);
  absl::StatusOr<std::string> GetStringLiteral(const clang::Expr* expr,
                                               const xls::SourceInfo& loc);
  // Returns true if built-in call generated
  absl::StatusOr<std::pair<bool, CValue>> GenerateIR_BuiltInCall(
      const clang::CallExpr* call, const xls::SourceInfo& loc);
  absl::StatusOr<CValue> GenerateIR_Call(const clang::CallExpr* call,
                                         const xls::SourceInfo& loc);

  absl::StatusOr<CValue> GenerateIR_Call(
      const clang::FunctionDecl* funcdecl,
      std::vector<const clang::Expr*> expr_args, TrackedBValue* this_inout,
      std::shared_ptr<LValue>* this_lval, const xls::SourceInfo& loc);

  absl::Status FailIfTypeHasDtors(const clang::CXXRecordDecl* cxx_record);
  bool LValueContainsOnlyChannels(const std::shared_ptr<LValue>& lvalue);

  absl::Status PushLValueSelectConditions(
      std::shared_ptr<LValue> lvalue, std::vector<TrackedBValue>& return_bvals,
      const xls::SourceInfo& loc);
  absl::StatusOr<std::shared_ptr<LValue>> PopLValueSelectConditions(
      std::list<TrackedBValue>& unpacked_returns,
      std::shared_ptr<LValue> lvalue_translated, const xls::SourceInfo& loc);
  absl::StatusOr<std::list<TrackedBValue>> UnpackTuple(
      TrackedBValue tuple_val, const xls::SourceInfo& loc);
  absl::StatusOr<TrackedBValue> Generate_LValue_Return(
      std::shared_ptr<LValue> lvalue, const xls::SourceInfo& loc);
  absl::StatusOr<CValue> Generate_LValue_Return_Call(
      std::shared_ptr<LValue> lval_untranslated, TrackedBValue unpacked_return,
      std::shared_ptr<CType> return_type, TrackedBValue* this_inout,
      std::shared_ptr<LValue>* this_lval,
      const absl::flat_hash_map<IOChannel*, IOChannel*>&
          caller_channels_by_callee_channel,
      const xls::SourceInfo& loc);
  absl::Status TranslateAddCallerChannelsByCalleeChannel(
      std::shared_ptr<LValue> caller_lval, std::shared_ptr<LValue> callee_lval,
      absl::flat_hash_map<IOChannel*, IOChannel*>*
          caller_channels_by_callee_channel,
      const xls::SourceInfo& loc);

  bool IOChannelInCurrentFunction(IOChannel* to_find,
                                  const xls::SourceInfo& loc);

  absl::Status ValidateLValue(std::shared_ptr<LValue> lval,
                              const xls::SourceInfo& loc);

  absl::StatusOr<std::shared_ptr<LValue>> TranslateLValueChannels(
      std::shared_ptr<LValue> outer_lval,
      const absl::flat_hash_map<IOChannel*, IOChannel*>&
          inner_channels_by_outer_channel,
      const xls::SourceInfo& loc);

  static std::shared_ptr<LValue> RemoveBValuesFromLValue(
      std::shared_ptr<LValue> outer_lval, const xls::SourceInfo& body_loc);

  static void CleanUpBValuesInTopFunction(GeneratedFunction& func);

  // This is a work-around for non-const operator [] needing to return
  //  a reference to the object being modified.
  absl::StatusOr<bool> ApplyArrayAssignHack(
      const clang::CXXOperatorCallExpr* op_call, const xls::SourceInfo& loc,
      CValue* output);

  struct PreparedBlock {
    const GeneratedFunction* xls_func;
    std::vector<TrackedBValue> args;
    // Not used for direct-ins
    absl::flat_hash_map<IOChannel*, ChannelBundle>
        xls_channel_by_function_channel;
    absl::flat_hash_map<const IOOp*, int64_t> arg_index_for_op;
    absl::flat_hash_map<const IOOp*, int64_t> return_index_for_op;
    absl::flat_hash_map<const clang::NamedDecl*, int64_t>
        return_index_for_static;
    absl::flat_hash_map<const clang::NamedDecl*, xls::StateElement*>
        state_element_for_variable;
    TrackedBValue orig_token;
    TrackedBValue token;
    bool contains_fsm = false;
  };

  struct ExternalChannelInfo {
    const clang::NamedDecl* decl;
    std::shared_ptr<CChannelType> channel_type;
    InterfaceType interface_type;
    bool extra_return = false;
    bool is_input = false;
    std::optional<xls::FlopKindProto> flop_kind = std::nullopt;
    ChannelBundle external_channels;
    xls::ChannelStrictness strictness =
        xls::ChannelStrictness::kProvenMutuallyExclusive;
  };

  absl::StatusOr<xls::Proc*> GenerateIR_Block(
      xls::Package* package, const HLSBlock& block,
      const std::shared_ptr<CType>& this_type,
      const clang::CXXRecordDecl* this_decl,
      std::list<ExternalChannelInfo>& top_decls,
      const xls::SourceInfo& body_loc, int top_level_init_interval,
      bool force_static, bool member_references_become_channels);

  absl::StatusOr<xls::Proc*> GenerateIR_Block(
      xls::Package* package, std::string_view block_name,
      const absl::flat_hash_map<const clang::NamedDecl*, ChannelBundle>&
          top_channel_injections,
      const std::shared_ptr<CType>& this_type,
      const clang::CXXRecordDecl* this_decl,
      const std::list<ExternalChannelInfo>& top_decls,
      const xls::SourceInfo& body_loc, int top_level_init_interval,
      bool force_static, bool member_references_become_channels,
      const GeneratedFunction* caller_sub_function = nullptr);

  // Verifies the function prototype in the Clang AST and HLSBlock are sound.
  absl::Status GenerateIRBlockCheck(
      const HLSBlock& block, const std::list<ExternalChannelInfo>& top_decls,
      const xls::SourceInfo& body_loc);

  // Creates xls::Channels in the package
  absl::Status GenerateExternalChannels(
      std::list<ExternalChannelInfo>& top_decls,
      absl::flat_hash_map<const clang::NamedDecl*, ChannelBundle>*
          top_channel_injections,
      const xls::SourceInfo& loc);

  absl::StatusOr<CValue> GenerateTopClassInitValue(
      const std::shared_ptr<CType>& this_type,
      // Can be nullptr
      const clang::CXXRecordDecl* this_decl, const xls::SourceInfo& body_loc);

  // Prepares IO channels for generating XLS Proc
  // definition can be null, and then channels_by_name can also be null. They
  // are only used for direct-ins
  // Returns ownership of dummy function for top proc
  absl::StatusOr<std::unique_ptr<GeneratedFunction>> GenerateIRBlockPrepare(
      PreparedBlock& prepared, xls::ProcBuilder& pb, int64_t next_return_index,
      const std::shared_ptr<CType>& this_type,
      // Can be nullptr
      const clang::CXXRecordDecl* this_decl,
      const std::list<ExternalChannelInfo>& top_decls,
      // Can be nullptr
      const GeneratedFunction* caller_sub_function,
      const xls::SourceInfo& body_loc);

  // Generates a dummy no-op with condition 0 for channels in
  // unused_external_channels_
  absl::Status GenerateDefaultIOOps(PreparedBlock& prepared,
                                    xls::ProcBuilder& pb,
                                    const xls::SourceInfo& body_loc);
  absl::Status GenerateDefaultIOOp(xls::Channel* channel, bool is_send,
                                   TrackedBValue token,
                                   std::vector<TrackedBValue>& final_tokens,
                                   xls::ProcBuilder& pb,
                                   const xls::SourceInfo& loc);

  std::optional<ChannelBundle> GetChannelBundleForOp(
      const IOOp& op, const xls::SourceInfo& loc) override;

  // ---- Old FSM ---
  struct InvokeToGenerate {
    const IOOp& op;
    TrackedBValue extra_condition;
  };

  TrackedBValue ConditionWithExtra(xls::BuilderBase& builder,
                                   TrackedBValue condition,
                                   std::optional<TrackedBValue> extra_condition,
                                   const xls::SourceInfo& op_loc);

  struct State {
    int64_t index = -1;
    std::list<InvokeToGenerate> invokes_to_generate;
    const PipelinedLoopSubProc* sub_proc = nullptr;
    TrackedBValue in_this_state;
    std::set<ChannelBundle> channels_used;
  };

  absl::StatusOr<GenerateFSMInvocationReturn> GenerateOldFSMInvocation(
      PreparedBlock& prepared, xls::ProcBuilder& pb, int nesting_level,
      const xls::SourceInfo& body_loc);

  struct LayoutFSMStatesReturn {
    absl::flat_hash_map<const IOOp*, const State*> state_by_io_op;
    std::vector<std::unique_ptr<State>> states;
    bool has_pipelined_loop = false;
  };

  absl::StatusOr<LayoutFSMStatesReturn> LayoutFSMStates(
      PreparedBlock& prepared, xls::ProcBuilder& pb,
      const xls::SourceInfo& body_loc);
  std::set<ChannelBundle> GetChannelsUsedByOp(
      const IOOp& op, const PipelinedLoopSubProc* sub_procp,
      const xls::SourceInfo& loc);

  struct SubFSMReturn {
    TrackedBValue first_iter;
    TrackedBValue exit_state_condition;
    TrackedBValue return_value;
    absl::btree_multimap<const xls::StateElement*, NextStateValue>
        extra_next_state_values;
    TrackedBValue token_out;
  };
  // Generates a sub-FSM for a state containing a sub-proc
  // Ignores the associated IO ops (context send/receive)
  // (Currently used for pipelined loops)
  absl::StatusOr<SubFSMReturn> GenerateSubFSM(
      PreparedBlock& outer_prepared,
      const std::list<Translator::InvokeToGenerate>& invokes_to_generate,
      TrackedBValue origin_token, xls::ProcBuilder& pb,
      const State& outer_state, const std::string& fsm_prefix,
      absl::flat_hash_map<const IOOp*, TrackedBValue>& op_tokens,
      TrackedBValue first_ret_val, int outer_nesting_level,
      const xls::SourceInfo& body_loc);

  // Generates only the listed ops. A token network will be created between
  // only these ops.
  absl::StatusOr<TrackedBValue> GenerateIOInvokesWithAfterOps(
      IOSchedulingOption option, TrackedBValue origin_token,
      const std::list<InvokeToGenerate>& invokes_to_generate,
      TrackedBValueMap<const IOOp*>& op_tokens, TrackedBValue& last_ret_val,
      PreparedBlock& prepared, xls::ProcBuilder& pb,
      const xls::SourceInfo& body_loc);

  absl::StatusOr<TrackedBValue> GenerateIOInvoke(const InvokeToGenerate& invoke,
                                                 TrackedBValue before_token,
                                                 PreparedBlock& prepared,
                                                 TrackedBValue& last_ret_val,
                                                 xls::ProcBuilder& pb);
  // ---- / Old FSM ---

  absl::StatusOr<TrackedBValue> GetOpCondition(const IOOp& op,
                                               TrackedBValue op_out_value,
                                               xls::ProcBuilder& pb) final;

  // Returns new token
  absl::StatusOr<TrackedBValue> GenerateTrace(TrackedBValue trace_out_value,
                                              TrackedBValue before_token,
                                              TrackedBValue condition,
                                              const IOOp& op,
                                              xls::ProcBuilder& pb) final;

  absl::StatusOr<GenerateIOReturn> GenerateIO(
      const IOOp& op, TrackedBValue before_token, TrackedBValue op_out_value,
      xls::ProcBuilder& pb,
      const std::optional<ChannelBundle>& optional_channel_bundle,
      std::optional<TrackedBValue> extra_condition = std::nullopt) final;

  struct IOOpReturn {
    bool generate_expr;
    CValue value;
  };
  // Checks if an expression is an IO op, and if so, generates the value
  //  to replace it in IR generation.
  absl::StatusOr<IOOpReturn> InterceptIOOp(const clang::Expr* expr,
                                           const xls::SourceInfo& loc,
                                           CValue assignment_value = CValue());

  // IOOp must have io_call, and op members filled in
  // This will add a parameter for IO input if needed,
  // Returns permanent IOOp pointer
  absl::StatusOr<IOOp*> AddOpToChannel(IOOp& op, IOChannel* channel_param,
                                       const xls::SourceInfo& loc,
                                       bool mask = false);

  absl::StatusOr<std::optional<const IOOp*>> GetPreviousOp(
      const IOOp& op, const xls::SourceInfo& loc);

  absl::StatusOr<TrackedBValue> AddConditionToIOReturn(
      const IOOp& op, TrackedBValue retval, const xls::SourceInfo& loc);

  absl::Status NewContinuation(const IOOp& op, bool slice_before);
  absl::Status AddFeedbacksForSlice(GeneratedFunctionSlice& slice,
                                    const xls::SourceInfo& loc);
  absl::StatusOr<std::vector<NATIVE_BVAL>>
  ConvertBValuesToContinuationOutputsForCurrentSlice(
      absl::flat_hash_map<const ContinuationValue*,
                          std::vector<TrackedBValue*>>&
          bvalues_by_continuation_output,
      absl::flat_hash_map<const TrackedBValue*, ContinuationValue*>&
          continuation_outputs_by_bval,
      absl::flat_hash_map<const TrackedBValue*, std::string>&
          name_found_for_bval,
      absl::flat_hash_map<const TrackedBValue*, const clang::NamedDecl*>&
          decls_by_bval_top_context,
      int64_t* total_bvals_out, const xls::SourceInfo& loc);
  absl::Status AddContinuationsToNewSlice(
      const IOOp& after_op, GeneratedFunctionSlice& last_slice,
      GeneratedFunctionSlice& new_slice,
      const absl::flat_hash_map<const ContinuationValue*,
                                std::vector<TrackedBValue*>>&
          bvalues_by_continuation_output,
      const absl::flat_hash_map<const TrackedBValue*, ContinuationValue*>&
          continuation_outputs_by_bval,
      const absl::flat_hash_map<const TrackedBValue*, std::string>&
          name_found_for_bval,
      const absl::flat_hash_map<const TrackedBValue*, const clang::NamedDecl*>&
          decls_by_bval_top_context,
      int64_t total_bvals, const xls::SourceInfo& loc);
  absl::Status FinishSlice(NATIVE_BVAL return_bval, const xls::SourceInfo& loc);
  // TODO(seanhaskell): Move into FinishSlice() once old FSM is removed.
  absl::Status RemoveMaskedOpParams(GeneratedFunction& func,
                                    const xls::SourceInfo& loc);
  absl::Status FinishLastSlice(TrackedBValue return_bval,
                               const xls::SourceInfo& loc);
  absl::Status OptimizeContinuations(GeneratedFunction& func,
                                     const xls::SourceInfo& loc);

  // This function is a temporary adapter for the old FSM generation style.
  // It creates a single function containing all slices and fills it into the
  // GeneratedFunction::function field.
  absl::Status GenerateFunctionSliceWrapper(GeneratedFunction& func,
                                            const xls::SourceInfo& loc);

  absl::StatusOr<std::shared_ptr<LValue>> CreateChannelParam(
      const clang::NamedDecl* channel_name,
      const std::shared_ptr<CChannelType>& channel_type, bool declare_variable,
      const xls::SourceInfo& loc);
  IOChannel* AddChannel(const IOChannel& new_channel,
                        const xls::SourceInfo& loc);
  absl::StatusOr<std::shared_ptr<CChannelType>> GetChannelType(
      const clang::QualType& channel_type, clang::ASTContext& ctx,
      const xls::SourceInfo& loc);
  absl::StatusOr<int64_t> GetIntegerTemplateArgument(
      const clang::TemplateArgument& arg, clang::ASTContext& ctx,
      const xls::SourceInfo& loc);

  absl::StatusOr<bool> ExprIsChannel(const clang::Expr* object,
                                     const xls::SourceInfo& loc);
  absl::StatusOr<bool> TypeIsChannel(clang::QualType param,
                                     const xls::SourceInfo& loc);
  // Returns nullptr if the parameter isn't a channel
  struct ConditionedIOChannel {
    IOChannel* channel;
    TrackedBValue condition;
  };
  absl::Status GetChannelsForExprOrNull(
      const clang::Expr* object, std::vector<ConditionedIOChannel>* output,
      const xls::SourceInfo& loc, TrackedBValue condition = TrackedBValue());
  absl::Status GetChannelsForLValue(const std::shared_ptr<LValue>& lvalue,
                                    std::vector<ConditionedIOChannel>* output,
                                    const xls::SourceInfo& loc,
                                    TrackedBValue condition = TrackedBValue());
  absl::Status GenerateIR_Compound(const clang::Stmt* body,
                                   clang::ASTContext& ctx);
  absl::Status GenerateIR_Stmt(const clang::Stmt* stmt, clang::ASTContext& ctx);
  absl::Status GenerateIR_ReturnStmt(const clang::ReturnStmt* rts,
                                     clang::ASTContext& ctx,
                                     const xls::SourceInfo& loc);
  absl::Status GenerateIR_StaticDecl(const clang::VarDecl* vard,
                                     const clang::NamedDecl* namedecl,
                                     const xls::SourceInfo& loc);
  absl::StatusOr<CValue> GenerateIR_LocalChannel(
      const clang::NamedDecl* namedecl,
      const std::shared_ptr<CChannelType>& channel_type,
      const xls::SourceInfo& loc);

  absl::Status CheckInitIntervalValidity(int initiation_interval_arg,
                                         const xls::SourceInfo& loc);

  // Determines whether to unroll or pipeline a loop, and calls the appropriate
  // subroutine.
  //
  // init, cond, and inc can be nullptr
  absl::Status GenerateIR_Loop(
      bool always_first_iter, const clang::Stmt* loop_stmt,
      clang::ArrayRef<const clang::AnnotateAttr*> attrs,
      const clang::Stmt* init, const clang::Expr* cond_expr,
      const clang::Stmt* inc, const clang::Stmt* body,
      const clang::PresumedLoc& presumed_loc, const xls::SourceInfo& loc,
      clang::ASTContext& ctx);

  struct GenerateIR_LoopResult {
    std::optional<int64_t> proven_iteration_count = std::nullopt;
    std::optional<int64_t> proven_max_iteration_count = std::nullopt;
  };

  // Does a trial unroll to check if a loop has a fixed number of iterations
  // for optimization purposes, calling GenerateIR_LoopImplImpl as necessary.
  //
  // if max_iters is specified, it must be > 0, and in new FSM mode,
  // the loop will be pipelined, that is, begin and end IO ops will be added.
  //
  // init, cond, and inc can be nullptr
  // TODO(seanhaskell): Remove trial_unroll_init with old FSM
  absl::Status GenerateIR_LoopImpl(
      bool always_first_iter, bool warn_inferred_loop_type,
      const clang::Stmt* init, const clang::Stmt* trial_unroll_init,
      const clang::Expr* cond_expr, const clang::Stmt* inc,
      const clang::Stmt* body, std::optional<int64_t> max_iters,
      bool propagate_break_up, clang::ASTContext& ctx,
      const xls::SourceInfo& loc);

  // init, cond, and inc can be nullptr
  // if max_iters is specified, it must be > 0, and in new FSM mode,
  // the loop will be pipelined, that is, begin and end IO ops will be added.
  absl::StatusOr<GenerateIR_LoopResult> GenerateIR_LoopImplImpl(
      bool always_first_iter, bool warn_inferred_loop_type,
      const clang::Stmt* init, const clang::Expr* cond_expr,
      const clang::Stmt* inc, const clang::Stmt* body,
      std::optional<int64_t> max_iters, bool propagate_break_up,
      bool omit_conditions_in_unrolling, clang::ASTContext& ctx,
      const xls::SourceInfo& loc);

  // init, cond, and inc can be nullptr
  absl::Status GenerateIR_PipelinedLoopOldFSM(
      bool always_first_iter, bool warn_inferred_loop_type,
      const clang::Stmt* init, const clang::Expr* cond_expr,
      const clang::Stmt* inc, const clang::Stmt* body,
      int64_t initiation_interval_arg, int64_t unroll_factor,
      bool schedule_asap, clang::ASTContext& ctx, const xls::SourceInfo& loc);

  // init, cond, and inc can be nullptr
  absl::Status GenerateIR_PipelinedLoopNewFSM(
      bool always_first_iter, bool warn_inferred_loop_type,
      const clang::Stmt* init, const clang::Expr* cond_expr,
      const clang::Stmt* inc, const clang::Stmt* body,
      int64_t initiation_interval_arg, int64_t unroll_factor,
      bool schedule_asap, clang::ASTContext& ctx, const xls::SourceInfo& loc);

  absl::StatusOr<IOOp*> GenerateIR_AddLoopBegin(const xls::SourceInfo& loc);
  absl::Status GenerateIR_AddLoopEndJump(const clang::Expr* cond_expr,
                                         IOOp* begin_op,
                                         const xls::SourceInfo& loc);

  // init is only used for trial unrolling, should already have been generated
  // by the caller.
  absl::StatusOr<PipelinedLoopSubProc> GenerateIR_PipelinedLoopBody(
      const clang::Expr* cond_expr, const clang::Stmt* init,
      const clang::Stmt* inc, const clang::Stmt* body, int64_t init_interval,
      int64_t unroll_factor, bool always_first_iter, clang::ASTContext& ctx,
      std::string_view name_prefix, xls::Type* context_struct_xls_type,
      xls::Type* context_lvals_xls_type,
      const std::shared_ptr<CStructType>& context_cvars_struct_ctype,
      LValueMap<const clang::NamedDecl*>* lvalues_out,
      const absl::flat_hash_map<const clang::NamedDecl*, uint64_t>&
          context_field_indices,
      const std::vector<const clang::NamedDecl*>& variable_fields_order,
      bool* uses_on_reset, const xls::SourceInfo& loc);

  struct PipelinedLoopContentsReturn {
    TrackedBValue token_out;
    TrackedBValue do_break;
    TrackedBValue first_iter;
    TrackedBValue out_tuple;
    absl::btree_multimap<const xls::StateElement*, NextStateValue>
        extra_next_state_values;
  };

  // If not nullptr, state_element_for_variable is used in generating the loop
  // body, and updated for any new state elements created inside.
  absl::StatusOr<PipelinedLoopContentsReturn> GenerateIR_PipelinedLoopContents(
      const PipelinedLoopSubProc& pipelined_loop_proc, xls::ProcBuilder& pb,
      TrackedBValue token_in, TrackedBValue received_context_tuple,
      TrackedBValue in_state_condition, bool in_fsm,
      absl::flat_hash_map<const clang::NamedDecl*, xls::StateElement*>*
          state_element_for_variable = nullptr,
      int nesting_level = -1);

  absl::Status SendLValueConditions(
      const std::shared_ptr<LValue>& lvalue,
      std::vector<TrackedBValue>* lvalue_conditions,
      const xls::SourceInfo& loc);
  absl::StatusOr<std::shared_ptr<LValue>> TranslateLValueConditions(
      const std::shared_ptr<LValue>& outer_lvalue,
      TrackedBValue lvalue_conditions_tuple, const xls::SourceInfo& loc,
      int64_t* at_index = nullptr);
  absl::Status GenerateIR_Switch(const clang::SwitchStmt* switchst,
                                 clang::ASTContext& ctx,
                                 const xls::SourceInfo& loc);

  struct ResolvedInheritance {
    std::shared_ptr<CField> base_field;
    std::shared_ptr<const CStructType> resolved_struct;
    const clang::NamedDecl* base_field_name;
  };

  absl::StatusOr<ResolvedInheritance> ResolveInheritance(
      std::shared_ptr<CType> sub_type, std::shared_ptr<CType> to_type);
  absl::StatusOr<CValue> ResolveCast(const CValue& sub,
                                     const std::shared_ptr<CType>& to_type,
                                     const xls::SourceInfo& loc);

  absl::StatusOr<TrackedBValue> GenTypeConvert(CValue const& in,
                                               std::shared_ptr<CType> out_type,
                                               const xls::SourceInfo& loc);
  absl::StatusOr<TrackedBValue> GenBoolConvert(CValue const& in,
                                               const xls::SourceInfo& loc);
  absl::StatusOr<CValue> HandleConstructors(const clang::CXXConstructExpr* ctor,
                                            const xls::SourceInfo& loc);
  absl::StatusOr<TrackedBValue> BuildCArrayTypeValue(
      std::shared_ptr<CType> t, TrackedBValue elem_val,
      const xls::SourceInfo& loc);
  absl::StatusOr<CValue> CreateDefaultCValue(const std::shared_ptr<CType>& t,
                                             const xls::SourceInfo& loc);
  absl::StatusOr<xls::Value> CreateDefaultRawValue(std::shared_ptr<CType> t,
                                                   const xls::SourceInfo& loc);
  absl::StatusOr<TrackedBValue> CreateDefaultValue(std::shared_ptr<CType> t,
                                                   const xls::SourceInfo& loc);
  absl::StatusOr<CValue> CreateInitListValue(
      const std::shared_ptr<CType>& t, const clang::InitListExpr* init_list,
      const xls::SourceInfo& loc);
  absl::StatusOr<CValue> CreateInitValue(const std::shared_ptr<CType>& ctype,
                                         const clang::Expr* initializer,
                                         const xls::SourceInfo& loc);
  absl::StatusOr<std::shared_ptr<LValue>> CreateReferenceValue(
      const clang::Expr* initializer, const xls::SourceInfo& loc);
  absl::StatusOr<CValue> GetOnReset(const xls::SourceInfo& loc);
  absl::StatusOr<bool> DeclIsOnReset(const clang::NamedDecl* decl);
  absl::StatusOr<CValue> GetIdentifier(const clang::NamedDecl* decl,
                                       const xls::SourceInfo& loc,
                                       bool record_access = true);

  absl::StatusOr<CValue> TranslateVarDecl(const clang::VarDecl* decl,
                                          const xls::SourceInfo& loc);
  absl::StatusOr<CValue> TranslateEnumConstantDecl(
      const clang::EnumConstantDecl* decl, const xls::SourceInfo& loc);
  absl::Status Assign(const clang::NamedDecl* lvalue, const CValue& rvalue,
                      const xls::SourceInfo& loc,
                      bool force_no_lvalue_assign = false);
  absl::Status Assign(const clang::Expr* lvalue, const CValue& rvalue,
                      const xls::SourceInfo& loc);
  absl::Status Assign(std::shared_ptr<LValue> lvalue, const CValue& rvalue,
                      const xls::SourceInfo& loc);

  absl::Status AssignMember(const clang::Expr* lvalue,
                            const clang::NamedDecl* member,
                            const CValue& rvalue, const xls::SourceInfo& loc);
  absl::Status AssignMember(const clang::NamedDecl* lvalue,
                            const clang::NamedDecl* member,
                            const CValue& rvalue, const xls::SourceInfo& loc);

  absl::StatusOr<const clang::NamedDecl*> GetThisDecl(
      const xls::SourceInfo& loc, bool for_declaration = false);
  absl::StatusOr<CValue> PrepareRValueWithSelect(
      const CValue& lvalue, const CValue& rvalue,
      const TrackedBValue& relative_condition, const xls::SourceInfo& loc);

  absl::Status DeclareVariable(const clang::NamedDecl* lvalue,
                               const CValue& rvalue, const xls::SourceInfo& loc,
                               bool check_unique_ids = true);

  absl::Status DeclareStatic(const clang::NamedDecl* lvalue,
                             const ConstValue& init,
                             const std::shared_ptr<LValue>& init_lvalue,
                             const xls::SourceInfo& loc,
                             bool check_unique_ids = true);

  // If the decl given is a forward declaration, the definition with a body will
  // be returned. This is done in multiple places because the ParamVarDecls vary
  // in each declaration.
  absl::StatusOr<const clang::Stmt*> GetFunctionBody(
      const clang::FunctionDecl*& funcdecl);

  absl::StatusOr<FunctionInProgress> GenerateIR_Function_Header(
      GeneratedFunction& sf, const clang::FunctionDecl* funcdecl,
      std::string_view name_override = "", bool force_static = false,
      bool member_references_become_channels = false);
  absl::Status GenerateIR_Function_Body(GeneratedFunction& sf,
                                        const clang::FunctionDecl* funcdecl,
                                        const FunctionInProgress& header);

  absl::Status GenerateIR_Ctor_Initializers(
      const clang::CXXConstructorDecl* constructor);

  absl::Status GenerateThisLValues(const clang::RecordDecl* this_struct_decl,
                                   std::shared_ptr<CType> thisctype,
                                   bool member_references_become_channels,
                                   const xls::SourceInfo& loc);

  const clang::CXXThisExpr* IsThisExpr(const clang::Expr* expr);
  const clang::Expr* RemoveParensAndCasts(const clang::Expr* expr);

  struct StrippedType {
    StrippedType(clang::QualType base, bool is_ref)
        : base(base), is_ref(is_ref) {}
    clang::QualType base;
    bool is_ref;
  };

  absl::StatusOr<StrippedType> StripTypeQualifiers(clang::QualType t);
  absl::Status ScanStruct(const clang::RecordDecl* sd);

  absl::StatusOr<std::shared_ptr<CType>> InterceptBuiltInStruct(
      const clang::RecordDecl* sd);

  absl::StatusOr<std::shared_ptr<CType>> TranslateTypeFromClang(
      clang::QualType t, const xls::SourceInfo& loc,
      bool array_as_tuple = false) final;
  absl::StatusOr<xls::Type*> TranslateTypeToXLS(std::shared_ptr<CType> t,
                                                const xls::SourceInfo& loc);
  absl::StatusOr<std::shared_ptr<CType>> ResolveTypeInstance(
      std::shared_ptr<CType> t) final;
  absl::StatusOr<std::shared_ptr<CType>> ResolveTypeInstanceDeeply(
      std::shared_ptr<CType> t);
  absl::StatusOr<bool> FunctionIsInSyntheticInt(
      const clang::FunctionDecl* decl);

  absl::StatusOr<int64_t> EvaluateInt64(const clang::Expr& expr,
                                        const class clang::ASTContext& ctx,
                                        const xls::SourceInfo& loc);
  absl::StatusOr<bool> EvaluateBool(const clang::Expr& expr,
                                    const class clang::ASTContext& ctx,
                                    const xls::SourceInfo& loc);
  absl::StatusOr<xls::Value> EvaluateNode(xls::Node* node,
                                          const xls::SourceInfo& loc,
                                          bool do_check = true);
  absl::StatusOr<xls::Value> EvaluateBVal(TrackedBValue bval,
                                          const xls::SourceInfo& loc,
                                          bool do_check = true);
  absl::StatusOr<int64_t> EvaluateBValInt64(TrackedBValue bval,
                                            const xls::SourceInfo& loc,
                                            bool do_check = true);
  absl::StatusOr<Z3_lbool> CheckAssumptions(
      absl::Span<xls::Node*> positive_nodes,
      absl::Span<xls::Node*> negative_nodes, Z3_solver& solver,
      xls::solvers::z3::IrTranslator& z3_translator);

  // bval can be invalid, in which case it is interpreted as 1
  // Short circuits the BValue
  absl::StatusOr<bool> BitMustBe(bool assert_value, TrackedBValue& bval,
                                 Z3_solver& solver,
                                 xls::solvers::z3::IrTranslator* z3_translator,
                                 const xls::SourceInfo& loc);

  absl::StatusOr<ConstValue> TranslateBValToConstVal(const CValue& bvalue,
                                                     const xls::SourceInfo& loc,
                                                     bool do_check = true);

  absl::StatusOr<xls::Op> XLSOpcodeFromClang(clang::BinaryOperatorKind clang_op,
                                             const CType& left_type,
                                             const CType& result_type,
                                             const xls::SourceInfo& loc);
  std::string XLSNameMangle(clang::GlobalDecl decl) const;

  TrackedBValue MakeFunctionReturn(const std::vector<TrackedBValue>& bvals,
                                   const xls::SourceInfo& loc);
  TrackedBValue GetFunctionReturn(TrackedBValue val, int index,
                                  int expected_returns,
                                  const clang::FunctionDecl* func,
                                  const xls::SourceInfo& loc);

  absl::Status GenerateMetadataCPPName(const clang::NamedDecl* decl_in,
                                       xlscc_metadata::CPPName* name_out);

  absl::Status GenerateMetadataType(
      const clang::QualType& type_in, xlscc_metadata::Type* type_out,
      absl::flat_hash_set<const clang::NamedDecl*>& aliases_used);

  absl::StatusOr<xlscc_metadata::IntType> GenerateSyntheticInt(
      std::shared_ptr<CType> ctype) final;

  // StructUpdate builds and returns a new CValue for a struct with the
  // value of one field changed. The other fields, if any, take their values
  // from struct_before, and the new value for the field named by field_name is
  // set to rvalue. The type of structure built is specified by type.
  absl::StatusOr<CValue> StructUpdate(CValue struct_before, CValue rvalue,
                                      const clang::NamedDecl* field_name,
                                      const CStructType& type,
                                      const xls::SourceInfo& loc);
  // Creates an BValue for a struct of type stype from field BValues given in
  //  order within bvals.
  TrackedBValue MakeStructXLS(const std::vector<TrackedBValue>& bvals,
                              const CStructType& stype,
                              const xls::SourceInfo& loc);

  // Creates a Value for a struct of type stype from field BValues given in
  //  order within bvals.
  xls::Value MakeStructXLS(const std::vector<xls::Value>& vals,
                           const CStructType& stype);

  // Returns the BValue for the field with index "index" from a BValue for a
  //  struct of type "type"
  // This version cannot be static because it needs access to the
  //  FunctionBuilder from the context
  TrackedBValue GetStructFieldXLS(TrackedBValue val, int64_t index,
                                  const CStructType& type,
                                  const xls::SourceInfo& loc) final;

  // Returns a BValue for a copy of array_to_update with slice_to_write replaced
  // at start_index.
  absl::StatusOr<TrackedBValue> UpdateArraySlice(TrackedBValue array_to_update,
                                                 TrackedBValue start_index,
                                                 TrackedBValue slice_to_write,
                                                 const xls::SourceInfo& loc);

  absl::StatusOr<CValue> GetArrayElement(const CValue& arr_val,
                                         TrackedBValue index_bval,
                                         const xls::SourceInfo& loc);

  absl::StatusOr<CValue> UpdateArrayElement(const CValue& arr_val,
                                            TrackedBValue index_bval,
                                            const CValue& rvalue,
                                            const xls::SourceInfo& loc);

  int64_t ArrayBValueWidth(TrackedBValue array_bval);

  // Creates a properly ordered list of next values to pass to
  // ProcBuilder::Build()
  absl::StatusOr<xls::Proc*> BuildWithNextStateValueMap(
      xls::ProcBuilder& pb, TrackedBValue token,
      const absl::btree_multimap<const xls::StateElement*, NextStateValue>&
          next_state_values,
      const xls::SourceInfo& loc);

 private:
  // Gets the appropriate XLS type for a struct. For example, it might be an
  //  xls::Tuple, or if #pragma hls_notuple was specified, it might be
  //  the single field's type
  absl::StatusOr<xls::Type*> GetStructXLSType(
      const std::vector<xls::Type*>& members, const CStructType& type,
      const xls::SourceInfo& loc);

  // "Flexible Tuple" functions
  // These functions will create a tuple if there is more than one
  //  field, or they will pass through the value if there
  //  is exactly 1 value. 0 values is unsupported.
  // This makes the generated IR less cluttered, as extra single-item tuples
  //  aren't generated.

  // Wraps bvals in a "flexible tuple"
  TrackedBValue MakeFlexTuple(const std::vector<TrackedBValue>& bvals,
                              const xls::SourceInfo& loc);
  // Gets the XLS type for a "flexible tuple" made from these elements
  xls::Type* GetFlexTupleType(const std::vector<xls::Type*>& members);
  // Gets the value of a field in a "flexible tuple"
  // val is the "flexible tuple" value
  // index is the index of the field
  // n_fields is the total number of fields
  // op_name is passed to the FunctionBuilder
  TrackedBValue GetFlexTupleField(TrackedBValue val, int64_t index,
                                  int64_t n_fields, const xls::SourceInfo& loc,
                                  std::string_view op_name = "");
  // Changes the value of a field in a "flexible tuple"
  // tuple_val is the "flexible tuple" value
  // new_val is the value to set the field to
  // index is the index of the field
  // n_fields is the total number of fields
  TrackedBValue UpdateFlexTupleField(TrackedBValue tuple_val,
                                     TrackedBValue new_val, int index,
                                     int n_fields, const xls::SourceInfo& loc);

  absl::StatusOr<bool> TypeMustHaveRValue(const CType& type);

  void FillLocationProto(const clang::SourceLocation& location,
                         xlscc_metadata::SourceLocation* location_out);
  void FillLocationRangeProto(const clang::SourceRange& range,
                              xlscc_metadata::SourceLocationRange* range_out);

  absl::StatusOr<bool> IsSubBlockDirectInParam(
      const clang::FunctionDecl* funcdecl, int64_t param_index);

  std::unique_ptr<CCParser> parser_;

  std::string LocString(const xls::SourceInfo& loc);
  xls::SourceInfo GetLoc(const clang::Stmt& stmt) final;
  xls::SourceInfo GetLoc(const clang::Decl& decl) final;
  absl::StatusOr<xls::ChannelStrictness> GetChannelStrictness(
      const clang::NamedDecl& decl, const ChannelOptions& channel_options,
      absl::flat_hash_map<std::string, xls::ChannelStrictness>&
          unused_strictness_options);
  inline std::string LocString(const clang::Decl& decl) {
    return LocString(GetLoc(decl));
  }
  clang::PresumedLoc GetPresumedLoc(const clang::Stmt& stmt);
  clang::PresumedLoc GetPresumedLoc(const clang::Decl& decl);

  std::vector<const clang::AnnotateAttr*> GetClangAnnotations(
      const clang::Decl& decl);

  // May update stmt pointer to un-annotated statement
  std::vector<const clang::AnnotateAttr*> GetClangAnnotations(
      const clang::Stmt*& stmt);

  bool DeclHasAnnotation(const clang::NamedDecl& decl, std::string_view name);
  bool HasAnnotation(clang::ArrayRef<const clang::AnnotateAttr*> attrs,
                     std::string_view name);

  // Returns std::nullopt if the annotation is not found
  // If default_value is specified, then the parameter is optional
  absl::StatusOr<std::optional<int64_t>>
  GetAnnotationWithNonNegativeIntegerParam(
      const clang::Decl& decl, std::string_view name,
      const xls::SourceInfo& loc,
      std::optional<int64_t> default_value = std::nullopt);

  absl::StatusOr<std::optional<int64_t>>
  GetAnnotationWithNonNegativeIntegerParam(
      clang::ArrayRef<const clang::AnnotateAttr*> attrs, std::string_view name,
      const xls::SourceInfo& loc, clang::ASTContext& ctx,
      std::optional<int64_t> default_value = std::nullopt);

  absl::StatusOr<xls::solvers::z3::IrTranslator*> GetZ3Translator(
      xls::FunctionBase* func) ABSL_ATTRIBUTE_LIFETIME_BOUND;

  absl::flat_hash_map<xls::FunctionBase*,
                      std::unique_ptr<xls::solvers::z3::IrTranslator>>
      z3_translators_;
};

}  // namespace xlscc

// See TrackedBValue
#undef BValue

#endif  // XLS_CONTRIB_XLSCC_TRANSLATOR_H_
