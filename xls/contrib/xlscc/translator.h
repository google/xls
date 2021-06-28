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

#ifndef PLATFORMS_HLS_XLS_FRONTEND_XLSCC_TRANSLATOR_H_
#define PLATFORMS_HLS_XLS_FRONTEND_XLSCC_TRANSLATOR_H_

#include <cstdint>
#include <memory>
#include <stack>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "clang/include/clang/AST/AST.h"
#include "clang/include/clang/AST/ASTConsumer.h"
#include "clang/include/clang/AST/ASTContext.h"
#include "clang/include/clang/AST/Attr.h"
#include "clang/include/clang/AST/Decl.h"
#include "clang/include/clang/AST/DeclCXX.h"
#include "clang/include/clang/AST/Expr.h"
#include "clang/include/clang/AST/Mangle.h"
#include "clang/include/clang/AST/RecursiveASTVisitor.h"
#include "clang/include/clang/AST/Stmt.h"
#include "clang/include/clang/AST/Type.h"
#include "clang/include/clang/Basic/IdentifierTable.h"
#include "clang/include/clang/Basic/SourceLocation.h"
#include "clang/include/clang/Frontend/CompilerInstance.h"
#include "clang/include/clang/Frontend/FrontendActions.h"
#include "clang/include/clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/include/clang/Tooling/CommonOptionsParser.h"
#include "clang/include/clang/Tooling/Tooling.h"
#include "llvm/include/llvm/ADT/APInt.h"
#include "xls/common/thread.h"
#include "xls/contrib/xlscc/hls_block.pb.h"
#include "xls/contrib/xlscc/metadata_output.pb.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/passes/inlining_pass.h"

namespace xlscc {

class Translator;

// Base class for immutable objects representing XLS[cc] value types
// These are not 1:1 with clang::Types, and do not represent qualifiers
//  such as const and reference.
class CType {
 public:
  virtual ~CType() = 0;
  virtual bool operator==(const CType& o) const = 0;
  bool operator!=(const CType& o) const;

  virtual int GetBitWidth() const;
  virtual explicit operator std::string() const;
  virtual xls::Type* GetXLSType(xls::Package* package) const;
  virtual bool StoredAsXLSBits() const;
  virtual absl::Status GetMetadata(xlscc_metadata::Type* output) const;
};

// C/C++ void
class CVoidType : public CType {
 public:
  ~CVoidType() override;

  int GetBitWidth() const override;
  explicit operator std::string() const override;
  absl::Status GetMetadata(xlscc_metadata::Type* output) const override;

  bool operator==(const CType& o) const override;
};

// __xls_bits special built-in type
class CBitsType : public CType {
 public:
  explicit CBitsType(int width);
  ~CBitsType() override;

  int GetBitWidth() const override;
  explicit operator std::string() const override;
  absl::Status GetMetadata(xlscc_metadata::Type* output) const override;

  bool operator==(const CType& o) const override;
  bool StoredAsXLSBits() const override;

 private:
  int width_;
};

// Any native integral type: char, short, int, long, etc
class CIntType : public CType {
 public:
  ~CIntType() override;
  CIntType(int width, bool is_signed);

  int GetBitWidth() const override;
  explicit operator std::string() const override;
  absl::Status GetMetadata(xlscc_metadata::Type* output) const override;

  bool operator==(const CType& o) const override;

  xls::Type* GetXLSType(xls::Package* package) const override;
  bool StoredAsXLSBits() const override;

  inline int width() const { return width_; }
  inline bool is_signed() const { return is_signed_; }

 private:
  const int width_;
  const bool is_signed_;
};

// C++ bool
class CBoolType : public CType {
 public:
  ~CBoolType() override;

  int GetBitWidth() const override;
  explicit operator std::string() const override;
  absl::Status GetMetadata(xlscc_metadata::Type* output) const override;
  bool operator==(const CType& o) const override;
  bool StoredAsXLSBits() const override;
};

// C/C++ struct field
class CField {
 public:
  CField(const clang::NamedDecl* name_, int index, std::shared_ptr<CType> type);

  const clang::NamedDecl* name() const;
  int index() const;
  std::shared_ptr<CType> type() const;
  absl::Status GetMetadata(xlscc_metadata::StructField* output) const;

 private:
  const clang::NamedDecl* name_;
  int index_;
  std::shared_ptr<CType> type_;
};

// C/C++ struct
class CStructType : public CType {
 public:
  CStructType(std::vector<std::shared_ptr<CField>> fields, bool no_tuple_flag);

  int GetBitWidth() const override;
  explicit operator std::string() const override;
  absl::Status GetMetadata(xlscc_metadata::Type* output) const override;
  bool operator==(const CType& o) const override;

  // Returns true if the #pragma no_notuple directive was given for the struct
  bool no_tuple_flag() const;
  const std::vector<std::shared_ptr<CField>>& fields() const;
  const absl::flat_hash_map<const clang::NamedDecl*, std::shared_ptr<CField>>&
  fields_by_name() const;
  // Get the full CField struct by name.
  // returns nullptr if the field is not found
  std::shared_ptr<CField> get_field(const clang::NamedDecl* name) const;

 private:
  bool no_tuple_flag_;
  std::vector<std::shared_ptr<CField>> fields_;
  absl::flat_hash_map<const clang::NamedDecl*, std::shared_ptr<CField>>
      fields_by_name_;
};

// An alias for a type that can be instantiated. Typically this is reduced to
//  another CType via Translator::ResolveTypeInstance()
//
// The reason for this to exist is that two types may have exactly
//  the same content and structure, but still be considered different in C/C++
//  because of their different typenames.
// For example, structs A and B still do not have the same type:
// struct A { int x; int y; };
// struct B { int x; int y; };
//
// They may also have different template parameters. CInstantiableTypeAliases
//  for Foo<true> and Foo<false> are not equal.
// template<bool Tp>
// struct Foo { int bar; };
class CInstantiableTypeAlias : public CType {
 public:
  explicit CInstantiableTypeAlias(const clang::NamedDecl* base);

  const clang::NamedDecl* base() const;

  bool operator==(const CType& o) const override;
  absl::Status GetMetadata(xlscc_metadata::Type* output) const override;

  explicit operator std::string() const override;
  int GetBitWidth() const override;

 private:
  const clang::NamedDecl* base_;
};

// C/C++ native array
class CArrayType : public CType {
 public:
  CArrayType(std::shared_ptr<CType> element, int size);
  bool operator==(const CType& o) const override;
  int GetBitWidth() const override;
  explicit operator std::string() const override;
  absl::Status GetMetadata(xlscc_metadata::Type* output) const override;

  int GetSize() const;
  std::shared_ptr<CType> GetElementType() const;

 private:
  std::shared_ptr<CType> element_;
  int size_;
};

// Immutable object representing an XLS[cc] value. The class associates an
//  XLS IR value expression with an XLS[cc] type (derived from C/C++).
// This class is necessary because an XLS "bits[16]" might be a native short
//  native unsigned short, __xls_bits, or a class containing a single __xls_bits
//  And each of these types implies different translation behaviors.
class CValue {
 public:
  CValue() {}
  CValue(xls::BValue value, std::shared_ptr<CType> type,
         bool disable_type_check = false)
      : value_(value), type_(std::move(type)) {
    XLS_CHECK(disable_type_check || !type_->StoredAsXLSBits() ||
              value.BitCountOrDie() == type_->GetBitWidth());
  }

  xls::BValue value() const { return value_; }
  std::shared_ptr<CType> type() const { return type_; }

 private:
  xls::BValue value_;
  std::shared_ptr<CType> type_;
};

enum class OpType { kNull = 0, kSend, kRecv };

// Tracks information about an __xls_channel parameter to a function
struct IOChannel {
  // Type of item the channel transfers
  std::shared_ptr<CType> item_type;
  // Direction of the port (in/out)
  OpType channel_op_type;
  // The total number of IO ops on the channel within the function
  // (IO ops are conditional, so this is the maximum in a real invocation)
  int total_ops;
};

// Tracks information about an IO op on an __xls_channel parameter to a function
struct IOOp {
  OpType op;

  // Clang AST node for the parameter
  const clang::ParmVarDecl* channel;

  // Clang AST node for the call containing the op.
  // This may be a read() or write() call, or a call to a subroutine containing
  //  read() or write() calls.
  const clang::CallExpr* io_call;

  // For calls to subroutines with IO inside
  const IOOp* sub_op;

  // Input __xls_channel parameters take tuple types containing a value for
  //  each read() op. This is the index of this op in the tuple.
  int channel_tuple_index;

  // Value to return from the function for this operation.
  xls::BValue ret_value;
};

// Tracks information about a call to read() or write() on and __xls_channel,
//  or to a subroutine containing read() or write().
struct IOCall {
  // Can turn into more than one op through unrolling
  std::vector<IOOp*> ops;

  // A counter used in generating IR
  int generated_ops;
};

// Encapsulates values produced when generating IR for a function
struct GeneratedFunction {
  xls::Function* xls_func = nullptr;

  absl::flat_hash_map<const clang::ParmVarDecl*, IOChannel> io_channels;
  absl::flat_hash_map<const clang::CallExpr*, IOCall> op_by_call;

  // All the IO Ops occurring within the function. Order matters here,
  //  as it is assumed that write() ops will depend only on values produced
  //  by read() ops occurring *before* them in the list.
  // Also, The XLS token will be threaded through the IO ops (Send, Receive)
  //  in the order specified in this list.
  // Use list for safe pointers to values
  std::list<IOOp> io_ops;

  // Global values built with this FunctionBuilder
  absl::flat_hash_map<const clang::NamedDecl*, CValue> global_values;
};

// Encapsulates a context for translating Clang AST to XLS IR.
// This is roughly equivalent to a "scope" in C++. There will typically
//  be at least one context pushed into the context stack for each C++ scope.
// The Translator::PopContext() function will propagate certain values, such
//  as new CValues for assignments to variables declared outside the scope,
//  up to the next context / outer scope.
struct TranslationContext {
  xls::BValue not_condition(const xls::SourceLocation& loc) {
    if (!condition.valid()) {
      return fb->Literal(xls::UBits(0, 1), loc);
    } else {
      return fb->Not(condition, loc);
    }
  }

  xls::BValue condition_bval(const xls::SourceLocation& loc) {
    if (!condition.valid()) {
      return fb->Literal(xls::UBits(1, 1), loc);
    } else {
      return condition;
    }
  }

  void and_condition(xls::BValue and_condition,
                     const xls::SourceLocation& loc) {
    if (!condition.valid()) {
      condition = and_condition;
    } else {
      condition = fb->And(condition, and_condition, loc);
    }
  }

  void print_vars() {
    std::cerr << "Context {" << std::endl;
    std::cerr << "  vars:" << std::endl;
    for (const auto& var : variables) {
      std::cerr << "  -- " << var.first->getNameAsString() << ": "
                << var.second.value().ToString() << std::endl;
    }
    std::cerr << "}" << std::endl;
  }

  xls::Package* package;
  std::shared_ptr<CType> return_type;
  xls::BuilderBase* fb;

  // Information being gathered about function currently being processed
  GeneratedFunction* sf;

  // Value for special "this" variable, used in translating class methods
  CValue this_val;

  absl::flat_hash_map<const clang::NamedDecl*, CValue> variables;

  xls::BValue return_val;
  xls::BValue last_return_condition;
  // For "control flow": assignments after a return are conditional on this
  xls::BValue have_returned_condition;

  // Condition for assignments
  xls::BValue condition;
  xls::BValue original_condition;

  // Assign new CValues to variables without applying "condition" above.
  // Used for loop unrolling.
  bool unconditional_assignment = false;

  // For unsequenced assignment check
  absl::flat_hash_set<const clang::NamedDecl*> forbidden_rvalues;
  absl::flat_hash_set<const clang::NamedDecl*> forbidden_lvalues;

  // Set to true if we are evaluating under a multi-argument node in the AST.
  // Constructors, unary operators, etc, don't need unsequenced assignment
  // checks.
  bool in_fork = false;

  // These flags control the behavior of break and continue statements
  bool in_for_body = false;
  bool in_switch_body = false;

  // Used in translating for loops
  xls::BValue break_condition;
  xls::BValue continue_condition;

  // Switch stuff
  // hit_break is set when a break is encountered inside of a switch body.
  // This signals from GenerateIR_Stmt() to GenerateIR_Switch().
  bool hit_break = false;
  // For checking for conditional breaks. If a break occurs in a context
  //  with a condition that's not equal to the enclosing "switch condition",
  //  ie that specified by the enclosing case or default, then a conditional
  //  break is detected, which is unsupported and an error.
  xls::BValue switch_cond;

  // Don't create side-effects when exploring the tree for unsequenced
  // assignments
  bool mask_assignments = false;

  // Ignore pointer qualifiers in type translation. Normally pointers
  //  cause an unsupported error, but when this flag is true,
  //  "Foo*" is translated as "Foo".
  // This mode was created to handle the "this" pointer, which is mandatory
  //  to handle classes.
  bool ignore_pointers = false;
};

class Translator {
 public:
  explicit Translator(int max_unroll_iters = 1000);
  ~Translator();

  // This function uses Clang to parse a source file and then walks its
  //  AST to discover global constructs. It will also scan the file
  //  and includes, recursively, for #pragma statements.
  //
  // Among these are functions, which can be used as entry points
  //  for translation to IR.
  //
  // source_filename must be .cc
  // Retains references to the TU until ~Translator()
  absl::Status ScanFile(absl::string_view source_filename,
                        absl::Span<absl::string_view> command_line_args);

  // Call after ScanFile, as the top function may be specified by #pragma
  // If none was found, an error is returned
  absl::StatusOr<std::string> GetEntryFunctionName() const;

  absl::Status SelectTop(absl::string_view top_function_name);

  // Generates IR as an XLS function, that is, a pure function without
  //  IO / state / side effects.
  // If top_function is 0 or "" then top must be specified via pragma
  // force_static=true Means the function is not generated with a "this"
  //  parameter & output. It is generated as if static was specified in the
  //  method prototype.
  absl::StatusOr<GeneratedFunction*> GenerateIR_Top_Function(
      xls::Package* package, bool force_static = false);

  // Generates IR as an HLS block / XLS proc.
  absl::StatusOr<xls::Proc*> GenerateIR_Block(xls::Package* package,
                                              const HLSBlock& block);

  // Ideally, this would be done using the opt_main tool, but for now
  //  codegen is done by XLS[cc] for combinational blocks.
  absl::Status InlineAllInvokes(xls::Package* package);

  // Generate some useful metadata after either GenerateIR_Top_Function() or
  //  GenerateIR_Block() has run.
  absl::StatusOr<xlscc_metadata::MetadataOutput> GenerateMetadata();

 private:
  friend class CInstantiableTypeAlias;

  // This object is used to push a new translation context onto the stack
  //  and then to pop it via RAII. This guard provides options for which bits of
  //  context to propagate up when popping it from the stack.
  struct PushContextGuard {
    PushContextGuard(Translator& translator, const xls::SourceLocation& loc)
        : translator(translator), loc(loc) {
      translator.PushContext();
    }
    PushContextGuard(Translator& translator, xls::BValue and_condition,
                     const xls::SourceLocation& loc)
        : translator(translator), loc(loc) {
      translator.PushContext();
      translator.context().and_condition(and_condition, loc);
    }
    ~PushContextGuard() {
      translator.PopContext(propagate_up, propagate_break_up,
                            propagate_continue_up, loc);
    }

    Translator& translator;
    // These are Translator::PopContext() parameters
    bool propagate_up = true;
    bool propagate_break_up = true;
    bool propagate_continue_up = true;
    xls::SourceLocation loc;
  };

  // This guard ignores assignment to variables for unsequenced assignment
  //  checking for a period determined by RAII.
  // Assignments to variables during this period do not become errors.
  struct DisallowAssignmentGuard {
    explicit DisallowAssignmentGuard(Translator& translator)
        : translator(translator), enabled(true) {
      forbidden_rvalues_saved = translator.context().forbidden_rvalues;
      forbidden_lvalues_saved = translator.context().forbidden_lvalues;
    }
    ~DisallowAssignmentGuard() {
      if (enabled) {
        translator.context().forbidden_rvalues = forbidden_rvalues_saved;
        translator.context().forbidden_lvalues = forbidden_lvalues_saved;
      }
    }

    absl::flat_hash_set<const clang::NamedDecl*> forbidden_rvalues_saved;
    absl::flat_hash_set<const clang::NamedDecl*> forbidden_lvalues_saved;
    Translator& translator;
    bool enabled;
  };

  // This guard makes assignments unconditional, regardless of scope, for a
  //  period determined by RAII.
  struct UnconditionalAssignmentGuard {
    explicit UnconditionalAssignmentGuard(Translator& translator)
        : translator(translator),
          prev_val(translator.context().unconditional_assignment) {
      translator.context().unconditional_assignment = true;
    }
    ~UnconditionalAssignmentGuard() {
      translator.context().unconditional_assignment = prev_val;
    }

    Translator& translator;
    bool prev_val;
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
    explicit MaskAssignmentsGuard(Translator& translator)
        : translator(translator),
          prev_val(translator.context().mask_assignments) {
      translator.context().mask_assignments = true;
    }
    ~MaskAssignmentsGuard() {
      translator.context().mask_assignments = prev_val;
    }

    Translator& translator;
    bool prev_val;
  };

  class LibToolVisitor : public clang::RecursiveASTVisitor<LibToolVisitor> {
   private:
    clang::ASTContext* ast_context_;
    Translator& translator_;

   public:
    explicit LibToolVisitor(clang::CompilerInstance* CI, Translator& translator)
        : ast_context_(&(CI->getASTContext())), translator_(translator) {}
    virtual ~LibToolVisitor() {}
    virtual bool VisitFunctionDecl(clang::FunctionDecl* func) {
      return translator_.LibToolVisitFunction(func);
    }
  };
  class LibToolASTConsumer : public clang::ASTConsumer {
   private:
    std::unique_ptr<LibToolVisitor> visitor_;

   public:
    explicit LibToolASTConsumer(clang::CompilerInstance* CI,
                                Translator& translator)
        : visitor_(new LibToolVisitor(CI, translator)) {}

    void HandleTranslationUnit(clang::ASTContext& Context) override {
      visitor_->TraverseDecl(Context.getTranslationUnitDecl());
    }
  };
  class LibToolFrontendAction : public clang::ASTFrontendAction {
   public:
    explicit LibToolFrontendAction(Translator& translator)
        : translator_(translator) {}
    void EndSourceFileAction() override;
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
        clang::CompilerInstance& CI, clang::StringRef /*file*/) override {
      return std::unique_ptr<clang::ASTConsumer>(
          new LibToolASTConsumer(&CI, translator_));
    }

   private:
    Translator& translator_;
  };
  class DiagnosticInterceptor : public clang::TextDiagnosticPrinter {
   public:
    DiagnosticInterceptor(Translator& translator, llvm::raw_ostream& os,
                          clang::DiagnosticOptions* diags,
                          bool OwnsOutputStream = false)
        : clang::TextDiagnosticPrinter(os, diags, OwnsOutputStream),
          translator_(translator) {}
    void HandleDiagnostic(clang::DiagnosticsEngine::Level level,
                          const clang::Diagnostic& info) override {
      if (level >= clang::DiagnosticsEngine::Level::Error) {
        translator_.libtool_visit_status_ = absl::FailedPreconditionError(
            absl::StrFormat("Unable to parse text with clang (libtooling)\n"));
      }
      // Print the message
      clang::TextDiagnosticPrinter::HandleDiagnostic(level, info);
    }

   private:
    Translator& translator_;
  };
  class LibToolThread {
   public:
    LibToolThread(absl::string_view source_filename,
                  absl::Span<absl::string_view> command_line_args,
                  Translator& translator);

    void Start();
    void Join();

   private:
    void Run();

    absl::optional<xls::Thread> thread_;
    absl::string_view source_filename_;
    absl::Span<absl::string_view> command_line_args_;
    Translator& translator_;
  };
  bool LibToolVisitFunction(clang::FunctionDecl* func);
  absl::Status libtool_visit_status_ = absl::OkStatus();

  std::unique_ptr<LibToolThread> libtool_thread_;
  std::unique_ptr<absl::BlockingCounter> libtool_wait_for_parse_;
  std::unique_ptr<absl::BlockingCounter> libtool_wait_for_destruct_;

  // The maximum number of iterations before loop unrolling fails.
  const int max_unroll_iters_;

  // This should be true except for debugging.
  // When this flag is true, the translator generates expressions
  //  right to left as well as left to right, so as to detect all
  //  unsequenced modification and access in expressions.
  const bool unsequenced_gen_backwards_ = true;

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

  void print_types() {
    std::cerr << "Types {" << std::endl;
    for (const auto& var : inst_types_) {
      std::cerr << "  -- " << std::string(*var.first) << ": "
                << std::string(*var.second) << std::endl;
    }
    std::cerr << "}" << std::endl;
  }

  // The translator assumes NamedDecls are unique. This set is used to
  //  generate an error if that assumption is violated.
  absl::flat_hash_set<const clang::NamedDecl*> check_unique_ids_;

  using PragmaLoc = std::tuple<std::string, int>;
  enum Pragma { Pragma_Null = 0, Pragma_NoTuples, Pragma_Unroll, Pragma_Top };
  absl::flat_hash_map<PragmaLoc, Pragma> hls_pragmas_;
  absl::flat_hash_set<std::string> files_scanned_for_pragmas_;

  absl::StatusOr<Pragma> FindPragmaForLoc(const clang::PresumedLoc& ploc);

  // Scans for top-level function candidates
  absl::Status VisitFunction(const clang::FunctionDecl* funcdecl);
  absl::Status ScanFileForPragmas(std::string filename);

  const clang::FunctionDecl* top_function_ = nullptr;
  absl::string_view top_function_name_ = "";

  // For source location
  absl::flat_hash_map<std::string, int> file_numbers_;
  int next_file_number_ = 1;

  int next_asm_number_ = 1;

  mutable std::unique_ptr<clang::MangleContext> mangler_;

  TranslationContext& PushContext();
  void PopContext(bool propagate_up, bool propagate_break_up,
                  bool propagate_continue_up, const xls::SourceLocation& loc);

  std::stack<TranslationContext> context_stack_;

  TranslationContext& context();

  absl::StatusOr<CValue> Generate_UnaryOp(const clang::UnaryOperator* uop,
                                          const xls::SourceLocation& loc);
  absl::StatusOr<CValue> Generate_Synthetic_ByOne(
      xls::Op xls_op, bool is_pre, CValue sub_value,
      const clang::Expr* sub_expr,  // For assignment
      const xls::SourceLocation& loc);
  absl::StatusOr<CValue> Generate_BinaryOp(
      clang::BinaryOperator::Opcode clang_op, bool is_assignment,
      std::shared_ptr<CType> result_type, const clang::Expr* lhs,
      const clang::Expr* rhs, const xls::SourceLocation& loc);
  absl::StatusOr<CValue> Generate_TernaryOp(std::shared_ptr<CType> result_type,
                                            const clang::Expr* cond_expr,
                                            const clang::Expr* true_expr,
                                            const clang::Expr* false_expr,
                                            const xls::SourceLocation& loc);
  absl::StatusOr<CValue> GenerateIR_Expr(const clang::Expr* expr,
                                         const xls::SourceLocation& loc);
  absl::StatusOr<CValue> GenerateIR_MemberExpr(const clang::MemberExpr* expr,
                                               const xls::SourceLocation& loc);
  absl::StatusOr<CValue> GenerateIR_Call(const clang::CallExpr* call,
                                         const xls::SourceLocation& loc);

  absl::StatusOr<CValue> GenerateIR_Call(
      const clang::FunctionDecl* funcdecl,
      std::vector<const clang::Expr*> expr_args, xls::BValue* this_inout,
      const xls::SourceLocation& loc, bool force_no_fork = false);

  // This is a work-around for non-const operator [] needing to return
  //  a reference to the object being modified.
  absl::StatusOr<bool> ApplyArrayAssignHack(
      const clang::CXXOperatorCallExpr* op_call, const xls::SourceLocation& loc,
      CValue* output);

  struct PreparedBlock {
    GeneratedFunction* xls_func;
    std::vector<xls::BValue> args;
    absl::flat_hash_map<const clang::ParmVarDecl*, xls::Channel*> fifo_by_param;
    absl::flat_hash_map<const IOOp*, int> arg_index_for_op;
    absl::flat_hash_map<const IOOp*, int> return_index_for_op;
    absl::flat_hash_map<std::string, HLSChannel> channels_by_name;
    xls::BValue token;
  };

  // Verifies the function prototype in the Clang AST and HLSBlock are sound.
  absl::Status GenerateIRBlockCheck(PreparedBlock& prepared,
                                    const HLSBlock& block,
                                    const clang::FunctionDecl* definition,
                                    const xls::SourceLocation& body_loc);

  // Prepares IO channels for generating XLS Proc
  absl::Status GenerateIRBlockPrepare(PreparedBlock& prepared,
                                      xls::ProcBuilder& pb,
                                      const HLSBlock& block,
                                      const clang::FunctionDecl* definition,
                                      xls::Package* package,
                                      const xls::SourceLocation& body_loc);

  struct IOOpReturn {
    bool generate_expr;
    CValue value;
    xls::BValue condition;
  };
  // Checks if an expression is an IO op, and if so, generates the value
  //  to replace it in IR generation.
  absl::StatusOr<IOOpReturn> InterceptIOOp(const clang::Expr* expr,
                                           const xls::SourceLocation& loc);

  absl::Status DeepScanForIO(const clang::Stmt* body, clang::ASTContext& ctx);

  absl::StatusOr<xls::Channel*> CreateChannel(const HLSChannel& hls_channel,
                                              std::shared_ptr<CType> ctype,
                                              xls::Package* package,
                                              int port_order,
                                              const xls::SourceLocation& loc);

  // Adds subroutine call's IO ops to context().sf->io_ops
  absl::Status TranslateIOOps(const clang::CallExpr* call,
                              const clang::FunctionDecl* callee,
                              const std::list<IOOp>& callee_ops,
                              clang::SourceManager& sm);

  // IOOp must have io_call, channel, and op members filled in
  absl::Status AddOpToChannel(IOOp op, const xls::SourceLocation& loc);
  absl::StatusOr<bool> ExprIsChannel(const clang::Expr* object,
                                     const xls::SourceLocation& loc);
  absl::StatusOr<bool> TypeIsChannel(const clang::QualType& param,
                                     const xls::SourceLocation& loc);

  absl::StatusOr<CValue> GetChannelInputValue(const IOOp& op,
                                              const xls::SourceLocation& loc);

  absl::Status GenerateIR_Compound(const clang::Stmt* body,
                                   clang::ASTContext& ctx);
  absl::Status GenerateIR_Stmt(const clang::Stmt* stmt, clang::ASTContext& ctx);
  absl::Status GenerateIR_UnrolledFor(const clang::ForStmt* stmt,
                                      clang::ASTContext& ctx,
                                      const xls::SourceLocation& loc,
                                      bool deep_scan);
  absl::Status GenerateIR_Switch(const clang::SwitchStmt* switchst,
                                 clang::ASTContext& ctx,
                                 const xls::SourceLocation& loc);

  struct ResolvedInheritance {
    std::shared_ptr<CField> base_field;
    std::shared_ptr<const CStructType> resolved_struct;
    const clang::NamedDecl* base_field_name;
  };

  absl::StatusOr<ResolvedInheritance> ResolveInheritance(
      std::shared_ptr<CType> sub_type, std::shared_ptr<CType> to_type);

  absl::StatusOr<xls::BValue> GenTypeConvert(CValue const& in,
                                             std::shared_ptr<CType> out_type,
                                             const xls::SourceLocation& loc);
  absl::StatusOr<xls::BValue> GenBoolConvert(CValue const& in,
                                             const xls::SourceLocation& loc);

  absl::StatusOr<xls::BValue> CreateDefaultValue(
      std::shared_ptr<CType> t, const xls::SourceLocation& loc);
  absl::StatusOr<xls::BValue> CreateInitValue(
      const CType& t, const clang::InitListExpr* init_list,
      const xls::SourceLocation& loc);
  absl::StatusOr<CValue> GetIdentifier(const clang::NamedDecl* decl,
                                       const xls::SourceLocation& loc,
                                       bool for_lvalue = false);
  absl::StatusOr<CValue> TranslateVarDecl(const clang::VarDecl* decl,
                                          const xls::SourceLocation& loc);
  absl::Status Assign(const clang::NamedDecl* lvalue, const CValue& rvalue,
                      const xls::SourceLocation& loc);
  absl::Status Assign(const clang::Expr* lvalue, const CValue& rvalue,
                      const xls::SourceLocation& loc);
  absl::Status AssignThis(const CValue& rvalue, const xls::SourceLocation& loc);
  absl::StatusOr<CValue> PrepareRValueForAssignment(const CValue& lvalue,
                                                    const CValue& rvalue);

  absl::Status DeclareVariable(const clang::NamedDecl* lvalue,
                               const CValue& rvalue,
                               const xls::SourceLocation& loc);

  absl::StatusOr<GeneratedFunction*> GenerateIR_Function(
      const clang::FunctionDecl* funcdecl, absl::string_view name_override = "",
      bool force_static = false);

  struct StrippedType {
    StrippedType(clang::QualType base, bool is_ref)
        : base(base), is_ref(is_ref) {}
    clang::QualType base;
    bool is_ref;
  };

  absl::StatusOr<StrippedType> StripTypeQualifiers(const clang::QualType& t);
  absl::Status ScanStruct(const clang::RecordDecl* sd);

  absl::StatusOr<std::shared_ptr<CType>> InterceptBuiltInStruct(
      const clang::RecordDecl* sd);

  absl::StatusOr<std::shared_ptr<CType>> TranslateTypeFromClang(
      const clang::QualType& t, const xls::SourceLocation& loc);
  absl::StatusOr<xls::Type*> TranslateTypeToXLS(std::shared_ptr<CType> t,
                                                const xls::SourceLocation& loc);
  absl::StatusOr<std::shared_ptr<CType>> ResolveTypeInstance(
      std::shared_ptr<CType> t);
  absl::StatusOr<GeneratedFunction*> TranslateFunctionToXLS(
      const clang::FunctionDecl* decl);

  absl::StatusOr<int64_t> EvaluateInt64(const clang::Expr& expr,
                                        const class clang::ASTContext& ctx,
                                        const xls::SourceLocation& loc);
  absl::StatusOr<bool> EvaluateBool(const clang::Expr& expr,
                                    const class clang::ASTContext& ctx,
                                    const xls::SourceLocation& loc);
  absl::StatusOr<xls::Value> EvaluateBVal(xls::BValue bval);

  absl::StatusOr<xls::Op> XLSOpcodeFromClang(clang::BinaryOperatorKind clang_op,
                                             const CType& left_type,
                                             const CType& result_type,
                                             const xls::SourceLocation& loc);
  std::string XLSNameMangle(clang::GlobalDecl decl) const;

  xls::BValue MakeFunctionReturn(const std::vector<xls::BValue>& bvals,
                                 const xls::SourceLocation& loc);
  xls::BValue GetFunctionReturn(xls::BValue val, int index,
                                int expected_returns,
                                const clang::FunctionDecl* func,
                                const xls::SourceLocation& loc);

  absl::Status GenerateMetadataType(const clang::QualType& type_in,
                                    xlscc_metadata::Type* type_out);

  // StructUpdate builds and returns a new BValue for a struct with the
  // value of one field changed. The other fields, if any, take their values
  // from struct_before, and the new value for the field named by field_name is
  // set to rvalue. The type of structure built is specified by type.
  absl::StatusOr<xls::BValue> StructUpdate(xls::BValue struct_before,
                                           CValue rvalue,
                                           const clang::NamedDecl* field_name,
                                           const CStructType& type,
                                           const xls::SourceLocation& loc);
  // Creates an BValue for a struct of type stype from field BValues given in
  //  order within bvals.
  xls::BValue MakeStructXLS(const std::vector<xls::BValue>& bvals,
                            const CStructType& stype,
                            const xls::SourceLocation& loc);
  // Returns the BValue for the field with index "index" from a BValue for a
  //  struct of type "type"
  xls::BValue GetStructFieldXLS(xls::BValue val, int index,
                                const CStructType& type,
                                const xls::SourceLocation& loc);

  // Gets the appropriate XLS type for a struct. For example, it might be an
  //  xls::Tuple, or if #pragma hls_notuple was specified, it might be
  //  the single field's type
  absl::StatusOr<xls::Type*> GetStructXLSType(
      const std::vector<xls::Type*>& members, const CStructType& type,
      const xls::SourceLocation& loc);

  // "Flexible Tuple" functions
  // These functions will create a tuple if there is more than one
  //  field, or they will pass through the value if there
  //  is exactly 1 value. 0 values is unsupported.
  // This makes the generated IR less cluttered, as extra single-item tuples
  //  aren't generated.

  // Wraps bvals in a "flexible tuple"
  xls::BValue MakeFlexTuple(const std::vector<xls::BValue>& bvals,
                            const xls::SourceLocation& loc);
  // Gets the XLS type for a "flexible tuple" made from these elements
  xls::Type* GetFlexTupleType(const std::vector<xls::Type*>& members);
  // Gets the value of a field in a "flexible tuple"
  // val is the "flexible tuple" value
  // index is the index of the field
  // n_fields is the total number of fields
  xls::BValue GetFlexTupleField(xls::BValue val, int index, int n_fields,
                                const xls::SourceLocation& loc);
  // Changes the value of a field in a "flexible tuple"
  // tuple_val is the "flexible tuple" value
  // new_val is the value to set the field to
  // index is the index of the field
  // n_fields is the total number of fields
  xls::BValue UpdateFlexTupleField(xls::BValue tuple_val, xls::BValue new_val,
                                   int index, int n_fields,
                                   const xls::SourceLocation& loc);

  xls::SourceLocation GetLoc(clang::SourceManager& sm, const clang::Stmt& stmt);
  clang::PresumedLoc GetPresumedLoc(clang::SourceManager& sm,
                                    const clang::Stmt& stmt);
  xls::SourceLocation GetLoc(const clang::Decl& decl);
  clang::PresumedLoc GetPresumedLoc(const clang::Decl& decl);
  xls::SourceLocation GetLoc(clang::SourceManager& sm, const clang::Expr& expr);
  xls::SourceLocation GetLoc(const clang::PresumedLoc& loc);
  std::string LocString(const xls::SourceLocation& loc);
  inline std::string LocString(const clang::Decl& decl) {
    return LocString(GetLoc(decl));
  }
};

}  // namespace xlscc

#endif  // PLATFORMS_HLS_XLS_FRONTEND_XLSCC_TRANSLATOR_H_
