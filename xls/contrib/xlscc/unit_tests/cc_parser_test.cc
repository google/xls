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

#include "xls/contrib/xlscc/cc_parser.h"

#include <cstdint>
#include <string>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "clang/include/clang/AST/Attr.h"
#include "clang/include/clang/AST/Decl.h"
#include "clang/include/clang/AST/Expr.h"
#include "clang/include/clang/AST/Stmt.h"
#include "clang/include/clang/Basic/LLVM.h"
#include "clang/include/clang/Basic/SourceLocation.h"
#include "llvm/include/llvm/Support/Casting.h"
#include "xls/common/status/matchers.h"
#include "xls/contrib/xlscc/metadata_output.pb.h"
#include "xls/contrib/xlscc/unit_tests/unit_test.h"
#include "xls/ir/source_location.h"

namespace {

class CCParserTest : public XlsccTestBase {
 public:
};

void ExpectIntrinsicWithIntegerArg(const clang::CallExpr* call,
                                   std::string_view name,
                                   int64_t arg_expected) {
  ASSERT_EQ(call->getDirectCallee()->getNameAsString(), name);
  ASSERT_EQ(call->getNumArgs(), 1);
  const clang::Expr* arg = call->getArg(0)->IgnoreCasts();
  const clang::IntegerLiteral* integer_literal =
      dyn_cast<clang::IntegerLiteral>(arg);
  ASSERT_NE(integer_literal, nullptr);
  EXPECT_EQ(integer_literal->getValue().getSExtValue(), arg_expected);
}

template <typename ClangT>
const ClangT* GetStmtInFunction(const clang::FunctionDecl* func) {
  const clang::Stmt* body = func->getBody();
  if (body == nullptr) {
    return nullptr;
  }

  for (const clang::Stmt* body_st : body->children()) {
    const clang::LabelStmt* label =
        clang::dyn_cast<const clang::LabelStmt>(body_st);
    if (label != nullptr) {
      body_st = label->getSubStmt();
    }

    const ClangT* ret = clang::dyn_cast<const ClangT>(body_st);
    if (ret != nullptr) {
      return ret;
    }
  }

  return nullptr;
}

const clang::CallExpr* FindCallBefore(const clang::FunctionDecl* func,
                                      const clang::Stmt* before_stmt) {
  CHECK(func->getBody() != nullptr);
  const clang::CallExpr* prev = nullptr;
  for (const clang::Stmt* stmt : func->getBody()->children()) {
    if (stmt == before_stmt) {
      return prev;
    }
    const clang::CallExpr* call = clang::dyn_cast<const clang::CallExpr>(stmt);
    if (call != nullptr) {
      prev = call;
    }
  }
  return nullptr;
}

TEST_F(CCParserTest, Basic) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #pragma hls_top
    int foo(int a, int b) {
      const int foo = a + b;
      return foo;
    }
  )";

  XLS_ASSERT_OK(ScanTempFileWithContent(cpp_src, {}, &parser));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  EXPECT_NE(top_ptr, nullptr);
}

TEST_F(CCParserTest, Basic2) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #pragma hls_design top
    int foo(int a, int b) {
      const int foo = a + b;
      return foo;
    }
  )";

  XLS_ASSERT_OK(ScanTempFileWithContent(cpp_src, {}, &parser));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  EXPECT_NE(top_ptr, nullptr);
}

TEST_F(CCParserTest, TopNotFound) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    int foo(int a, int b) {
      const int foo = a + b;
      return foo;
    }
  )";

  XLS_ASSERT_OK(ScanTempFileWithContent(cpp_src, {}, &parser));
  EXPECT_THAT(parser.GetTopFunction().status(),
              xls::status_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(CCParserTest, SourceMeta) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #pragma hls_top
    int foo(int a, int b) {
      const int foo = a + b;
      return foo;
    }
  )";

  XLS_ASSERT_OK(ScanTempFileWithContent(cpp_src, {}, &parser));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  ASSERT_NE(top_ptr, nullptr);

  xls::SourceInfo loc = parser.GetLoc(*top_ptr);

  xlscc_metadata::MetadataOutput output;
  parser.AddSourceInfoToMetadata(output);
  ASSERT_EQ(output.sources_size(), 1);
  EXPECT_EQ(loc.locations.size(), 1);
  EXPECT_EQ(static_cast<int32_t>(loc.locations[0].fileno()),
            static_cast<int32_t>(output.sources(0).number()));
}

TEST_F(CCParserTest, Pragma) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #pragma hls_top
    int foo(int a, int b) {
      const int foo = a + b;
      return foo;
    }
  )";

  XLS_ASSERT_OK(ScanTempFileWithContent(cpp_src, {}, &parser));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  ASSERT_NE(top_ptr, nullptr);
}

TEST_F(CCParserTest, Annotation) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    [[hls_top]]
    int foo(int a, int b) {
      const int foo = a + b;
      return foo;
    }
  )";

  XLS_ASSERT_OK(ScanTempFileWithContent(cpp_src, {}, &parser));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  ASSERT_NE(top_ptr, nullptr);
}

TEST_F(CCParserTest, PragmaPipelineInitInterval) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #pragma hls_top
    int foo(int a, int b) {
      int foo = a;
      #pragma hls_pipeline_init_interval 3
      for(int i=0;i<2;++i) {
        foo += b;
      }
      return foo;
    }
  )";

  XLS_ASSERT_OK(ScanTempFileWithContent(cpp_src, {}, &parser));
  XLS_ASSERT_OK_AND_ASSIGN(const clang::FunctionDecl* top_ptr,
                           parser.GetTopFunction());
  ASSERT_NE(top_ptr, nullptr);

  auto* for_stmt = GetStmtInFunction<clang::ForStmt>(top_ptr);
  ASSERT_NE(for_stmt, nullptr);

  const clang::CallExpr* call = FindCallBefore(top_ptr, for_stmt);
  ASSERT_NE(call, nullptr);

  ExpectIntrinsicWithIntegerArg(call, "__xlscc_pipeline",
                                /*arg_expected=*/3);
}

TEST_F(CCParserTest, PragmaPipelineInitIntervalDouble) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #pragma hls_top
    int foo(int a, int b) {
      int foo = a;
      #pragma hls_pipeline_init_interval 10
      #pragma hls_pipeline_init_interval 3
      for(int i=0;i<2;++i) {
        foo += b;
      }
      return foo;
    }
  )";

  XLS_ASSERT_OK(ScanTempFileWithContent(cpp_src, {}, &parser));
  XLS_ASSERT_OK_AND_ASSIGN(const clang::FunctionDecl* top_ptr,
                           parser.GetTopFunction());
  ASSERT_NE(top_ptr, nullptr);

  auto* for_stmt = GetStmtInFunction<clang::ForStmt>(top_ptr);
  ASSERT_NE(for_stmt, nullptr);

  const clang::CallExpr* call = FindCallBefore(top_ptr, for_stmt);
  ASSERT_NE(call, nullptr);

  ExpectIntrinsicWithIntegerArg(call, "__xlscc_pipeline", /*arg_expected=*/3);
}

TEST_F(CCParserTest, UnknownPragma) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #pragma hls_top
    int foo(int a, int b) {
      int foo = a;
      #pragma foo
      for(int i=0;i<2;++i) {
        foo += b;
      }
      return foo;
    }
  )";

  XLS_EXPECT_OK(ScanTempFileWithContent(cpp_src, {}, &parser));
}

TEST_F(CCParserTest, InvalidPragmaArg) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #pragma hls_top
    int foo(int a, int b) {
      int foo = a;
      #pragma hls_pipeline_init_interval hey
      for(int i=0;i<2;++i) {
        foo += b;
      }
      return foo;
    }
  )";

  EXPECT_THAT(
      ScanTempFileWithContent(cpp_src, {}, &parser),
      xls::status_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(CCParserTest, InvalidPragmaArg2) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #pragma hls_top
    int foo(int a, int b) {
      int foo = a;
      #pragma hls_pipeline_init_interval -22
      for(int i=0;i<2;++i) {
        foo += b;
      }
      return foo;
    }
  )";

  EXPECT_THAT(
      ScanTempFileWithContent(cpp_src, {}, &parser),
      xls::status_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(CCParserTest, CommentedPragma) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #pragma hls_top
    int foo(int a, int b) {
      int foo = a;
      //#pragma hls_pipeline_init_interval -22
      for(int i=0;i<2;++i) {
        foo += b;
      }
      return foo;
    }
  )";

  XLS_ASSERT_OK(ScanTempFileWithContent(cpp_src, {}, &parser));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  ASSERT_NE(top_ptr, nullptr);

  auto* for_stmt = GetStmtInFunction<clang::ForStmt>(top_ptr);
  ASSERT_NE(for_stmt, nullptr);

  const clang::CallExpr* call = FindCallBefore(top_ptr, for_stmt);
  EXPECT_EQ(call, nullptr);
}

TEST_F(CCParserTest, IfdefdPragmaFalse) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #pragma hls_top
    int foo(int a, int b) {
      int foo = a;
#if 0
      #pragma hls_pipeline_init_interval -22
#endif
      for(int i=0;i<2;++i) {
        foo += b;
      }
      return foo;
    }
  )";

  XLS_ASSERT_OK(ScanTempFileWithContent(cpp_src, {}, &parser));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  ASSERT_NE(top_ptr, nullptr);

  auto* for_stmt = GetStmtInFunction<clang::ForStmt>(top_ptr);
  ASSERT_NE(for_stmt, nullptr);

  const clang::CallExpr* call = FindCallBefore(top_ptr, for_stmt);
  EXPECT_EQ(call, nullptr);
}

TEST_F(CCParserTest, IfdefdPragmaTrue) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #pragma hls_top
    int foo(int a, int b) {
      int foo = a;
#if 1
      #pragma hls_pipeline_init_interval -22
#endif
      for(int i=0;i<2;++i) {
        foo += b;
      }
      return foo;
    }
  )";

  EXPECT_THAT(
      ScanTempFileWithContent(cpp_src, {}, &parser),
      xls::status_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(CCParserTest, SourceManagerInitialized) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #pragma hls_top
    int foo(int a, int b) {
      return a+b;
    }
  )";

  XLS_ASSERT_OK(ScanTempFileWithContent(cpp_src, {}, &parser));
  ASSERT_NE(parser.sm_, nullptr);
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  ASSERT_NE(top_ptr, nullptr);
  parser.GetPresumedLoc(*top_ptr);
}

TEST_F(CCParserTest, FoundOnReset) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #include "/xls_builtin.h"
    #pragma hls_top
    int foo(int a, int b) {
      return a+b;
    }
  )";

  XLS_ASSERT_OK(ScanTempFileWithContent(cpp_src, {}, &parser));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* on_reset_ptr, parser.GetXlsccOnReset());
  EXPECT_NE(on_reset_ptr, nullptr);
}

TEST_F(CCParserTest, NameOverPragma) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    int bar(int a, int b) {
      const int foo = a + b;
      return foo+1;
    }
    #pragma hls_design top
    int foo(int a, int b) {
      const int foo = a + b;
      return foo;
    }
  )";

  XLS_ASSERT_OK(
      ScanTempFileWithContent(cpp_src, {}, &parser, /*top_name=*/"bar"));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  ASSERT_NE(top_ptr, nullptr);
  EXPECT_EQ(top_ptr->getNameAsString(), "bar");
}

TEST_F(CCParserTest, UnrollYes) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    int bar(int (&a)[5], int b) {
      #pragma hls_unroll yes
      for (int i = 0; i < 5; ++i) a[i] = b;
      return true;
    }
  )";

  XLS_ASSERT_OK(
      ScanTempFileWithContent(cpp_src, {}, &parser, /*top_name=*/"bar"));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  ASSERT_NE(top_ptr, nullptr);

  auto* for_stmt = GetStmtInFunction<clang::ForStmt>(top_ptr);
  ASSERT_NE(for_stmt, nullptr);

  const clang::CallExpr* call = FindCallBefore(top_ptr, for_stmt);
  ASSERT_NE(call, nullptr);

  ExpectIntrinsicWithIntegerArg(call, "__xlscc_unroll", /*arg_expected=*/1);
}

TEST_F(CCParserTest, Unroll2) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    int bar(int (&a)[5], int b) {
      #pragma hls_unroll 2
      for (int i = 0; i < 5; ++i) a[i] = b;
      return true;
    }
  )";

  XLS_ASSERT_OK(
      ScanTempFileWithContent(cpp_src, {}, &parser, /*top_name=*/"bar"));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  ASSERT_NE(top_ptr, nullptr);

  clang::PresumedLoc func_loc = parser.GetPresumedLoc(*top_ptr);
  clang::PresumedLoc loop_loc(func_loc.getFilename(), func_loc.getFileID(),
                              func_loc.getLine() + 2, func_loc.getColumn(),
                              func_loc.getIncludeLoc());

  auto* for_stmt = GetStmtInFunction<clang::ForStmt>(top_ptr);
  ASSERT_NE(for_stmt, nullptr);

  const clang::CallExpr* call = FindCallBefore(top_ptr, for_stmt);
  ASSERT_NE(call, nullptr);

  ExpectIntrinsicWithIntegerArg(call, "__xlscc_unroll", /*arg_expected=*/2);
}

TEST_F(CCParserTest, Unroll2WithCommentBefore) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    int bar(int (&a)[5], int b) {
      #pragma hls_unroll 2
      // infinity and beyond
      for (int i = 0; i < 5; ++i) a[i] = b;
      return true;
    }
  )";

  XLS_ASSERT_OK(
      ScanTempFileWithContent(cpp_src, {}, &parser, /*top_name=*/"bar"));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  ASSERT_NE(top_ptr, nullptr);

  auto* for_stmt = GetStmtInFunction<clang::ForStmt>(top_ptr);
  ASSERT_NE(for_stmt, nullptr);

  const clang::CallExpr* call = FindCallBefore(top_ptr, for_stmt);
  ASSERT_NE(call, nullptr);

  ExpectIntrinsicWithIntegerArg(call, "__xlscc_unroll", /*arg_expected=*/2);
}

TEST_F(CCParserTest, Unroll2WithComment) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    int bar(int (&a)[5], int b) {
      #pragma hls_unroll 2  // be or not to be
      for (int i = 0; i < 5; ++i) a[i] = b;
      return true;
    }
  )";

  XLS_ASSERT_OK(
      ScanTempFileWithContent(cpp_src, {}, &parser, /*top_name=*/"bar"));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  ASSERT_NE(top_ptr, nullptr);

  auto* for_stmt = GetStmtInFunction<clang::ForStmt>(top_ptr);
  ASSERT_NE(for_stmt, nullptr);

  const clang::CallExpr* call = FindCallBefore(top_ptr, for_stmt);
  ASSERT_NE(call, nullptr);

  ExpectIntrinsicWithIntegerArg(call, "__xlscc_unroll", /*arg_expected=*/2);
}

TEST_F(CCParserTest, UnrollZero) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    int bar(int (&a)[5], int b) {
      #pragma hls_unroll 0
      for (int i = 0; i < 5; ++i) a[i] = b;
      return true;
    }
  )";

  XLS_ASSERT_OK(
      ScanTempFileWithContent(cpp_src, {}, &parser, /*top_name=*/"bar"));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  ASSERT_NE(top_ptr, nullptr);

  auto* for_stmt = GetStmtInFunction<clang::ForStmt>(top_ptr);
  ASSERT_NE(for_stmt, nullptr);

  const clang::CallExpr* call = FindCallBefore(top_ptr, for_stmt);
  EXPECT_EQ(call, nullptr);
}

TEST_F(CCParserTest, UnrollNo) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    int bar(int (&a)[5], int b) {
      #pragma hls_unroll no 
      for (int i = 0; i < 5; ++i) a[i] = b;
      return true;
    }
  )";

  XLS_ASSERT_OK(
      ScanTempFileWithContent(cpp_src, {}, &parser, /*top_name=*/"bar"));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  ASSERT_NE(top_ptr, nullptr);

  auto* for_stmt = GetStmtInFunction<clang::ForStmt>(top_ptr);
  ASSERT_NE(for_stmt, nullptr);

  const clang::CallExpr* call = FindCallBefore(top_ptr, for_stmt);
  EXPECT_EQ(call, nullptr);
}

TEST_F(CCParserTest, UnrollBadNumber) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    int bar(int (&a)[5], int b) {
      #pragma hls_unroll -4
      for (int i = 0; i < 5; ++i) a[i] = b;
      return true;
    }
  )";

  EXPECT_THAT(
      ScanTempFileWithContent(cpp_src, {}, &parser, /*top_name=*/"bar"),
      xls::status_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                    testing::HasSubstr("must")));
}

TEST_F(CCParserTest, DoubleTopName) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    namespace blah {
    int bar(int a, int b) {
      const int foo = a + b;
      return foo+1;
    }
    }  // namespace
    #pragma hls_design top
    int bar(int a, int b) {
      const int foo = a + b;
      return foo;
    }
  )";

  EXPECT_THAT(
      ScanTempFileWithContent(cpp_src, {}, &parser, /*top_name=*/"bar"),
      xls::status_testing::StatusIs(absl::StatusCode::kAlreadyExists,
                                    testing::HasSubstr("Two top functions")));
}

TEST_F(CCParserTest, DoubleTopPragma) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #pragma hls_design top
    int foo(int a, int b) {
      const int foo = a + b;
      return foo+1;
    }
    #pragma hls_top
    int bar(int a, int b) {
      const int foo = a + b;
      return foo;
    }
  )";

  EXPECT_THAT(
      ScanTempFileWithContent(cpp_src, {}, &parser),
      xls::status_testing::StatusIs(absl::StatusCode::kAlreadyExists,
                                    testing::HasSubstr("Two top functions")));
}

TEST_F(CCParserTest, PragmaZeroExtend) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #pragma hls_top
    int foo(int a, int b) {
      #pragma hls_array_allow_default_pad
      int x[4] = {5};
      return x[3];
    }
  )";
  XLS_ASSERT_OK(ScanTempFileWithContent(cpp_src, {}, &parser));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  ASSERT_NE(top_ptr, nullptr);

  auto* decl_stmt = GetStmtInFunction<clang::DeclStmt>(top_ptr);
  ASSERT_NE(decl_stmt, nullptr);

  ASSERT_TRUE(decl_stmt->isSingleDecl());

  const clang::Decl* decl = decl_stmt->getSingleDecl();
  ASSERT_NE(decl, nullptr);

  const clang::AnnotateAttr* attr = decl->getAttr<clang::AnnotateAttr>();

  EXPECT_EQ(attr->getAnnotation().str(), "hls_array_allow_default_pad");
}

TEST_F(CCParserTest, ChannelRead) {
  xlscc::CCParser parser;
  constexpr std::string_view cpp_src = R"(
    #pragma hls_top
    int foo(__xls_channel<int>& chan) {
      return chan.read();
    }
  )";
  XLS_ASSERT_OK(ScanTempFileWithContent(cpp_src, {}, &parser));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  EXPECT_NE(top_ptr, nullptr);
}

TEST_F(CCParserTest, ChannelWrite) {
  xlscc::CCParser parser;
  constexpr std::string_view cpp_src = R"(
    #pragma hls_top
    void foo(__xls_channel<int>& chan, int value) {
      chan.write(value);
      return;
    }
  )";
  XLS_ASSERT_OK(ScanTempFileWithContent(cpp_src, {}, &parser));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  EXPECT_NE(top_ptr, nullptr);
}

TEST_F(CCParserTest, MemoryRead) {
  xlscc::CCParser parser;
  constexpr std::string_view cpp_src = R"(
    #pragma hls_top
    int foo(const __xls_memory<int, 1024>& mem) {
      return mem[0] + mem.read(1);
    }
  )";
  XLS_ASSERT_OK(ScanTempFileWithContent(cpp_src, {}, &parser));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  EXPECT_NE(top_ptr, nullptr);
}

TEST_F(CCParserTest, MemoryWrite) {
  xlscc::CCParser parser;
  constexpr std::string_view cpp_src = R"(
    #pragma hls_top
    void foo(__xls_memory<int, 1024>& mem, int value) {
      mem[0] = value;
      mem.write(3, value);
      return;
    }
  )";
  XLS_ASSERT_OK(ScanTempFileWithContent(cpp_src, {}, &parser));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  EXPECT_NE(top_ptr, nullptr);
}

TEST_F(CCParserTest, TopClass) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    class ABlock {
     public:
      #pragma hls_top
      int foo(int a, int b) {
        const int foo = a + b;
        return foo;
      }
    };
  )";

  XLS_ASSERT_OK(ScanTempFileWithContent(cpp_src, {}, &parser,
                                        /*top_name=*/"my_package",
                                        /*top_class_name=*/"ABlock"));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  EXPECT_NE(top_ptr, nullptr);
}

TEST_F(CCParserTest, DesignUnknown) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #pragma hls_design unknown
    int foo(int a, int b) {
      const int foo = a + b;
      return foo;
    }
  )";

  XLS_ASSERT_OK(ScanTempFileWithContent(cpp_src, {}, &parser));
  absl::StatusOr<const clang::FunctionDecl*> top = parser.GetTopFunction();
  EXPECT_THAT(top, xls::status_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(CCParserTest, PragmasInDefines) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #define HLS_PRAGMA(x) _Pragma(x)
    HLS_PRAGMA("hls_top")
    int foo(int a, int b) {
      const int foo = a + b;
      return foo;
    }
  )";

  XLS_ASSERT_OK(ScanTempFileWithContent(cpp_src, {}, &parser));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  EXPECT_NE(top_ptr, nullptr);
}

TEST_F(CCParserTest, PragmasInDefinesHonorComments) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #define HLS_PRAGMA(x) _Pragma(x)
    HLS_PRAGMA(/* testing */"hls_top")
    int foo(int a, int b) {
      const int foo = a + b;
      return foo;
    }
  )";

  XLS_ASSERT_OK(ScanTempFileWithContent(cpp_src, {}, &parser));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  EXPECT_NE(top_ptr, nullptr);
}

TEST_F(CCParserTest, UnrollNoParameters) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    int bar(int (&a)[5], int b) {
      #pragma hls_unroll
      for (int i = 0; i < 5; ++i) a[i] = b;
      return true;
    }
  )";

  XLS_ASSERT_OK(
      ScanTempFileWithContent(cpp_src, {}, &parser, /*top_name=*/"bar"));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  ASSERT_NE(top_ptr, nullptr);

  auto* for_stmt = GetStmtInFunction<clang::ForStmt>(top_ptr);
  ASSERT_NE(for_stmt, nullptr);

  const clang::CallExpr* call = FindCallBefore(top_ptr, for_stmt);
  ASSERT_NE(call, nullptr);

  ExpectIntrinsicWithIntegerArg(call, "__xlscc_unroll", /*arg_expected=*/1);
}

TEST_F(CCParserTest, PragmaPipelineInitIntervalParameterMustBeNumber) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #pragma hls_top
    int foo(int a, int b) {
      int foo = a;
      #pragma hls_pipeline_init_interval test
      for(int i=0;i<2;++i) {
        foo += b;
      }
      return foo;
    }
  )";
  EXPECT_THAT(ScanTempFileWithContent(cpp_src, {}, &parser),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kFailedPrecondition,
                  testing::HasSubstr("Must be an integer >= 1.")));
}

TEST_F(CCParserTest, PragmaUnrollParametersMustBeNumberIfNotYesOrNo) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    int bar(int (&a)[5], int b) {
      #pragma hls_unroll test
      for (int i = 0; i < 5; ++i) a[i] = b;
      return true;
    }
  )";

  EXPECT_THAT(ScanTempFileWithContent(cpp_src, {}, &parser, /*top_name=*/"bar"),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kFailedPrecondition,
                  testing::HasSubstr("Must be 'yes', 'no', or an integer.")));
}

TEST_F(CCParserTest, TemplateArgsCanBeInferred) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    class x {
      public:
      template <typename T>
      T foo(T a) {
        return a;
      }
    };
    template <typename T>
    T bar(T a) {
      x x_inst;
      return x_inst.template foo(a);
    }
    int top(int a) {
      return bar(a);
    }
  )";

  XLS_EXPECT_OK(
      ScanTempFileWithContent(cpp_src, {}, &parser, /*top_name=*/"top"));
}

// TODO(seanhaskell): Enable once only annotations are used b/371085056
TEST_F(CCParserTest, DISABLED_PragmaInDefineAppliesOnlyInDefine) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #define some_macro(x) { \
        _Pragma("hls_unroll yes") \
          int i = 0;             \
          while (i < 2) {   \
            x[i] += 1;                                         \
            ++i;                             \
          }                                                       \
        }

    int bar(int (&a)[5], int b) {
      some_macro(a);
      some_macro(a);

      for (int i = 0; i < 5; ++i) a[i] = b;
      return true;
    }
  )";

  XLS_ASSERT_OK(
      ScanTempFileWithContent(cpp_src, {}, &parser, /*top_name=*/"bar"));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  ASSERT_NE(top_ptr, nullptr);

  auto* for_stmt = GetStmtInFunction<clang::ForStmt>(top_ptr);
  ASSERT_NE(for_stmt, nullptr);

  const clang::CallExpr* call = FindCallBefore(top_ptr, for_stmt);
  EXPECT_EQ(call, nullptr);
}

TEST_F(CCParserTest, ChannelStrictnessUnimplemented) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #define HLS_PRAGMA(x) _Pragma(x)
    HLS_PRAGMA("hls_top")
    int foo(int a, int b,
            #pragma hls_channel_strictness proven_mutually_exclusive
            __xls_channel<int>& chan) {
      const int foo = a + b;
      return foo;
    }
  )";

  EXPECT_THAT(
      ScanTempFileWithContent(cpp_src, {}, &parser, /*top_name=*/"bar"),
      xls::status_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(CCParserTest, DesignTooManyArgs) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #pragma hls_design top foo
    int foo(int a, int b) {
      const int foo = a + b;
      return foo;
    }
  )";

  EXPECT_THAT(
      ScanTempFileWithContent(cpp_src, {}, &parser),
      xls::status_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                    testing::HasSubstr("1 argument")));
}

TEST_F(CCParserTest, UnrollTooManyArgs) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #pragma hls_top
    int foo(int a, int b) {
      #pragma hls_unroll 1 2 3
      for(int i=0;i<4;++i) {
        a += b;
      }
      return a;
    }
  )";

  EXPECT_THAT(
      ScanTempFileWithContent(cpp_src, {}, &parser),
      xls::status_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                    testing::HasSubstr("1 argument")));
}

}  // namespace
