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
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "clang/include/clang/AST/Attr.h"
#include "clang/include/clang/AST/AttrIterator.h"
#include "clang/include/clang/AST/Attrs.inc"
#include "clang/include/clang/AST/Decl.h"
#include "clang/include/clang/AST/Expr.h"
#include "clang/include/clang/AST/Stmt.h"
#include "clang/include/clang/Basic/LLVM.h"
#include "llvm/include/llvm/Support/Casting.h"
#include "xls/common/status/matchers.h"
#include "xls/contrib/xlscc/metadata_output.pb.h"
#include "xls/contrib/xlscc/unit_tests/unit_test.h"
#include "xls/ir/channel.h"
#include "xls/ir/source_location.h"

namespace {

class CCParserTest : public XlsccTestBase {
 public:
};

const clang::AnnotateAttr* GetLastAnnotation(
    const clang::ArrayRef<const clang::Attr*> attrs,
    llvm::StringRef annotation) {
  const clang::AnnotateAttr* last_annotation = nullptr;
  for (auto it = attrs.rbegin(); it != attrs.rend(); ++it) {
    const clang::Attr* attr = *it;
    if (const clang::AnnotateAttr* annotate =
            llvm::dyn_cast<clang::AnnotateAttr>(attr);
        annotate != nullptr && annotate->getAnnotation() == annotation) {
      last_annotation = annotate;
      break;
    }
  }
  return last_annotation;
}

void ExpectAnnotateWithoutArgs(const clang::ArrayRef<const clang::Attr*> attrs,
                               llvm::StringRef annotation) {
  const clang::AnnotateAttr* last_annotation =
      GetLastAnnotation(attrs, annotation);
  ASSERT_NE(last_annotation, nullptr);
  EXPECT_EQ(last_annotation->args_size(), 0);
}

void ExpectAnnotateWithIntegerArg(
    const clang::ArrayRef<const clang::Attr*> attrs, llvm::StringRef annotation,
    int64_t arg_expected) {
  const clang::AnnotateAttr* last_annotation =
      GetLastAnnotation(attrs, annotation);
  ASSERT_NE(last_annotation, nullptr);
  ASSERT_GE(last_annotation->args_size(), 1);
  EXPECT_EQ(last_annotation->args_size(), 1);
  const clang::Expr* argument = *last_annotation->args_begin();
  const clang::IntegerLiteral* literal_argument =
      llvm::dyn_cast<clang::IntegerLiteral>(argument);
  ASSERT_NE(literal_argument, nullptr);
  ASSERT_LE(literal_argument->getValue().getSignificantBits(), 64);
  EXPECT_EQ(literal_argument->getValue().getSExtValue(), arg_expected);
}

template <typename ClangT>
const ClangT* GetStmtInCompoundStmt(const clang::CompoundStmt* stmt) {
  for (const clang::Stmt* body_st : stmt->children()) {
    if (const clang::LabelStmt* label =
            clang::dyn_cast<clang::LabelStmt>(body_st);
        label != nullptr) {
      body_st = label->getSubStmt();
    }

    const ClangT* ret;
    if (const clang::CompoundStmt* cmpnd_stmt =
            clang::dyn_cast<clang::CompoundStmt>(body_st);
        cmpnd_stmt != nullptr) {
      ret = GetStmtInCompoundStmt<ClangT>(cmpnd_stmt);
    } else {
      ret = clang::dyn_cast<const ClangT>(body_st);
    }
    if (ret != nullptr) {
      return ret;
    }
  }

  return nullptr;
}

template <typename ClangT>
const ClangT* GetStmtInFunction(const clang::FunctionDecl* func) {
  const clang::Stmt* body = func->getBody();
  if (body == nullptr) {
    return nullptr;
  }

  for (const clang::Stmt* body_st : body->children()) {
    if (const clang::LabelStmt* label =
            clang::dyn_cast<clang::LabelStmt>(body_st);
        label != nullptr) {
      body_st = label->getSubStmt();
    }

    const ClangT* ret;
    if (const clang::CompoundStmt* cmpnd_stmt =
            clang::dyn_cast<clang::CompoundStmt>(body_st);
        cmpnd_stmt != nullptr) {
      ret = GetStmtInCompoundStmt<ClangT>(cmpnd_stmt);
    } else {
      ret = clang::dyn_cast<const ClangT>(body_st);
    }
    if (ret != nullptr) {
      return ret;
    }
  }

  return nullptr;
}

template <typename ClangT>
const clang::AttributedStmt* GetAttributedStmtInCompoundStmt(
    const clang::CompoundStmt* stmt) {
  for (const clang::Stmt* body_st : stmt->children()) {
    if (const clang::LabelStmt* label =
            clang::dyn_cast<clang::LabelStmt>(body_st);
        label != nullptr) {
      body_st = label->getSubStmt();
    }
    if (const clang::CompoundStmt* cmpnd_stmt =
            clang::dyn_cast<clang::CompoundStmt>(body_st);
        cmpnd_stmt != nullptr) {
      const clang::AttributedStmt* ret =
          GetAttributedStmtInCompoundStmt<ClangT>(cmpnd_stmt);
      if (ret == nullptr) {
        continue;
      }
      return ret;
    }
    const clang::AttributedStmt* attributed =
        clang::dyn_cast<clang::AttributedStmt>(body_st);
    if (attributed == nullptr) {
      continue;
    }
    if (llvm::isa<ClangT>(attributed->getSubStmt())) {
      return attributed;
    }
  }

  return nullptr;
}

template <typename ClangT>
const clang::AttributedStmt* GetAttributedStmtInFunction(
    const clang::FunctionDecl* func) {
  const clang::Stmt* body = func->getBody();
  if (body == nullptr) {
    return nullptr;
  }

  for (const clang::Stmt* body_st : body->children()) {
    if (const clang::LabelStmt* label =
            clang::dyn_cast<clang::LabelStmt>(body_st);
        label != nullptr) {
      body_st = label->getSubStmt();
    }
    if (const clang::CompoundStmt* cmpnd_stmt =
            clang::dyn_cast<clang::CompoundStmt>(body_st);
        cmpnd_stmt != nullptr) {
      const clang::AttributedStmt* ret =
          GetAttributedStmtInCompoundStmt<ClangT>(cmpnd_stmt);
      if (ret == nullptr) {
        continue;
      }
      return ret;
    }
    const clang::AttributedStmt* attributed =
        clang::dyn_cast<clang::AttributedStmt>(body_st);
    if (attributed == nullptr) {
      continue;
    }
    if (llvm::isa<ClangT>(attributed->getSubStmt())) {
      return attributed;
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
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(CCParserTest, Block) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #pragma hls_block
    int bar(int a, int b) {
      const int foo = a + b;
      return foo+1;
    }
  )";

  XLS_ASSERT_OK(
      ScanTempFileWithContent(cpp_src, {}, &parser, /*top_name=*/"bar"));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  ASSERT_NE(top_ptr, nullptr);
  EXPECT_EQ(top_ptr->getNameAsString(), "bar");
  top_ptr->specific_attrs<clang::AnnotateAttr>();

  const clang::AttrVec& attrs = top_ptr->getAttrs();
  ASSERT_EQ(attrs.size(), 1);
  const clang::Attr* attr = attrs.data()[0];
  const clang::AnnotateAttr* annotate =
      llvm::dyn_cast<clang::AnnotateAttr>(attr);
  ASSERT_NE(annotate, nullptr);
  EXPECT_EQ(annotate->getAnnotation(), "hls_block");
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

  const clang::AttributedStmt* attributed =
      GetAttributedStmtInFunction<clang::ForStmt>(top_ptr);
  ASSERT_NE(attributed, nullptr);
  ExpectAnnotateWithIntegerArg(attributed->getAttrs(),
                               "hls_pipeline_init_interval", 3);
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

  const clang::AttributedStmt* attributed =
      GetAttributedStmtInFunction<clang::ForStmt>(top_ptr);
  ASSERT_NE(attributed, nullptr);
  ExpectAnnotateWithIntegerArg(attributed->getAttrs(),
                               "hls_pipeline_init_interval", 3);
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

  EXPECT_THAT(ScanTempFileWithContent(cpp_src, {}, &parser),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
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

  EXPECT_THAT(ScanTempFileWithContent(cpp_src, {}, &parser),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
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

  EXPECT_THAT(ScanTempFileWithContent(cpp_src, {}, &parser),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
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

  const clang::AttributedStmt* attributed =
      GetAttributedStmtInFunction<clang::ForStmt>(top_ptr);
  ASSERT_NE(attributed, nullptr);
  ExpectAnnotateWithoutArgs(attributed->getAttrs(), "hls_unroll");
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

  const clang::AttributedStmt* attributed =
      GetAttributedStmtInFunction<clang::ForStmt>(top_ptr);
  ASSERT_NE(attributed, nullptr);
  ExpectAnnotateWithIntegerArg(attributed->getAttrs(), "hls_unroll",
                               /*arg_expected=*/2);
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

  const clang::AttributedStmt* attributed =
      GetAttributedStmtInFunction<clang::ForStmt>(top_ptr);
  ASSERT_NE(attributed, nullptr);
  ExpectAnnotateWithIntegerArg(attributed->getAttrs(), "hls_unroll",
                               /*arg_expected=*/2);
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

  const clang::AttributedStmt* attributed =
      GetAttributedStmtInFunction<clang::ForStmt>(top_ptr);
  ASSERT_NE(attributed, nullptr);
  ExpectAnnotateWithIntegerArg(attributed->getAttrs(), "hls_unroll",
                               /*arg_expected=*/2);
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

  const clang::AttributedStmt* attributed =
      GetAttributedStmtInFunction<clang::ForStmt>(top_ptr);
  ASSERT_NE(attributed, nullptr);
  ExpectAnnotateWithIntegerArg(attributed->getAttrs(), "hls_unroll",
                               /*arg_expected=*/0);
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

  const clang::AttributedStmt* attributed =
      GetAttributedStmtInFunction<clang::ForStmt>(top_ptr);
  ASSERT_EQ(attributed, nullptr);
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

  EXPECT_THAT(ScanTempFileWithContent(cpp_src, {}, &parser, /*top_name=*/"bar"),
              absl_testing::StatusIs(absl::StatusCode::kAlreadyExists,
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

  EXPECT_THAT(ScanTempFileWithContent(cpp_src, {}, &parser),
              absl_testing::StatusIs(absl::StatusCode::kAlreadyExists,
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
  EXPECT_THAT(top, absl_testing::StatusIs(absl::StatusCode::kNotFound));
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
  EXPECT_THAT(
      ScanTempFileWithContent(cpp_src, {}, &parser),
      absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                             testing::HasSubstr("must be an integer >= 1")));
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
              absl_testing::StatusIs(
                  absl::StatusCode::kFailedPrecondition,
                  testing::HasSubstr("must be 'yes', 'no', or an integer.")));
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

TEST_F(CCParserTest, PragmaInDefineAppliesOnlyInDefine) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #define some_macro(x) {          \
          int i = 0;                 \
          _Pragma("hls_unroll yes")  \
          while (i < 2) {            \
            x[i] += 1;               \
            ++i;                     \
          }                          \
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

  EXPECT_EQ(GetStmtInFunction<clang::WhileStmt>(top_ptr), nullptr);

  const clang::AttributedStmt* attributed_while =
      GetAttributedStmtInFunction<clang::WhileStmt>(top_ptr);
  EXPECT_NE(attributed_while, nullptr);
  if (attributed_while != nullptr) {
    ExpectAnnotateWithoutArgs(attributed_while->getAttrs(), "hls_unroll");
  }

  auto* for_stmt = GetStmtInFunction<clang::ForStmt>(top_ptr);
  ASSERT_NE(for_stmt, nullptr);
}

TEST_F(CCParserTest, ChannelStrictnessWorks) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    #define HLS_PRAGMA(x) _Pragma(x)
    HLS_PRAGMA("hls_top")
    int foo(int a, int b,
            [[xlscc::hls_channel_strictness(runtime_mutually_exclusive)]]
            __xls_channel<int>& chan,
            [[xlscc::hls_channel_strictness(arbitrary_static_order)]]
            __xls_channel<int>& chan2) {
      const int foo = a + b;
      return foo;
    }
  )";

  XLS_ASSERT_OK(
      ScanTempFileWithContent(cpp_src, {}, &parser, /*top_name=*/"bar"));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_ptr, parser.GetTopFunction());
  ASSERT_NE(top_ptr, nullptr);

  const clang::ParmVarDecl* chan_decl = top_ptr->getParamDecl(2);
  ASSERT_NE(chan_decl, nullptr);
  ASSERT_EQ(chan_decl->getName(), "chan");

  ExpectAnnotateWithIntegerArg(
      chan_decl->getAttrs(), "hls_channel_strictness",
      /*arg_expected=*/
      static_cast<int64_t>(xls::ChannelStrictness::kRuntimeMutuallyExclusive));

  const clang::ParmVarDecl* chan2_decl = top_ptr->getParamDecl(3);
  ASSERT_NE(chan2_decl, nullptr);
  ASSERT_EQ(chan2_decl->getName(), "chan2");

  ExpectAnnotateWithIntegerArg(
      chan2_decl->getAttrs(), "hls_channel_strictness",
      /*arg_expected=*/
      static_cast<int64_t>(xls::ChannelStrictness::kArbitraryStaticOrder));
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

  EXPECT_THAT(ScanTempFileWithContent(cpp_src, {}, &parser),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                     testing::HasSubstr("1 argument")));
}

}  // namespace
