// Copyright 2025 The XLS Authors
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

#include "clang/include/clang/AST/Decl.h"
#include "clang/include/clang/AST/Stmt.h"

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
