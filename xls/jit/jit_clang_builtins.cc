// Copyright 2024 The XLS Authors
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

#include "xls/jit/jit_clang_builtins.h"

#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/include/llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/CoreContainers.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/include/llvm/IR/DataLayout.h"
#include "llvm/include/llvm/Support/Error.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"

namespace xls {

// TODO(allight): We could probably pull this stuff from headers but this is
// frankly easier.
extern "C" {
__int128 __divti3(__int128, __int128);
__int128 __modti3(__int128, __int128);
unsigned __int128 __udivti3(unsigned __int128, unsigned __int128);
unsigned __int128 __umodti3(unsigned __int128, unsigned __int128);
}

namespace {
template <typename T>
absl::StatusOr<llvm::orc::ExecutorSymbolDef> Sym(T from_ptr) {
  XLS_RET_CHECK(from_ptr != nullptr);
  return llvm::orc::ExecutorSymbolDef(
      llvm::orc::ExecutorAddr::fromPtr(from_ptr),
      llvm::JITSymbolFlags::Callable);
}
}  // namespace

absl::Status AddCompilerRtSymbols(llvm::orc::JITDylib& lib,
                                  const llvm::DataLayout& layout) {
  llvm::orc::SymbolMap rt_symbols;
  llvm::orc::ExecutionSession& es = lib.getExecutionSession();
  llvm::orc::MangleAndInterner intern(es, layout);
  // TODO(allight): Is there a better way to do this? Technically just
  // hard-referencing them is sufficient.
  XLS_ASSIGN_OR_RETURN(rt_symbols[intern("__divti3")], Sym(__divti3),
                       _ << "No __divti3");
  XLS_ASSIGN_OR_RETURN(rt_symbols[intern("__udivti3")], Sym(__udivti3),
                       _ << "No __udivti3");
  XLS_ASSIGN_OR_RETURN(rt_symbols[intern("__modti3")], Sym(__modti3),
                       _ << "No __modti3");
  XLS_ASSIGN_OR_RETURN(rt_symbols[intern("__umodti3")], Sym(__umodti3),
                       _ << "No __umodti3");
  llvm::Error failed = lib.define(llvm::orc::absoluteSymbols(rt_symbols));
  if (!failed) {
    return absl::OkStatus();
  }
  return absl::InternalError(
      absl::StrCat("LLVM Failed to define compiler-rt symbols: ",
                   llvm::toString(std::move(failed))));
}
}  // namespace xls
