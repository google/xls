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

#include "xls/contrib/mlir/util/interpreter.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/include/llvm/ADT/StringRef.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/MLIRContext.h"
#include "mlir/include/mlir/IR/OwningOpRef.h"
#include "mlir/include/mlir/Parser/Parser.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"

namespace mlir::xls {
namespace {
using ::testing::Eq;
using ::testing::HasSubstr;
using ::xls::status_testing::IsOkAndHolds;
using ::xls::status_testing::StatusIs;

class TestInterpreterContext
    : public InterpreterContext<TestInterpreterContext, int64_t> {
 public:
  absl::StatusOr<bool> GetValueAsPred(int64_t value) { return value; }
  absl::StatusOr<int64_t> GetValueAsSwitchCase(int64_t value) { return value; }
};

class TestInterpreter
    : public Interpreter<TestInterpreter, TestInterpreterContext, int64_t,
                         mlir::arith::AddIOp, mlir::arith::ConstantOp,
                         mlir::arith::CmpIOp> {
 public:
  using Interpreter::Interpret;

  absl::Status Interpret(mlir::arith::AddIOp op, TestInterpreterContext& ctx) {
    XLS_ASSIGN_OR_RETURN(std::vector<int64_t> operands,
                         ctx.Get(op.getOperands()));
    ctx.Set(op.getResult(), operands[0] + operands[1]);
    return absl::OkStatus();
  }

  absl::Status Interpret(mlir::arith::CmpIOp op, TestInterpreterContext& ctx) {
    XLS_ASSIGN_OR_RETURN(std::vector<int64_t> operands,
                         ctx.Get(op.getOperands()));
    ctx.Set(op.getResult(), operands[0] < operands[1] ? 1 : 0);
    return absl::OkStatus();
  }

  absl::Status Interpret(mlir::arith::ConstantOp op,
                         TestInterpreterContext& ctx) {
    ctx.Set(op.getResult(), 1);
    return absl::OkStatus();
  }
};

absl::StatusOr<int64_t> RunBinaryFunc(llvm::StringRef module_str, int64_t a,
                                      int64_t b, TestInterpreter interpreter) {
  mlir::MLIRContext ctx;
  ctx.loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(module_str, &ctx);
  if (!module) {
    return absl::InternalError("Failed to parse module");
  }
  mlir::func::FuncOp host_func =
      module->lookupSymbol<mlir::func::FuncOp>("main");

  XLS_ASSIGN_OR_RETURN(std::vector<int64_t> results,
                       interpreter.InterpretFunc(host_func, {a, b}));
  return results[0];
}

TEST(InterpreterTest, Simple) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<1xi32> {
      %0 = arith.addi %arg0, %arg1 : tensor<1xi32>
      func.return %0 : tensor<1xi32>
    }
  })";
  ASSERT_THAT(RunBinaryFunc(kModule, 4, 5, TestInterpreter()),
              IsOkAndHolds(Eq(9)));
}

TEST(InterpreterTest, Call) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<1xi32> {
      %0 = func.call @f(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
      func.return %0 : tensor<1xi32>
    }
    func.func @f(%arg0: tensor<1xi32>) -> tensor<1xi32> {
      %0 = arith.addi %arg0, %arg0 : tensor<1xi32>
      func.return %0 :tensor<1xi32>
    }
  })";
  ASSERT_THAT(RunBinaryFunc(kModule, 4, 5, TestInterpreter()),
              IsOkAndHolds(Eq(8)));
}

class InPlaceTestInterpreterContext
    : public InterpreterContext<TestInterpreterContext,
                                std::shared_ptr<int64_t>> {};

class InPlaceTestInterpreter
    : public Interpreter<InPlaceTestInterpreter, InPlaceTestInterpreterContext,
                         std::shared_ptr<int64_t>, mlir::arith::AddIOp,
                         mlir::arith::SubIOp> {
 public:
  using Interpreter::Interpret;

  absl::Status Interpret(mlir::arith::AddIOp op,
                         InPlaceTestInterpreterContext& ctx) {
    XLS_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<int64_t>> operands,
                         ctx.Get(op.getOperands()));
    int64_t c = *operands[0] + *operands[1];

    ctx.Set(op.getResult(), std::make_shared<int64_t>(c));
    return absl::OkStatus();
  }

  absl::Status Interpret(mlir::arith::SubIOp op,
                         InPlaceTestInterpreterContext& ctx) {
    XLS_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<int64_t>> operands,
                         ctx.Get(op.getOperands()));
    if (!ctx.InputMayBeReused(op.getOperand(0), op.getOperation())) {
      return absl::InternalError("InputMayBeReused");
    }
    *operands[0] -= *operands[1];

    ctx.Set(op.getResult(), operands[0]);
    return absl::OkStatus();
  }
};

absl::StatusOr<int64_t> RunBinaryFunc(llvm::StringRef module_str, int64_t a,
                                      int64_t b,
                                      InPlaceTestInterpreter interpreter) {
  mlir::MLIRContext ctx;
  ctx.loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(module_str, &ctx);
  if (!module) {
    return absl::InternalError("Failed to parse module");
  }
  mlir::func::FuncOp host_func =
      module->lookupSymbol<mlir::func::FuncOp>("main");

  XLS_ASSIGN_OR_RETURN(
      std::vector<std::shared_ptr<int64_t>> results,
      interpreter.InterpretFunc(host_func, {std::make_shared<int64_t>(a),
                                            std::make_shared<int64_t>(b)}));
  return *results[0];
}

TEST(InterpreterTest, InPlaceReusable) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<1xi32> {
      %0 = arith.subi %arg0, %arg1 : tensor<1xi32>
      func.return %0 : tensor<1xi32>
    }
  })";
  ASSERT_THAT(RunBinaryFunc(kModule, 4, 5, InPlaceTestInterpreter()),
              IsOkAndHolds(Eq(-1)));
}

TEST(InterpreterTest, InPlaceNotReusable) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<1xi32> {
      %0 = arith.subi %arg0, %arg1 : tensor<1xi32>
      %1 = arith.addi %0, %arg0 : tensor<1xi32>
      func.return %1 : tensor<1xi32>
    }
  })";
  ASSERT_THAT(
      RunBinaryFunc(kModule, 4, 5, InPlaceTestInterpreter()),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("InputMayBeReused")));
}

}  // namespace

}  // namespace mlir::xls
