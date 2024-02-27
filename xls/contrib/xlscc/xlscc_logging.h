// Copyright 2022 The XLS Authors
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

#ifndef XLSCC_LOGGING_H_
#define XLSCC_LOGGING_H_

#define XLSCC_CHECK(condition, loc) \
  XLS_CHECK(condition) << ErrorMessage(loc, "")
#define XLSCC_QCHECK(condition, loc) QCHECK(condition) << ErrorMessage(loc, "")
#define XLSCC_DCHECK(condition, loc) DCHECK(condition) << ErrorMessage(loc, "")

#define XLSCC_CHECK_EQ(val1, val2, loc) \
  XLS_CHECK_EQ(val1, val2) << ErrorMessage(loc, "")
#define XLSCC_CHECK_NE(val1, val2, loc) \
  XLS_CHECK_NE(val1, val2) << ErrorMessage(loc, "")
#define XLSCC_CHECK_LE(val1, val2, loc) \
  XLS_CHECK_LE(val1, val2) << ErrorMessage(loc, "")
#define XLSCC_CHECK_LT(val1, val2, loc) \
  XLS_CHECK_LT(val1, val2) << ErrorMessage(loc, "")
#define XLSCC_CHECK_GE(val1, val2, loc) \
  XLS_CHECK_GE(val1, val2) << ErrorMessage(loc, "")
#define XLSCC_CHECK_GT(val1, val2, loc) \
  XLS_CHECK_GT(val1, val2) << ErrorMessage(loc, "")
#define XLSCC_QCHECK_EQ(val1, val2, loc) \
  QCHECK_EQ(val1, val2) << ErrorMessage(loc, "")
#define XLSCC_QCHECK_NE(val1, val2, loc) \
  QCHECK_NE(val1, val2) << ErrorMessage(loc, "")
#define XLSCC_QCHECK_LE(val1, val2, loc) \
  QCHECK_LE(val1, val2) << ErrorMessage(loc, "")
#define XLSCC_QCHECK_LT(val1, val2, loc) \
  QCHECK_LT(val1, val2) << ErrorMessage(loc, "")
#define XLSCC_QCHECK_GE(val1, val2, loc) \
  QCHECK_GE(val1, val2) << ErrorMessage(loc, "")
#define XLSCC_QCHECK_GT(val1, val2, loc) \
  QCHECK_GT(val1, val2) << ErrorMessage(loc, "")

#define XLSCC_DCHECK_EQ(val1, val2, loc) \
  XLSCC_CHECK_EQ(val1, val2) << ErrorMessage(loc, "")
#define XLSCC_DCHECK_NE(val1, val2, loc) \
  XLSCC_CHECK_NE(val1, val2) << ErrorMessage(loc, "")
#define XLSCC_DCHECK_LE(val1, val2, loc) \
  XLSCC_CHECK_LE(val1, val2) << ErrorMessage(loc, "")
#define XLSCC_DCHECK_LT(val1, val2, loc) \
  XLSCC_CHECK_LT(val1, val2) << ErrorMessage(loc, "")
#define XLSCC_DCHECK_GE(val1, val2, loc) \
  XLSCC_CHECK_GE(val1, val2) << ErrorMessage(loc, "")
#define XLSCC_DCHECK_GT(val1, val2, loc) \
  XLSCC_CHECK_GT(val1, val2) << ErrorMessage(loc, "")

#define XLSCC_CHECK_OK(val, loc) XLS_CHECK_EQ(val) << ErrorMessage(loc, "")
#define XLSCC_QCHECK_OK(val, loc) QCHECK_EQ(val) << ErrorMessage(loc, "")
#define XLSCC_DCHECK_OK(val, loc) DCHECK_EQ(val) << ErrorMessage(loc), ""

#endif  // XLSCC_COMMON_LOGGING_LOGGING_H_
