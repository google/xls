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

#ifndef XLS_DSLX_CPP_AST_BUILTIN_TYPES_INC_
#define XLS_DSLX_CPP_AST_BUILTIN_TYPES_INC_

// Form is:
//   X(enum_value, cap_ident, lang_str, signed, width)
#define XLS_DSLX_BUILTIN_TYPE_EACH(X)                   \
  X(kBool, BOOL, "bool", false, 1)                      \
  /* unsigned */                                        \
  X(kU1, U1, "u1", false, 1)                            \
  X(kU2, U2, "u2", false, 2)                            \
  X(kU3, U3, "u3", false, 3)                            \
  X(kU4, U4, "u4", false, 4)                            \
  X(kU5, U5, "u5", false, 5)                            \
  X(kU6, U6, "u6", false, 6)                            \
  X(kU7, U7, "u7", false, 7)                            \
  X(kU8, U8, "u8", false, 8)                            \
  X(kU9, U9, "u9", false, 9)                            \
  X(kU10, U10, "u10", false, 10)                        \
  X(kU11, U11, "u11", false, 11)                        \
  X(kU12, U12, "u12", false, 12)                        \
  X(kU13, U13, "u13", false, 13)                        \
  X(kU14, U14, "u14", false, 14)                        \
  X(kU15, U15, "u15", false, 15)                        \
  X(kU16, U16, "u16", false, 16)                        \
  X(kU17, U17, "u17", false, 17)                        \
  X(kU18, U18, "u18", false, 18)                        \
  X(kU19, U19, "u19", false, 19)                        \
  X(kU20, U20, "u20", false, 20)                        \
  X(kU21, U21, "u21", false, 21)                        \
  X(kU22, U22, "u22", false, 22)                        \
  X(kU23, U23, "u23", false, 23)                        \
  X(kU24, U24, "u24", false, 24)                        \
  X(kU25, U25, "u25", false, 25)                        \
  X(kU26, U26, "u26", false, 26)                        \
  X(kU27, U27, "u27", false, 27)                        \
  X(kU28, U28, "u28", false, 28)                        \
  X(kU29, U29, "u29", false, 29)                        \
  X(kU30, U30, "u30", false, 30)                        \
  X(kU31, U31, "u31", false, 31)                        \
  X(kU32, U32, "u32", false, 32)                        \
  X(kU33, U33, "u33", false, 33)                        \
  X(kU34, U34, "u34", false, 34)                        \
  X(kU35, U35, "u35", false, 35)                        \
  X(kU36, U36, "u36", false, 36)                        \
  X(kU37, U37, "u37", false, 37)                        \
  X(kU38, U38, "u38", false, 38)                        \
  X(kU39, U39, "u39", false, 39)                        \
  X(kU40, U40, "u40", false, 40)                        \
  X(kU41, U41, "u41", false, 41)                        \
  X(kU42, U42, "u42", false, 42)                        \
  X(kU43, U43, "u43", false, 43)                        \
  X(kU44, U44, "u44", false, 44)                        \
  X(kU45, U45, "u45", false, 45)                        \
  X(kU46, U46, "u46", false, 46)                        \
  X(kU47, U47, "u47", false, 47)                        \
  X(kU48, U48, "u48", false, 48)                        \
  X(kU49, U49, "u49", false, 49)                        \
  X(kU50, U50, "u50", false, 50)                        \
  X(kU51, U51, "u51", false, 51)                        \
  X(kU52, U52, "u52", false, 52)                        \
  X(kU53, U53, "u53", false, 53)                        \
  X(kU54, U54, "u54", false, 54)                        \
  X(kU55, U55, "u55", false, 55)                        \
  X(kU56, U56, "u56", false, 56)                        \
  X(kU57, U57, "u57", false, 57)                        \
  X(kU58, U58, "u58", false, 58)                        \
  X(kU59, U59, "u59", false, 59)                        \
  X(kU60, U60, "u60", false, 60)                        \
  X(kU61, U61, "u61", false, 61)                        \
  X(kU62, U62, "u62", false, 62)                        \
  X(kU63, U63, "u63", false, 63)                        \
  X(kU64, U64, "u64", false, 64)                        \
  /* signed */                                          \
  X(kS1, S1, "s1", true, 1)                             \
  X(kS2, S2, "s2", true, 2)                             \
  X(kS3, S3, "s3", true, 3)                             \
  X(kS4, S4, "s4", true, 4)                             \
  X(kS5, S5, "s5", true, 5)                             \
  X(kS6, S6, "s6", true, 6)                             \
  X(kS7, S7, "s7", true, 7)                             \
  X(kS8, S8, "s8", true, 8)                             \
  X(kS9, S9, "s9", true, 9)                             \
  X(kS10, S10, "s10", true, 10)                         \
  X(kS11, S11, "s11", true, 11)                         \
  X(kS12, S12, "s12", true, 12)                         \
  X(kS13, S13, "s13", true, 13)                         \
  X(kS14, S14, "s14", true, 14)                         \
  X(kS15, S15, "s15", true, 15)                         \
  X(kS16, S16, "s16", true, 16)                         \
  X(kS17, S17, "s17", true, 17)                         \
  X(kS18, S18, "s18", true, 18)                         \
  X(kS19, S19, "s19", true, 19)                         \
  X(kS20, S20, "s20", true, 20)                         \
  X(kS21, S21, "s21", true, 21)                         \
  X(kS22, S22, "s22", true, 22)                         \
  X(kS23, S23, "s23", true, 23)                         \
  X(kS24, S24, "s24", true, 24)                         \
  X(kS25, S25, "s25", true, 25)                         \
  X(kS26, S26, "s26", true, 26)                         \
  X(kS27, S27, "s27", true, 27)                         \
  X(kS28, S28, "s28", true, 28)                         \
  X(kS29, S29, "s29", true, 29)                         \
  X(kS30, S30, "s30", true, 30)                         \
  X(kS31, S31, "s31", true, 31)                         \
  X(kS32, S32, "s32", true, 32)                         \
  X(kS33, S33, "s33", true, 33)                         \
  X(kS34, S34, "s34", true, 34)                         \
  X(kS35, S35, "s35", true, 35)                         \
  X(kS36, S36, "s36", true, 36)                         \
  X(kS37, S37, "s37", true, 37)                         \
  X(kS38, S38, "s38", true, 38)                         \
  X(kS39, S39, "s39", true, 39)                         \
  X(kS40, S40, "s40", true, 40)                         \
  X(kS41, S41, "s41", true, 41)                         \
  X(kS42, S42, "s42", true, 42)                         \
  X(kS43, S43, "s43", true, 43)                         \
  X(kS44, S44, "s44", true, 44)                         \
  X(kS45, S45, "s45", true, 45)                         \
  X(kS46, S46, "s46", true, 46)                         \
  X(kS47, S47, "s47", true, 47)                         \
  X(kS48, S48, "s48", true, 48)                         \
  X(kS49, S49, "s49", true, 49)                         \
  X(kS50, S50, "s50", true, 50)                         \
  X(kS51, S51, "s51", true, 51)                         \
  X(kS52, S52, "s52", true, 52)                         \
  X(kS53, S53, "s53", true, 53)                         \
  X(kS54, S54, "s54", true, 54)                         \
  X(kS55, S55, "s55", true, 55)                         \
  X(kS56, S56, "s56", true, 56)                         \
  X(kS57, S57, "s57", true, 57)                         \
  X(kS58, S58, "s58", true, 58)                         \
  X(kS59, S59, "s59", true, 59)                         \
  X(kS60, S60, "s60", true, 60)                         \
  X(kS61, S61, "s61", true, 61)                         \
  X(kS62, S62, "s62", true, 62)                         \
  X(kS63, S63, "s63", true, 63)                         \
  X(kS64, S64, "s64", true, 64)                         \
  /* keywords used to construct arbitrary sizes */      \
  X(kBits, BITS, "bits", false, 0)                      \
  X(kToken, TOKEN, "token", std::nullopt, 0)            \
  X(kUN, UN, "uN", false, 0)                            \
  X(kSN, SN, "sN", true, 0)                             \
  X(kXN, XN, "xN", std::nullopt, 0)                     \
  /* other */                                           \
  X(kChannelIn, CHANNEL_IN, "chan in", std::nullopt, 0) \
  X(kChannelOut, CHANNEL_OUT, "chan out", std::nullopt, 0)
/* done */

#endif  // XLS_DSLX_CPP_AST_BUILTIN_TYPES_INC_
