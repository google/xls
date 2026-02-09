#ifndef GXL_PARSER_H
#define GXL_PARSER_H

#include "graph.h"
#include "tinyxml2.h"
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

XLSGraph parse_gxl(const std::string &filename);
bool export_gxl(const XLSGraph &graph, const std::string &filename);

#endif // GXL_PARSER_H
