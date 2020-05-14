// Copyright 2020 Google LLC
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

// Verifies we can compile and link something that depends
// on ncurses, and runs a simple ncurses program.

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <curses.h>
#include <string>

static char msg[] = " --- ncurses_unittest: PASS --- ";

static void test_ncurses() {
  char *term = getenv("TERM");
  if (term == NULL || !strcmp(term, "unknown")) {
    setenv("TERM", "vt100", 1);
  }
  initscr();
  noecho();
  cbreak();
  nonl();
  curs_set(0);

  int c = (COLS - sizeof(msg) -1) / 2;
  move(LINES/2, c >= 0 ? c : 0);

  if (has_colors())
  {
    start_color();
    init_pair(1, COLOR_RED,     COLOR_BLACK);
    init_pair(2, COLOR_GREEN,   COLOR_BLACK);
    init_pair(3, COLOR_YELLOW,  COLOR_BLACK);
    init_pair(4, COLOR_BLUE,    COLOR_BLACK);
    init_pair(5, COLOR_CYAN,    COLOR_BLACK);
    init_pair(6, COLOR_MAGENTA, COLOR_BLACK);
    init_pair(7, COLOR_WHITE,   COLOR_BLACK);
  }

  for (int n = 0; n < sizeof(msg)-1; n++)
  {
    addch(msg[n]);
    attrset(COLOR_PAIR(n % 8));
    refresh();
    usleep(10000);
  }

  endwin();
}

int main(int argc, char **argv) {
  test_ncurses();
  printf("PASS\\n");
  return 0;
}
