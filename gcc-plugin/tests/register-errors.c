/* GCC-StarPU
   Copyright (C) 2011 Institut National de Recherche en Informatique et Automatique

   GCC-StarPU is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   GCC-StarPU is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GCC-StarPU.  If not, see <http://www.gnu.org/licenses/>.  */

/* Test error handling for `#pragma starpu register ...'.  */

#undef NDEBUG

#include <lib.h>

int
main (int argc, char *argv[])
{
#pragma starpu initialize

#pragma starpu register /* (error "unterminated") */

#pragma starpu register argv 234 junk right here /* (error "junk after") */

  static int x[123] __attribute__ ((unused));
#pragma starpu register x 234 /* (note "can be omitted") *//* (error "differs from actual size") */

  size_t x_size __attribute__ ((unused)) = sizeof x / sizeof x[0];
#pragma starpu register x x_size /* (note "can be omitted") *//* (error "known at compile-time") */

#pragma starpu register does_not_exit 123  /* (error "unbound variable") */
#pragma starpu register argv does_not_exit /* (error "unbound variable") */

#pragma starpu register argv /* (error "cannot determine size") */

#pragma starpu register argc /* (error "neither a pointer nor an array") */

  return EXIT_SUCCESS;
}
