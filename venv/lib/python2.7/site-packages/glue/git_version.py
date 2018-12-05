# This file is part of the Grid LSC User Environment (GLUE)
#
# GLUE is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.

version = '1.60.0'

# git information
id = '30db5c2a8b324ae53500aed512db7220fd4e7f4b'
date = '2018-09-17 20:15:47 +0000'
branch = 'glue-1-60-branch'
tag = 'glue-release-1.60.0'
author = 'Ryan Fisher <ryan.fisher@ligo.org>'
builder = 'Ryan Fisher <ryan.fisher@ligo.org>'
committer = 'Ryan Fisher <ryan.fisher@ligo.org>'
status = 'CLEAN: All modifications committed'
verbose_msg = """Branch: glue-1-60-branch
Tag: glue-release-1.60.0
Id: 30db5c2a8b324ae53500aed512db7220fd4e7f4b

Builder: Ryan Fisher <ryan.fisher@ligo.org>
Build date: 2018-10-30 21:25:05 +0000
Repository status: CLEAN: All modifications committed"""

import warnings

class VersionMismatchError(ValueError):
    pass

def check_match(foreign_id, onmismatch="raise"):
    """
    If foreign_id != id, perform an action specified by the onmismatch
    kwarg. This can be useful for validating input files.

    onmismatch actions:
      "raise": raise a VersionMismatchError, stating both versions involved
      "warn": emit a warning, stating both versions involved
    """
    if onmismatch not in ("raise", "warn"):
        raise ValueError(onmismatch + " is an unrecognized value of onmismatch")
    if foreign_id == '30db5c2a8b324ae53500aed512db7220fd4e7f4b':
        return
    msg = "Program id (30db5c2a8b324ae53500aed512db7220fd4e7f4b does not match given id (%s)." % foreign_id
    if onmismatch == "raise":
        raise VersionMismatchError(msg)

    # in the backtrace, show calling code
    warnings.warn(msg, UserWarning)


