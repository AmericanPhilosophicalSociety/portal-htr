#
# Copyright 2026 David Ragnar Nelson

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
Utility functions
"""
import torch


def compute_intersection(point, box):
    '''Returns true if a point is inside a box'''

    # vector representation of the box's corners
    ab = box[1] - box[0]
    bc = box[2] - box[1]

    # vectors representing corners to point
    ap = point - box[0]
    bp = point - box[1]

    # calculate dot products of vertices
    val = 0 <= ab @ ap <= ab @ ab and 0 <= bc @ bp <= bc @ bc

    return val



def compute_box_center(box):
    a = box[0]
    b = box[2]

    cx = (a[0] + b[0]) / 2
    cy = (a[1] + b[1]) / 2

    # need to return a tensor so the math works
    return torch.tensor([cx, cy])
