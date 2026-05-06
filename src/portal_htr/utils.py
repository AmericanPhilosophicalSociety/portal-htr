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
import numpy as np


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


def get_vector(line):
    '''From two coordinates of a line, return the line's vector'''
    a, b = line[0]
    c, d = line[-1]

    return (c - a, b - d)


def rotate_segment(im, bbox):
    '''Rotate a segment to prepare it for extraction from the image'''

    vector = get_vector([bbox[-1], bbox[0]])
    angle = np.arctan2(vector[1], vector[0])
    # determine box width and height
    # if height longer than width, assume box is vertical
    box_width = bbox[0][0] - bbox[2][0]
    box_height = bbox[0][1] - bbox[1][1]
    if box_height > box_width:
        angle = angle + np.pi / 2
    w, h = im.size
    center = torch.tensor([w / 2, h / 2])
    if angle != 0.0:
        angle = -angle
        # provide transformation to adjust for planar geometry of the image
        bbox[:, 1] = h - bbox[:, 1]
        # displace image from center
        bbox = bbox - center
        # transformation matrix - set dtype to float32 to avoid issues with torch tensor
        s = np.sin(angle, dtype=np.float32)
        c = np.cos(angle, dtype=np.float32)
        T = torch.tensor([
            [c, -s],
            [s, c]
        ])
        # matrix multiplication
        rotated_bbox = (T@bbox.T).T
        angle = np.rad2deg(angle)
        im = im.rotate(angle, expand=True)
        w, h = im.size
        rotated_center = torch.tensor([w / 2, h / 2])
        # undo previous displacements
        rotated_bbox = rotated_bbox + rotated_center
        rotated_bbox[:, 1] = h - rotated_bbox[:, 1]
        # since image has been rotated, cannot assume coordinates correspond to position
        x_coords, y_coords = zip(*rotated_bbox)
    else:
        x_coords, y_coords = zip(*bbox)
    left = min(x_coords).item()
    right = max(x_coords).item()
    upper = min(y_coords).item()
    lower = max(y_coords).item()

    return (left, upper, right, lower), im
