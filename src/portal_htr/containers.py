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
Utilities for representing HTR data.
"""


class HTRBBox:
    '''Covenience container for converting OBB bounded HTR results to bbox'''
    def __init__(self, obb):
        self.obb = obb

    @property
    def bbox(self):
        x_coords, y_coords = zip(*self.obb)
        x_min = round(min(x_coords).item())
        y_min = round(min(y_coords).item())
        x_max = round(max(x_coords).item())
        y_max = round(max(y_coords).item())
        return (x_min, y_min, x_max, y_max)


class HTRLine(HTRBBox):
    def __init__(self, obb, text, confidence):
        self.obb = obb
        self.text = text
        self.confidence = confidence


class HTRRegion(HTRBBox):
    def __init__(self, obb, lines=None):
        self.obb = obb
        self.lines = lines

    def add_line(box, text, confidence):
        '''Convenience method for adding lines to the container'''
        text_line = HTRLine(box, text, confidence)
        self.lines.append(text_line)
        

class HTRPage:
    def __init__(self, name, size, regions=None):
        self.name = name
        self.height = size[0]
        self.width = size[1]
        self.regions = regions

    def add_region(box, lines=None):
        text_region = HTRRegion(box, lines=lines)
        self.regions.append(text_region)
