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
Utilities for performing segmentation with YOLO
"""
import torch
from collections import defaultdict

from ultralytics import YOLO
from .utils import compute_intersection, compute_box_center


def reading_order(results):
    '''Heuristic for determining reading order'''
    # filter based on class
    classes = results.obb.cls
    boxes = results.obb.xyxyxyxy

    # lines are assigned class 3
    line_mask = torch.isin(classes, torch.tensor([3.]))
    lines = boxes[line_mask]

    # class typology for regions:
    # 0. generic block
    # 1. left side of a two-page spread
    # 2. right side of a two-page spread
    # out of order here to facilitate sorting later
    class_options = [1., 2., 0.]
    # dict to keep track of class results
    regions = defaultdict(list)
    for c in class_options:
        selected_cls = torch.tensor([c])
        mask = torch.isin(classes, selected_cls)
        filtered_boxes = boxes[mask]
        print(filtered_boxes.shape)
        if len(filtered_boxes) != 0:
            regions[c].extend(filtered_boxes)
    
    left = regions[1.]
    right = regions[2.]
    block = regions[0.]

    region_order = [left, right, block]
    # region_order = [block, left, right]

    ordered_regions = []
    # reorder regions from top to bottom, left to right
    for r in region_order:
        if len(r) > 1:
            r = torch.stack(r)
            reordered_region = r[r[:, -1][:, 1].argsort()]
            ordered_regions.extend(reordered_region)
        else:
            r = r[0]
            ordered_regions.append(r)


    # make a container to hold the lines, with one extra for unassigned lines
    target_length = len(ordered_regions) + 1
    ordered_lines = [[] for n in range(target_length)]
    # ordered_lines = [None] * (len(ordered_regions) + 1)

    # iterate through lines and assign them to first possible region
    for line in lines:
        for n, region in enumerate(ordered_regions):
            line_center = compute_box_center(line)
            intersection = compute_intersection(line_center, region)
            if intersection:
                ordered_lines[n].append(line)
                break
        # if no match, assign to dummy last region
        else:
            ordered_lines[n + 1].append(line)

    for n, lineset in enumerate(ordered_lines):
        print(lineset)
        if len(lineset) > 1:
            lineset = torch.stack(lineset)
            lineset = lineset[lineset[:, -1][:, 1].argsort()]
            ordered_lines[n] = lineset

    return ordered_regions, ordered_lines


def _segment_page(page, model=None):
    # if model, use custom model, otherwise load default YOLO OBB model
    if model:
        model = YOLO(model)
    else:
        model = YOLO('yolo26n-obb.pt')

    # possible to pass in additional parameters, but not currently implemented
    results = model(page, imgsz=1280)
    return results


def segment_page(page, model=None):
    results = _segment_page(page, model=model)
    results = results[0].cpu()
    ordered_regions, ordered_lines = reading_order(results)
    return results.orig_shape, (ordered_regions, ordered_lines)
