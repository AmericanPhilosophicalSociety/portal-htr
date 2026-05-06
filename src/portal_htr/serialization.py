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
Utilities for serializing HTR results.
"""
from jinja2 import Environment, PackageLoader


def serialize(doc):
    '''Serialize the HTR output into hOCR format'''

    env = Environment(
        loader=PackageLoader('portal_htr'),
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=True
    )

    # Jinja does not compute class properties so convert the container to a dict for serialization
    # page = {
        # 'name': doc.name,
        # 'height': doc.height,
        # 'width': doc.width,
        # 'regions': [],
    # }

    # for region in doc.regions:
        # region_bbox = region.bbox
        # region_dict = {
            # 'bbox': region_bbox,
            # 'lines': [],
        # }
        # print('Number of lines:', len(region.lines))
        # for n, line in enumerate(region.lines):
            # print('Line number:', n)
            # print('Pred:', line.text)
            # line_bbox = line.bbox
            # line_dict = {
                # 'text': line.text,
                # 'confidence': line.confidence,
                # 'bbox': line_bbox
            # }
            # region_dict['lines'].append(line_dict)
        # page['regions'].append(region_dict)
    
    template = env.get_template('hocr.html')
    return template.render(page=doc)
