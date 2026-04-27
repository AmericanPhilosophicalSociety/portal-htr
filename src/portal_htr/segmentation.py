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
Utilities for performing segmentation in kraken
"""
from PIL import Image
from kraken.tasks import SegmentationTaskModel
from kraken.configs import SegmentationInferenceConfig


def segment_page(page, model=None):
    # if model, use custom model, otherwise load default kraken model
    if model:
        model = SegmentationTaskModel.load_model(model)
    else:
        model = SegmentationTaskModel.load_model()
    # passing parameters to config not yet implemented
    config = SegmentationInferenceConfig()
    im = Image.open(page)
    segmentation = model.predict(im, config)
    # return size because we need it for serialization
    return segmentation, im.size
