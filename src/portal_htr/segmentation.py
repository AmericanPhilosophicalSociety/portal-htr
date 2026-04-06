#
# Copyright 2026 David Ragnar Nelson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
