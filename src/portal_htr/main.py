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
Main entrypoint for end-to-end HTR operations
"""
from .segmentation import segment_page
from .trocr import predict
from .islandora import download_book
from kraken.serialization import serialize
from pathlib import Path
import click


def inference(page, seg_model, processor, rec_model):
    '''Perform segmentation and recognition on a single page'''
    segmentation, size = segment_page(page, seg_model)
    rec = predict(page, segmentation, processor, rec_model)
    segmentation.lines = rec
    return segmentation, size


def to_hocr(segmentation, size, file_base):
    hocr_xml = serialize(segmentation, image_size=size, template='portal_hocr', template_source='custom', sub_line_segmentation=False)
    with open(f'{file_base}.html', 'w') as f:
        f.write(hocr_xml)


def ocr_book(nid):
    book_data = download_book(nid)
    for child_nid, image in book_data:
        img_type = image.format
        filename = Path(f'{child_nid}.{img_type}')
        image.save(filename)
        segmentation, size = inference(
            filename,
            None,
            'drnelson6/trocr-18th-c-english',
            'drnelson6/trocr-18th-c-english'
        )
        to_hocr(segmentation, size, child_nid)
        filename.unlink()


@click.command()
@click.option('--file', '-f', help='Path to a file with Drupal nodes')
@click.argument('nodes', nargs=-1)
def cli(file, nodes):
    """
    Program to prepare hOCR files of images from an Islandora site

    Args: Drupal nodes from which to generate hOCR files
    """
    if len(nodes) > 0 and file:
        raise click.BadOptionUsage(file, "Please provdie either a file with nodes or a list of nodes.")
    if len(nodes) == 0 and not file:
        raise click.UsageError("Please provide either a list of nodes or a file with a list of nodes")
    if file:
        with open(file, 'r') as f:
            files = f.read().split('\n')
            # discard any whitespace
        nodes = [f for f in files if not f == '']
    for node in nodes:
        ocr_book(node)
