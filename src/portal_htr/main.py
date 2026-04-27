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
            'revcity_seg.mlmodel',
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
        click.echo(f'Processed {node}.')
