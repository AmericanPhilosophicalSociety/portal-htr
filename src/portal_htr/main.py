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
from .segmentation import load_yolo, segment_page
from .trocr import load_model, predict
from .islandora import download_book
from .containers import HTRLine, HTRRegion, HTRPage
from .serialization import serialize
from pathlib import Path
import click


def inference(page, seg_model, processor, rec_model):
    '''Perform segmentation and recognition on a single page'''
    size, segmentation = segment_page(page, seg_model)
    regions, lines = segmentation
    if len(lines) == 0:
        print(f'No text detected on {page}.')
        return None
    preds = []
    confidences = []
    for lineset in lines:
        pred, conf = predict(page, lineset, processor, rec_model)
        preds.extend(pred)
        confidences.extend(conf)

    n = 0
    prepared_regions = []
    # check for orphan lines
    if len(regions) != len(lines):
        orphan_lines = lines.pop(-1)
    for region, lineset in zip(regions, lines):
        prepared_lines = []
        for line in lineset:
            text = preds[n]
            score = confidences[n]
            prepared_line = HTRLine(line, text, score)
            prepared_lines.append(prepared_line)
            n = n + 1
        prepared_region = HTRRegion(region, lines=prepared_lines)
        prepared_regions.append(prepared_region)

    # orphan lines are assigned a dummy region equal to the line itself
    for line in orphan_lines:
        text = preds[n]
        score = confidences[n]
        prepared_line = HTRLine(line, text, score)
        dummy_region = HTRRegion(line, lines=[prepared_line])
        prepared_regions.append(dummy_region)
        n = n + 1
    
    prepared_page = HTRPage(page, size, regions=prepared_regions)

    return prepared_page


def to_hocr(page):
    hocr_xml = serialize(page)
    file_base = page.name.stem.split('.')[0]
    with open(f'{file_base}.html', 'w') as f:
        f.write(hocr_xml)


def ocr_book(nid, seg_model, processor, rec_model):
    book_data = download_book(nid)
    for child_nid, image in book_data:
        img_type = image.format
        filename = Path(f'{child_nid}.{img_type}')
        image.save(filename)
        page = inference(
            filename,
            seg_model,
            processor,
            rec_model
        )
        if page:
            to_hocr(page)
        filename.unlink()


@click.command()
@click.option('--file', '-f', help='Path to a file with Drupal nodes')
@click.option('--seg-model', '-s', default=None)
@click.option('--processor', '-p', default='american-philosophical-society/trocr-18th-c-english-obb')
@click.option('--rec-model', '-r', default='american-philosophical-society/trocr-18th-c-english-obb')
@click.argument('nodes', nargs=-1)
def cli(file, seg_model, processor, rec_model, nodes):
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
    yolo_model = load_yolo(model=seg_model)
    processor, rec_model = load_model(processor, rec_model)
    for node in nodes:
        ocr_book(node, yolo_model, processor, rec_model)
        click.echo(f'Processed {node}.')
