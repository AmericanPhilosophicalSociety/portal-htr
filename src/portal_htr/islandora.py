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
Utilities for retrieving data from Islandora and preparing data for ingest.
"""
import requests
from PIL import Image
from dotenv import load_dotenv
import os
import io
from time import sleep


load_dotenv()
baseurl = str(os.getenv('PORTAL'))
username = str(os.getenv('USER'))
password = str(os.getenv('PASSWORD'))

# avoid decompression bomb error
Image.MAX_IMAGE_PIXELS = None


def connect_drupal(auth):
    s = requests.session()
    s.auth = auth
    return s


def fetch_child_nids(session, host, nid):
    '''Return NID, title and MID for all pages in book object'''
    url = f'{host}/show-children-api/'
    request = session.get(url + str(nid))
    request.raise_for_status()
    data = request.json()
    if len(data) == 0:
        raise ValueError("Invalid NID")
    else:
        sorted_data = sorted(data, key=lambda x: int(x['field_weight_value']))
        return sorted_data


def fetch_file_paths(host, nid):
    '''Generator function to sequentially fetch image urls'''
    url = f'{host}/node/{nid}/manifest'
    # don't use session because it will fail if you use auth
    manifest = requests.get(url).json()
    image_urls = []
    for i in manifest['sequences'][0]['canvases']:
        url = i['images'][0]['resource']['@id']
        image_urls.append(url)

    return image_urls

def load_image(session, url):
    '''Download an image from the portal and store in local memory'''
    request = requests.get(url, stream=True)
    request.raise_for_status()
    image_data = Image.open(io.BytesIO(request.content))
    return image_data


def download_book(nid, auth=(username, password), host=baseurl):
    '''Generator function to sequentially download images in a book'''
    session = connect_drupal(auth)
    book_data = fetch_child_nids(session, host, nid)
    filepaths = fetch_file_paths(host, nid)
    for i, fp in zip(book_data, filepaths):
        child_nid = i['nid']
        image = load_image(session, fp)
        yield child_nid, image 
