import cv2
import sys
import numpy as np
from pathlib import Path
from docarray import Document
from docarray import DocumentArray

'''
Usage
python client.py DALLE_PORT DIFFUSION UPSCALE_PORT SEGMENT_PORT

where DALLE_PORT, DIFFUSION_PORT, UPSCALE_PORT, SEGMENT_PORT are from the jina log:
    INFO   dalle/rep-0@180585 start server bound to 0.0.0.0:56768
    INFO   diffusion/rep-0@305004 start server bound to 0.0.0.0:55470
    INFO   upscaler/rep-0@178370 start server bound to 0.0.0.0:62240
    INFO   clipseg/rep-0@180633 start server bound to 0.0.0.0:57042

That is, from the jina log when running
    python3 flow_parser.py --enable-stable-diffusion --enable-clipseg
    python3 -m jina flow --uses flow.tmp.yml
'''

def to_doc(prompt, search, name, dal_port, dif_port, scal_port, clip_port, n=1, c=2):
    folder = "spices"
    dalle_url = f'grpc://127.0.0.1:{dal_port}'
    dif_url = f'grpc://127.0.0.1:{dif_port}'
    scale_url = f'grpc://127.0.0.1:{scal_port}'
    seg_url = f'grpc://127.0.0.1:{clip_port}'
    doc = Document(text=prompt).post(
        dalle_url, parameters={'num_images': n}
    )
    Path(folder).mkdir(parents=True, exist_ok=True)

    # Upscale best match
    fav = doc.matches[0]
    fav.embedding = doc.embedding

    print('EMBEDDING')
    print(doc.embedding)
    print(fav.embedding)

    # Stable diffusion TODO
#    diffused = fav.post(dif_url, parameters={'skip_rate': 0.5, 'num_images': 4}, target_executor='diffusion').matches

    # Upscale matches
    fav = fav.post(f'{scale_url}/upscale')
    fav.load_uri_to_image_tensor(1024, 1024)
    fav.save_image_tensor_to_file(f'{folder}/{name}.png')

    # Segment the upscaled version
    fav.convert_image_tensor_to_uri()
    mask_in = Document(text=search, uri = fav.uri)
    mask_out = mask_in.post(f'{seg_url}/segment', {
        'thresholding_type': 'adaptive_gaussian',
        'adaptive_thresh_block_size': 32,
        'adaptive_thresh_c': c,
        'invert': True
    }).matches[0]

    # TODO: test
    mask_file = f'{folder}/{name}-mask.png'
    mask_out.save_uri_to_file(mask_file)
    mask_blur = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
    mask_blur[:, :, 3] = cv2.blur(mask_blur[:,:,3], (32, 32)) 
    mask_blur[:, :, 3] = 255*(mask_blur[:,:,3] >= 127).astype(np.uint8)
    cv2.imwrite(mask_file, mask_blur)


def yield_prompts(prefix, spices):
    for spice in spices:
        yield (
            f'Photograph in pantry of a name-brand plastic spice jar with a {prefix} for the {spice} ',
            f'Photograph in pantry of a name-brand plastic spice jar with a {prefix} for the {spice} ',
            f'Photograph in pantry of a name-brand plastic spice jar with a {prefix} for the {spice} ',
        )

PORTS = sys.argv[1:5]
SEASONINGS = [
'baharat seasoning', 'chili powder', 'chinese five-spice powder',
'curry powder', 'dukkah', 'garam masala', 'herbes de provence',
'mojo seasoning', 'old bay seasoning', 'pickling spice',
'pumpkin pie spice', 'ras el hanout', 'za\'atar seasoning'
]
SPICES = SEASONINGS + [
'allspice', 'ancho powder', 'annatto seeds', 'black pepper',
'cardamom', 'carom seeds', 'cayenne pepper', 'celery seeds',
'chervil', 'chia seeds', 'chipotle powder', 'cinnamon',
'coriander', 'cumin', 'fenugreek', 'flax seeds', 'garlic powder',
'ginger', 'gochugaru', 'grains of paradise', 'ground cloves',
'kosher salt', 'loomi', 'mace', 'mahlab', 'mustard powder',
'nutmeg', 'paprika', 'pickling salt', 'saffron', 'sea salt',
'smoked paprika', 'star anise', 'sumac', 'turmeric'
]
SPICES.sort()
PREFIX = 'small square paper label'
SEARCH = 'small square paper label'
TRIES = 16
VERSION = '2'
C = 5

for (prompts, spice) in zip(yield_prompts(PREFIX, SPICES), SPICES):
    key = spice.replace(' ', '-')
    print('Rendering', key)
    for (i, prompt) in enumerate(prompts):
        to_doc(
            prompt, SEARCH, f'v-{VERSION}-prompt-{i}-spice-{key}',
            PORTS[0], PORTS[1], PORTS[2], PORTS[3], n=TRIES, c=C
        )
