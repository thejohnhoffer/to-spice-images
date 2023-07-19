import os
import re
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

def to_doc(prompt, search, folder, name, dal_port, dif_port, scal_port, clip_port, check=1, n=1, c=2):
    dalle_url = f'grpc://127.0.0.1:{dal_port}'
    dif_url = f'grpc://127.0.0.1:{dif_port}'
    scale_url = f'grpc://127.0.0.1:{scal_port}'
    seg_url = f'grpc://127.0.0.1:{clip_port}'

    # Create output directory
    Path(folder).mkdir(parents=True, exist_ok=True)
    print(f'Will check {check} of {n} variants')

    # Generate the original n images
    doc = Document(text=prompt).post(
        dalle_url, parameters={'num_images': n}
    )
    subset = doc.matches[:check]

    # Consider a subset of images
    mask_file_temp = f'{folder}/{name}-mask-temp.png'
    mask_file = f'{folder}/{name}-mask.png'
    image_file = f'{folder}/{name}.png'
    out_mask_matrix = None
    out_image_doc = None
    out_max_area = 0
    max_ratio = 1/3
    
    print('EMBEDDING')
    print(doc.embedding)

    for match in subset:

        # Copy any embedding to match
        match.embedding = doc.embedding

        # Segment the match
        mask_in = Document(text=search, uri = match.uri)
        mask_out = mask_in.post(f'{seg_url}/segment', {
            'thresholding_type': 'adaptive_gaussian',
            'adaptive_thresh_block_size': 32,
            'adaptive_thresh_c': c,
            'invert': True
        }).matches[0]

        # Reduce mask resolution to 32x32
        mask_out.save_uri_to_file(mask_file_temp)
        mask_matrix = cv2.imread(mask_file_temp, cv2.IMREAD_UNCHANGED)
        mask_matrix[:, :, 3] = cv2.blur(mask_matrix[:,:,3], (16, 16)) 
        mask_matrix[:, :, 3] = 255*(mask_matrix[:,:,3] >= 127).astype(np.uint8)

        # Measure max area
        contours = cv2.findContours(mask_matrix[:, :, 3], 1, 2)[0]
        max_area = max_contour_area(contours)
        ratio = 3 * max_area / mask_matrix.size

        # Verify improved area
        too_big = ratio > max_ratio
        too_small = max_area < out_max_area
        if out_max_area > 0 and (too_small or too_big):
            continue

        # Select match
        out_image_doc = match
        out_max_area = max_area
        out_mask_matrix = mask_matrix

    try:
        os.remove(mask_file_temp)
    except:
        pass

    print(f'Max mask area: {out_max_area}')

    # Upscale and save output image
    out_image_doc = out_image_doc.post(f'{scale_url}/upscale')
    out_image_doc.load_uri_to_image_tensor(1024, 1024)
    out_image_doc.save_image_tensor_to_file(image_file)

    # Upscaled and save output mask
    out_mask_matrix = cv2.resize(out_mask_matrix, (1024, 1024))
    out_mask_matrix[:,:,:3] = out_image_doc.tensor[:,:,::-1]
    cv2.imwrite(mask_file, out_mask_matrix)

    print(f'Wrote {mask_file}')


# Select contour with largest area
def max_contour_area(contours):

    # No contours in sample
    if len(contours) == 0:
        return 0

    # create an empty list
    cnt_area = []
     
    # loop through all the contours
    for i in range(0,len(contours),1):
        # for each contour, use OpenCV to calculate the area of the contour
        cnt_area.append(cv2.contourArea(contours[i]))
 
    # Sort our list of contour areas in descending order
    list.sort(cnt_area, reverse=True)
    return cnt_area[0]


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
C = 5
TRIES = 64
VERSION = '3-2-0'
CHECK = min(TRIES, 7)
PREFIX = 'small square paper label'
SEARCH = 'small square paper label'
FOLDER = f'spices-v-{VERSION}-check-{CHECK}-of-{TRIES}'

print(f'Rendering {FOLDER}')
for (prompts, spice) in zip(yield_prompts(PREFIX, SPICES), SPICES):
    key = re.sub(r'[^a-z0-9]', '-', spice)
    print(f'Rendering {key}')
    for (i, prompt) in enumerate(prompts):
        to_doc(
            prompt, SEARCH, FOLDER, f'v-{VERSION}-prompt-{i}-spice-{key}',
            PORTS[0], PORTS[1], PORTS[2], PORTS[3],
            check=CHECK, n=TRIES, c=C
        )
