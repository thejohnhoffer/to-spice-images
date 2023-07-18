import sys
from pathlib import Path
from docarray import Document
from docarray import DocumentArray

'''
Usage
python client.py DALLE_PORT UPSCALE_PORT SEGMENT_PORT

where DALLE_PORT, UPSCALE_PORT, SEGMENT_PORT are from the jina log:
    INFO   dalle/rep-0@180585 start server bound to 0.0.0.0:52986
    INFO   upscaler/rep-0@178370 start server bound to 0.0.0.0:62298
    INFO   clipseg/rep-0@180633 start server bound to 0.0.0.0:58522

That is, from the jina log when running
    python3 flow_parser.py --enable-stable-diffusion --enable-clipseg
    python3 -m jina flow --uses flow.tmp.yml
'''

def to_doc(prompt, name, port1, port2, port3, n=1):
    folder = "spices"
    dalle_url = f'grpc://127.0.0.1:{port1}'
    scale_url = f'grpc://127.0.0.1:{port2}'
    seg_url = f'grpc://127.0.0.1:{port3}'
    doc = Document(text=prompt).post(
        dalle_url, parameters={'num_images': n}
    )
    Path(folder).mkdir(parents=True, exist_ok=True)

    # Upscale best match
    fav = doc.matches[0]
    fav.embedding = doc.embedding
    fav = fav.post(f'{scale_url}/upscale')
    fav.load_uri_to_image_tensor(1024, 1024)
    fav.save_image_tensor_to_file(f'{folder}/{name}.png')

    # Segment the upscaled version
    fav.convert_image_tensor_to_uri()
    mask_in = Document(text=prompt, uri = fav.uri)
    mask_out = mask_in.post(f'{seg_url}/segment', {
        'thresholding_type': 'adaptive_gaussian',
        'adaptive_thresh_block_size': 16,
        'adaptive_thresh_c': 0.75,
        'invert': True
    }).matches[0]
    mask_out.save_uri_to_file(f'{folder}/{name}-mask.png')

ports = sys.argv[1:4]
spices = [
    'red pepper flakes',
    'ground ginger'
]
PROMPTS = [
    f'Photograph of paper label of the name-brand {spices[0]} spice jar in the pantry',
    f'Photograph of paper label of the name-brand {spices[1]} spice jar in the pantry',
]
for (i, p) in enumerate(PROMPTS):
    to_doc(p, f'test-{i}', ports[0], ports[1], ports[2], n=2)
