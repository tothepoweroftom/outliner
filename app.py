from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import UJSONResponse
# import gpt_2_simple as gpt2
# import tensorflow as tf
import uvicorn
import os
import gc
import json

import re
import io
import os
import base64
from io import BytesIO
from PIL import Image
# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
client = vision.ImageAnnotatorClient()

middleware = [
    Middleware(CORSMiddleware,    
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"])
]

app = Starlette(debug=True, middleware=middleware)


# Needed to avoid cross-domain issues
response_header = {
    'Access-Control-Allow-Origin': '*'
}



generate_count = 0

def getLabels(content):
    image = types.Image(content=content)
    metadata = {}
    objectlabels = []
        # Performs label detection on the image file
    response = client.label_detection(image)
    labels = response.label_annotations
    print('Labels:')
    textlabels = []
    for label in labels:
        print(label.description)
        textlabels.append(label.description)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))


    objects = client.object_localization(
        image=image).localized_object_annotations

    print('Number of objects found: {}'.format(len(objects)))
    for object_ in objects:
        obj = {}
        obj["name"] = object_.name
        obj["score"] = object_.score
        print('\n{} (confidence: {})'.format(object_.name, object_.score))
        print('Normalized bounding polygon vertices: ')
        obj["vertices"] = []
        for vertex in object_.bounding_poly.normalized_vertices:
            obj["vertices"].append({'x': str(vertex.x), 'y': str(vertex.y)})
        objectlabels.append(obj)

    metadata["labels"] = textlabels
    metadata["objects"] = objectlabels
    return metadata


@app.route('/', methods=['GET', 'POST', 'HEAD'])
async def homepage(request):


    # global generate_count
    # global sess

    params = await request.json()
    imageBase64 = params.get('img')
    image_data = re.sub('^data:image/.+;base64,', '',imageBase64)
    base64Data = base64.b64decode(image_data)
    # im = Image.open()
    labels = getLabels(base64Data)
    output = json.dumps(labels)
    # print(imageBase64)

    return UJSONResponse({'labels': output},
                            headers=response_header)



if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
