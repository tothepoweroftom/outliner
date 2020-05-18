from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import UJSONResponse
import uvicorn
import os
import gc
import json
import re
import io
import base64
from io import BytesIO
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
import imageio
from scipy import misc
import skimage
import cv2

import requests


g_mean = np.array(([126.88,120.24,112.19])).reshape([1,1,3])


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

g_mean = np.array(([126.88,120.24,112.19])).reshape([1,1,3])

def loadCheckpoint(sess):
    saver = tf.train.import_meta_graph('./meta_graph/my-model.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./salience_model'))


def rgba2rgb(img):
	return img[:,:,:3]*np.expand_dims(img[:,:,3],2)

sess = tf.Session(config=tf.ConfigProto())
loadCheckpoint(sess)





def encode(image) -> str:

    # # convert image to bytes
    # with BytesIO() as output_bytes:
    #     PIL_image = Image.fromarray(skimage.img_as_ubyte(image))
    #     PIL_image.save(output_bytes, 'JPEG') # Note JPG is not a vaild type here
    #     bytes_data = output_bytes.getvalue()

    # encode bytes to base64 string
    base64_str = str(base64.b64encode(image))

    return base64_str


def main(sess, imagedata):
	
    image_batch = tf.get_collection('image_batch')[0]
    pred_mattes = tf.get_collection('mask')[0]

    image = Image.open(BytesIO(imagedata))
    rgb = np.array(image)
    rgb_ = rgb

    if rgb.shape[2]==4:
        rgb = rgba2rgb(rgb)
    rgbcopy = rgb.copy()

    origin_shape = rgb.shape[:2]
    img = np.zeros(rgb.shape,dtype=np.uint8)
    img.fill(255) # or img[:] = 255
    rgb = np.expand_dims(misc.imresize(rgb.astype(np.uint8),[320,320,3],interp="nearest").astype(np.float32)-g_mean,0)

    feed_dict = {image_batch:rgb}
    pred_alpha = sess.run(pred_mattes,feed_dict = feed_dict)
    final_alpha = misc.imresize(np.squeeze(pred_alpha),origin_shape)
    final_alpha = final_alpha/255
    # alpha3 = np.stack([final_alpha]*3, axis=2)
    mask = final_alpha.reshape(*final_alpha.shape, 1)

    blended = (mask) * rgb_


    imageio.imsave(os.path.join('./test','alpha.png'),blended)
    _, im_arr = cv2.imencode('.png', blended)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    # url = "https://api.imgbb.com/1/upload"
    # payload = {
    #     "key": 'd2be788bd7cde38e2a8ca07a9abf49c7',
    #     "image":im_b64,
    # }


    # res = requests.post(url, payload)
    # print(res)
    
    return im_b64

@app.route('/', methods=['GET', 'POST', 'HEAD'])
async def homepage(request):


    # global generate_count
    global sess

    params = await request.json()
    imageBase64 = params.get('img')
    image_data = re.sub('^data:image/.+;base64,', '',imageBase64)
    base64Data = base64.b64decode(image_data)
    output = main(sess, base64Data)
    

    return UJSONResponse({'image': output},
                            headers=response_header)



if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
