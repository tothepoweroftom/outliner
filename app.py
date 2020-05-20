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
import random



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

def oldMethod(image):
    bg = Image.open("./backgrounds/background.png")
    background = np.array(bg)
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
    mask = final_alpha.reshape(*final_alpha.shape, 1)
    background = misc.imresize(background, origin_shape)
    blended = (mask) * rgb_ + (1. - mask)*background
    output = background + blended
    return output

def simple_threshold(im, threshold=128):
    return ((im > threshold) * 255).astype("uint8")

def main(sess, imagedata, color):
	
    image_batch = tf.get_collection('image_batch')[0]
    pred_mattes = tf.get_collection('mask')[0]

    image = Image.open(BytesIO(imagedata))
    num = random.randint(1,4)
    bg = Image.open("./backgrounds/"+color+str(num)+".png")
    background = np.array(bg)
    rgb = np.array(image)

    rgb_ = rgb

    if rgb.shape[2]==4:
        rgb = rgba2rgb(rgb)
    
    if rgb_.shape[2]==3:
        rgba = rgb2rgba(rgb_)
        rgb_ = rgba

    
    origin_shape = rgb.shape[:2]
    img = np.zeros(rgb.shape,dtype=np.uint8)
    img.fill(255) # or img[:] = 255
    rgb = np.expand_dims(misc.imresize(rgb.astype(np.uint8),[320,320,3],interp="nearest").astype(np.float32)-g_mean,0)

    feed_dict = {image_batch:rgb}
    pred_alpha = sess.run(pred_mattes,feed_dict = feed_dict)
    final_alpha = misc.imresize(np.squeeze(pred_alpha),origin_shape)
    # final_alpha = simple_threshold(final_alpha, threshold=150)

    final_alpha = final_alpha/255
    mask = final_alpha.reshape(*final_alpha.shape, 1)
    # resize background
    background = misc.imresize(background, origin_shape)



    # Here we add the images

    print(mask.shape)
    print(background.shape)
    print(rgb_.shape)


    blended = ((1.-mask)*background) + (mask*rgb_)   
    im = Image.fromarray(blended.astype("uint8"))

    buf = io.BytesIO()
    im.save(buf, format='PNG')
    byte_im = buf.getvalue()
    im_b64 = base64.b64encode(byte_im)

    
    return im_b64

@app.route('/', methods=['GET', 'POST', 'HEAD'])
async def homepage(request):


    global generate_count
    global sess

    params = await request.json()
    imageBase64 = params.get('img')
    color = params.get('color', 'r')
    image_data = re.sub('^data:image/.+;base64,', '',imageBase64)
    base64Data = base64.b64decode(image_data)
    output = main(sess, base64Data, color)
    
    gc.collect()
    return UJSONResponse({'image': output},
                            headers=response_header)



if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
