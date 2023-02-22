import json
from flask import Flask, jsonify, make_response, render_template, request, send_file
from itsdangerous import base64_encode
from jinja2 import Environment
import io
import base64
from diffusers import StableDiffusionImg2ImgPipeline
import torch
import requests
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import glob 
from typing import Optional, List, Union
from flask_ngrok import run_with_ngrok
# import redis
# from redis import Redis
# from rq import Queue
# from rq.job import Job
# from queue_image_generator import *

# Connect to Redis instance
# r = Redis(host='localhost', port=6379)
# q = Queue('high', connection = r)
## Run redis-server in terminal
## Run rq worker in terminal

# Load model
img2imgpipe = StableDiffusionImg2ImgPipeline.from_pretrained("nitrosocke/mo-di-diffusion", torch_dtype=torch.float16)
img2imgpipe.to("cuda")

def multiple_rounds_img2img(
  init_image: Image,
  prompt: str,  
  negative_prompt: str,
  strength_array: List[float],
  guidance_array: Union[List[float], List[int]],
  final_images_to_return: Optional[int] = 5,
  num_rounds: Optional[int] = 4,
  seed: Optional[int] = 123) -> List:

  # Parameter checking
  ## init_image
  assert isinstance(init_image, Image.Image), "init_image must be an Image"

  ## prompt & negative_prompt
  assert isinstance(prompt, str) and len(prompt) > 0, "Prompt provided must be a comma separated string and cannot be an empty string" 
  assert isinstance(negative_prompt, str), "Negative Prompt provided must be a comma separated string"

  ## num rounds
  assert num_rounds > 0, "num_rounds must be greater than 0"

  ## strength_array & guidance array
  assert len(strength_array) == num_rounds, 'strength_array length must be identical to num_rounds'
  assert len(guidance_array) == num_rounds, 'guidance_array length must be identical to num_rounds'

  ## final_images_to_return
  assert final_images_to_return > 0, "final_images_to_return must be greater than 0"

  ## seed
  assert isinstance(seed, int), "seed must be an integer"
  
  # Main Body
  torch.manual_seed(seed)
  output_image_array = [init_image]

  for idx in list(range(0, num_rounds - 1)):
    
    img2imgpipeline = img2imgpipe(prompt = prompt,
                          image=output_image_array[idx],
                          strength=strength_array[idx],
                          guidance_scale=guidance_array[idx],
                          num_inference_steps=400,
                          num_images_per_prompt = 1,
                          negative_prompt = negative_prompt)

    output_image_array.append( img2imgpipeline.images[0] )

    # For final round of inference
    torch.manual_seed(seed)
    img2imgpipeline_final = img2imgpipe(prompt = prompt,
                            image=output_image_array[-1],
                            strength=strength_array[-1],
                            guidance_scale=guidance_array[-1],
                            num_inference_steps=400,
                            num_images_per_prompt = final_images_to_return,
                            negative_prompt = negative_prompt)

    return img2imgpipeline_final.images

# Set flask app and set to ngrok
app = Flask(__name__)
run_with_ngrok(app)


@app.route('/generate_image_rest_direct', methods=["POST"])
def generate_image_direct():
    
    # Get the JSON data from the POST request
    data = request.json

    # Get the base64-encoded image data and decode it
    image_b64 = data.get('initial_image', '')
    

    # Get parameters
    prompt = data.get('prompt', '')
    negative_prompt = data.get('negative', '')
    final_images_to_return = int(data.get('final_images_to_return', ''))
    seed = int(data.get('seed', ''))

    print(prompt)
    print(negative_prompt)
    print(final_images_to_return)
    print(seed)
    
    # Load the image data into a PIL Image object
    image_data = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(image_data))
   
    print(type(img))
    
    returned_imgs = multiple_rounds_img2img(
    init_image = img,
    prompt = prompt,
    negative_prompt = negative_prompt,
    strength_array = [0.7, 0.6],
    guidance_array = [20.0, 18],
    final_images_to_return = final_images_to_return,
    num_rounds = 2,
    seed = seed)

    print(returned_imgs)
    print(type(returned_imgs))

    # Convert generated image to bytestring
    buffered = io.BytesIO()
    returned_imgs[0].save(buffered, format = "PNG")
    returned_imgs_data = buffered.getvalue()
    returned_imgs_b64 = base64.b64encode(returned_imgs_data).decode('utf-8')

    response_data = {"generated_image": returned_imgs_b64}

    json_response = jsonify(response_data)

    return json_response
    


if __name__ == '__main__':
    app.run()
