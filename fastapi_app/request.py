import requests
import json
import base64
from PIL import Image
from itsdangerous import base64_encode
import io
import base64

# Load the image data from a file
with open('../images/safari_png/IMG_0927.png', 'rb') as f:
    image_data = f.read()
f.close()


# Encode the image data as a base64 string
image_b64 = base64.b64encode(image_data).decode('utf-8')

# Define the payload data as a dictionary
payload = {"initial_image": image_b64,
"prompt": "a stuffed brown meerkat dressed in a zebra suit, cartoon, Pixar, Disney character, 3D render, modern disney style",
"negative_prompt": "disfigured, misaligned, ugly, blurry, grumpy, grey, dark, big eyes, person, human, fuzzy, furry",
"final_images_to_return": 3,
"seed": 123}

# Convert the payload data to a JSON string
json_payload = json.dumps(payload)

# Define the headers to include the content type as JSON
headers = {'Content-Type': 'application/json'}

# Send the POST request to the server
ngrok_url = "http://3745-34-141-229-210.ngrok.io"
response = requests.post(f'{ngrok_url}/generate_image_rest_direct', data=json_payload, headers=headers)


# Handle the response
print(response.status_code)
response_json = response.json()

print(response_json.keys())
generated_imgs = response_json['generated_images']
generated_image_data = [base64.b64decode(i) for i in generated_imgs]

# Load the image data into a PIL Image object
generated_image = [Image.open(io.BytesIO(i)) for i in generated_image_data]

# Show returned images
[i.show() for i in generated_image]

