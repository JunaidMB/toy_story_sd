from PIL import Image
import glob
import os
import pillow_heif
pillow_heif.register_heif_opener()

# Convert heic files to PNG
# List filenames
images_filename = glob.glob("./safari/IMG_*")

labels = [i.split("/")[-1].split(".")[0] for i in images_filename]

# Make an output 
output_dir = "./safari_png/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open images
images = [Image.open(i) for i in images_filename]

# Resize images
resized_images = [i.resize((512, 512)) for i in images]

# Convert the images to RGB
resized_images_rgb = [i.convert("RGB") for i in resized_images]

# Save images
for idx, image in enumerate(resized_images_rgb):
    image.save( "".join([output_dir, labels[idx], ".png"]) )

