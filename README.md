# Toy Story Creator Stable Diffusion 

The main notebook in this project is `CLIP_img2img_instruct_workflow.ipynb`. This notebook is to be run with a GPU runtime in Google Colab. It reads input images from the images directory (which must be read into Colab) and uses the Hugging Face Diffusers Pipelines to perform image to image transformation using Stable Diffusion. There is also support for using CLIP interrogators to generate image captions before applying the image to image pipeline. 

Also played with instruct pix2pix with mixed results.

The goal is to generate animated versions for the input images of toys.

## Next Steps

1. Explore Background transformation - for some images, the transformation is poor. If we could generate images via text to image and use them as backgrounds to the transformed toys, it would give a stronger impression of the toys being cartoonised. One issue to address is that the background must feature naturally in the image aspect ratio, switching a background without consideration for where the character model will be gives unrealistic output.

2. Explore inpainting and instruct pix2pix for background editing. Playground AI has a good example, illustrated [here](https://www.youtube.com/watch?v=-I9-2XK3kOs)

3. Build API: Either with Flask or FASTAPI. Need to create a service that takes an input image and returns a range of transformed images via stable diffusion. Need to think about how to design this, do we send an image as a POST request? If so we need to implement task queues and streaming body responses. Or do we upload input images to a COS bucket with a namespace and save the model outputs to a different directory in the COS bucket? All APIs will require GPU support.

4. Gradio app on Hugging face spaces as an alternative to an API?

5. Finetune our own stable diffusion model with Disney Pixar style images.

6. Try other Stable Diffusion models on Hugging Face - new models are coming out daily and the models tried in this notebook could be outdated.

7. Explore other Neural Style Transfer methods to Stable Diffusion - should ideally be as easy as using HuggingFace Pipelines.


