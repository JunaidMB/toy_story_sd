# Running the Flask App

Guidance on how to run this in conjunction with colab (to take advantage of the GPUs) is [here](https://www.youtube.com/watch?v=wBCEDCiQh3Q). This is a prototype. Once the API endpoint is running, use the `request.py` script to test it.

## Next steps
1. Implement task queues for a long running process
2. Ammend inference function to fix 5 number of steps vs 2
3. Find a way to run the app with GPUs properly vs a collab notebook
4. Rebuild the app with fastapi to use pydantic basemodel
5. Make into a webapp and integrate with a frontend