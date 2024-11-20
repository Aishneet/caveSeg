import time
import numpy as np
from pycoral.adapters.common import input_size, set_input
from pycoral.adapters.classify import get_classes
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import detect

# Define the path to the TFLite model
model_path = 'big_big_deepthwise_unet_full_integer_quant_edgetpu.tflite'

# Load the model using the Edge TPU interpreter
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']
input_type = input_details[0]['dtype']

# Generate random input data
def generate_random_input(input_shape, input_type):
    if np.issubdtype(input_type, np.integer):
        info = np.iinfo(input_type)
        return np.random.randint(info.min, info.max+1, input_shape, dtype=input_type)
    elif np.issubdtype(input_type, np.floating):
        return np.random.random(input_shape).astype(input_type)
    else:
        raise ValueError("Unsupported input type")

# Set the duration for running the model (in seconds)
duration = 5 * 60  # 5 minutes
end_time = time.time() + duration
count =0
while time.time() < end_time:
    # Generate random input data
    input_data = generate_random_input(input_shape, input_type)

    # Set the input tensor
    set_input(interpreter, input_data)

    # Run inference
    interpreter.invoke()
    count+=1

    # Get the results (optional)
    #results = detect.get_objects(interpreter, 0.05)
    #if not results:
        #print("No objects detected")
    print("count:", count)

print('Inference completed.')

                                             