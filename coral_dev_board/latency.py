import numpy as np
import time
from pycoral.adapters.common import input_size, set_input
from pycoral.adapters.classify import get_classes
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import detect

# Path to your TFLite model
MODEL_PATH = 'your_model.tflite'

# Load the TFLite model and allocate tensors
interpreter = make_interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input shape and dtype
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

# Function to generate random input data
def generate_random_input():
    info = np.iinfo(input_dtype)
    return np.random.randint(info.min, info.max+1, input_shape, dtype=input_dtype)

# Number of inferences to run
NUM_INFERENCES = 100
latencies = []

for _ in range(NUM_INFERENCES):
    # Generate random input data
    input_data = generate_random_input()

    # Set the tensor to point to the input data to be inferred
    set_input(interpreter, input_data)

    # Run inference and measure time
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()

    # Calculate latency and append to list
    latency = end_time - start_time
    latencies.append(latency)

# Calculate average latency
average_latency = sum(latencies) / len(latencies)

print(f'Average latency over {NUM_INFERENCES} inferences: {average_latency:.6f} seconds')
