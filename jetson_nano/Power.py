import time
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import os
import sys
import lightweight_dataframes as dataframes

if len(sys.argv) < 2:
  print("File not specified")
  print("Usage:",sys.argv[0], "<model.trt>")
  exit()
print("Benchmarking power for", sys.argv[1])

filename = sys.argv[1]
model_name = filename[:-4]

val_x = np.random.rand(150,1,3,3,128)
start_time = time.time()
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(TRT_LOGGER)
with open(filename, 'rb') as f:
    engine_bytes = f.read()
    engine = runtime.deserialize_cuda_engine(engine_bytes)


context = engine.create_execution_context()
batch_size = 1

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    # def __str__(self):
    #     return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    # def __repr__(self):
    #     return self.__str__()

inputs = []
outputs = []
bindings = []
stream = cuda.Stream()
for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * batch_size
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    # Allocate host and device buffers
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    # Append the device buffer to device bindings.
    bindings.append(int(device_mem))
    # Append to the appropriate list.
    if engine.binding_is_input(binding):
        inputs.append(HostDeviceMem(host_mem, device_mem))
    else:
        outputs.append(HostDeviceMem(host_mem, device_mem))

modelLoadTime = (time.time() - start_time)
currentBaseTime = time.time()
def infer(input_img):
    input_img = input_img.flatten()
    np.copyto(inputs[0].host, input_img)
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]
inference_time = 300
df = dataframes.createDataFrame(columnNames=["model_name", "file_name", "num_inferences", "total_time (s)", "load_time (s)", "alloc_time (s)", "inference_time (s)"])
num_infer = 0
avg_time = 0

allocationTime = (time.time() - currentBaseTime)
inference_start_time = time.time()
while time.time() - inference_start_time < inference_time:
    inp_id = np.random.randint(100)
    input_img = val_x[inp_id]
    outs = infer(input_img)
    num_infer+=1

inferenceTime = time.time() - inference_start_time
totalTime = time.time() - start_time
print(filename)
print("-----------------")
print("POWER     PROFILE")
print("")
print("Number of samples analyzed:\t", num_infer)
print("Model Load Time:           \t", modelLoadTime, "(s)")
print("Model Alloc Time:          \t", allocationTime, "(s)")
print("Model Inerence Time:       \t", inferenceTime, "(s)")
print("Total Program Runtime:     \t", totalTime, "(s)")


df = dataframes.append_row(df, {"model_name":model_name,"file_name":filename, "num_inferences":num_infer, "total_time (s)":totalTime, "load_time (s)":modelLoadTime, "alloc_time (s)":allocationTime, "inference_time (s)":inferenceTime})
dataframes.to_csv(df, model_name + "_power_timeline.csv")

