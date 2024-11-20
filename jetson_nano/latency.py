import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import os
import time
import sys
import lightweight_dataframes as dataframes

if len(sys.argv) < 2:
  print("File not specified")
  print("Usage:",sys.argv[0], "<model.trt>")
  exit()


filename = sys.argv[1]
model_name = filename[:-3]


inp_rand = np.random.rand(150,1,128)
print(inp_rand[1, :, :].shape)

currentBaseTime = time.time()
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(TRT_LOGGER)
with open('./' + filename, 'rb') as f:
    engine_bytes = f.read()
    engine = runtime.deserialize_cuda_engine(engine_bytes)


context = engine.create_execution_context()
batch_size = 1
interpreterReadTime1 = (time.time() - currentBaseTime)*1000

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

inputs = []
outputs = []
bindings = []
currentBaseTime = time.time()
stream = cuda.Stream()
for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * batch_size
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    # Allocate host and device buffers
    host_mem = cuda.pagelocked_empty(-1*size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    # Append the device buffer to device bindings.
    bindings.append(int(device_mem))
    # Append to the appropriate list.
    if engine.binding_is_input(binding):
        inputs.append(HostDeviceMem(host_mem, device_mem))
    else:
        outputs.append(HostDeviceMem(host_mem, device_mem))

allocationTime1 = (time.time() - currentBaseTime)*1000 # ms

def infer(input_ids):
    input_ids = input_ids.flatten()
    np.copyto(inputs[0].host, input_ids)
    #np.copyto(inputs[1].host, type_ids)
    #np.copyto(inputs[2].host, mask_ids)
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]
df = dataframes.createDataFrame(columnNames=["model_name", "avg inference time (ms)","alloc_time (ms)", "total_inference_time (s)", "load_time (ms)", "num_samples", "file_name"])
import time
num_samples = inp_rand.shape[0]
avg_time = 0
for run in range(40):
    for i in range (int(num_samples)):
        start_time = time.time()
        x_sample = inp_rand[i,:,:].astype(np.int32)
        outs = infer(x_sample)
        infer_time = time.time() - start_time
        avg_time+=infer_time

avg_infer_time = avg_time / (40 * num_samples)*1000
print(filename)
print("Model Load Time is", interpreterReadTime1, "(ms)")
print("Allocation Time is", allocationTime1, "(ms)")
print("Inference Time is", avg_time, "(secs)")
print("Total Time is", interpreterReadTime1/1000 + allocationTime1/1000 + avg_time, "(secs)")
print("Avg Time for one inference:", avg_infer_time, "(ms)")
df = dataframes.append_row(df, {"model_name":model_name,"avg inference time (ms)":avg_infer_time, "alloc_time (ms)":allocationTime1, "total_inference_time (s)":avg_time, "load_time (ms)":interpreterReadTime1, "num_samples":num_samples, "file_name":filename})
dataframes.to_csv(df, model_name + "_benchmark.csv")
