import tensorflow as tf

if tf.test.is_gpu_available():
    # Get the name of the current GPU
    gpu_name = tf.test.gpu_device_name()
    print(f"Training on GPU: {gpu_name}")
else:
    print("Training on CPU")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    # Get the name of each GPU
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. No GPUs detected.")

print(torch.backends.cudnn.version())