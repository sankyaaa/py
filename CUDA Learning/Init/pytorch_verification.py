import torch

mat_size = 4
#Variable on CPU
x = torch.rand(mat_size, mat_size)
print(x,"on",x.device)

#variable on GPU
if torch.cuda.is_available():
    device=torch.device("cuda")
    x_gpu = x.to(device)
    print(x_gpu)
else:
    print("failed to detect cuda")

''' Expected output (random values)

tensor([[0.0113, 0.2387, 0.9091, 0.0092],
        [0.3004, 0.1877, 0.4133, 0.4684],
        [0.0954, 0.3185, 0.4998, 0.9194],
        [0.0219, 0.0143, 0.1788, 0.4887]]) on cpu
tensor([[0.0113, 0.2387, 0.9091, 0.0092],
        [0.3004, 0.1877, 0.4133, 0.4684],
        [0.0954, 0.3185, 0.4998, 0.9194],
        [0.0219, 0.0143, 0.1788, 0.4887]], device='cuda:0')

'''