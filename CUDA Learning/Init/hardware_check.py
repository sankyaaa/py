import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available()) #To check if CUDA is available

#To list details of available GPU's
for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i).name)

''' Expected output

<torch version>
<cuda version>
True
<GPU name>

'''