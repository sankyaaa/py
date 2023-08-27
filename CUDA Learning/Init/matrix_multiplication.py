import torch
import time

if torch.cuda.is_available(): #If gpu is available
    device=torch.device("cuda") #Assign gpu as device
else:
    device=torch.device("cpu") #Assign cpu as device
print(device, 'device detected\n')

mat_size = 20 * 20
x = torch.rand(mat_size, mat_size)
y = torch.rand(mat_size, mat_size)

start_time = time.perf_counter()  # Save the start time

torch.matmul(x,y)

end_time = time.perf_counter()  # Save the end time
elapsed_time = end_time - start_time # Calculate the elapsed time
print(x.device, "has taken",elapsed_time,"sec ==", round(elapsed_time,5))
print('start time: ',start_time,'\t end time: ',end_time)

x_gpu = x.to(device)
y_gpu = y.to(device)

'''Initialization takes time so running in loop to check time taken on next iterations''' 
for i in range(5):
    gpu_start_time = time.perf_counter()  # Save the start time
    torch.cuda.synchronize()

    torch.matmul(x_gpu,y_gpu)
    torch.cuda.synchronize()

    gpu_end_time = time.perf_counter()  # Save the end time
    gpu_elapsed_time = gpu_end_time - gpu_start_time # Calculate the elapsed time
    print(x_gpu.device, "has taken",gpu_elapsed_time,"sec ==", round(gpu_elapsed_time,5))
    print('start time: ',gpu_start_time,'\t end time: ',gpu_end_time)

''' Expected output (random values)

cuda device detected

cpu has taken 0.0035392000008869218 sec == 0.00354
start time:  12260.0294992       end time:  12260.0330384
cuda:0 has taken 1.623187400000461 sec == 1.62319
start time:  12262.4321731       end time:  12264.0553605
cuda:0 has taken 0.00043550000009418 sec == 0.00044
start time:  12264.0560651       end time:  12264.0565006
cuda:0 has taken 0.00021019999985583127 sec == 0.00021
start time:  12264.0569347       end time:  12264.0571449
cuda:0 has taken 0.00044809999963035807 sec == 0.00045
start time:  12264.0575676       end time:  12264.0580157
cuda:0 has taken 0.000659799999993993 sec == 0.00066
start time:  12264.0588311       end time:  12264.0594909

'''