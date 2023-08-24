To get list gpu list
    $ nvidia-smi --list-gpus
    >>GPU 0: NVIDIA GeForce RTX 3080 Ti (UUID: GPU-*******-****-****-****-************)

Install cuda from https://pytorch.org/get-started/locally/

inputs:
    Operating System : Linux
    Architecture : x86_64
    Distribution : Ubuntu
    Version : 22.04
    Installer Type : runfile(local)

You will get below commands
    $ wget https://developer.download.nvidia.com/compute/cuda/12.2.1/local_installers/cuda_12.2.1_535.86.10_linux.run

    $ sudo sh cuda_12.2.1_535.86.10_linux.run

    $ python -c 'import torch; print torch.cuda.is_available()'