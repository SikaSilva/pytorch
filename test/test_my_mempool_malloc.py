import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch

using_mem_pool = True

device = torch.device("cuda:0")

if using_mem_pool:
    mem_pool = torch.cuda.MemPool()
    with torch.cuda.use_mem_pool(mem_pool, device=device):
        t1 = torch.full((1, 1), 1.0, dtype=torch.float32, requires_grad=True, device=device)
        t2 = torch.full((2, 2), 2.0, dtype=torch.float32, requires_grad=True, device=device)
        t3 = torch.full((3, 3), 3.0, dtype=torch.float32, requires_grad=True, device=device)
        print(t3)

        del t1
        del t2
        del t3
        torch.cuda.empty_cache()
else:
    t1 = torch.full((1, 1), 1.0, dtype=torch.float32, requires_grad=True, device=device)
    t2 = torch.full((2, 2), 2.0, dtype=torch.float32, requires_grad=True, device=device)
    t3 = torch.full((3, 3), 3.0, dtype=torch.float32, requires_grad=True, device=device)
    print(t3)

    del t1
    del t2
    del t3
    torch.cuda.empty_cache()
    