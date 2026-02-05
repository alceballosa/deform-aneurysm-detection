import time

import torch


def hold_gpu():
    print("Entering function")
    start = time.time()
    x1 = torch.randn(100000, 10000, device="cuda:0")
    x2 = torch.randn(100000, 10000, device="cuda:1")
    x3 = torch.randn(100000, 10000, device="cuda:2")
    x4 = torch.randn(100000, 10000, device="cuda:3")
    print("GPU operation starting")
    while True:
        x1 = x1 * x1
        x2 = x2 * x2
        x3 = x3 * x3
        x4 = x4 * x4
        total_time = time.time() - start
        #rint(f"Total time: {total_time:.2f} seconds")
        if time.time() - start > 1200:
            print("GPU operation ending")

            break

    
print("Starting to use gpu")
hold_gpu()  
