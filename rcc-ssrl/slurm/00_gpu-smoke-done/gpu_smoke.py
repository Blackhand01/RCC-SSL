#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, torch, torch.nn as nn

def main():
    print("PyTorch:", torch.__version__)
    cuda_ok = torch.cuda.is_available()
    print("CUDA available:", cuda_ok)
    ngpu = torch.cuda.device_count()
    print("GPU count:", ngpu)
    # Rispetta eventuale binding di SLURM/CUDA_VISIBLE_DEVICES
    dev_index = int(os.environ.get("LOCAL_RANK", "0"))
    if ngpu > 0:
        # Se SLURM ha impostato bind, l'indice 0 è la tua GPU allocata
        torch.cuda.set_device(dev_index if dev_index < ngpu else 0)
        device = torch.device("cuda")
        print("Using device:", torch.cuda.get_device_name(torch.cuda.current_device()))
        print("Device index (within job):", torch.cuda.current_device())
        print("Memory (MB):", torch.cuda.get_device_properties(0).total_memory/1024**2)

    else:
        device = torch.device("cpu")
        print("Falling back to CPU.")

    # Modello dummy piccolo
    model = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 1024)).to(device)
    x = torch.randn(256, 1024, device=device)
    # Mini warmup/forward
    iters = 50 if cuda_ok else 5
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters):
            y = model(x)
    torch.cuda.synchronize() if cuda_ok else None
    print("OK: forward finished in %.3fs" % (time.time()-t0))

if __name__ == "__main__":
    main()
