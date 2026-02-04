import torch

# mixed distance
def mixed_distance_torch(C, P, S, m):
    """
    C: (K, 5)  -> [L, A, B, Y, X]
    P: (H, W, 5)
    """
    dc = (P[..., :3] - C[:, None, None, :3]).pow(2).sum(dim=-1)
    ds = (P[..., 3:] - C[:, None, None, 3:]).pow(2).sum(dim=-1)
    D = torch.sqrt(dc + (m*m / (S*S)) * ds)
    return D

# segmentation
import cv2
import math
import torch

def SLIC_GPU(
    filename,
    k,
    m,
    threshold=0.1,
    max_iters=10,
    device="cuda"
):
    # --- Load image ---
    im = cv2.imread(filename)
    H, W, _ = im.shape

    im_lab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    im_lab = torch.tensor(im_lab, dtype=torch.float32, device=device)

    N = H * W
    S = int(math.sqrt(N / k))

    # --- Initialize cluster centers ---
    ys = torch.arange(S//2, H, S, device=device)
    xs = torch.arange(S//2, W, S, device=device)

    centers = []
    for y in ys:
        for x in xs:
            l, a, b = im_lab[int(y), int(x)]
            centers.append(torch.tensor([l, a, b, y, x], device=device))

    Ck = torch.stack(centers)   # (K,5)
    K = Ck.shape[0]

    # --- Pixel feature grid ---
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )

    P = torch.stack([
        im_lab[..., 0],
        im_lab[..., 1],
        im_lab[..., 2],
        yy,
        xx
    ], dim=-1)  # (H,W,5)

    labels = torch.zeros((H, W), dtype=torch.long, device=device)
    prev_error = 1e20

    for it in range(max_iters):
        # --- Distance computation (GPU) ---
        D = mixed_distance_torch(Ck, P, S, m)  # (K,H,W)

        labels = torch.argmin(D, dim=0)

        # --- Update centers ---
        error = 0.0
        for k_i in range(K):
            mask = labels == k_i
            if mask.sum() == 0:
                continue
            new_center = P[mask].mean(dim=0)
            error += torch.norm(Ck[k_i] - new_center)
            Ck[k_i] = new_center

        improvement = abs(prev_error - error.item()) / prev_error
        prev_error = error.item()

        print(f"Iter {it}, error={error.item():.3f}")

        if improvement <= threshold:
            break

    return labels

# boundary visualization
import numpy as np

def show_segmentation_gpu(filename, savename, labels, color=(0,0,255)):
    labels = labels.cpu().numpy()
    im = cv2.imread(filename)

    H, W = labels.shape
    for y in range(1, H):
        im[y][labels[y] != labels[y-1]] = color
    for x in range(1, W):
        im[:, x][labels[:, x] != labels[:, x-1]] = color

    cv2.imwrite(savename, im)
    return im

