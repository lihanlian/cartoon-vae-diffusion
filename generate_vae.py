import os
import torch
from torchvision.utils import save_image
from vae import VAE
from utils import set_seed

# --- 0) Reproducibility ---
SEED = 42
set_seed(SEED)

# --- 1) Config ---
IMAGE_SIZE= 64          # must match training
Z_DIM     = 512
CKPT_PATH = f"models/vae/vae_zdim{Z_DIM}.pth"
OUT_DIR   = f"runs/vae"
N_SAMPLES = 25           # e.g. 8×8 grid

os.makedirs(OUT_DIR, exist_ok=True)

# --- 2) Load model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = VAE(in_channels=3, image_size=IMAGE_SIZE, z_dim=Z_DIM).to(device)
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.eval()

# --- 3) Sample & save ---
with torch.no_grad():
    # set seed again right before sampling
    torch.manual_seed(SEED)

    z = torch.randn(N_SAMPLES, Z_DIM, device=device)
    samples = model.dec(z)           # in [-1,1]
    samples = (samples + 1) * 0.5    # to [0,1]
    save_image(
        samples,
        os.path.join(OUT_DIR, "sample_grid.png"),
        nrow=int(N_SAMPLES**0.5),
        pad_value=1.0
    )
    print(f"→ Wrote samples to {OUT_DIR}/sample_grid.png")
