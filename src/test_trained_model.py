from model import UNet
import torch
from pathlib import Path


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = (
    Path(__file__).parent.parent
    / "model"
    / "unet_model1_best.pth"
)


print("Model path:")
print(MODEL_PATH)

print()


# ============================================================
# CREATE MODEL
# ============================================================

model = UNet()


# ============================================================
# LOAD TRAINED CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# ============================================================
# LOAD STATE DICTIONARY
# ============================================================

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict
)


# ============================================================
# EVALUATION MODE
# ============================================================

model.to(DEVICE)

model.eval()


print("========================================")
print("TRAINED MODEL LOADED SUCCESSFULLY")
print("========================================")

print()

print("Device:", DEVICE)

print(
    "Model file exists:",
    MODEL_PATH.exists()
)


# ============================================================
# TEST INPUT
# ============================================================

x = torch.randn(
    1,
    1,
    512,
    512
)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    output = model(x)


# ============================================================
# OUTPUT
# ============================================================

print()

print("Input shape :")
print(x.shape)

print()

print("Output shape:")
print(output.shape)

print()

print("Raw output minimum:")
print(output.min().item())

print()

print("Raw output maximum:")
print(output.max().item())

print()

print("========================================")
print("TEST COMPLETED")
print("========================================")