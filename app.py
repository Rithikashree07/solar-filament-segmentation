import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Solar Filament Segmentation",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ Solar Filament Segmentation")
st.write("U-Net based solar filament segmentation using H-alpha images")


# ============================================================
# MODEL
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(1, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(self.pool1(e1))

        e3 = self.enc3(self.pool2(e2))

        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "unet_model1_best.pth"

if not MODEL_PATH.exists():
    st.error(f"Model not found:\n{MODEL_PATH}")
    st.stop()


@st.cache_resource
def load_model():

    model = UNet()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model, checkpoint


model, checkpoint = load_model()

st.success("✅ Trained U-Net model loaded successfully")

if "epoch" in checkpoint:
    st.info(
        f"Model checkpoint: Epoch {checkpoint['epoch']}"
    )

if "val_dice" in checkpoint:
    st.info(
        f"Validation Dice: {checkpoint['val_dice']:.4f}"
    )


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a solar H-alpha image",
    type=["jpg", "jpeg", "png"]
)

threshold = st.slider(
    "Segmentation Threshold",
    0.10,
    0.90,
    0.35,
    0.05
)


# ============================================================
# PREDICT
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    st.write(
        f"Original image size: {image.width} × {image.height}"
    )

    if image.size != (2048, 2048):
        image = image.resize(
            (2048, 2048),
            Image.Resampling.BILINEAR
        )

    image_array = (
        np.array(image, dtype=np.float32) / 255.0
    )

    st.image(
        image_array,
        caption="Input Solar Image",
        width=500
    )

    if st.button(
        "🔍 Predict Solar Filament",
        type="primary"
    ):

        probability_map = np.zeros(
            (2048, 2048),
            dtype=np.float32
        )

        progress = st.progress(0)

        with torch.no_grad():

            for row in range(4):

                for col in range(4):

                    y = row * 512
                    x = col * 512

                    patch = image_array[
                        y:y+512,
                        x:x+512
                    ]

                    tensor = torch.from_numpy(
                        patch.copy()
                    ).float()

                    tensor = tensor.unsqueeze(0).unsqueeze(0)

                    output = model(tensor)

                    probability = torch.sigmoid(
                        output
                    )[0, 0].numpy()

                    probability_map[
                        y:y+512,
                        x:x+512
                    ] = probability

                    progress.progress(
                        ((row * 4 + col + 1) / 16)
                    )

        progress.empty()

        # ====================================================
        # MASK
        # ====================================================

        predicted_mask = (
            probability_map >= threshold
        ).astype(np.uint8)

        filament_pixels = int(
            predicted_mask.sum()
        )

        total_pixels = predicted_mask.size

        filament_percentage = (
            filament_pixels / total_pixels
        ) * 100

        # ====================================================
        # OVERLAY
        # ====================================================

        rgb = np.stack(
            [image_array] * 3,
            axis=-1
        )

        overlay = rgb.copy()

        overlay[predicted_mask == 1, 0] = 1.0
        overlay[predicted_mask == 1, 1] *= 0.25
        overlay[predicted_mask == 1, 2] *= 0.25

        # ====================================================
        # RESULTS
        # ====================================================

        st.success("✅ Prediction completed!")

        st.subheader("Segmentation Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(
                image_array,
                caption="Original Image",
                use_container_width=True
            )

        with col2:
            st.image(
                predicted_mask * 255,
                caption="Predicted Filament Mask",
                use_container_width=True
            )

        with col3:
            st.image(
                overlay,
                caption="Filament Overlay",
                use_container_width=True
            )

        # ====================================================
        # METRICS
        # ====================================================

        st.subheader("Detection Results")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Filament Pixels",
                f"{filament_pixels:,}"
            )

        with c2:
            st.metric(
                "Filament Area",
                f"{filament_percentage:.2f}%"
            )

        with c3:
            st.metric(
                "Threshold",
                f"{threshold:.2f}"
            )

        st.subheader("Model Probability")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Minimum",
                f"{probability_map.min():.6f}"
            )

        with c2:
            st.metric(
                "Maximum",
                f"{probability_map.max():.6f}"
            )

        with c3:
            st.metric(
                "Mean",
                f"{probability_map.mean():.6f}"
            )

        if filament_pixels > 0:
            st.success(
                "☀️ Solar filament region detected."
            )
        else:
            st.warning(
                "No filament detected at this threshold."
            )