import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
from tkinter import filedialog, Tk, Button, Label, Canvas, PhotoImage
from PIL import Image, ImageTk
from patchify import patchify
from tensorflow.keras.models import load_model
from unetr_2d import build_unetr_2d
from metrics import dice_loss, dice_coef

# Config (same as training)
cf = {
    "image_size": 256,
    "num_channels": 3,
    "num_layers": 12,
    "hidden_dim": 128,
    "mlp_dim": 32,
    "num_heads": 6,
    "dropout_rate": 0.1,
    "patch_size": 16
}
cf["num_patches"] = (cf["image_size"] ** 2) // (cf["patch_size"] ** 2)
cf["flat_patches_shape"] = (
    cf["num_patches"],
    cf["patch_size"] * cf["patch_size"] * cf["num_channels"]
)

# Load model
model = build_unetr_2d(cf)
model.compile(loss=dice_loss, optimizer="adam", metrics=[dice_coef, "accuracy"])
model.load_weights(r"C:\Users\LENOVO\AMIT AI\Amit-1\myenv\Scripts\Final Project BME\Brain-Tumor-Segmentation-using-UNET-Transformer-computer-vision-main\Brain-Tumor-Segmentation-using-UNET-Transformer-computer-vision-main\model.keras")

# Tkinter window
root = Tk()
root.title("Brain Tumor Segmentation")

# To keep reference of images
original_img_tk = None
mask_img_tk = None

canvas_orig = Canvas(root, width=cf["image_size"], height=cf["image_size"])
canvas_orig.pack(padx=5, pady=5)
canvas_mask = Canvas(root, width=cf["image_size"], height=cf["image_size"])
canvas_mask.pack(padx=5, pady=5)

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
    image_norm = image / 255.0

    patches = patchify(image_norm, (cf["patch_size"], cf["patch_size"], cf["num_channels"]), step=cf["patch_size"])
    patches = patches.reshape(cf["flat_patches_shape"]).astype(np.float32)
    patches = np.expand_dims(patches, axis=0)
    return patches, image

def postprocess_mask(mask):
    mask = mask[0, :, :, 0]
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask

def show_image_on_canvas(canvas, image_array):
    global original_img_tk, mask_img_tk
    image_pil = Image.fromarray(image_array)
    image_tk = ImageTk.PhotoImage(image_pil)
    canvas.create_image(0, 0, anchor='nw', image=image_tk)

    # Keep reference to avoid garbage collection
    if canvas == canvas_orig:
        original_img_tk = image_tk
    else:
        mask_img_tk = image_tk

def select_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    patches, original = preprocess_image(file_path)
    prediction = model.predict(patches)
    mask = postprocess_mask(prediction)

    show_image_on_canvas(canvas_orig, original)
    show_image_on_canvas(canvas_mask, mask)

# GUI Widgets
label = Label(root, text="Select a brain MRI image to segment:")
label.pack(pady=10)

button = Button(root, text="Browse Image", command=select_image)
button.pack(pady=5)

root.mainloop()
