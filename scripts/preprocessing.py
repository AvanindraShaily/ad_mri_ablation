import os
import sys
import numpy as np
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)
if not hasattr(np, 'issubsctype'):
    np.issubsctype = np.issubdtype
from PIL import Image
from scipy.ndimage import zoom
import dtcwt

dataset_name = "OASIS" # "OASIS" or "Alzheimer (Preprocessed Data)"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(PROJECT_ROOT, dataset_name)
OUTPUT_DIR = os.path.join(INPUT_DIR, "dtcwt_preprocessed")

NLEVELS = 2
TARGET_SIZE = 128


def dtcwt_preprocess(image_path, nlevels=NLEVELS, target_size=TARGET_SIZE):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)

    img_array = img_array / 255.0

    transform = dtcwt.Transform2d()
    result = transform.forward(img_array, nlevels=nlevels)

    channels = []
    for level in range(nlevels):
        highpass = result.highpasses[level]
        magnitudes = np.abs(highpass)

        for orientation in range(6):
            subband = magnitudes[:, :, orientation]
            # Resize
            scale_h = target_size / subband.shape[0]
            scale_w = target_size / subband.shape[1]
            resized = zoom(subband, (scale_h, scale_w), order=1)
            channels.append(resized)

    lowpass = result.lowpass
    scale_h = target_size / lowpass.shape[0]
    scale_w = target_size / lowpass.shape[1]
    lowpass_resized = zoom(np.abs(lowpass), (scale_h, scale_w), order=1)
    channels.append(lowpass_resized)

    output = np.stack(channels, axis=0).astype(np.float32)
    return output

def main():
    class_names = [
        d for d in os.listdir(INPUT_DIR)
        if os.path.isdir(os.path.join(INPUT_DIR, d)) and d != "dtcwt_preprocessed"
    ]

    for cls_name in class_names:
        cls_input_dir = os.path.join(INPUT_DIR, cls_name)
        cls_output_dir = os.path.join(OUTPUT_DIR, cls_name)
        os.makedirs(cls_output_dir, exist_ok=True)

        image_files = sorted([
            f for f in os.listdir(cls_input_dir)
        ])

        for img_name in image_files:
            img_path = os.path.join(cls_input_dir, img_name)
            out_name = img_name.replace('jpg', 'npy')
            out_path = os.path.join(cls_output_dir, out_name)
            wavelet_data = dtcwt_preprocess(img_path)
            np.save(out_path, wavelet_data)
            print(out_path)

if __name__ == "__main__":
    main()
