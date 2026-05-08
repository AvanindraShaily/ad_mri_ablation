import os
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
import pywt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(PROJECT_ROOT, "OASIS")
OUTPUT_DIR = os.path.join(INPUT_DIR, "haar_preprocessed")

TARGET_SIZE = 128


def haar_preprocess(image_path, target_size=TARGET_SIZE):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img, dtype=np.float32) / 255.0

    LL, (LH, HL, HH) = pywt.dwt2(img_array, 'haar')

    channels = [img_array, img_array, img_array]

    for subband in [LL, LH, HL]:
        scale_h = target_size / subband.shape[0]
        scale_w = target_size / subband.shape[1]
        resized = zoom(subband, (scale_h, scale_w), order=1)
        channels.append(resized)

    for i in range(3):
        scale_h = target_size / channels[i].shape[0]
        scale_w = target_size / channels[i].shape[1]
        channels[i] = zoom(channels[i], (scale_h, scale_w), order=1)

    output = np.stack(channels, axis=0).astype(np.float32)
    return output


def main():
    class_names = [
        d for d in os.listdir(INPUT_DIR)
        if os.path.isdir(os.path.join(INPUT_DIR, d))
        and d not in ("dtcwt_preprocessed", "haar_preprocessed")
    ]

    for cls_name in class_names:
        cls_input_dir = os.path.join(INPUT_DIR, cls_name)
        cls_output_dir = os.path.join(OUTPUT_DIR, cls_name)
        os.makedirs(cls_output_dir, exist_ok=True)

        image_files = sorted(os.listdir(cls_input_dir))

        for img_name in image_files:
            img_path = os.path.join(cls_input_dir, img_name)
            out_name = os.path.splitext(img_name)[0] + '.npy'
            out_path = os.path.join(cls_output_dir, out_name)
            wavelet_data = haar_preprocess(img_path)
            np.save(out_path, wavelet_data)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()