import argparse
import time
from tqdm import tqdm
import cv2 as cv
import os

from utils import save_image
from qr_detector import QrDetector


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process QR code images.')
    parser.add_argument('input_dir', help='path to directory with input images')
    parser.add_argument('output_dir', help='path to directory with output images')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    input_images_basenames = os.listdir(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    detector = QrDetector()

    start_processing_time = time.time()
    n_recognized_qr = 0

    for basename in tqdm(input_images_basenames):
        input_path = os.path.join(input_dir, basename)
        output_path = os.path.join(output_dir, basename)

        image = cv.imread(input_path)
        output_image = detector(image)

        if output_image is None:
            continue
        n_recognized_qr += 1
        save_image(output_image, output_path)

    end_processing_time = time.time()

    avg_time_per_image = (end_processing_time - start_processing_time) / len(input_images_basenames)

    print(f'Average seconds per image: {avg_time_per_image:.2f}\nRecognized {n_recognized_qr}/{len(input_images_basenames)} = {n_recognized_qr/len(input_images_basenames):.2f} images. ')
