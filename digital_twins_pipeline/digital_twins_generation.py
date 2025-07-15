import argparse
import os
import sys
from utils.log import log_info, log_verbose, set_log_level
from .image_to_digital_pipeline import ImageToDigitalTwinsPipeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path_list_txt', type=str)
    parser.add_argument('--digital_twins_dir', type=str)
    args = parser.parse_args()

    set_log_level(True)

    os.makedirs(args.digital_twins_dir, exist_ok=True)

    with open(args.image_path_list_txt, 'r') as f:
        image_path_list = [line.strip() for line in f]
    if len(image_path_list) <= 0:
        sys.exit(0)

    img_path_set = set()
    img_dir = os.path.dirname(image_path_list[0])

    filtered_image_path_list = []
    for img_path in image_path_list:
        if img_path not in img_path_set:
            filtered_image_path_list.append(img_path)
            img_path_set.add(img_path)

    log_info(f"Image list: {len(filtered_image_path_list)}, {filtered_image_path_list}")
    if len(filtered_image_path_list) <= 0:
        sys.exit(0)

    pipeline = ImageToDigitalTwinsPipeline()
    pipeline.image_to_digital_twins(filtered_image_path_list, args.digital_twins_dir)
