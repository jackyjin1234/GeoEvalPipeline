#!/usr/bin/env python3

import cv2
import numpy as np
from pathlib import Path


def downscale(image_path, output_left_path, output_right_path):
    """
    Downscale a 2048x1024 Cityscapes image by splitting and resizing.
    
    Args:
        image_path: Path to input 2048x1024 image
        output_left_path: Path to save left 512x512 image
        output_right_path: Path to save right 512x512 image
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    height, width = img.shape[:2]
    if width != 2048 or height != 1024:
        raise ValueError(f"Expected 2048x1024 image, got {width}x{height}")
    
    left_half = img[:, :1024]
    right_half = img[:, 1024:]
    
    left_resized = cv2.resize(left_half, (512, 512), interpolation=cv2.INTER_AREA)
    right_resized = cv2.resize(right_half, (512, 512), interpolation=cv2.INTER_AREA)
    
    cv2.imwrite(str(output_left_path), left_resized)
    cv2.imwrite(str(output_right_path), right_resized)
    
    return left_resized, right_resized


def upscale(left_image_path, right_image_path, output_path):
    """
    Upscale two 512x512 images to create a 2048x1024 merged image.
    
    Args:
        left_image_path: Path to left 512x512 image
        right_image_path: Path to right 512x512 image
        output_path: Path to save merged 2048x1024 image
    """
    left_img = cv2.imread(str(left_image_path))
    right_img = cv2.imread(str(right_image_path))
    
    if left_img is None:
        raise ValueError(f"Could not load left image from {left_image_path}")
    if right_img is None:
        raise ValueError(f"Could not load right image from {right_image_path}")
    
    left_h, left_w = left_img.shape[:2]
    right_h, right_w = right_img.shape[:2]
    
    if left_w != 512 or left_h != 512:
        raise ValueError(f"Expected 512x512 left image, got {left_w}x{left_h}")
    if right_w != 512 or right_h != 512:
        raise ValueError(f"Expected 512x512 right image, got {right_w}x{right_h}")
    
    left_upscaled = cv2.resize(left_img, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    right_upscaled = cv2.resize(right_img, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    
    merged = np.hstack([left_upscaled, right_upscaled])
    
    cv2.imwrite(str(output_path), merged)
    
    return merged


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python image_processor.py <mode>")
        print("  mode: 'downscale' or 'upscale' or 'test'")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "downscale":
        input_path = input("Enter input image path: ")
        left_output = input("Enter left output path: ")
        right_output = input("Enter right output path: ")
        
        downscale(input_path, left_output, right_output)
        print(f"Downscaled {input_path} to {left_output} and {right_output}")
        
    elif mode == "upscale":
        left_input = input("Enter left image path: ")
        right_input = input("Enter right image path: ")
        output_path = input("Enter merged output path: ")
        
        upscale(left_input, right_input, output_path)
        print(f"Upscaled {left_input} and {right_input} to {output_path}")
        
    elif mode == "test":
        test_image = "/home/jackyjin/cityscapesScripts/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png"
        if Path(test_image).exists():
            print(f"Testing with {test_image}")
            downscale(test_image, "test_left.png", "test_right.png")
            print("Downscaled to test_left.png and test_right.png")
            
            upscale("test_left.png", "test_right.png", "test_merged.png")
            print("Upscaled to test_merged.png")
        else:
            print(f"Test image not found: {test_image}")
    
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)