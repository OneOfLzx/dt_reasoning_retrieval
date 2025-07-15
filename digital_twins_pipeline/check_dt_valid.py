#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path

INVALID_OBJECT_NUM = 10

def check_dt_validity(test_info_path, dt_dir):
    """
    Check the validity of DT files by examining semantic, caption, and description fields.
    Also check if all three required files (normal DT, coarse DT, and mask JSON) exist.
    
    Args:
        test_info_path (str): Path to the test_info.json file
        dt_dir (str): Path to the directory containing DT files
    
    Returns:
        dict: Statistics about invalid DTs
    """
    # Load test info
    with open(test_info_path, 'r', encoding='utf-8') as f:
        test_info = json.load(f)
    
    total_images = len(test_info)
    print(f"Total images in test_info: {total_images}")
    
    # Statistics
    stats = {
        "total_images": total_images,
        "empty_image_semantic": 0,
        "empty_image_caption": 0,
        "empty_image_embedding": 0,
        "empty_object_description": 0,
        "missing_normal_dt": 0,
        "missing_coarse_dt": 0,
        "missing_mask_json": 0,
        "inconsistent_semantic": 0,
        "inconsistent_caption": 0,
        "coarse_obj_not_in_normal": 0,
        "inconsistent_obj_fields": 0,
        "normal_obj_not_in_mask": 0,
        "insufficient_objects": 0,  # New counter for DTs with fewer than INVALID_OBJECT_NUM objects
        "invalid_images": set(),
        "errors": []
    }
    
    # Check each image
    for idx, item in enumerate(test_info):
        image_path = item["image_path"]
        image_name = os.path.basename(image_path)
        image_name_without_ext = os.path.splitext(image_name)[0]
        
        # Get DT paths
        dt_path = os.path.join(dt_dir, f"{image_name_without_ext}.json")
        coarse_dt_path = os.path.join(dt_dir, f"{image_name_without_ext}.json") #os.path.join(dt_dir, f"{image_name_without_ext}_coarse_grained.json")
        mask_json_path = os.path.join(dt_dir, f"{image_name_without_ext}_mask.json")
        
        is_valid = True
        
        # Check if all required files exist
        if not os.path.exists(dt_path):
            stats["missing_normal_dt"] += 1
            stats["invalid_images"].add(image_name)
            print(f"Missing normal DT: {dt_path}")
            is_valid = False
        
        if not os.path.exists(coarse_dt_path):
            stats["missing_coarse_dt"] += 1
            stats["invalid_images"].add(image_name)
            print(f"Missing coarse DT: {coarse_dt_path}")
            is_valid = False
        
        if not os.path.exists(mask_json_path):
            stats["missing_mask_json"] += 1
            stats["invalid_images"].add(image_name)
            print(f"Missing mask JSON: {mask_json_path}")
            is_valid = False
        
        # If any file is missing, continue to the next image
        if not is_valid:
            continue
        
        # Load all three files
        try:
            with open(dt_path, 'r', encoding='utf-8') as f:
                dt_data = json.load(f)
            
            with open(coarse_dt_path, 'r', encoding='utf-8') as f:
                coarse_dt_data = json.load(f)
            
            with open(mask_json_path, 'r', encoding='utf-8') as f:
                mask_data = json.load(f)
        except Exception as e:
            stats["errors"].append(f"Error loading files for {image_name}: {str(e)}")
            print(f"Error loading files for {image_name}: {str(e)}")
            continue
        
        # Check if normal DT has at least INVALID_OBJECT_NUM objects
        if len(dt_data.get("objects_info", [])) < INVALID_OBJECT_NUM:
            stats["insufficient_objects"] += 1
            stats["invalid_images"].add(image_name)
            print(f"Insufficient objects in {dt_path}: found {len(dt_data.get('objects_info', []))}, need at least {INVALID_OBJECT_NUM}")
            
        # Check normal DT for empty fields
        if dt_data.get("image_info", {}).get("semantic", "") == "":
            stats["empty_image_semantic"] += 1
            stats["invalid_images"].add(image_name)
            print(f"Empty image semantic in {dt_path}")
        
        if dt_data.get("image_info", {}).get("caption", "") == "":
            stats["empty_image_caption"] += 1
            stats["invalid_images"].add(image_name)
            print(f"Empty image caption in {dt_path}")

        if dt_data.get("image_info", {}).get("image_embedding", []) == []:
            stats["empty_image_embedding"] += 1
            stats["invalid_images"].add(image_name)
            print(f"Empty image embedding in {dt_path}")

        for obj in dt_data.get("objects_info", []):
            if obj.get("description", "") == "":
                stats["empty_object_description"] += 1
                stats["invalid_images"].add(image_name)
                print(f"Empty object description: obj_id={obj.get('id')} in {dt_path}")
                break
        
        # Check coarse DT for empty fields
        if coarse_dt_data.get("image_info", {}).get("semantic", "") == "":
            stats["empty_image_semantic"] += 1
            stats["invalid_images"].add(image_name)
            print(f"Empty image semantic in {coarse_dt_path}")
        
        if coarse_dt_data.get("image_info", {}).get("caption", "") == "":
            stats["empty_image_caption"] += 1
            stats["invalid_images"].add(image_name)
            print(f"Empty image caption in {coarse_dt_path}")
        
        if dt_data.get("image_info", {}).get("image_embedding", []) == []:
            stats["empty_image_embedding"] += 1
            stats["invalid_images"].add(image_name)
            print(f"Empty image embedding in {dt_path}")

        for obj in coarse_dt_data.get("objects_info", []):
            if obj.get("description", "") == "":
                stats["empty_object_description"] += 1
                stats["invalid_images"].add(image_name)
                print(f"Empty object description: obj_id={obj.get('id')} in {coarse_dt_path}")
                break
        
        # 1. Check if coarse DT semantic and caption match normal DT
        normal_semantic = dt_data.get("image_info", {}).get("semantic", "")
        coarse_semantic = coarse_dt_data.get("image_info", {}).get("semantic", "")
        if normal_semantic != coarse_semantic:
            stats["inconsistent_semantic"] += 1
            stats["invalid_images"].add(image_name)
            print(f"Inconsistent semantic between normal and coarse DT for {image_name}")
        
        normal_caption = dt_data.get("image_info", {}).get("caption", "")
        coarse_caption = coarse_dt_data.get("image_info", {}).get("caption", "")
        if normal_caption != coarse_caption:
            stats["inconsistent_caption"] += 1
            stats["invalid_images"].add(image_name)
            print(f"Inconsistent caption between normal and coarse DT for {image_name}")
        
        # 2. Check if coarse DT objects exist in normal DT with same field values
        normal_objects = {obj.get("id"): obj for obj in dt_data.get("objects_info", [])}
        
        # Automatically determine fields to check by getting common fields between objects
        if dt_data.get("objects_info") and coarse_dt_data.get("objects_info"):
            normal_obj_fields = set(dt_data["objects_info"][0].keys())
            coarse_obj_fields = set(coarse_dt_data["objects_info"][0].keys())
            common_fields = normal_obj_fields.intersection(coarse_obj_fields)
            # Remove fields that are expected to be different or don't need comparison
            fields_to_exclude = set()
            fields_to_check = list(common_fields - fields_to_exclude)
        else:
            print(f"[Error]: No objects info in {dt_path} or {coarse_dt_path}")
            fields_to_check = []
        
        for coarse_obj in coarse_dt_data.get("objects_info", []):
            coarse_obj_id = coarse_obj.get("id")
            
            if coarse_obj_id not in normal_objects:
                stats["coarse_obj_not_in_normal"] += 1
                stats["invalid_images"].add(image_name)
                print(f"Object ID {coarse_obj_id} in coarse DT not found in normal DT for {image_name}")
                continue
            
            normal_obj = normal_objects[coarse_obj_id]
            
            # Check if object fields match
            for field in fields_to_check:
                if coarse_obj.get(field) != normal_obj.get(field):
                    stats["inconsistent_obj_fields"] += 1
                    stats["invalid_images"].add(image_name)
                    print(f"Inconsistent {field} for object ID {coarse_obj_id} in {image_name}")
                    break
        
        # 3. Check if all normal DT objects have corresponding masks in mask JSON
        # Handle the mask JSON format based on the actual structure
        mask_ids = set()
        if isinstance(mask_data, dict) and "objects_mask_info" in mask_data:
            # Format: {"objects_mask_info": [{"id": 0, "mask": "..."}, ...]}
            mask_ids = {mask.get("id") for mask in mask_data.get("objects_mask_info", [])}
        elif isinstance(mask_data, dict) and "image_info" in mask_data:
            # Format: {"image_info": {...}, "objects_mask_info": [{"id": 0, "mask": "..."}, ...]}
            mask_ids = {mask.get("id") for mask in mask_data.get("objects_mask_info", [])}
        elif isinstance(mask_data, list):
            # Format: [{"id": 0, "mask": "..."}, ...]
            mask_ids = {mask.get("id") for mask in mask_data}
        
        for obj in dt_data.get("objects_info", []):
            obj_id = obj.get("id")
            if obj_id not in mask_ids:
                stats["normal_obj_not_in_mask"] += 1
                stats["invalid_images"].add(image_name)
                print(f"Object ID {obj_id} in normal DT not found in mask JSON for {image_name}")
        
        # 4. Check if the image_embedding in normal DT is the same as the image_embedding in coarse DT
        normal_image_embedding = dt_data.get("image_info", {}).get("image_embedding", [])
        coarse_image_embedding = coarse_dt_data.get("image_info", {}).get("image_embedding", [])
        if normal_image_embedding != coarse_image_embedding:
            stats["inconsistent_image_embedding"] += 1
            stats["invalid_images"].add(image_name)
            print(f"Inconsistent image embedding between normal and coarse DT for {image_name}")

        # Print progress
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{total_images} images")
    
    # Convert set to list for JSON serialization
    stats["invalid_images"] = list(stats["invalid_images"])
    stats["total_invalid_images"] = len(stats["invalid_images"])
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Check DT validity')
    parser.add_argument('--query_json_path', type=str, required=True)
    parser.add_argument('--dt_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str)
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.query_json_path):
        print(f"Error: test_info file not found at {args.query_json_path}")
        return
    
    if not os.path.exists(args.dt_dir):
        print(f"Error: DT directory not found at {args.dt_dir}")
        return
    
    # Run validation
    print(f"Checking DT validity...")
    print(f"Test info: {args.query_json_path}")
    print(f"DT directory: {args.dt_dir}")
    
    stats = check_dt_validity(args.query_json_path, args.dt_dir)
    
    # Print summary
    print("\n===== Summary =====")
    print(f"Total images: {stats['total_images']}")
    print(f"Images with empty image semantic: {stats['empty_image_semantic']}")
    print(f"Images with empty image caption: {stats['empty_image_caption']}")
    print(f"Images with empty image embedding: {stats['empty_image_embedding']}")
    print(f"Images with empty object description: {stats['empty_object_description']}")
    print(f"Images missing normal DT: {stats['missing_normal_dt']}")
    print(f"Images missing coarse DT: {stats['missing_coarse_dt']}")
    print(f"Images missing mask JSON: {stats['missing_mask_json']}")
    print(f"Images with insufficient objects (< {INVALID_OBJECT_NUM}): {stats['insufficient_objects']}")
    print(f"Images with inconsistent semantic: {stats['inconsistent_semantic']}")
    print(f"Images with inconsistent caption: {stats['inconsistent_caption']}")
    print(f"Images with coarse objects not in normal DT: {stats['coarse_obj_not_in_normal']}")
    print(f"Images with inconsistent object fields: {stats['inconsistent_obj_fields']}")
    print(f"Images with normal objects not in mask: {stats['normal_obj_not_in_mask']}")
    print(f"Total invalid images: {stats['total_invalid_images']}")
    print(f"Total errors: {len(stats['errors'])}")
    

    with open(args.query_json_path, "r") as f:
        test_info = json.load(f)

    with open(os.path.join(args.output_dir, "invalid_images.txt"), "w", encoding="utf-8") as f:
        for img_name in stats["invalid_images"]:
            for item in test_info:
                if os.path.basename(item["image_path"]) == img_name:
                    f.write(item["image_path"] + "\n")
                    break

    # Save report
    with open(os.path.join(args.output_dir, "check_dt_valid.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"Report saved to {args.output_dir}")


if __name__ == "__main__":
    main()
