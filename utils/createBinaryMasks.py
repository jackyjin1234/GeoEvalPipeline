#!/usr/bin/python
#
# Creates binary masks for selected categories from Cityscapes annotations
# 
# Usage: createBinaryMasks.py [OPTIONS] <input json> <output directory>
# Options:
#   -h                    print help text
#   -c, --categories      comma-separated list of categories (e.g., "vehicle,human")
#   -l, --labels          comma-separated list of specific labels (e.g., "car,person") 
#   -s, --suffix          suffix for output files (default: "mask")
#   --combine             create single combined mask for all selected categories
#   --list-categories     list all available categories and exit
#   --list-labels         list all available labels and exit
#
# Examples:
#   # Create binary masks for vehicles and humans
#   createBinaryMasks.py -c "vehicle,human" input.json ./masks/
#   
#   # Create mask for specific labels
#   createBinaryMasks.py -l "car,truck,bus" input.json ./masks/
#   
#   # Create single combined mask
#   createBinaryMasks.py -c "vehicle" --combine input.json ./masks/
#
#   python3 -m cityscapesscripts.preparation.createBinaryMasks -l 
#   "car,person" testcity_000000_000019_gt_polygons.json test_masks/

from __future__ import print_function, absolute_import, division
import os, sys, getopt, glob
import numpy as np
from PIL import Image, ImageDraw

# cityscapes imports  
from cityscapesscripts.helpers.annotation import Annotation
from cityscapesscripts.helpers.labels import labels, name2label, category2labels
from cityscapesscripts.helpers.csHelpers import printError, ensurePath, getCoreImageFileName

def printHelp():
    print('{} [OPTIONS] inputJson outputDir'.format(os.path.basename(sys.argv[0])))
    print('')
    print('Creates binary masks for selected categories from Cityscapes annotations')
    print('')
    print('Options:')
    print(' -h                    Print this help')
    print(' -c, --categories      Comma-separated categories (vehicle,human,nature,flat,construction,object,sky)')
    print(' -l, --labels          Comma-separated specific labels (car,person,building,etc.)')  
    print(' -s, --suffix          Suffix for output files (default: "mask")')
    print('     --combine         Create single combined mask for all selected categories')
    print('     --list-categories List all available categories and exit')
    print('     --list-labels     List all available labels and exit')
    print('')
    print('Examples:')
    print(' # Create binary masks for vehicles and humans')
    print(' {} -c "vehicle,human" input.json ./masks/'.format(os.path.basename(sys.argv[0])))
    print('')
    print(' # Create mask for specific labels')
    print(' {} -l "car,truck,bus" input.json ./masks/'.format(os.path.basename(sys.argv[0])))
    print('')
    print(' # Create single combined mask')
    print(' {} -c "vehicle" --combine input.json ./masks/'.format(os.path.basename(sys.argv[0])))

def listCategories():
    print("Available categories:")
    for category, label_list in category2labels.items():
        if category == 'void':  # Skip void category
            continue
        print("  {}: {}".format(category, ', '.join([l.name for l in label_list if not l.ignoreInEval])))

def listLabels():
    print("Available labels (excluding ignored ones):")
    for label in labels:
        if not label.ignoreInEval and label.id >= 0:
            print("  {} (category: {})".format(label.name, label.category))

def getSelectedLabels(categories=None, label_names=None):
    """Get list of label objects based on category or label selection"""
    selected_labels = []
    
    if categories:
        for category in categories:
            if category not in category2labels:
                printError("Unknown category: {}. Use --list-categories to see available options.".format(category))
            for label in category2labels[category]:
                if not label.ignoreInEval and label.id >= 0:
                    selected_labels.append(label)
    
    if label_names:
        for label_name in label_names:
            if label_name not in name2label:
                printError("Unknown label: {}. Use --list-labels to see available options.".format(label_name))
            label = name2label[label_name]
            if not label.ignoreInEval and label.id >= 0:
                selected_labels.append(label)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_labels = []
    for label in selected_labels:
        if label.id not in seen:
            seen.add(label.id)
            unique_labels.append(label)
    
    return unique_labels

def createBinaryMask(annotation, selected_labels, combine_masks=False):
    """Create binary mask(s) from annotation for selected labels"""
    size = (annotation.imgWidth, annotation.imgHeight)
    
    if combine_masks:
        # Single mask for all selected labels
        mask = Image.new("L", size, 0)  # 0 = background
        drawer = ImageDraw.Draw(mask)
        label_ids = {label.id for label in selected_labels}
        
        for obj in annotation.objects:
            if obj.deleted:
                continue
                
            label_name = obj.label
            if label_name.endswith('group'):
                label_name = label_name[:-len('group')]
            
            if label_name in name2label:
                label = name2label[label_name]
                if label.id in label_ids:
                    try:
                        drawer.polygon(obj.polygon, fill=255)  # 255 = foreground
                    except:
                        print("Failed to draw polygon with label {}".format(label_name))
                        
        return {'combined': mask}
    else:
        # Separate mask for each selected label
        masks = {}
        for label in selected_labels:
            mask = Image.new("L", size, 0)
            drawer = ImageDraw.Draw(mask)
            
            for obj in annotation.objects:
                if obj.deleted:
                    continue
                    
                label_name = obj.label
                if label_name.endswith('group'):
                    label_name = label_name[:-len('group')]
                
                if label_name in name2label and name2label[label_name].id == label.id:
                    try:
                        drawer.polygon(obj.polygon, fill=255)
                    except:
                        print("Failed to draw polygon with label {}".format(label_name))
            
            masks[label.name] = mask
            
        return masks

def processSingleFile(json_file, output_dir, categories=None, label_names=None, 
                     suffix="mask", combine_masks=False):
    """Process a single JSON annotation file"""
    
    # Get selected labels
    selected_labels = getSelectedLabels(categories, label_names)
    if not selected_labels:
        printError("No valid labels selected")
    
    # Load annotation
    annotation = Annotation()
    annotation.fromJsonFile(json_file)
    
    # Create binary masks
    masks = createBinaryMask(annotation, selected_labels, combine_masks)
    
    # Get base filename
    base_name = getCoreImageFileName(json_file)
    
    # Save masks
    ensurePath(output_dir)
    for mask_name, mask in masks.items():
        if combine_masks:
            output_filename = "{}_{}.png".format(base_name, suffix)
        else:
            output_filename = "{}_{}_{}.png".format(base_name, mask_name.replace(' ', '_'), suffix)
        
        output_path = os.path.join(output_dir, output_filename)
        mask.save(output_path)
        print("Saved: {}".format(output_path))

def processBatch(cityscapes_path, output_dir, categories=None, label_names=None,
                suffix="mask", combine_masks=False):
    """Process all annotation files in Cityscapes dataset"""
    
    # Search patterns for annotation files
    search_fine = os.path.join(cityscapes_path, "gtFine", "*", "*", "*_gt*_polygons.json")
    search_coarse = os.path.join(cityscapes_path, "gtCoarse", "*", "*", "*_gt*_polygons.json")
    
    files_fine = glob.glob(search_fine)
    files_coarse = glob.glob(search_coarse)
    files = sorted(files_fine + files_coarse)
    
    if not files:
        printError("No annotation files found. Check CITYSCAPES_DATASET path.")
    
    print("Processing {} annotation files".format(len(files)))
    
    # Get selected labels once
    selected_labels = getSelectedLabels(categories, label_names)
    if not selected_labels:
        printError("No valid labels selected")
    
    print("Selected labels: {}".format([l.name for l in selected_labels]))
    
    # Process each file
    for i, json_file in enumerate(files):
        try:
            processSingleFile(json_file, output_dir, categories, label_names, 
                            suffix, combine_masks)
        except Exception as e:
            print("Failed to process {}: {}".format(json_file, str(e)))
            continue
            
        # Progress indicator
        progress = (i + 1) * 100 // len(files)
        print("\rProgress: {:>3}%".format(progress), end='')
        sys.stdout.flush()
    
    print("\nCompleted processing {} files".format(len(files)))

def main(argv):
    categories = None
    label_names = None
    suffix = "mask"
    combine_masks = False
    
    try:
        opts, args = getopt.getopt(argv, "hc:l:s:", 
                                 ["help", "categories=", "labels=", "suffix=", 
                                  "combine", "list-categories", "list-labels"])
    except getopt.GetoptError:
        printError('Invalid arguments')
    
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            printHelp()
            sys.exit(0)
        elif opt == '--list-categories':
            listCategories()
            sys.exit(0)
        elif opt == '--list-labels':
            listLabels()
            sys.exit(0)
        elif opt in ('-c', '--categories'):
            categories = [cat.strip() for cat in arg.split(',') if cat.strip()]
        elif opt in ('-l', '--labels'):
            label_names = [label.strip() for label in arg.split(',') if label.strip()]
        elif opt in ('-s', '--suffix'):
            suffix = arg
        elif opt == '--combine':
            combine_masks = True
    
    # Check if we have selection criteria
    if not categories and not label_names:
        printError("Must specify either --categories or --labels. Use --list-categories or --list-labels to see options.")
    
    # Handle arguments
    if len(args) == 0:
        # Batch processing mode - use CITYSCAPES_DATASET environment variable
        if 'CITYSCAPES_DATASET' not in os.environ:
            printError("No input file specified and CITYSCAPES_DATASET environment variable not set")
        
        cityscapes_path = os.environ['CITYSCAPES_DATASET']
        output_dir = input("Enter output directory: ").strip()
        if not output_dir:
            printError("Output directory required for batch processing")
            
        processBatch(cityscapes_path, output_dir, categories, label_names, suffix, combine_masks)
        
    elif len(args) == 2:
        # Single file processing mode
        json_file = args[0]
        output_dir = args[1]
        
        if not os.path.exists(json_file):
            printError("Input JSON file does not exist: {}".format(json_file))
        
        processSingleFile(json_file, output_dir, categories, label_names, suffix, combine_masks)
    else:
        printError("Invalid number of arguments")

if __name__ == "__main__":
    main(sys.argv[1:])