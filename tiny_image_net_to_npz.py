import os
import sys
import numpy
import gc
import argparse
from PIL import Image

# Download Tiny ImageNet at http://cs231n.stanford.edu/tiny-imagenet-200.zip
# Takes folders and creates NPZ file.

RESIZE_SIZE = 64
CROP_SIZE = 56
IMG_EXT = ["jpg", "jpeg", "png"]
WNIDS_FILE = "tiny_image_net_wnids.txt"


def read_wnids(path):
    lines = open(path, 'r').readlines()
    counter = 0
    wnids = dict()
    for line in lines:
        norm_line = line.strip().lower()
        if len(norm_line) < 9:
            continue
        wnids[norm_line] = counter
        counter += 1
    print("Number of categories = {:d}".format(counter))
    return wnids


def try_read_image(path):
    if not os.path.exists(path):
        return None
    try:
        image = Image.open(path)
        image = image.convert("RGB")
        min_size = min(image.size)
        factor = float(RESIZE_SIZE) / min_size
        w = int(factor * image.size[0] + 0.5)
        h = int(factor * image.size[1] + 0.5)
        resized_image = image.resize((w, h), Image.ANTIALIAS)
        pad_x = (resized_image.size[0] - CROP_SIZE) // 2
        pad_y = (resized_image.size[1] - CROP_SIZE) // 2
        crop_image = resized_image.crop((pad_x, pad_y, CROP_SIZE + pad_x, CROP_SIZE + pad_y))
        crop_arr = numpy.array(crop_image, dtype="float32")
        return crop_arr
    except:
        return None


def read_images(folder):
    print("Processing folder {}".format(folder))
    image_acc = []
    filenames_acc = []
    files = os.listdir(folder)
    for filename in files:
        ext = filename.split(".")[-1].lower()
        if ext not in IMG_EXT:
            continue
        path = os.path.join(folder, filename)
        image = try_read_image(path)
        if image is None:
            continue
        image_acc.append(image)
        filenames_acc.append(filename)
    gc.collect(0)
    return image_acc, filenames_acc


def read_labels(labels_path, filenames, wnids):
    print("Reading {}".format(labels_path))
    filename_to_wnid = dict()
    lines = open(labels_path).readlines()
    for line in lines:
        items = line.split(" ")
        filename_to_wnid[items[0]] = items[1]
    labels = []
    for filename in filenames:
        wnid = filename_to_wnid[filename]
        label = wnids[wnid]
        labels.append(label)
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny ImageNet -> NPZ")
    parser.add_argument('--inp', type=str, help="Input folder with images", dest="inp")
    parser.add_argument('--mode', type=str, help="TRAIN or VAL", dest="mode")
    parser.add_argument('--out', type=str, help="Output NPZ file", dest="out")
    args = parser.parse_args()

    import pathlib
    script_dir = str(pathlib.Path(__file__).resolve().parents[0])
    os.chdir(script_dir)

    mode = args.mode.lower()
    if not os.path.exists(args.inp):
        print("Cannot find input folder at {}".format(args.inp))
        sys.exit()
    if mode not in ["train", "val"]:
        print("Unknown mode {}".format(mode))
        sys.exit()
    if os.path.exists(args.out):
        print("Output file already exists at {}".format(args.out))
        sys.exit()

    wnids = read_wnids(os.path.join(script_dir, WNIDS_FILE))

    if mode == "train":
        images = []
        labels = []
        subfolders = os.listdir(args.inp)
        for subfolder in subfolders:
            sub_images, _ = read_images(os.path.join(args.inp, subfolder + "/images"))
            sub_label = wnids[subfolder]
            sub_labels = [sub_label] * len(sub_images)

            images.extend(sub_images)
            labels.extend(sub_labels)
    if mode == "val":
        images, filenames = read_images(os.path.join(args.inp, "images"))
        labels = read_labels(filenames, args.inp)

    x = numpy.stack(images)
    y = numpy.stack(labels)
    print("X shape", x.shape)
    print("Y shape", y.shape)

    print("Saving results...")
    numpy.savez(args.out, x, y)

    print("Done.")