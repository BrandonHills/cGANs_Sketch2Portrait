import os
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='faces/A')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='faces/B')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='faces/A_B')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images',type=int, default=1000000)
parser.add_argument('--use_AB', dest='use_AB', help='if true: (0001_A, 0001_B) to (0001_AB)',action='store_true')
args = parser.parse_args()

# Telling the user which args were chosen.
for arg in vars(args):
    print('[%s] = ' % arg,  getattr(args, arg))

# Returns a list of the names of the entries in a directory path.
print("args.fold_A:", args.fold_A)
splits = os.listdir(args.fold_A)
print("splits:", splits)

if ".DS_Store" in splits:
    print(".DS_Store Removed...")
    splits.remove(".DS_Store")

for sp in splits:
    img_fold_A = os.path.join(args.fold_A, sp)
    print("img_fold_A",img_fold_A)
    img_fold_B = os.path.join(args.fold_B, sp)
    img_list = os.listdir(img_fold_A)

    if args.use_AB:
        img_list = [img_path for img_path in img_list if '_A.' in img_path]

    num_imgs = min(args.num_imgs, len(img_list))
    print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
    img_fold_AB = os.path.join(args.fold_AB, sp)
    if not os.path.isdir(img_fold_AB):
        os.makedirs(img_fold_AB)
    print('split = %s, number of images = %d' % (sp, num_imgs))

    for n in range(num_imgs):
        name_A = img_list[n]
        path_A = os.path.join(img_fold_A, name_A)

        if args.use_AB:
            name_B = name_A.replace('_A.', '_B.')
        else:
            name_B = name_A
        path_B = os.path.join(img_fold_B, name_B)

        print("Current Image:", path_A, path_B)

        if os.path.isfile(path_A) and os.path.isfile(path_B):
            name_AB = name_A
            if args.use_AB:
                name_AB = name_AB.replace('_A.', '.') # remove _A
            path_AB = os.path.join(img_fold_AB, name_AB)
            im_A = cv2.cvtColor(cv2.imread(path_A), cv2.COLOR_BGR2RGB)
            im_B = cv2.imread(path_B)

            print("img_fold_A:", img_fold_A)
            v = (img_fold_A == "faces/A/real")
            print ("img_fold_A == faces/A/real", v)
            
            # For Real images:
            if img_fold_A == "faces/A/real":
                im_B = im_B[:,239:(1024-239),:]
                im_B = cv2.resize(im_B, (im_A.shape[1], im_A.shape[0]))
            print("im_A.shape, im_B.shape:", im_A.shape, im_B.shape)

            

            im_AB = np.concatenate([im_A, im_B], 1)
            cv2.imwrite(path_AB, im_AB)
