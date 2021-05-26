#coding:utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2,glob,math
import matplotlib.pyplot as plt
import os.path as osp
from tqdm import tqdm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

ROOT_PATH='/data/03_Datasets/airbus-ship-detection'
masks = pd.read_csv(osp.join(ROOT_PATH,'train_ship_segmentations_v2.csv'))

print(os.listdir(ROOT_PATH))
train = os.listdir(osp.join(ROOT_PATH,'train_v2'))
print(len(train))
test = os.listdir(osp.join(ROOT_PATH,'test_v2'))
print(len(test))


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def decode_mask2rect(imgid,masks):
   
    img_masks = masks.loc[masks['ImageId'] == imgid, 'EncodedPixels'].tolist()
    #import pdb;pdb.set_trace()
    # Take the individual ship masks and create a single mask array for all ships
    if np.nan in img_masks:
        return None
    rect_list=[]
    annotname='ship'
    for mask in img_masks: 
        ship_mask=rle_decode(mask)
        ys,xs=np.where(ship_mask==1)
        xmin,ymin,xmax,ymax=xs.min(),ys.min(),xs.max(),ys.max()
        rect_list.append((annotname,[xmin,ymin,xmax,ymax]))
    return rect_list

def rescale_img_label(rect_list,insize,outsize):
    new_rectlist=[]
    for rect in rect_list:
        annotname=rect[0]
        xmin,ymin,xmax,ymax=rect[1]

        xmin,xmax=int(xmin*outsize/insize),int(xmax*outsize/insize)
        ymin,ymax=int(ymin*outsize/insize),int(ymax*outsize/insize)

        new_rectlist.append((annotname,[xmin,ymin,xmax,ymax]))
    return new_rectlist

def rect2txt(rect_list,out_path):
    out_file=open(out_path,'w',encoding='utf-8')
    for rect in rect_list:
        annot_name=rect[0]
        rect_area=rect[1] #xmin,ymin,xmax,ymax
        out_file.write('{} {} {} {} {}\n'.format(rect_area[0],rect_area[1],rect_area[2],rect_area[3],annot_name))
    out_file.close()

def mkdatasets(img_dir,masks,img_outdir,label_outdir,outimg_size,vis_label=True):
    imglist=glob.glob(os.path.join(img_dir,'*.jpg'))
    for imgpath in tqdm(imglist):
        imgid=os.path.basename(imgpath)
        basename=os.path.splitext(imgid)[0]
        img=cv2.imread(imgpath)
        rect_list=decode_mask2rect(imgid,masks)
        if rect_list==None:
            continue
        insize=img.shape[0]

        new_rectlist=rescale_img_label(rect_list,insize,outimg_size)

        new_img=cv2.resize(img,(outimg_size,outimg_size))
        cv2.imwrite(osp.join(img_outdir,'{}.jpg'.format(basename)),new_img)
        savetxt=os.path.join(label_outdir,'{}.txt'.format(basename))

        rect2txt(new_rectlist,savetxt)
        if vis_label:
            show_img=new_img.copy()
            label_dirname=os.path.dirname(label_outdir)
            vis_labeldir=os.path.join(label_dirname,'vis_label')
            for rect in new_rectlist:
                annotname,points=rect
                xmin,ymin,xmax,ymax=points
                cv2.rectangle(show_img,(xmin,ymin),(xmax,ymax),(0,0,255),2,2)
            vis_labelname=os.path.join(vis_labeldir,'{}.jpg'.format(basename))
            cv2.imwrite(vis_labelname,show_img)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description="decode kaggle ship datasets")

    parser.add_argument('--input_imgdir',help="directory of input images")
    parser.add_argument('--outdir',help="directory of output labels")
    parser.add_argument('--outimgsize',default=512,type=int,help="the output images size")

    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    out_imgdir=osp.join(args.outdir,'image')
    out_labeldir=osp.join(args.outdir,'label')

    if not os.path.exists(out_imgdir):
        os.mkdir(out_imgdir)  
    if not os.path.exists(out_labeldir):
        os.mkdir(out_labeldir)  

    mkdatasets(args.input_imgdir,masks,out_imgdir,out_labeldir,args.outimgsize)



        






        
