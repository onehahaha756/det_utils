#-*- coding: utf-8 -*-
import os
import os.path as osp
import glob
import codecs,shutil
import random

import  xml.dom.minidom
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image


multil_type=['*.jpg','*.png','*.tif']
dataset_annotname='ship'

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def CasiaTxt2polygons(txt_path,annot_name=dataset_annotname):
    '''
    casia ship v1 annot format:[[x1,y1,x2,y2,x3,y3,x4,y4],[....]]
    '''
    polygon_list = list()
    txt_file=open(txt_path,encoding='utf-8')

    for line in txt_file:
        #import pdb;pdb.set_trace()
        line=line.strip()
        line=line.replace('[','')
        line=line.replace(']','').replace(' ','')
        row_list=line.split(',')
        polygon=[int(float(x)) for x in row_list[:8]]

        polygon_list.append((annot_name,polygon))

    txt_file.close()
    return polygon_list

def txt2polygons(txt_path):
    '''
    '''
    polygon_list = list()
    txt_file=codecs.open(txt_path,encoding='utf-8')

    for line in txt_file:
        #import pdb;pdb.set_trace()
        line=line.strip()
        row_list=line.split()
        polygon=[int(float(x)) for x in row_list[:8]]
        annot_name=row_list[-2]

        polygon_list.append((annot_name,polygon))

    txt_file.close()
    return polygon_list


def get_grid_list(img, roi_size=(400, 400), overlap_size=(50, 50)):
    ''' Calculate the bounding edges of cropping grids
    return:' xmin xmax ymin ymax'
    '''
    img_h, img_w = img.shape[0:2]
    #import pdb;pdb.set_trace()
    row_crops = (img_h - roi_size[1]) // (roi_size[1] - overlap_size[1])
    col_crops = (img_w - roi_size[0]) // (roi_size[0] - overlap_size[0])

    grid_list = [] # ymin, ymax, xmin, xmax
    for i_iter in range(row_crops * col_crops):

        x_crop_idx = i_iter % col_crops   #行
        y_crop_idx = i_iter//col_crops

        xmin=x_crop_idx*(roi_size[0]-overlap_size[0])
        ymin=y_crop_idx*(roi_size[1]-overlap_size[1])
        xmax=xmin+roi_size[0]
        ymax=ymin+roi_size[1]

        grid_list.append((xmin,xmax,ymin,ymax))

    return grid_list

def polygons2rect(polygon_list):
    rect_list=[]
    for polygon in polygon_list:
        annot_name=polygon[0]
        rect=polygon[1:]
        rect_array=np.array(rect).reshape((-1,2))
       # import pdb;pdb.set_trace()
        xmin=rect_array[:,0].min()
        xmax=rect_array[:,0].max()
        ymin=rect_array[:,1].min()
        ymax=rect_array[:,1].max()

        rect_list.append((annot_name,(xmin,ymin,xmax,ymax)))

    return rect_list

def rect2txt(rect_list,out_path):
    out_file=open(out_path,'w',encoding='utf-8')
    for rect in rect_list:
        annot_name=rect[0]
        rect_area=rect[1] #xmin,ymin,xmax,ymax
        out_file.write('{} {} {} {} {}'.format(rect_area[0],rect_area[1],rect_area[2],rect_area[3],annot_name))
    out_file.close()

def match_grid_rect(cur_grid,rect):
    '''
    if rect in grid then return True and the new rect in subpatch
    cur_grid: [xmin:xmax,ymin:ymax]
    rect :[xmin,ymin,xmax,ymax]
    return
    rect_in_grid : type bool
    new_rect : new rect position in subpatch
    '''
    annot_name=rect[0]
    #import pdb;pdb.set_trace()
    xmin,ymin,xmax,ymax=rect[1]
    grid_xmin,grid_xmax,grid_ymin,grid_ymax=cur_grid
    rect_in_grid=False
    new_rect=None
    if xmax<grid_xmax and xmin>grid_xmin and ymax<grid_ymax and ymin >grid_ymin:
        new_rect=[annot_name,[xmin-grid_xmin,ymin-grid_ymin,xmax-grid_xmin,ymax-grid_ymin]]
        rect_in_grid=True
    return rect_in_grid,new_rect

def cutsubimg(img,polygons,save_dir,cut_size=512,overlap=50):
    pass

def split_traintest(img_dir,save_dir):
    '''
    split a image dir to train and test list
    input: imgdir
    do: write a trainfile and a testfile
    '''
    train_file=open(os.path.join(save_dir,'train.txt'),'w')
    test_file=open(os.path.join(save_dir,'test.txt'),'w')

    imglist=os.listdir(img_dir)
    for imgpath in imglist:
        basename=osp.splitext(imgpath)[0]
        if random.random()<0.3:
            test_file.write('{}\n'.format(basename))
        else:
            train_file.write('{}\n'.format(basename))
    train_file.close()
    test_file.close()

def write_imglist(img_dir,savetxt):
    savefile=open(savetxt,'w',encoding='utf-8')

    imglist=os.listdir(img_dir)
    for imgpath in imglist:
        basename=osp.splitext(imgpath)[0]
        savefile.write('{}\n'.format(basename))
    savefile.close()

def vis_labels(img,rect_list,visname):
    show_img=img.copy()
    for rect in rect_list:
        # import pdb;pdb.set_trace()
        annot_name=rect[0]
        x1,y1,x2,y2=rect[1]
        #import pdb;pdb.set_trace()
        w=abs(x2-x1)
        h=abs(y2-y1)
        cv2.rectangle(show_img,(x1,y1),(x2,y2),(0,0,255),5)
        w_h='{}*{}'.format(w,h)
        cv2.putText(show_img,w_h,(x1-2,y1-2),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    #import pdb;pdb.set_trace()
    cv2.imwrite(visname, show_img)

def cutimage2detect(input_imgdir,input_annotdir,save_dir,cut_size=512,overlap=50,vis=True,show_origin=False):
    '''
    input_imgdir: images 
    input_annotdir: txt format annots
    save_dir:output dataset dir
            ./train.txt
            ./test.txt
            ./images
            ./labels
    '''
    target_lable_dir=os.path.join(save_dir,'labelDota')
    target_imagetrain_dir=os.path.join(save_dir,'image','train')
    target_imagetest_dir=os.path.join(save_dir,'image','test')
    target_vis_dir=os.path.join(save_dir,'vis_label')
    target_origin_vis_dir=os.path.join(save_dir,'vis_origin_label')

    origin_trainlist=os.path.join(save_dir,'origin_train.txt')
    origin_testlist=os.path.join(save_dir,'origin_test.txt')

    if not osp.exists(save_dir):
        os.mkdir(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    if not os.path.exists(target_lable_dir):
        os.mkdir(target_lable_dir)
    if not os.path.exists(target_imagetrain_dir):
        os.makedirs(target_imagetrain_dir)
    if not os.path.exists(target_imagetest_dir):
        os.makedirs(target_imagetest_dir)
    if not os.path.exists(target_vis_dir):
        os.mkdir(target_vis_dir)
    #show origin labels
    if show_origin:
        if not os.path.exists(target_origin_vis_dir):
            os.mkdir(target_origin_vis_dir)
    #save train and test originlist
    origin_trainfile=open(origin_trainlist,'w',encoding='utf-8')
    origin_testfile=open(origin_testlist,'w',encoding='utf-8')

    imglist=[]
    for imgtype in multil_type:
        imglist+=glob.glob(osp.join(input_imgdir,imgtype))

    assert len(imglist) >0,'imgdir is empty please check your image directory path!'
    
    target_imgdir=target_imagetrain_dir
    for imgpath in tqdm(imglist):
            
        basename,suffix=osp.splitext(osp.basename(imgpath))
        annotpath=osp.join(input_annotdir,'{}.txt'.format(basename))
        if random.random()<0.3:
            target_imgdir=target_imagetest_dir
            #save origin test list
            origin_testfile.write('{}\n'.format(basename))
        else:
            target_imgdir=target_imagetrain_dir
            #save origin train list
            origin_trainfile.write('{}\n'.format(basename))
        
        img=cv2.imread(imgpath)
        
        polygon_list=txt2polygons(annotpath)

        rect_list=polygons2rect(polygon_list)

        grid_list=get_grid_list(img,(cut_size,cut_size),(overlap,overlap))

        #
        for i,cur_grid in enumerate(grid_list):
            exsit_object=False
            new_rectlist=[]
            #match rect withe grid
            for rect in rect_list:
                rect_in_grid,new_rect=match_grid_rect(cur_grid,rect)
                if rect_in_grid:
                    exsit_object=True
                    new_rectlist.append(new_rect)
                   #import pdb;pdb.set_trace()
            if exsit_object:
                save_name=basename+'{}'.format(i)

                save_labelname=osp.join(target_lable_dir,'{}_{}.txt'.format(basename,i))
                save_imgname=osp.join(target_imgdir,'{}_{}.jpg'.format(basename,i))
                save_img=img[cur_grid[2]:cur_grid[3],cur_grid[0]:cur_grid[1],:]

                #print('save image {}'.format(save_imgname))
                rect2txt(new_rectlist,save_labelname)

                cv2.imwrite(save_imgname,save_img)

                if vis:
                    vis_imgname=osp.join(target_vis_dir,'{}_{}.jpg'.format(basename,i))
                    vis_labels(save_img,new_rectlist,vis_imgname)
            else:
                #print('save img ',i)
                save_imgname=osp.join(target_imgdir,'{}_{}.jpg'.format(basename,i))
                save_img=img[cur_grid[2]:cur_grid[3],cur_grid[0]:cur_grid[1],:]
                cv2.imwrite(save_imgname,save_img)

        if show_origin:
            showname=save_imgname=osp.join(target_origin_vis_dir,'{}.jpg'.format(basename))
            vis_labels(img,rect_list,showname)

    origin_trainfile.close()
    origin_testfile.close()

    traintxt=os.path.join(save_dir,'train.txt')
    testtxt=os.path.join(save_dir,'test.txt')
    write_imglist(target_imagetrain_dir,traintxt)
    write_imglist(target_imagetest_dir,testtxt)
    #split_traintest(target_image_dir,save_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="cut remote images into small patches ")

    parser.add_argument('--input_imgdir',help="directory of input images")
    parser.add_argument('--input_annotdir',help="directory of the txt format annot")
    parser.add_argument('--save_dir',help="directory of output")
    parser.add_argument('--vis_label',default=False,type=str2bool,help="show origin labels")
    parser.add_argument('--resplit',default=False,type=str2bool,help="do not cut the dataset ,but only resplit train and testdata")
    parser.add_argument('-c','--cut_size',type=int,default=512,help="cut patch sizes")
    parser.add_argument('-l','--overlap',type=int,default=50,help="overlap for cut")


    args = parser.parse_args()
    #only resplit the dataset,do not recut
    if args.resplit:
        target_image_dir=os.path.join(args.save_dir,'image')
        #split_traintest(target_image_dir,args.save_dir)
    else:
        cutimage2detect(args.input_imgdir,args.input_annotdir,args.save_dir,args.cut_size,args.overlap,args.vis_label)




