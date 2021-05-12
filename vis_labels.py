#coding:utf-8
import json
import cv2,glob,os
import os.path as osp
import numpy as np

def json2rect(annot_file):
    f=open(annot_file,'r',encoding='utf-8')
    json_data=json.load(f,encoding='utf-8')
    rect_list=[]
    point_lists=json_data['shapes']
    for rect in point_lists:
        x1,y1=int(rect['points'][0][0]),int(rect['points'][0][1])
        x2,y2=int(rect['points'][1][0]),int(rect['points'][1][1])
        annot_name=rect['label']
        rect_list.append((annot_name,x1,y1,x2,y2))
    return rect_list
    # import pdb;pdb.set_trace()
def txt2polygon(annot_file):
    f=open(annot_file,'r',encoding='utf-8')
    for annot in f.readlines():
        import pdb;pdb.set_trace()
        annot=0
def show_poppy_labels(annot_dir,out_dir):
    annot_list = glob.glob(osp.join(annot_dir,'*.json'))
    for annot_path in annot_list:
        base_path=osp.splitext(annot_path)[0]
        basename=osp.splitext(osp.basename(annot_path))[0]

        img_path=base_path+'.jpg'
        vis_path=osp.join(out_dir,'{}_vis.jpg'.format(basename))

        img =cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),cv2.IMREAD_COLOR)

        rect_list = json2rect(annot_path)

        for rect in rect_list:
            # import pdb;pdb.set_trace()
            x1,y1=rect[1:3]
            x2,y2=rect[3:]
            w=abs(x2-x1)
            h=abs(y2-y1)
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),5)

            w_h='{}*{}'.format(w,h)
            cv2.putText(img,w_h,(x1-2,y1-2),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.imencode('.jpg', img)[1].tofile(vis_path)
def cut_img(img,cx,cy,height,width):
    # this x,y is diffrent from annot xy, this xy is cutted img xy
    y1,y2=cy-height//2,cy+height//2
    x1,x2=cx-width//2,cx+width//2

    h,w,c=img.shape

    #judge whether out of the image boundary
    if x1<0:
        x1=0;x2=width
    if x2>w:
        x2=w;x1=w-width
    if y1<0:
        y1=0;y2=height
    if y2>h:
        y2=h;y1=h-height

    cutted_img=img[y1:y2,x1:x2]
    return x1,y1,cutted_img



def get_cutted_rect(rect_list,x_shift,y_shift,height,width):
    cut_rect_list=[]
    for rect in rect_list:
        annot_name,x1,y1,x2,y2=rect 

        cut_x1,cut_x2=x1-x_shift,x2-x_shift
        cut_y1,cut_y2=y1-y_shift,y2-y_shift

        cut_x1=np.clip(cut_x1,0,width)
        cut_x2=np.clip(cut_x2,0,width)

        cut_y1=np.clip(cut_y1,0,height)
        cut_y2=np.clip(cut_y2,0,height)

        if abs(cut_x2-cut_x1)*abs(cut_y2-cut_y1)>0:
            cut_rect_list.append((annot_name,cut_x1,cut_y1,cut_x2,cut_y2))
    return cut_rect_list

# make cutted labels
def cut_label_imgs(annot_dir,out_dir,width=608,height=608):
    cut_vis_dir=osp.join(out_dir,'cut_labels_vis')
    cut_out_dir=osp.join(out_dir,'cut_labels')

    annot_list = glob.glob(osp.join(annot_dir,'*.json'))

    if not osp.exists(cut_vis_dir):
        os.mkdir(cut_vis_dir)
    if not osp.exists(cut_out_dir):
        os.mkdir(cut_out_dir)

    index=0

    for annot_path in annot_list:
        base_path=osp.splitext(annot_path)[0]
        img_path=base_path+'.jpg'

        img =cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),cv2.IMREAD_COLOR)

        rect_list = json2rect(annot_path)

        for i,rect in enumerate(rect_list):

            x1,y1=rect[1:3]
            x2,y2=rect[3:]

            cx=(x1+x2)//2
            cy=(y1+y2)//2


            cutted_img=np.zeros((height,width,3))
            x_shift,y_shift,cutted_img=cut_img(img,cx,cy,height,width)

            cut_rect_list=get_cutted_rect(rect_list,x_shift,y_shift,height,width)

            cut_img_path=osp.join(cut_out_dir,'{:0>5}.jpg'.format(index))
            cut_label_path=osp.splitext(cut_img_path)[0]+'.txt'

            cv2.imwrite(cut_img_path,cutted_img)

            label_file=open(cut_label_path,'w',encoding='utf-8')
            for cut_rect in cut_rect_list:
                label_file.write('{},{},{},{},{}\n'.format(annot_name,cut_x1,cut_y1,cut_x2,cut_y2))
            label_file.close()
            index=index+1
def vis_cutted_img_labes(cutted_dir):

    for cut_rect in cut_rect_list:
        annot_name,cut_x1,cut_y1,cut_x2,cut_y2=cut_rect
        label_file.write('{},{},{},{},{}\n'.format(annot_name,cut_x1,cut_y1,cut_x2,cut_y2))
        cv2.rectangle(cutted_img2,(cut_x1,cut_y1),(cut_x2,cut_y2),(0,0,255),5)


annot_dir=r"E:\master\01_project\pop\origin_annot_pics\origin"
out_dir=r"E:\master\01_project\pop\origin_annot_pics\vis_labels\cutted_labels"
# show_poppy_labels(annot_dir,out_dir)
cut_label_imgs(annot_dir,out_dir,width=608,height=608)