#coding:utf-8
import os,glob
import os.path as osp
import numpy as np
from cut2dota import split_traintest,rect2txt,polygons2rect
dataset_annotname='ship'

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
        if len(polygon)<8:
            continue
        polygon_list.append((annot_name,polygon))

    txt_file.close()
    return polygon_list

def Polygons2DotaTxt(polygon_list,save_txt):
    save_file=open(save_txt,'w',encoding='utf-8')
    for polygon in polygon_list:
        annot_name,points=polygon
        x1,y1,x2,y2,x3,y3,x4,y4=points
        save_file.write('{} {} {} {} {} {} {} {} {} {}\n' \
                        .format(x1,y1,x2,y2,x3,y3,x4,y4,annot_name,0))
    save_file.close()

# def polygons2rect(polygon_list):
#     rect_list=[]
#     for polygon in polygon_list:
#         annot_name=polygon[0]
#         rect=polygon[1:]
#         rect_array=np.array(rect).reshape((-1,2))
#        # import pdb;pdb.set_trace()
#         xmin=rect_array[:,0].min()
#         xmax=rect_array[:,0].max()
#         ymin=rect_array[:,1].min()
#         ymax=rect_array[:,1].max()

#         rect_list.append((annot_name,(xmin,ymin,xmax,ymax)))

#     return rect_list

# def rect2txt(rect_list,out_path):
#     out_file=open(out_path,'w',encoding='utf-8')
#     for rect in rect_list:
#         annot_name=rect[0]
#         rect_area=rect[1] #xmin,ymin,xmax,ymax
#         out_file.write('{} {} {} {} {}'.format(rect_area[0],rect_area[1],rect_area[2],rect_area[3],annot_name))
#     out_file.close()

def main(data_dir,out_dir):
    '''
    casia data_dir format is : ./image
                               ./label
    '''
    out_labelDota_dir=osp.join(out_dir,'labelDota')
    out_Rectlabel_dir=osp.join(out_dir,'labelRect')
    if not osp.exists(out_labelDota_dir):
        os.makedirs(out_labelDota_dir)
    if not osp.exists(out_Rectlabel_dir):
        os.makedirs(out_Rectlabel_dir)

    annot_dir=osp.join(data_dir,'label')
    img_dir=osp.join(data_dir,'image')
    annot_list=glob.glob(osp.join(annot_dir,'*.txt'))
    #import pdb;pdb.set_trace()

    for annot_path in annot_list:
        polygon_list=CasiaTxt2polygons(annot_path)
        rect_list=polygons2rect(polygon_list)

        basename=osp.splitext(osp.basename(annot_path))[0]
        save_polygon_txt=osp.join(out_labelDota_dir,'{}.txt'.format(basename))
        save_rect_txt=osp.join(out_Rectlabel_dir,'{}.txt'.format(basename))
        # import pdb;pdb.set_trace()    
        Polygons2DotaTxt(polygon_list,save_polygon_txt)
        rect2txt(rect_list,save_rect_txt)
    split_traintest(img_dir,out_dir)
if __name__=='__main__':
    import argparse

    parser=argparse.ArgumentParser(description='turn Casia label to dota format labels')

    parser.add_argument('--data_dir',help='the directory of the casia dataset')
    parser.add_argument('--out_dir',default='labelDota',help='the directory of the dota format labels')
    #parser.add_argument('--out_Rectlabel_dir',default='labelRect',help='the directory of the rect format labels')

    args=parser.parse_args()


    main(args.data_dir,args.out_dir)