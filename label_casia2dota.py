#coding:utf-8
import os,glob
import os.path as osp

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

        polygon_list.append((annot_name,polygon))

    txt_file.close()
    return polygon_list

def Polygons2DotaTxt(polygon_list,save_txt):
    save_file=open(save_txt,'w',encoding='utf-8')
    for polygon in polygon_list:
        annot_name,points=polygon
        x1,y1,x2,y2,x3,y3,x4,y4=points
        save_file.write('{} {} {} {} {} {} {} {} {} {}' \
                        .format(x1,y1,x2,y2,x3,y3,x4,y4,annot_name,0))
    save_file.close()

def main(data_dir,out_label_dir):
    '''
    casia data_dir format is : ./image
                               ./label
    '''
    annot_dir=osp.join(data_dir,'label')
    annot_list=glob.glob(osp.join(annot_dir,'*.txt'))
    import pdb;pdb.set_trace()

    for annot_path in annot_list:
        polygon_list=CasiaTxt2polygons(annot_path)
        basename=osp.basename(annot_path)
        save_txt=osp.join(out_label_dir,'{}'.format(basename))
        Polygons2DotaTxt(polygon_list,save_txt)
if __name__=='__main__':
    import argparse

    parser=argparse.ArgumentParser(description='turn Casia label to dota format labels')

    parser.add_argument('--data_dir',help='the directory of the casia dataset')
    parser.add_argument('--out_label_dir',default='labelDota',help='the directory of the dota format labels')
    #parser.add_argument('--annot_name',default='ship',help='default annot name')

    args=parser.parse_args()
    if not osp.exists(args.out_label_dir):
        os.makedirs(args.out_label_dir)

    main(args.data_dir,args.out_label_dir)