#coding:utf-8
import os,glob,pickle
import os.path as osp
from typing import cast
import numpy as np

GTCLASS=['0'] #only one class
def txt2polygons(txt_path):
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

        if len(row_list)>10:
            annot_name=row_list[-2]
        else :
            annot_name='0'
        polygon_list.append((annot_name,polygon))

    txt_file.close()
    return polygon_list

def polygons2rect(polygon_list):
    '''
    turn rotation labels to rectangle
    get the (minx miny),(maxx,maxy) to be lefttop and rightbottom
    '''
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

def LoadTxtGt_casia(annot_dir):
    '''
    annot_dir: txt labels dir
    return: 
    gt_dict:{basename:[[annot_name,[xmin,ymin,xmax,ymax]],
                       [annot_name,[xmin,ymin,xmax,ymax]],...],
             basename:[...],
             ...
            }
    '''
    annot_list = glob.glob(osp.join(annot_dir,'*.txt'))
    gt_dict={}
    for annot_path in annot_list:
        basename=osp.splitext(osp.basename(annot_path))[0]
        polygon_list=txt2polygons(annot_path)
        rect_list=polygons2rect(polygon_list)
        gt_dict[basename]=rect_list
    return gt_dict

def LoadDetfile(det_path):
    '''
    load .pkl format detection results
    detfile:{basename:[[x1,y1,x2,y2,score,cls],[x1,y1,x2,y2,score,cls]...],
             bashename:[...] 
            }
    return: detfile

    '''
    detfile=open(det_path,'rb')
    det_dict = pickle.load(detfile)


    return det_dict

def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def casia_eval(annot_dir,det_path,imagesetfile,classname,ovthresh=0.5,conf_thre=0.3,use_07_metric=True):
    '''

    '''
    gt_dict=LoadTxtGt_casia(annot_dir)
    det_dict=LoadDetfile(det_path)

    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]  
    cls_det=[] 
    cls_gt={} 
    #select groud truth of this cls 
    GtNmus=0
    for imagename in imagenames:
        if not imagename in gt_dict.keys():
            continue
        gts=gt_dict[imagename]
        cls_bboxes=[]
        for gt in gts:
            #import pdb;pdb.set_trace()
            if gt[0]==classname:
                detflag=0
                GtNmus+=1
                gtbbox=gt[1]
                cls_bboxes.append([gtbbox,detflag])
        #make sure image has the cls object
        if len(cls_bboxes)>0:
            cls_gt[imagename]=cls_bboxes
    
    #select detections of this cls
    for imagename in imagenames:
        if not imagename in det_dict.keys():
            continue
        dets=det_dict[imagename]
        for det in dets:
            #import pdb;pdb.set_trace()
            if det[-1]==GTCLASS.index(classname):
                cls_det.append(([imagename]+det[:]))
    if len(cls_det)>1:
        #get detction confidence and bbox
        '''
        cls_det:[[imagename,x1,y1,x2,y2,cof,cls],...]
        '''

        imageids=np.array([x[0] for x in cls_det])
        confidence=np.array([float(x[-2]) for x in cls_det])
        BBox=np.array([[float(z) for z in x[1:-2]] for x in cls_det])
       
        # rm confidence below confidence threshhold
        select_mask=confidence>conf_thre
        confidence=confidence[select_mask]
        BBox=BBox[select_mask]
        imageids=imageids[select_mask]

        #sort by confidence
        sorted_ind=np.argsort(-confidence)
        sorted_scores=np.sort(confidence)
        BBox=BBox[sorted_ind,:]
        imageids=imageids[sorted_ind]   

        #mark TPs and FPs
        nd = len(BBox)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        for d in range(nd):
            #import pdb;pdb.set_trace()
            '''
            cls_gtbboxs:[[(x1,y1,x2,y2),detflag],...]
            '''
            imgname=imageids[d]
            cls_gtbboxs=cls_gt[imgname]

            BBGT=np.array([bbox[0] for bbox in cls_gtbboxs])
            bb=BBox[d,:].astype(float)
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])

            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih
            uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not cls_gt[imgname][jmax][-1]:
                    tp[d] = 1.
                    cls_gt[imgname][jmax][-1] = 1
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.
        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        rec = tp / float(GtNmus)
        ap=voc_ap(rec, prec, use_07_metric)

        return rec,prec,ap


def calculat_Precision(annot_dir,det_path,imagesetfile,ovthresh=0.5,conf_thre=0.3):
    '''
    for single cls detection evaluation
    just caculate total presion and recall
    '''
    gt_dict=LoadTxtGt_casia(annot_dir)
    det_dict=LoadDetfile(det_path)

    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    #import pdb;pdb.set_trace()
    TP=0
    FP=0
    GtNums=0 # groudtruth nums
    for imagename in imagenames:
        if not imagename in det_dict.keys():
            continue
        #load gts and dets
        gts=gt_dict[imagename]
        dets=det_dict[imagename]
        # for single class
        GtNums+=len(gts)
        #
        detected_flag=np.zeros(len(gts))

        #get gt bbox
        BBGT=np.array([x[1] for x in gts])
        #get detction confidence and bbox
        confidence=np.array([float(x[-2]) for x in dets])
        BBox=np.array([[float(z) for z in x[:-2]] for x in dets])

        # rm confidence below confidence threshhold
        select_mask=confidence>conf_thre
        confidence=confidence[select_mask]
        BBox=BBox[select_mask]

        #import pdb;pdb.set_trace()
        #sort by confidence
        sorted_ind=np.argsort(-confidence)
        sorted_scores=np.sort(confidence)
        BBox=BBox[sorted_ind,:]


        nd=BBox.shape[0]
        tp=np.zeros(nd)
        fp=np.zeros(nd)
        # mark dets and mark TPs and FPs
        for d in range(nd):
            bb=BBox[d,:].astype(float)
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])

            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih
            uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not detected_flag[jmax]:
                    tp[d] = 1.
                    detected_flag[jmax] = 1
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.
        #import pdb;pdb.set_trace()
        TP+=np.sum(tp)
        FP+=np.sum(fp)
    Recall=TP/float(GtNums)
    Precision= TP/np.maximum(TP + FP, np.finfo(np.float64).eps)  
    return Recall,Precision
#def cal_ap()    

if __name__=='__main__':
    annot_dir='/data/03_Datasets/CasiaDatasets/ship/label'
    det_path='/data/02_code_implement/ssd.pytorch/MixShip/MixShip_iter5700/detections.pkl'
    imagesetfile='/data/02_code_implement/ssd.pytorch/MixShip/MixShip_iter5700/inference_imgnames.txt'
    overthre=0.5
    conf_thre=0.1
    clss=GTCLASS[0] #label is '0'
    #Recall,Precision=calculat_Precision(annot_dir,det_path,imagesetfile,overthre,conf_thre)
    rec,prec,ap=casia_eval(annot_dir,det_path,imagesetfile,clss,overthre,conf_thre)
    print('*'*15+\
         '\niou overthre:{}\nConfidence thre:{}\nAP:{}\nMaxRecall:{} \nMinPrecision: {}'\
        .format(overthre,conf_thre,ap,rec[-1],prec[-1]))
    #import pdb;pdb.set_trace()


