#-*-coding:utf-8-*-
import os,sys
import glob
import config as CF
import numpy as np
from PIL import Image
import random
#import argparse
#通过将训练集按照指定格式放入input_data文件夹下，
#将测试集放到test文件夹下。
#运行python3 input_data.py 函数，给出训练集和测试集图片的路径。
def reforce(im,convert_RGB):
    out1=im
    out2=im
    out3=im
    out4=im
    out5=im
    out1 = out1.transpose(Image.FLIP_TOP_BOTTOM)
    out2 =out2.transpose(Image.FLIP_LEFT_RIGHT)
    out3 = out3.transpose(Image.ROTATE_90)
    out4 = out4.transpose(Image.ROTATE_180)
    out5 = out5.transpose(Image.ROTATE_270)
    if convert_RGB:
        out1 = out1.convert("RGB")
        out2 = out2.convert("RGB")
        out3 = out3.convert("RGB")
        out4 = out4.convert("RGB")
        out5 = out5.convert("RGB")
        im = im.convert("RGB")
    out1=np.array(out1)
    out2=np.array(out2)
    out3=np.array(out3)
    out4=np.array(out4)
    out5=np.array(out5)
    imag=np.array(im)
    return [imag,out1,out2,out3,out4,out5]

def reforce_fake_rgb(im):
    out1=im
    out2=im
    out3=im
    out4=im
    out5=im
    out1 = out1.transpose(Image.FLIP_TOP_BOTTOM)
    out1=to_fake_grb(out1)
    out1=np.array(out1)
    out2 =out2.transpose(Image.FLIP_LEFT_RIGHT)
    out2=to_fake_grb(out2)
    out2=np.array(out2)
    out3 = out3.transpose(Image.ROTATE_90)
    out3=to_fake_grb(out3)
    out3=np.array(out3)
    out4 = out4.transpose(Image.ROTATE_180)
    out4=to_fake_grb(out4)
    out4=np.array(out4)
    out5 = out5.transpose(Image.ROTATE_270)
    out5=to_fake_grb(out5)
    out5=np.array(out5)
    im=to_fake_grb(im)
    imag=np.array(im)
    return [imag,out1,out2,out3,out4,out5]

def path2rgb(path,pix_x,pix_y,ref=False,convert_RGB=False):
    img = Image.open(path)
    try:
        img = img.resize([pix_x,pix_y])
    except Exception as e:
        print(e)
        return False
    try:
        if ref:
            im_lst=reforce(img,convert_RGB)
        else:
            if convert_RGB:
                img=img.convert("RGB")
            im_lst=[np.array(img)]
    except Exception as e:
        print(e)
        if ref:
            im_lst=reforce_fake_rgb(img)
        else:
            img=to_fake_grb(img)
            im_lst=[np.array(img)]
    return im_lst

def to_fake_grb(ar):
    ret=[]
    for aa in ar:
        tmp=[]
        for aaa in aa:
            tmp.append([aaa,aaa,aaa])
        ret.append(tmp)
    return ret

def give_batch(data,pix_x,pix_y,ref=False,convert_RGB=False):
    feature_batch=[]
    label_batch=[]
    for path,label in data:
        im_lst=path2rgb(path,pix_x,pix_y,ref=ref,convert_RGB=False)
        if not im_lst: continue 
        label_lst=[label]*len(im_lst)
        for fea,la in zip(im_lst,label_lst):
            feature_batch.append(fea)
            label_batch.append(la)
    return feature_batch,label_batch
