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
def to_onehot(num,length):
    ret=[0]*length
    ret[num-1]=1
    return ret

def get_train_files(file_dir="train"):
    print("正在获取训练集")
    if sys.platform.find("win")!=-1:
        file_dir=file_dir+"\\"
        sp="\\"
    else:
        file_dir=file_dir+"/"
        sp="/"
    #print(file_dir)
    #exit()
    if glob.glob(file_dir+"label2id")!=[]:
        print("删除遗留的label2id文件")
        os.remove(file_dir+"label2id")
    label_files=glob.glob(file_dir+"*")
    if sys.platform.find("win")!=-1:
        label_lst=[x.split("\\")[-1] for x in label_files]
    else:
        label_lst=[x.split("/")[-1] for x in label_files]
    total_data=[]
    t_label2id={}
    for idd,label in enumerate(label_lst):
        temp={}      
        temp["id"]=int(idd)
        temp["num"]=0
        t_label2id[label]=temp
        path_label=file_dir+label
        path_image=glob.glob(path_label+sp+"*")
        for path in path_image:
            #print(path)
            
            total_data.append([path,label])
            t_label2id[label]["num"]+=1
    print(t_label2id)
    print("训练集数据一共有%s个，一共有%s个类别，其中:"%(len(total_data),len(label_lst)))
    with open(file_dir+"label2id","w",encoding="utf-8") as f:
        for label,temp in t_label2id.items():
            idd=temp["id"]
            num=temp["num"]
            print("类别%s的id是%s，有样本%s个！"%(label,idd,num))
            f.write("%s\t%s\t%s\n"%(label,idd,num))
    train_data=[]
    for path_label in total_data:
        path,label=path_label
        train_data.append([path,t_label2id[label]["id"]])
    n_class=len(label_lst)
    tra = [[x[0],to_onehot(x[1],n_class)] for x in train_data]
    random.shuffle(tra)
    return tra,t_label2id,n_class

def get_test_files(t_label2id, file_dir="test"):
    print("正在获取指定测试集")
    if sys.platform.find("win")!=-1:
        file_dir=file_dir+"\\"
        sp="\\"
    else:
        file_dir=file_dir+"/"
        sp="/"
    label_files=glob.glob(file_dir+"*")
    if sys.platform.find("win")!=-1:
        label_lst=[x.split("\\")[-1] for x in label_files]
    else:
        label_lst=[x.split("/")[-1] for x in label_files]
    total_data=[]
    t_label2cnt={}
    for label in label_lst:
        try:
            idd=t_label2id[label]
        except Exception as e:
            print(e)
            print("训练集和测试集标签不一致，请检查测试集或者训练集")
            exit()
        path_label=file_dir+label
        path_image=glob.glob(path_label+sp+"*")
        for path in path_image:
            total_data.append([path,label])
            t_label2cnt[label]=t_label2cnt.get(label,0)+1
    print("测试集数据一共有%s个，一共有%s个类别，其中:"%(len(total_data),len(label_lst)))
    for label,cnt in t_label2cnt.items():
        print("类别%s的样本有%s个！"%(label,cnt))
    test_data=[]
    for path_label in total_data:
        path,label=path_label
        test_data.append([path,t_label2id[label]["id"]])
    n_class=len(label_lst)
    val = [[x[0],to_onehot(x[1],n_class)] for x in test_data]
    random.shuffle(val)
    return val
def cut_test_from_train(tra,ratio):
    print("正在从训练集中分割测试集")
    tra_new=[]
    val=[]
    for path_idd in tra:
        if random.random()<ratio:
            tra_new.append(path_idd)
        else:
            val.append(path_idd)
    random.shuffle(tra)
    random.shuffle(val)
    return tra,val

if __name__=="__main__":
    #pix_x,pix_y=CF.config["IMG_W"],CF.config["IMG_H"]
    tra,t_label2id,n_class=get_train_files(file_dir="train")
    val=get_test_files(t_label2id, file_dir="noise_test")
    print(tra[:10])
    print(val[:10])
