import numpy as np
import model as m
import input_data 
import feature_extra
import random
import logging
import config as CF
import tensorflow as tf
import time
import argparse
import glob
parser = argparse.ArgumentParser()
parser.description='训练还是生成bp'
parser.add_argument("--train", default=False,help="是否训练",type=bool)
parser.add_argument("--savePB", default=False,help="我是B",type=bool)
#parser.add_argument("--savePB", default=False,help="我是B",type=bool)
args = parser.parse_args()
is_train=args.train
is_savePB=args.savePB
if not is_train and not is_savePB:
   exit()


is_test=CF.config["is_test"]
pix_x,pix_y=CF.config["IMG_W"],CF.config["IMG_H"]
max_step=CF.config["MAX_STEP"]
batch_size=CF.config["BATCH_SIZE"]
step_to_test=CF.config["step_to_test"]
if is_train:
    if is_test:
        tra,t_label2id,n_class=input_data.get_train_files(file_dir="train")
        val=input_data.get_test_files(t_label2id, file_dir="test")
    else:
        tra,t_label2id,n_class=input_data.get_train_files(file_dir="train")
        tra,val=input_data.cut_test_from_train(tra,0.9)
    CF.config["N_CLASSES"]=n_class
    total_num=len(tra)
    logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(filename)s[line:%(lineno)d]%(levelname)s :%(message)s",
                    datefmt="%Y-%m_%d %H:%M:%S",
                    filename=CF.config["logging_name"],
                    filemode="w")

else:
    if glob.glob("train/label2id")==[]:
        print("没有训练，保存不了BP")
        exit()
    
    label2id_path=glob.glob("train/label2id")[0]
    CF.config["N_CLASSES"]=len(open(label2id_path,"r",encoding="utf-8").readlines())

sess=tf.Session()
model_1=m.model(sess=sess,config=CF.config,logging=logging)
model_1.print_var()


if is_train:
    for k in range(max_step*total_num):
        if k>0:
            random.shuffle(tra)
        for step in range(int(total_num/batch_size)-1):
            train_lst=tra[step*batch_size:batch_size*(step+1)]
        #st=time.time()
            batch_x,batch_y=feature_extra.give_batch(train_lst,pix_x,pix_y,ref=False)
        #print("处理这一批数据用了%s 秒"%(time.time()-st))
        #st=time.time()
            model_1.train(batch_x,batch_y,step)
            if (step+1)%step_to_test==0:
                if CF.config["num"] in ["vgg16","Alexnet"]:
                    valvgg=random.sample(val,batch_size)
                    batch_x,batch_y=feature_extra.give_batch(valvgg,pix_x,pix_y,ref=False)
                else:
                    batch_x,batch_y=feature_extra.give_batch(val,pix_x,pix_y,ref=False)
            #    print(batch_y)
                model_1.tes_acc(batch_x,batch_y)
if is_savePB:
    model_1.savePB()
