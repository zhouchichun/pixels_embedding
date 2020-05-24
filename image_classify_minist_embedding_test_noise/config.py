import glob
import os



#CONFIG={"num":"vgg16"}#["Alexnet","Googlenet","vgg16",1]
#k=CONFIG["num"]
#if k==1:
config={
        #变量声明
        "step_to_test":10,
        "is_test":True,
        "N_CLASSES":0,#数据类型  
        "ckpt":"ckpt_1/",
        #"logging_name":"a_simple_frame.log",
        "BATCH_SIZE" : 64,
        "CAPACITY" : 200,
        "MAX_STEP": 10000 , # 一般大于10K
        "learning_rate" : 0.005,  # 一般小于0.0001
        "ratio":0.9,#训练集与测试集的比率
        "train_dir": './input_data/' , # 训练样本的读入路径
        "logs_train_dir": './save' ,# logs存储路径
        "IMG_W":28,#resize图像
        "IMG_H":28,   
        "num":"1",#["1","vgg16","Alexnet","Googlenet"],分别代表模型1~4
        "img_embed_dim":64,
        'min_':-1000,
        'max_':1000
        }
config["ckpt"]="ckpt_"+config["num"]+"/"
config["logging_name"]="log_"+config["num"]+".log"

if glob.glob(config["ckpt"])==[]:
    os.mkdir(config["ckpt"])

