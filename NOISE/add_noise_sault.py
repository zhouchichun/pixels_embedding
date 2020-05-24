import PIL.Image as I
import numpy as np
import glob,os,sys
import random
def transform(max_,min_,im_ar):
    #print(im_ar)
    k=255/(max_-min_)
    b=255*min_/(min_-max_)
    ret=k*im_ar+b
    #print(ret)
    #exit()
    return ret.astype('uint8')

def norm(im_ar):
    #print(im_ar)
    mean=np.mean(im_ar)
    std=np.std(im_ar)
    ret=(im_ar-mean)/std
    ret=(ret-np.min(ret))*100
    ret=ret.astype('uint8')#transform(np.max(ret),np.min(ret),ret)
    #print(im_ar-ret)
    #exit()
    return ret

def add_noise(im_ar,pro,val):
    shape=im_ar.shape
    noise=np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if random.random()<pro:
                noise[i][j]=val
    noise=noise.astype('uint8')
    return im_ar+noise


if __name__=="__main__":
    mean,std=sys.argv[1:]
    mean=float(mean)
    std=float(std)
    path_name="test_noise_jiaoyan_%s_%s"%(mean,std)
    if glob.glob(path_name)==[]:
        os.mkdir(path_name)
    label_lst=glob.glob("test/*")
    
    for label in label_lst:
        print(label)
        la=label.split("/")[-1]
        to_path=path_name+"/"+la
        if glob.glob(to_path)==[]:
            os.mkdir(to_path)
        im_lst=glob.glob(label+"/*")
        for im in im_lst:
            to_save=to_path+"/"+im.split("/")[-1]
            im_image=I.open(im)
            im_array=np.array(im_image).astype('uint8')
            im_array_noise=add_noise(im_array,mean,std)
            im_noise=I.fromarray(im_array_noise)
            im_noise.save(to_save)
