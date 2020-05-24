#-*-coding:utf-8-*-
#from sklearn import preprocessing
import tensorflow.contrib.layers as layers
import tensorflow.contrib.framework as ops
import tensorflow as tf
import config as CF
import numpy as np
import gc
import pickle
class model(object):
    def __init__(self,sess,config,logging):
        self.sess=sess
        self.config=config
        self.logging=logging
        self._checkpoint_path=self.config["ckpt"]
        self.hight=CF.config["IMG_H"]
        self.width=CF.config["IMG_W"]
        self.n_classes=CF.config["N_CLASSES"]
        self.model_type=CF.config["num"]
        self.global_step=tf.Variable(0,trainable=False)
        self.build()
        self.print_var()
        self.loggingAll()
        self._saver=tf.train.Saver(tf.global_variables(),max_to_keep=2)
        self.initialize()
    def loggingAll(self):
        for name in dir(self):
            if name.find("_")==0 and name.find("_")==-1:
                self.logging.info("self.%s\t%s"%(name,str(getattr(self,name))))
#    def _input(self):
#        self.image_batch_i= tf.placeholder(tf.float32, shape=[None,self.hight, self.width, 3])
#        self.image_label_batch=tf.placeholder(tf.int32, shape=[None,self.n_classes])
#        
#        with tf.variable_scope('to_64') as scope:
#            #(none,h,w,3)
#            if self.hight==64 and self.width==64:
#                self.image_batch=self.image_batch_i
#            else:
#                size_x=self.hight-63
#                size_y=self.width-63
#                w_input= tf.get_variable('w_to_64',shape=[size_x, size_y, 3, 3],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
#                self.image_batch=tf.nn.conv2d(self.image_batch_i, w_input, strides=[1, 1, 1, 1], padding='VALID')
    def _input(self):
        self.image_batch_i= tf.placeholder(tf.float32, shape=[None,self.hight, self.width, 3])
        self.image_label_batch=tf.placeholder(tf.int32, shape=[None,self.n_classes])
        self.image_batch=self.image_batch_i

    def readCKPT(self):
        self.ckpt = tf.train.get_checkpoint_state(self._checkpoint_path)
        if self.ckpt:
            self.logging.info("reading training record from '%s'"%self.ckpt.model_checkpoint_path)
            self._saver.restore(self.sess,self.ckpt.model_checkpoint_path)
            return True
        return False   
 
    def build(self):
        self._input()
        if self.model_type=="1":
            print("选用模型1")
            if self.hight%4!=0 or self.width%4!=0:
                print("ERROR----像素值不合适，请设置像素值是4的倍数")
                exit()
            self.structure_1()
        elif self.model_type=="vgg16":
              print("选用模型vgg16")
              self.is_training = tf.placeholder(tf.bool)
              self.is_use_l2 = tf.placeholder(tf.bool)
              self.lam = tf.placeholder(tf.float32)
              self.keep_prob = tf.placeholder(tf.float32)
              self.structure_2()
        elif self.model_type=="Alexnet":
            print("选用模型Alexnet")
            self.keep_prob = tf.placeholder(tf.float32)
            if self.hight%4!=0 or self.width%4!=0:
                print("请检查像素，确保像素是4的倍数")
            
            self.weights = {
                            'wc1': tf.Variable(tf.random_normal([7, 7, 3, 96])),#[None,]
                            'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),#[None,h/4-5+1,w/4-5+1,]
                            'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
                            'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
                            'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
                            'wd2': tf.Variable(tf.random_normal([1024, 1024])),
                            'out': tf.Variable(tf.random_normal([1024, self.n_classes]))
                        }

            self.biases = {
                            'bc1': tf.Variable(tf.random_normal([96])),
                            'bc2': tf.Variable(tf.random_normal([256])),
                            'bc3': tf.Variable(tf.random_normal([384])),
                            'bc4': tf.Variable(tf.random_normal([384])),
                            'bc5': tf.Variable(tf.random_normal([256])),
                            'bd1': tf.Variable(tf.random_normal([1024])),
                            'bd2': tf.Variable(tf.random_normal([1024])),
                            'out': tf.Variable(tf.random_normal([self.n_classes]))
                        }
            self.alex_net(self.image_batch,self.weights, self.biases, self.keep_prob)
        elif self.model_type=="Googlenet":
            print("选用模型Googlenet")
            self.keep_prob = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder(tf.bool)
            self.restore_logits=None
            self.googlenet(self.image_batch)
        #self.tra_acc(self.image_batch,self.image_label_batch) 
        #self.tes_acc(self.image_batch,self.image_label_batch)
        self.loss(self.image_label_batch)
        self.tra_acc()
        #self.tes_acc()
        self.opt() 
        self.logging.info("model is built")
##############################################################################################################################
#第一个模型的结构
############################################################################################################################
    def structure_1(self):
        with tf.variable_scope('conv1') as scope:
            #[None,h,w,3]
            weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], stddev=1.0, dtype=tf.float32),
                                  name='weights', dtype=tf.float32)
            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[64]),
                                 name='biases', dtype=tf.float32)
            self.image_batch = tf.nn.conv2d(self.image_batch, weights, strides=[1, 1, 1, 1], padding='SAME') 
            self.image_batch = tf.nn.bias_add(self.image_batch, biases)
            self.image_batch = tf.nn.relu(self.image_batch, name=scope.name)
        
        with tf.variable_scope('pooling1_lrn') as scope:
            self.image_batch = tf.nn.max_pool(self.image_batch, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling1')
            #[None,h/2,w/2,64]
            self.image_batch = tf.nn.lrn(self.image_batch, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

        with tf.variable_scope('conv2') as scope:
            weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=0.1, dtype=tf.float32),
                                  name='weights', dtype=tf.float32)
    
            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[64]),
                                 name='biases', dtype=tf.float32)
    
            self.image_batch = tf.nn.conv2d(self.image_batch, weights, strides=[1, 1, 1, 1], padding='SAME')
            
            self.image_batch = tf.nn.bias_add(self.image_batch, biases)
            self.image_batch = tf.nn.relu(self.image_batch, name='conv2')
        
        with tf.variable_scope('pooling2_lrn') as scope:
            self.image_batch = tf.nn.lrn(self.image_batch, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
            self.image_batch = tf.nn.max_pool(self.image_batch, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling2')
            #[None,h/4,w/4,64]
        with tf.variable_scope('local3') as scope:
            self.image_batch = tf.reshape(self.image_batch,shape=[-1, 64*int(self.hight/4)*int(self.width/4)])
            dim = self.image_batch.get_shape()[1].value
            weights = tf.Variable(tf.truncated_normal(shape=[dim, 128], stddev=0.005, dtype=tf.float32),
                                  name='weights', dtype=tf.float32)
            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[128]),
                                 name='biases', dtype=tf.float32)
            
            self.image_batch = tf.nn.relu(tf.matmul(self.image_batch, weights) + biases, name=scope.name)
            #print(local3)
        with tf.variable_scope('local4') as scope:
            weights = tf.Variable(tf.truncated_normal(shape=[128, 128], stddev=0.005, dtype=tf.float32),
                                  name='weights', dtype=tf.float32)
            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[128]),
                                 name='biases', dtype=tf.float32)
    
            self.image_batch = tf.nn.relu(tf.matmul(self.image_batch, weights) + biases, name='local4')
            #print(local4)
        with tf.variable_scope('softmax_linear') as scope:
            weights = tf.Variable(tf.truncated_normal(shape=[128, self.n_classes], stddev=0.005, dtype=tf.float32),
                                  name='softmax_linear', dtype=tf.float32)
            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[self.n_classes]),
                                 name='biases', dtype=tf.float32)
            self.softmax_linear_i = tf.add(tf.matmul(self.image_batch, weights), biases, name='softmax_linear')
        with tf.variable_scope("loss_pb")as scope:
            self.softmax_linear = self.softmax_linear_i
###########################################################################################################################
#第二个模型
#############################################################################################################################
    def weight_variable(self,shape,n,use_l2,lam):
        weight = tf.Variable(tf.truncated_normal(shape, stddev=1 / n))
        if use_l2 is True:
            weight_loss = tf.multiply(tf.nn.l2_loss(weight), lam, name='weight_loss')
            tf.add_to_collection('losses', weight_loss)
        return weight

    def bias_variable(self, shape):
        bias = tf.Variable(tf.constant(0.1, shape=shape))
        return bias

    def conv2d(self, x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
    
    def structure_2(self):
        #image=[B,64,64,3]
        if self.hight%2!=0 or self.width%2!=0:
            print("像素有问题，请检查像素，确保其是2的倍数")
            self.logging.warn("像素有问题，请检查像素，确保其是2的倍数")
            exit()
        with tf.name_scope('conv1_layer'):
            w_conv1 =self.weight_variable([3, 3, 3, 64], 64, use_l2=False, lam=0)
            b_conv1 = self.bias_variable([64])
            self.image_batch = self.conv2d(self.image_batch, w_conv1)#[B,64,64,64]
            self.image_batch = tf.layers.batch_normalization(self.image_batch, training=self.is_training)        
            self.image_batch = tf.nn.relu(tf.nn.bias_add(self.image_batch, b_conv1))

            w_conv2 = self.weight_variable([3, 3, 64, 64], 64, use_l2=False, lam=0)
            b_conv2 = self.bias_variable([64])
            self.image_batch = self.conv2d(self.image_batch, w_conv2)
            self.image_batch = tf.layers.batch_normalization(self.image_batch, training=self.is_training)
            self.image_batch = tf.nn.relu(tf.nn.bias_add(self.image_batch, b_conv2))#[B,32,32,128]

            self.image_batch = self.max_pool_2x2(self.image_batch)
            #[None,h/2,w/2,64] 
        if self.hight%4==0 and self.width%4==0:
            print("第二层") 
            with tf.name_scope('conv2_layer'):
                w_conv3 = self.weight_variable([3, 3, 64, 128], 128, use_l2=False, lam=0)
                b_conv3 = self.bias_variable([128])
                self.image_batch = self.conv2d(self.image_batch, w_conv3)#[B,32,32,128]
                self.image_batch = tf.layers.batch_normalization(self.image_batch, training=self.is_training)
                self.image_batch = tf.nn.relu(tf.nn.bias_add(self.image_batch, b_conv3))
        
                w_conv4 = self.weight_variable([3, 3, 128, 128], 128, use_l2=False, lam=0)
                b_conv4 = self.bias_variable([128])
                self.image_batch = self.conv2d(self.image_batch, w_conv4)
                self.image_batch = tf.layers.batch_normalization(self.image_batch, training=self.is_training)
                self.image_batch = tf.nn.relu(tf.nn.bias_add(self.image_batch, b_conv4))#[B,32,32,128]
        
                self.image_batch = self.max_pool_2x2(self.image_batch)  # 112*112 -> 56*56#[B,16,16,128]
            #[None,h/4,w/4,128]
            if self.hight%8==0 and self.width%8==0:
                print("第三层")
                with tf.name_scope('conv3_layer'):
                    w_conv5 = self.weight_variable([3, 3, 128, 256], 256, use_l2=False, lam=0)
                    b_conv5 = self.bias_variable([256])
                    self.image_batch = self.conv2d(self.image_batch, w_conv5)#[B,16,16,256]
                    self.image_batch = tf.layers.batch_normalization(self.image_batch, training=self.is_training)
                    self.image_batch = tf.nn.relu(tf.nn.bias_add(self.image_batch, b_conv5))
        
                    w_conv6 = self.weight_variable([3, 3, 256, 256], 256, use_l2=False, lam=0)
                    b_conv6 = self.bias_variable([256])
                    self.image_batch = self.conv2d(self.image_batch, w_conv6)
                    self.image_batch = tf.layers.batch_normalization(self.image_batch, training=self.is_training)
                    self.image_batch = tf.nn.relu(tf.nn.bias_add(self.image_batch, b_conv6))
        
                    w_conv7 = self.weight_variable([3, 3, 256, 256], 256, use_l2=False, lam=0)
                    b_conv7 = self.bias_variable([256])
                    self.image_batch = self.conv2d(self.image_batch, w_conv7)
                    self.image_batch = tf.layers.batch_normalization(self.image_batch, training=self.is_training)
                    self.image_batch = tf.nn.relu(tf.nn.bias_add(self.image_batch, b_conv7))
            
                    self.image_batch = self.max_pool_2x2(self.image_batch)  # 56*56 -> 28*28
                if self.hight%16==0 and self.width%16==0: 
                    print("第四层")     
                    with tf.name_scope('conv4_layer'):
                        w_conv8 = self.weight_variable([3, 3, 256, 512], 512, use_l2=False, lam=0)
                        b_conv8 = self.bias_variable([512])
                        self.image_batch = self.conv2d(self.image_batch, w_conv8)
                        self.image_batch = tf.layers.batch_normalization(self.image_batch, training=self.is_training)
                        self.image_batch = tf.nn.relu(tf.nn.bias_add(self.image_batch, b_conv8))
        
                        w_conv9 = self.weight_variable([3, 3, 512, 512], 512, use_l2=False, lam=0)
                        b_conv9 = self.bias_variable([512])
                        self.image_batch = self.conv2d(self.image_batch, w_conv9)
                        self.image_batch = tf.layers.batch_normalization(self.image_batch, training=self.is_training)
                        self.image_batch = tf.nn.relu(tf.nn.bias_add(self.image_batch, b_conv9))
                        
                        w_conv10 = self.weight_variable([3, 3, 512, 512], 512, use_l2=False, lam=0)
                        b_conv10 = self.bias_variable([512])
                        self.image_batch = self.conv2d(self.image_batch, w_conv10)
                        self.image_batch = tf.layers.batch_normalization(self.image_batch, training=self.is_training)
                        self.image_batch = tf.nn.relu(tf.nn.bias_add(self.image_batch, b_conv10))

                        self.image_batch = self.max_pool_2x2(self.image_batch)  # 28*28 -> 14*14

                    if self.hight%32==0 and self.width%32==0:
                        print("第五层")
                        with tf.name_scope('conv5_layer'):
                            w_conv11 = self.weight_variable([3, 3, 512, 512], 512, use_l2=False, lam=0)
                            b_conv11 = self.bias_variable([512])
                            self.image_batch = self.conv2d(self.image_batch, w_conv11)
                            self.image_batch = tf.layers.batch_normalization(self.image_batch, training=self.is_training)
                            self.image_batch = tf.nn.relu(tf.nn.bias_add(self.image_batch, b_conv11)) 
            
                            w_conv12 = self.weight_variable([3, 3, 512, 512], 512, use_l2=False, lam=0)
                            b_conv12 =self. bias_variable([512])
                            self.image_batch = self.conv2d(self.image_batch, w_conv12)
                            self.image_batch = tf.layers.batch_normalization(self.image_batch, training=self.is_training)
                            self.image_batch = tf.nn.relu(tf.nn.bias_add(self.image_batch, b_conv12))
            
                            w_conv13 = self.weight_variable([3, 3, 512, 512], 512, use_l2=False, lam=0)
                            b_conv13 = self.bias_variable([512])
                            self.image_batch =self.conv2d(self.image_batch, w_conv13)
                            self.image_batch = tf.layers.batch_normalization(self.image_batch, training=self.is_training)
                            self.image_batch = tf.nn.relu(tf.nn.bias_add(self.image_batch, b_conv13))
            
                            self.image_batch = self.max_pool_2x2(self.image_batch)  # 14*14 -> 7*7#[B,2,2,512]
        dim1 = self.image_batch.get_shape()[1].value
        dim2 = self.image_batch.get_shape()[2].value
        dim3 = self.image_batch.get_shape()[3].value 
        
        with tf.name_scope('fc1_layer'):
            w_fc14 = self.weight_variable([dim1*dim2 *dim3, 1024], 1024, use_l2=self.is_use_l2, lam=self.lam)
            b_fc14 =self.bias_variable([1024])
            self.image_batch = tf.reshape(self.image_batch, [-1, dim1*dim2*dim3])
            self.image_batch = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.image_batch, w_fc14), b_fc14))#[B,4096]
            self.image_batch = tf.nn.dropout(self.image_batch, self.keep_prob)#?
        
        with tf.name_scope('fc2_layer'):
            w_fc15 = self.weight_variable([1024, 1024], 1024, use_l2=self.is_use_l2, lam=self.lam)
            b_fc15 =self.bias_variable([1024])
            self.image_batch = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.image_batch, w_fc15), b_fc15))
            self.image_batch = tf.nn.dropout(self.image_batch, self.keep_prob)        
        
        with tf.name_scope('output_layer'):
            w_fc16 = self.weight_variable([1024, self.n_classes], self.n_classes, use_l2=self.is_use_l2, lam=self.lam)
            b_fc16 = self.bias_variable([self.n_classes])
            self.image_batch = tf.matmul(self.image_batch, w_fc16) + b_fc16
            self.softmax_linear_i = tf.nn.softmax(self.image_batch)
        with tf.variable_scope("loss_pb")as scope:
            self.softmax_linear = self.softmax_linear_i 
########################################################################################################################################
#第三个模型
########################################################################################################################################
 
    def shuffle_1(self,*arrs):
        arrs = list(arrs)
        for i, arr in enumerate(arrs):
            assert len(arrs[0]) == len(arrs[i])
            arrs[i] = np.array(arr)
            print(arrs[i])
        p = np.random.permutation(len(arrs[0]))
        data_shape = arrs[0].shape
        new_data = np.empty(data_shape, np.float)
        new_label = np.empty(arrs[1].shape, np.float)
        data = arrs[0]
        label = arrs[1]
        for i in range(len(data)):
            tmp_data = data[p[i]]
            new_data[i] = tmp_data   
            tmp_label = label[p[i]]
            new_label[i] = tmp_label   
        return new_data, new_label
        
    def shuffle_2(self,*arrs):
        arrs = list(arrs)
        for i, arr in enumerate(arrs):
            assert len(arrs[0]) == len(arrs[i])
            arrs[i] = np.array(arr)
        p = np.random.permutation(len(arrs[0]))
        return tuple(arr[p] for arr in arrs)
    
    
    def shuffle_3(self,size):
        p = np.random.permutation(size)
        return p
    def conv2d_A(self, name, input, w, b, stride, padding='SAME'):
        x = input.get_shape()[-1]
        x = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding=padding)
        x = tf.nn.bias_add(x, b)
        data_result = tf.nn.relu(x, name=name)
        gc.collect()
        return data_result
    def max_pool(self,name, input, k, stride):
        return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)
    def norm(self,name, input, size=4):
        return tf.nn.lrn(input, size, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
     
    def alex_net(self,x, weights, biases,dropout):
        #[None,h,w,3]

        
        dropout=self.keep_prob 
        conv1 = self.conv2d_A('conv1', x, weights['wc1'], biases['bc1'], stride=2)
        #[None,h/2,w/2,96]
        if self.hight/2<32:
            pool1 = self.max_pool('pool1', conv1, k=3, stride=1)
        else:
            pool1 = self.max_pool('pool1', conv1, k=3, stride=2)
        #[None,h/4,w/4,96]
        norm1 = self.norm('norm1', pool1, size=5)    
        conv2 = self.conv2d_A('conv2', norm1, weights['wc2'], biases['bc2'], stride=1, padding="SAME")   
        pool2 = self.max_pool('pool2', conv2, k=3, stride=1)    
        norm2 = self.norm('norm2', pool2, size=5)
        conv3 = self.conv2d_A('conv3', norm2, weights['wc3'], biases['bc3'], stride=1, padding="VALID")
        conv4 = self.conv2d_A('conv4', conv3, weights['wc4'], biases['bc4'], stride=1, padding="VALID")        
        conv5 = self.conv2d_A('conv5', conv4, weights['wc5'], biases['bc5'], stride=1, padding="VALID")              
        pool5 = self.max_pool('pool5', conv5, k=3, stride=1)
        dim1 = pool5.get_shape()[1].value
        dim2 = pool5.get_shape()[2].value
        dim3 = pool5.get_shape()[3].value 
        
        weights["wd1"]= tf.Variable(tf.random_normal([dim1*dim2*dim3, 1024]))
        fc1 = tf.reshape(pool5, [-1,dim1*dim2*dim3])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1, name='fc1')
        drop1 = tf.nn.dropout(fc1, dropout)
        fc2 = tf.add(tf.matmul(drop1, weights['wd2']), biases['bd2'])
        fc2 = tf.nn.relu(fc2, name='fc2')
        drop2 = tf.nn.dropout(fc2, dropout)      
        self.softmax_linear_i= tf.add(tf.matmul(drop2, weights['out']), biases['out'])
        with tf.variable_scope("loss_pb")as scope:
            self.softmax_linear = self.softmax_linear_i
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
    def get_inception_layer(self, inputs, conv11_size, conv33_11_size, conv33_size,
                         conv55_11_size, conv55_size, pool11_size ):
        with tf.variable_scope("conv_1x1"):
            conv11 = layers.conv2d( inputs, conv11_size, [ 1, 1 ] )    
            
        with tf.variable_scope("conv_3x3"):
            conv33_11 = layers.conv2d( inputs, conv33_11_size, [ 1, 1 ] )
            conv33 = layers.conv2d( conv33_11, conv33_size, [ 3, 3 ] )
            
        with tf.variable_scope("conv_5x5"):
            conv55_11 = layers.conv2d( inputs, conv55_11_size, [ 1, 1 ] )
            conv55 = layers.conv2d( conv55_11, conv55_size, [ 5, 5 ] )
            
        with tf.variable_scope("pool_proj"):
            pool_proj = layers.max_pool2d( inputs, [ 3, 3 ], stride = 1 )
            pool11 = layers.conv2d( pool_proj, pool11_size, [ 1, 1 ] )
            
        if tf.__version__ == '0.11.0rc0':
            return tf.concat(3, [conv11, conv33, conv55, pool11])
        return tf.concat([conv11, conv33, conv55, pool11], 3)
    
    def aux_logit_layer(self,inputs, n_classes, is_training ):        
        with tf.variable_scope("pool2d"):
            pooled = layers.avg_pool2d(inputs, [ 2, 2 ], stride = 2 )  
            
        with tf.variable_scope("conv11"):
            conv11 = layers.conv2d( pooled, 128, [1, 1] )
            
        with tf.variable_scope("flatten"):
            flat = tf.reshape( conv11, [-1, 2048] )

        with tf.variable_scope("fc"):
            fc = layers.fully_connected( flat, 1024, activation_fn=None )
            
        with tf.variable_scope("drop"):
            drop = layers.dropout( fc, 0.3, is_training = self.is_training )
            
        with tf.variable_scope( "linear" ):
            linear = layers.fully_connected( drop, n_classes, activation_fn=None )
            
        with tf.variable_scope("soft"):
            soft = tf.nn.softmax( linear )
        return soft
    
     # 定义整个网络
    def googlenet(self,inputs):
        with tf.name_scope( "google_net", "googlenet", [inputs] ):
            with ops.arg_scope( [ layers.max_pool2d ], padding = 'SAME' ):
                conv0 = layers.conv2d( inputs, 64, [ 7, 7 ], stride = 1, scope = 'conv0' )
                pool0 = layers.max_pool2d(conv0, [3, 3], scope='pool0')
                conv1_a = layers.conv2d( pool0, 64, [ 1, 1 ], scope = 'conv1_a' )
                conv1_b = layers.conv2d( conv1_a, 192, [ 3, 3 ], scope = 'conv1_b' )              
                pool1 = layers.max_pool2d(conv1_b, [ 3, 3 ], scope='pool1')                
                
                with tf.variable_scope("inception_3a"):
                    inception_3a = self.get_inception_layer( pool1, 64, 96, 128, 16, 32, 32 )
                    
                with tf.variable_scope("inception_3b"):
                    inception_3b = self.get_inception_layer( inception_3a, 128, 128, 192, 32, 96, 64 )
                    
                pool2 = layers.max_pool2d(inception_3b, [ 3, 3 ], scope='pool2')
                
                with tf.variable_scope("inception_4a"):
                    inception_4a = self.get_inception_layer( pool2, 192, 96, 208, 16, 48, 64 )
                    
                #with tf.variable_scope("aux_logits_1"):
                    #aux_logits_1 = self.aux_logit_layer( inception_4a, self.n_classes, self.is_training )
                with tf.variable_scope("inception_4b"):
                    inception_4b = self.get_inception_layer( inception_4a, 160, 112, 224, 24, 64, 64 )
                    
                with tf.variable_scope("inception_4c"):
                    inception_4c = self.get_inception_layer( inception_4b, 128, 128, 256, 24, 64, 64 )
    
                with tf.variable_scope("inception_4d"):
                    inception_4d = self.get_inception_layer( inception_4c, 112, 144, 288, 32, 64, 64 )
    
                #with tf.variable_scope("aux_logits_2"):
                    #aux_logits_2 = self.aux_logit_layer( inception_4d, self.n_classes, self.is_training )                    
                with tf.variable_scope("inception_4e"):
                    inception_4e = self.get_inception_layer( inception_4d, 256, 160, 320, 32, 128, 128 )
                    
                pool3 = layers.max_pool2d(inception_4e, [ 3, 3 ], scope='pool3')               
                with tf.variable_scope("inception_5a"):
                    inception_5a = self.get_inception_layer( pool3, 256, 160, 320, 32, 128, 128 )  
                    
                with tf.variable_scope("inception_5b"):
                    inception_5b = self.get_inception_layer( inception_5a, 384, 192, 384, 48, 128, 128 ) 
                pool4 = layers.avg_pool2d(inception_5b, [ 2, 2 ], stride = 1, scope='pool4')
                reshape = tf.reshape( pool4, [-1, 1024*3*3] )
                dropout = layers.dropout( reshape,self.keep_prob, is_training = self.is_training )
                logits = layers.fully_connected( dropout, self.n_classes, activation_fn=None, scope='logits')
                predictions = tf.nn.softmax(logits, name='predictions')
        self.softmax_linear_i=predictions
        with tf.variable_scope("loss_pb")as scope:
            self.softmax_linear = self.softmax_linear_i         
        
        
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################        
    def loss(self,image_label_batch):
        with tf.variable_scope('loss'):
            #print(image_label_batch)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.softmax_linear, 
                                                                           labels=self.image_label_batch,
                                                                           name='xentropy_per_example')
            self.loss = tf.reduce_mean(cross_entropy, name='loss')
    #def loss(self):        
     #   self.predict=self.softmax_linear
       # self.loss=tf.nn.l2_loss(self.predict-self.y)
    def print_var(self):
        for item in dir(self):
            type_string=str(type(getattr(self,item)))
            print(item,type_string)
    def opt(self):
        self._opt=tf.train.AdadeltaOptimizer(self.config["learning_rate"])
        self._train_opt=self._opt.minimize(self.loss,global_step=self.global_step)
        
    def initialize(self):
        print ("now initialize")
        if not self.readCKPT():
            print ("no ckpt")
            self.sess.run(tf.global_variables_initializer())
        #self.sess.run(tf.global_variables_initializer())
    def train(self,train_batch,train_label_batch,i):
        if self.model_type=="1":
           feed_dict={self.image_batch_i:train_batch,self.image_label_batch:train_label_batch}
        elif self.model_type=="vgg16":       
            feed_dict={self.image_batch_i:train_batch, self.image_label_batch:train_label_batch,self.keep_prob:0.8, self.is_training:False,self.is_use_l2:True,self.lam:0.001}
        elif self.model_type=="Alexnet":
            feed_dict={self.keep_prob:0.8,self.image_batch_i:train_batch, self.image_label_batch:train_label_batch}
        elif self.model_type=="Googlenet":
            feed_dict={self.image_batch_i:train_batch, self.image_label_batch:train_label_batch,self.keep_prob:0.4,self.is_training:True}
            
        _,tra_loss,global_step,tra_acc,label_pre=self.sess.run([self._train_opt,self.loss,self.global_step,self.tra_accurate,self.softmax_linear],feed_dict=feed_dict)
        if i%10==0:
            print("loss is %s,global_step is %s,tra_acc is %.2f%%"%(tra_loss,global_step,tra_acc*100.0))
            self.logging.info("loss is %s,global_step is %s,tra_acc is %.2f%%,i is %s"%(tra_loss,global_step,tra_acc*100.0,i))
            #print(label_pre)
            #print(train_label_batch)
            #print("=====================")
            if i %1000==0:
                self._saver.save(self.sess,self._checkpoint_path+"checkpoint",global_step=global_step)
            

#    def test(self,test,test_label,i):
#        if self.model_type==1:
#            feed_dict={self.image_batch:test,self.image_label_batch:test_label}
#            loss,global_step,tes_acc=self.sess.run([self.loss,self.global_step,self.tra_accurate],feed_dict=feed_dict)
#            if i%2==0:
#                print("测试集：loss is %s,global_step is %s,tes_acc is %.2f%%,i is %s"%(loss,global_step,tes_acc*100.0,i))

    def tra_acc(self):
        with tf.variable_scope('train_accuracy'):     
            correct = tf.nn.in_top_k(self.softmax_linear, tf.argmax(self.image_label_batch,1), 1)      
            correct=tf.cast(correct,tf.float32) 
            self.tra_accurate=tf.reduce_mean(correct)
           
    def savePB(self):
        print ("保存BP",self.softmax_linear)
        name_pb=self.softmax_linear.name.split(":")[0]
        input_name=self.image_batch_i.name
        keep_node=self.keep_prob.name
        is_training=self.is_training.name
        self.config["input_node"]=input_name
        self.config["output_node"]=name_pb
        try:
            self.config["is_training"]=is_training
        except:
            pass
        self.config["keep_node"]=keep_node#{"input":input_name,"output":name_pb}
        with open("Flusk_config","wb")as f:
            pickle.dump(self.config,f)
        if self.readCKPT():
            output_node_names = [name_pb]
            output_graph_def = tf.graph_util.convert_variables_to_constants(self.sess,self.sess.graph_def,output_node_names=output_node_names)
            with tf.gfile.FastGFile("model.pb",mode="wb") as f:
                f.write(output_graph_def.SerializeToString())
            self.logging.info("pb file is saved")
        else:
            self.logging.warn("there is nothing to be save")
    

    def tes_acc(self,test,test_label):
        if self.model_type=="1":
            feed_dict={self.image_batch_i:test,self.image_label_batch:test_label}                
        elif self.model_type=="vgg16":
            self.logging.info("由于算法复杂，采用随机抽取 batch_size 个测试集样本进行测试")
            feed_dict={self.image_batch_i:test, self.image_label_batch:test_label,self.keep_prob:1, self.is_training:False,self.is_use_l2:True,self.lam:0.001}
        elif self.model_type=="Alexnet":
            feed_dict={self.image_batch_i:test, self.image_label_batch:test_label,self.keep_prob:1}
        elif self.model_type=="Googlenet":
            feed_dict={self.image_batch_i:test, self.image_label_batch:test_label,self.keep_prob:1,self.is_training:True}
        loss,accurate = self.sess.run([self.loss, self.tra_accurate],feed_dict=feed_dict)
        print("测试集：loss is %s,tes_acc is %.2f%%"%(loss,accurate*100.0))
        self.logging.info("测试集：loss is %s,tes_acc is %.2f%%"%(loss,accurate*100.0))
