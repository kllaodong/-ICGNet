import tensorflow as tf
import os
import tensorflow.contrib.slim as slim
# import mobilenet_v2
from FPN import resnet

ROOT_PATH='./FPN'
NET_NAME='resnet_v1_101'
if NET_NAME.startswith("resnet"):
    weights_name = NET_NAME
elif NET_NAME.startswith("MobilenetV2"):
    weights_name = "mobilenet/mobilenet_v2_1.0_224"
else:
    raise NotImplementedError
PRETRAINED_CKPT = ROOT_PATH + '/pretrained_weights/' + weights_name + '.ckpt'
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')
VERSION = 'FPN_Res101_20181201'

def build_base_network( input_img_batch):
    is_training=True
    base_network_name=NET_NAME  #目前只是使用了resnet_v1_101，
    if base_network_name.startswith('resnet_v1'):
        return resnet.resnet_base(input_img_batch, scope_name=base_network_name, is_training=is_training)
    #elif base_network_name.startswith('MobilenetV2'):
    #    return mobilenet_v2.mobilenetv2_base(input_img_batch, is_training=is_training)
    else:
        raise ValueError('Sry, we only support resnet or mobilenet_v2')
        
def build_whole_detection_network(input_img_batch):
    P_list = build_base_network(input_img_batch)  # [P2, P3, P4, P5, P6]
    return P_list

import cv2
import numpy as np

batch_size=16
def data_preprocess(path):
    src=cv2.resize(cv2.imread(path),(256,256))
    datas=[]
    for dd in range(batch_size):
        datas.append(src)
    #src=np.array(src)[np.newaxis,:]
    #src=np.tile(src,(batch_size,1))
    return np.array(datas)

def get_restorer():
    base_network_name=NET_NAME
    checkpoint_path = tf.train.latest_checkpoint(os.path.join(TRAINED_CKPT, VERSION))

    if checkpoint_path != None:
        restorer = tf.train.Saver()
        print("model restore from :", checkpoint_path)
    else:
        checkpoint_path = PRETRAINED_CKPT
        print("model restore from pretrained mode, path is :", checkpoint_path)

        model_variables = slim.get_model_variables()

        def name_in_ckpt_rpn(var):
            return var.op.name

        def name_in_ckpt_fastrcnn_head(var):
            '''
            Fast-RCNN/resnet_v1_50/block4 -->resnet_v1_50/block4
            Fast-RCNN/MobilenetV2/** -- > MobilenetV2 **
            :param var:
            :return:
            '''
            return '/'.join(var.op.name.split('/')[1:])
        nameInCkpt_Var_dict = {}
        for var in model_variables:
            if var.name.startswith(base_network_name):
                var_name_in_ckpt = name_in_ckpt_rpn(var)
                nameInCkpt_Var_dict[var_name_in_ckpt] = var
        restore_variables = nameInCkpt_Var_dict
        # for key, item in restore_variables.items():
        #     print("var_in_graph: ", item.name)
        #     print("var_in_ckpt: ", key)
        #     print(20*"___")
        restorer = tf.train.Saver(restore_variables)
        print(20 * "****")
        print("restore from pretrained_weighs in IMAGE_NET")
    return restorer, checkpoint_path


if __name__=='__main__':

    input=tf.placeholder(dtype=tf.float32,shape=[None,256,256,3],name='input')
    d=build_whole_detection_network(input)
    latent=resnet.get_latent(d)
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        data=data_preprocess('../1.jpg')
        sess.run(init)
        restorer,restore_ckpt=get_restorer()
        restorer.restore(sess, restore_ckpt)
        latent_code=sess.run(latent,feed_dict={input:data})
        print(latent_code.shape)

