# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 00:35:07 2021

@author: 元元吃汤圆
"""


import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import cv2
import os
import sklearn.metrics as sm
from scipy.cluster.vq import *
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC


def read_img_path():
    """
    获取图片路径
    """
    picture_list,label_list=[],[]
    root_dir='D:/data2'
    sub_dir_list=os.listdir(root_dir)
    for index,sub_dir in enumerate(sub_dir_list):
        sub_path=os.path.join(root_dir,sub_dir)
        small_dir_list=os.listdir(sub_path)
        print(small_dir_list)
        for small_dir in small_dir_list:
            small_dir_path=os.path.join(sub_path,small_dir)
            pic_list=os.listdir(small_dir_path)
            for pic in pic_list:
                if(pic.startswith('.')):
                    continue
                pic_path=os.path.join(small_dir_path,pic)
                picture_list.append(pic_path)
                label_list.append(index)
    print(picture_list,label_list)
    return picture_list,label_list

def read_img_test():
    """
    获取图片路径
    """
    picture_test,label_test=[],[]
    root_dir1='D:/data3'
    sub_dir_list1=os.listdir(root_dir1)
    for index,sub_dir in enumerate(sub_dir_list1):
        sub_path1=os.path.join(root_dir1,sub_dir)
        small_dir_list1=os.listdir(sub_path1)
        print(small_dir_list1)
        for small_dir in small_dir_list1:
            small_dir_path1=os.path.join(sub_path1,small_dir)
            pic_list1=os.listdir(small_dir_path1)
            for pic in pic_list1:
                if(pic.startswith('.')):
                    continue
                pic_path1=os.path.join(small_dir_path1,pic)
                picture_test.append(pic_path1)
                label_test.append(index)
    print(picture_test,label_test)
    return picture_test,label_test



def extract_feature(path_list):
    """
    对图片进行surf特征提取
    """
    features=[]
    surf = cv2.xfeatures2d.SURF_create(10)
    for image_path in path_list:
        raw_img=cv2.imread(image_path)
        image_rgb=cv2.cvtColor(raw_img,cv2.COLOR_BGR2RGB)
        image_stand=cv2.resize(image_rgb,(128,128),interpolation=cv2.INTER_CUBIC)
        _, fea = surf.detectAndCompute(image_stand, None)
        features.append(fea)
    return features


def kmeans_model(features,num):
    all_ele,num_clu=[],1
    for feature in features:
        for value in feature:
            all_ele.append(value)
    center,_=kmeans(all_ele,num,num_clu)
    final_fea=np.zeros((len(features),num))
    for index in range(len(features)):
        dic,dis=vq(features[index],center)
        for ele in dic:
            final_fea[index][ele]+=1
    return final_fea



if __name__=="__main__":
    picture_list,label_list=read_img_path()
    raw_train_features = extract_feature(picture_list)
    features = kmeans_model(raw_train_features,500)
    labels = np.reshape(np.array(label_list), (-1,))
    
    piclist_test,lablist_test=read_img_test()
    raw_train_test = extract_feature(piclist_test)
    features_test = kmeans_model(raw_train_test,500)
    labels_test = np.reshape(np.array(lablist_test), (-1,))
    
    
    # build svm model
    print('———————————————————————————————————————build svm model—————————————————————————————————————')
    svm_model = LinearSVC()
    svm_model.fit(features,labels)
    predict = svm_model.predict(features_test)
    svm_accuracy_value = accuracy_score(labels_test, predict)
    print('svm accuracy is :', svm_accuracy_value)
    cp = sm.classification_report(labels_test, predict)
    print(cp)

