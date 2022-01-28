
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from PIL import Image
import cv2
import os
import numpy as np
from keras.preprocessing.image import array_to_img, img_to_array, load_img
   
def display_images(directory, numOfImages = 5):
    file_list = glob.glob(directory + "/*/*")
    indicies = random.sample(range(len(file_list)), numOfImages * numOfImages)    
    fig, axes = plt.subplots(nrows=numOfImages,ncols=numOfImages, figsize=(15,15), sharex=True, sharey=True, frameon=False)
    for i,ax in enumerate(axes.flat):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #Pick a random picture from the file list
        imgplot = mpimg.imread(file_list[indicies[i]], 0)
        ax.imshow(imgplot)
        ax.text(10,20,file_list[indicies[i]].split("/")[-2], fontdict={"backgroundcolor": "black","color": "white" })
        ax.axis('off')
    plt.tight_layout(h_pad=0, w_pad=0)
    

def resize_image(file, size=299):
    img = Image.open(file)
    img = img.resize((size,size))
    img.save(file)

def crop_gaussian(file):
    img = Image.open(file)
    img = img_to_array(img)
    img = circle_crop(img)
    img=array_to_img(img)
    img = img.resize((299,299))
    img.save(file)
    
def circle_crop(img):   
    img = crop_image_from_gray(img)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img=cv2.addWeighted(img,4, cv2.GaussianBlur( img , (0,0) , 30) ,-4 ,128)
    return img

def crop(img):
    img = crop_image_from_gray(img)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    #img=cv2.addWeighted(img,4, cv2.GaussianBlur( img , (0,0) , 30) ,-4 ,128)
    return img
    
    
def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

