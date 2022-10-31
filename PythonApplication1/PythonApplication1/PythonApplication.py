import tkinter as tk
from tkinter import ttk
import sys
import os
from tkinter import *
from tkinter import messagebox,filedialog
import numpy as np
from PIL import Image, ImageTk ,ImageFilter
import cv2
import os
from scipy.ndimage import maximum_filter, minimum_filter
from sklearn.cluster import MeanShift, estimate_bandwidth
import decimal as d
import math
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
 
import keras
import cv2 as ocv
from keras.preprocessing import image
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator
from numpy.random import rand, shuffle
from PIL import Image, ImageTk
from itertools import count, cycle
 
         
 
class Application(tk.Frame):
    def __init__(self,master):
        super().__init__(master)
        self.pack()

        self.master.geometry("800x700")
        self.master.title("Image Processing ")
        self.master.configure(bg='gray12')
        self.create_widgets()
        self.flag=0 
         
    ## show processed img
    def show_new_image(self):
        self.image_bgr_resize = cv2.resize(self.new_image, self.new_size, interpolation=cv2.INTER_AREA)#
        self.image_bgr_resize = cv2.normalize(self.image_bgr_resize, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.image_PIL = Image.fromarray(self.image_bgr_resize) #Convert from RGB to PIL format
        self.image_tk2 = ImageTk.PhotoImage(self.image_PIL) #Convert to ImageTk format
        self.canvas1.create_image(450,240, image=self.image_tk2)
        self.canvas1.grid(column=0, row=1)

    ####  Spatial Filters ####
    def Average_filter(self):
        self.gray_img = self.image_rgb
        rows, cols = self.gray_img.shape
        self.new_image = cv2.blur(self.gray_img, (9,9))
        ######
        self.show_new_image()
    def Gaussian_filter (self):
        self.gray_img = self.image_rgb
        self.new_image = cv2.GaussianBlur(self.gray_img, (9, 9), 0)

        ######
        self.show_new_image()
    def lablacian_filter(self):
        self.gray_img = self.image_rgb
        self.new_image= cv2.Laplacian(self.gray_img,cv2.CV_64F) 
         ######
        self.show_new_image()

    def Min_filter(self):
        self.gray_img = self.image_rgb 
        self.new_image= minimum_filter(self.gray_img, (3,3)) 
        ######
        self.show_new_image() 
    def Max_filter(self):
        self.gray_img = self.image_rgb
        self.new_image=maximum_filter(self.gray_img, (3, 3)) 
        ######
        self.show_new_image() 
         
    def Median_filter(self):
        self.gray_img = self.image_rgb
        self.new_image= cv2.medianBlur(self.gray_img,9)
        ######
        self.show_new_image()
    def midpoint(self):
        self.gray_img = self.image_rgb 
        maxf = maximum_filter(self.gray_img, (3, 3))
        minf = minimum_filter(self.gray_img, (3, 3))
        self.new_image = (maxf + minf) / 2
        ####
        self.show_new_image() 
    def AlphaTrimmedMeanFilter(self):
        self.gray_img = self.image_rgb
        kernel = np.ones([5,5], dtype = np.uint8)
        kernel = kernel / np.sum(kernel)
        image_h, image_w = self.gray_img.shape    
        h,w =kernel.shape
        self.new_image=np.zeros_like(self.gray_img) 
        d = 5
        for i in range(h, image_h - h):
            for j in range(w, image_w - w):
                l = []
                for m in range(h):
                    for n in range(w):
                        l.append((kernel[m, n] * self.gray_img[i-h+m][j-w+n]))
                l = sorted(l)
                s = 0
                for k in range(d, len(l) - d):
                    s += l[k]
                self.new_image[i, j] = s
         ####
        self.show_new_image() 

    def addition(self):
       # image_rgb2=self.
        tk.messagebox.showinfo("OK", "Choose  the second image to make addition")
        self.filename2 = filedialog.askopenfilename()
        self.image_bgr2 = cv2.imread(self.filename2)
        self.image_bgr_resize2 = cv2.resize(self.image_bgr2, self.new_size, interpolation=cv2.INTER_AREA)
        self.image_rgb2 = cv2.cvtColor(self.image_bgr_resize2, cv2.COLOR_BGR2GRAY)
        self.img1 = self.image_rgb  
        self.img2 = self.image_rgb2
        self.new_image = cv2.addWeighted(self.img1, 0.5, self.img2, 0.4, 0)
        self.show_new_image() 
    def Subtraction(self):
       # image_rgb2=self.
        tk.messagebox.showinfo("OK", "Choose  the second image to make addition")
        self.filename2 = filedialog.askopenfilename()
        self.image_bgr2 = cv2.imread(self.filename2)
        self.image_bgr_resize2 = cv2.resize(self.image_bgr2, self.new_size, interpolation=cv2.INTER_AREA)
        self.image_rgb2 = cv2.cvtColor(self.image_bgr_resize2, cv2.COLOR_BGR2GRAY)
        self.img1 = self.image_rgb  
        self.img2 = self.image_rgb2
        self.new_image = cv2.subtract(self.img1, self.img2)
        self.show_new_image() 
   #### frequency Domain Filters #### 
     
    def ideal_low_pass(self):
        #print(self.original_image.shape)
        self.image=self.image_rgb
        imageFloat32 = np.float32(self.image)
        dft = cv2.dft(imageFloat32,flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        D0 = 15    #// ex 15      
        idealImage = self.IdealLowpassFilter(dft_shift, D0)
        idft_shift = np.fft.ifftshift(idealImage)
        idft = cv2.idft(idft_shift)
        self.new_image = cv2.magnitude(idft[:,:,0], idft[:,:,1])
        self.show_new_image()
    def IdealLowpassFilter(self, shiftedImage,D0):
        image_h, image_w, _ = shiftedImage.shape
        h, w = image_h // 2, image_w // 2
        # Prepare Filter
        H = np.zeros((image_h,image_w,2))
        for u in range(image_h):
            for v in range(image_w):
                D = np.sqrt(((u - h)**2) + ((v - w)**2))
                if D <= D0:
                    H[u, v] = 1
                else:
                    H[u, v] = 0
        im = shiftedImage * H
        return im
    ###
    def butterworth_low_pass(self): 
        self.image=self.image_rgb
        imageFloat32 = np.float32(self.image)
        dft = cv2.dft(imageFloat32,flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        D0 = 15 #d0=15        
        n = 2   #n=2      
        butterImage = self.ButterworthLowpassFilter(dft_shift, D0, n)
        idft_shift = np.fft.ifftshift(butterImage)
        idft = cv2.idft(idft_shift)
        self.new_image = cv2.magnitude(idft[:,:,0], idft[:,:,1])
        self.show_new_image()
    def ButterworthLowpassFilter(self, shiftedImage, D0, n):
        image_h, image_w, _ = shiftedImage.shape
        h, w = image_h // 2, image_w // 2
        # Prepare Filter
        H = np.zeros((image_h,image_w,2))
        P = 2 * n
        for u in range(image_h):
            for v in range(image_w):
                D = np.sqrt(((u - h)**2) + ((v - w)**2))
                H[u, v] = d.Decimal(1.0 / (1.0 + pow( D / D0, P)))
        im = shiftedImage * H
        return im

    ###
    def gussion_low_pass(self):
        self.image=self.image_rgb
        imageFloat32 = np.float32(self.image)
        dft = cv2.dft(imageFloat32,flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        D0 =  15        
        gussionImage = self.GuassionLowpassFilter(dft_shift, D0)
        idft_shift = np.fft.ifftshift(gussionImage)
        idft = cv2.idft(idft_shift)
        self.new_image = cv2.magnitude(idft[:,:,0], idft[:,:,1])
        self.show_new_image()
    def GuassionLowpassFilter(self, shiftedImage, D0):
        image_h, image_w, _ = shiftedImage.shape
        h, w = image_h // 2, image_w // 2
        # Prepare Filter
        H = np.zeros((image_h,image_w,2))
        for u in range(image_h):
            for v in range(image_w):
                D = np.sqrt(((u - h)**2) + ((v - w)**2))
                H[u, v] = d.Decimal(pow(math.e, (-D**2) / (2*D0**2)))
        im = shiftedImage * H
        return im
    ###
    def ideal_high_pass(self):
        self.image=self.image_rgb
        imageFloat32 = np.float32(self.image)
        dft = cv2.dft(imageFloat32,flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        D0 = 30      
        idealImage = self.IdealHighpassFilter(dft_shift, D0)
        idft_shift = np.fft.ifftshift(idealImage)
        idft = cv2.idft(idft_shift)
        self.new_image = cv2.magnitude(idft[:,:,0], idft[:,:,1])
        self.show_new_image()
    def IdealHighpassFilter(self, shiftedImage,D0):
        image_h, image_w, _ = shiftedImage.shape
        h, w = image_h // 2, image_w // 2
        # Prepare Filter
        H = np.zeros((image_h,image_w,2))
        for u in range(image_h):
            for v in range(image_w):
                D = np.sqrt(((u - h)**2) + ((v - w)**2))
                if D <= D0:
                    H[u, v] = 0
                else:
                    H[u, v] = 1
        im = shiftedImage * H
        return im
    ###
    def butterworth_high_pass(self):
        self.image=self.image_rgb
        imageFloat32 = np.float32(self.image)
        dft = cv2.dft(imageFloat32,flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        D0 = 30       
        n =  2        
        butterImage = self.ButterworthHighpassFilter(dft_shift, D0, n)
        idft_shift = np.fft.ifftshift(butterImage)
        idft = cv2.idft(idft_shift)
        self.new_image = cv2.magnitude(idft[:,:,0], idft[:,:,1])
        self.show_new_image()
    def ButterworthHighpassFilter(self, shiftedImage, D0, n):
        image_h, image_w, _ = shiftedImage.shape
        h, w = image_h // 2, image_w // 2
        # Prepare Filter
        H = np.zeros((image_h,image_w,2))
        P = 2 * n
        for u in range(image_h):
            for v in range(image_w):
                D = np.sqrt(((u - h)**2) + ((v - w)**2))
                H[u, v] = d.Decimal(1.0 / (1.0 + pow( D0 / D, P)))
        im = shiftedImage * H
        return im

    ### 
    def gussion_high_pass(self):
        self.image=self.image_rgb
        imageFloat32 = np.float32(self.image)
        dft = cv2.dft(imageFloat32,flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        D0 = 15       
        gussionImage = self.GuassionHighpassFilter(dft_shift, D0)
        idft_shift = np.fft.ifftshift(gussionImage)
        idft = cv2.idft(idft_shift)
        self.new_image = cv2.magnitude(idft[:,:,0], idft[:,:,1])
        self.show_new_image()
    def GuassionHighpassFilter(self, shiftedImage, D0):
        image_h, image_w, _ = shiftedImage.shape
        h, w = image_h // 2, image_w // 2
        # Prepare Filter
        H = np.zeros((image_h,image_w,2))
        for u in range(image_h):
            for v in range(image_w):
                D = np.sqrt(((u - h)**2) + ((v - w)**2))
                H[u, v] = 1 - d.Decimal(pow(math.e, (-D**2) / (2*D0**2)))
        im = shiftedImage * H
        return im 

    #Applying Gaussian blurring helps remove some of the high frequency edges in the image that we are not concerned with and allow us to obtain a more “clean” segmentation.
    def global_thresholding_binary(self):
        self.gray_img = self.image_rgb 
        blur = cv2.GaussianBlur(self.gray_img,(5,5),0)
        ret, self.new_image = cv2.threshold(blur,127, 255, cv2.THRESH_BINARY)
        ####
        self.show_new_image() 
    def otsu_thresholding(self):
        self.gray_img = self.image_rgb 
        blur = cv2.GaussianBlur(self.gray_img,(5,5),0)
        ret,self.new_image = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
         ####
        self.show_new_image()
    def adaptive_thresholding(self):
        self.gray_img = self.image_rgb 
        blur = cv2.GaussianBlur(self.gray_img,(5,5),0)
        self.new_image =  cv2.adaptiveThreshold(blur, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        ####
        self.show_new_image()

    def Arithmatic_Mean_Filter(self):
        self.gray_img = self.image_rgb 
        rows, cols=self.gray_img.shape
        ksize = 5
        padsize = int((ksize - 1) / 2)
        pad_img = cv2.copyMakeBorder(self.gray_img, *[padsize] * 4, cv2.BORDER_DEFAULT)
        self.new_image= np.zeros_like(self.gray_img) 
        for r in range(rows):
            for c in range(cols):
                self.new_image[r, c] = np.sum(pad_img[r:r + ksize, c:c + ksize])/(ksize * ksize)
         ####
        self.show_new_image()
####
    def GMF(self,img):
        m, n = img.shape
        self.new_image = np.zeros((m, n), np.uint8)  # output image set with placeholder values of all zeros
        val = 1  # variable to hold new pixel value
        i, j = 2, 2
        for i in range(m - 2):  # loop through each pixel in original image
            for j in range(n - 2):  # compute geometric mean of 3x3 window around pixel
                p = img[i - 1, j - 1]
                q = img[i - 1, j]
                r = img[i - 1, j + 1]
                s = img[i, j - 1]
                t = img[i, j]
                u = img[i, j + 1]
                v = img[i + 1, j - 1]
                w = img[i + 1, j]
                x = img[i + 1, j + 1]
                # print(img[i - 1, j - 1])
                val = np.product(p * q * r * s * t * u * v * w * x) ** (1 / 9)
                self.new_image[i, j] = val  # set output pixel to computed geometric mean
                val = 1;  # reset val for next pixel
        return self.new_image
    def Geometric_filter(self):
        self.gray_img = self.image_rgb.astype(float)  
        rows, cols=self.gray_img.shape
        ksize = 5
        padsize = int((ksize - 1) / 2)
        pad_img = cv2.copyMakeBorder(self.gray_img, *[padsize] * 4, cv2.BORDER_DEFAULT)
        self.new_image= np.zeros_like(self.gray_img)
        #Geometric mean filter
        for r in range(rows):
            for c in range(cols):
                self.new_image[r, c] = np.prod(pad_img[r:r+ksize, c:c+ksize])**(1/(ksize**2))
         ####
        self.show_new_image()
####
    def harmonic_Mean_Filter(self):
        self.gray_img = self.image_rgb.astype(float) 
        rows, cols=self.gray_img.shape
        ksize = 5
        padsize = int((ksize - 1) / 2)
        pad_img = cv2.copyMakeBorder(self.gray_img, *[padsize] * 4, cv2.BORDER_DEFAULT)
        self.new_image= np.zeros_like(self.gray_img)
        # harmonic Mean Filter
        for r in range(rows):
            for c in range(cols):
                self.new_image[r, c] = (ksize * ksize) / (np.sum(1 / pad_img[r:r + ksize, c:c + ksize]))
         ####
        self.show_new_image()
    def Contraharmonic_Filter(self):
        self.gray_img = self.image_rgb.astype(float) 
        rows, cols=self.gray_img.shape
        ksize = 5
        padsize = int((ksize - 1) / 2)
        pad_img = cv2.copyMakeBorder(self.gray_img, *[padsize] * 4, cv2.BORDER_DEFAULT)
        self.new_image= np.zeros_like(self.gray_img)
        q = 1.5
        for r in range(rows):
            for c in range(cols):
                self.new_image[r, c] = np.sum(np.power(pad_img[r:r + ksize, c:c + ksize], q + 1)) / np.sum(np.power(pad_img[r:r + ksize, c:c + ksize], q))
         ####
        self.show_new_image() 
    
    def Sobel_X(self):
        self.gray_img = self.image_rgb
        img_blur = cv2.GaussianBlur(self.gray_img, (3,3), 0)
        #Sobel x
        self.new_image = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
        ######
        self.show_new_image() 
    def Sobel_Y(self):
        self.gray_img = self.image_rgb
        img_blur = cv2.GaussianBlur(self.gray_img, (3,3), 0)
        #Sobel Y
        self.new_image = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
        ######
        self.show_new_image()
    def Sobel_XY(self):
        self.gray_img = self.image_rgb
        img_blur = cv2.GaussianBlur(self.gray_img, (3,3), 0)
        #Sobel Y
        self.new_image = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)
        ######
        self.show_new_image() 
    def Canny_Edge_Detection(self):
        self.gray_img = self.image_rgb
        img_blur = cv2.GaussianBlur(self.gray_img, (3,3), 0)
        self.new_image  = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
        ######
        self.show_new_image()

    def K_Means_Algorithm(self):
        self.img = self.image_bgr
        self.img=cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        pixel_values = self.img.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.95)
        k = 3
        _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # convert back to 8 bit values
        centers = np.uint8(centers)
        # flatten the labels array
        labels = labels.flatten()
        # convert all pixels to the color of the centroids
        segmented_image = centers[labels.flatten()]
        # reshape back to the original image dimension
        segmented_image = segmented_image.reshape(self.img.shape)
        self.new_image=segmented_image
         ######
        self.show_new_image()
    def Mean_Shift_Algorithm(self):
        self.img = self.image_bgr
        originShape = self.img.shape 
        flatImg=np.reshape(self.img, [-1, 3]) 
        bandwidth = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100)  
        ms = MeanShift(bandwidth = bandwidth, bin_seeding=True) 
        ms.fit(flatImg) 
        labels=ms.labels_   
        cluster_centers = ms.cluster_centers_    
        with np.printoptions(threshold=np.inf):
           print(cluster_centers)
        # Finding and diplaying the number of clusters    
        labels_unique = np.unique(labels)    
        n_clusters_ = len(labels_unique)  
        segmentedImg = cluster_centers[np.reshape(labels, originShape[:2])]
        segmentedImg = np.uint8(segmentedImg )
        self.new_image=segmentedImg
         ######
        self.show_new_image() 
    def Connected_Labels(self):
        self.gray_img = self.image_rgb
        img = cv2.threshold(self.gray_img, 127, 255, cv2.THRESH_BINARY)[1]
        # Applying cv2.connectedComponents()
        num_labels, labels = cv2.connectedComponents(img)
        label_hue = np.uint8(179 * labels / np.max(labels))
        blank_ch = 255 * np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        labeled_img[label_hue==0] = 0
        self.new_image = labeled_img
        ######
        self.show_new_image() 
    def snake(self): 
        from skimage.color import rgb2gray
        from skimage import data
        from skimage.filters import gaussian
        from skimage.segmentation import active_contour
         
        # Sample Image of scikit-image package
        self.gray_img = self.image_rgb  
        img = self.gray_img 
        s = np.linspace(0, 2*np.pi, 400)
        r = 100 + 100*np.sin(s)
        c = 220 + 100*np.cos(s)
        init = np.array([r, c]).T

        snake = active_contour(gaussian(img, 3, preserve_range=False),
                               init, alpha=0.015, beta=10, gamma=0.001)

        fig, ax = plt.subplots(figsize=(9,5))
        ax.imshow(img, cmap=plt.cm.gray)
        ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
        ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0]) 
        self.canvas = FigureCanvasTkAgg(fig,master = self.canvas1)  
        #canvas1.draw() 
        # placing the canvas on the Tkinter window
        self.canvas.get_tk_widget().pack() 
        # creating the Matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas1)
        self.toolbar.update() 
        self.flag=1
    def level_set(self):
        self.gray_img = self.image_rgb
        image1 = self.gray_img - np.mean(self.gray_img)
        imSmooth = cv2.GaussianBlur(image1, (5, 5), 0) 
        self.new_image=imSmooth
        self.show_new_image()
    def Watershed_Seg(self):
        self.gray_img = self.image_rgb
        ret, thresh = cv2.threshold(self.gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        k = np.ones((5, 5), dtype=np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k, iterations=2)
        distTransform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, fore = cv2.threshold(distTransform, 0.2 * distTransform.max(), 255, 0)
        bg = cv2.dilate(opening, k, iterations=3)
        fore = np.uint8(fore)
        ret, markets = cv2.connectedComponents(fore)
        unknown = cv2.subtract(bg, fore)
        markets = markets + 1
        markets[unknown == 255] = 0
        markets = cv2.watershed(self.image_bgr_resize, markets)
        self.image_bgr_resize[markets == -1] = [255, 0, 0]
        self.new_image=self.image_bgr_resize

        ######
        self.show_new_image() 
    def reigon_based(self):
        class Point(object):
            def __init__(self, x, y):
                self.x = x
                self.y = y
            def getX(self):
                return self.x
            def getY(self):
                return self.y

        def getGrayDiff(img, currentPoint, tmpPoint):
            return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))
        def selectConnects(p):
            if p != 0:
                connects = [Point(-1, -1),Point(0, -1),Point(1, -1),Point(1, 0),Point(1, 1),
                            Point(0, 1),Point(-1, 1),Point(-1, 0),]
            else:
                connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
            return connects

        def regionGrow(img, seeds, thresh, p=1):
            height, weight = img.shape
            seedMark = np.zeros(img.shape)
            seedList = []
            for seed in seeds:
                seedList.append(seed)
            label = 1
            connects = selectConnects(p)
            while len(seedList) > 0:
                currentPoint = seedList.pop(0)
                seedMark[currentPoint.x, currentPoint.y] = label
                for i in range(8):
                    tmpX = currentPoint.x + connects[i].x
                    tmpY = currentPoint.y + connects[i].y
                    if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                        continue
                    grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
                    if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                        seedMark[tmpX, tmpY] = label
                        seedList.append(Point(tmpX, tmpY))
            return seedMark

        def RegionGr():
            img=self.image_rgb
            seeds = [Point(10,10),Point(82,150),Point(20,300)]
            binaryImg = regionGrow(img,seeds,10)
            self.new_image=binaryImg 
        RegionGr()
        self.show_new_image()

    def Gaussian_noise(self):
        self.gray_img = self.image_rgb
        def gaussian_noise(size, mean=0, std=0.01):
         noise = np.multiply(np.random.normal(mean, std, size), 255) 
         return noise
        
        gau_noise = gaussian_noise(self.gray_img.shape, mean=0.0, std=0.01)

       # adding and clipping values below 0 or above 255
        img_gau = self.gray_img+gau_noise
        self.new_image=img_gau

        ######
        self.show_new_image() 
    def Impulse_noise(self):
        
        def impulsive_noise(image, prob=0.1, mode='salt_and_pepper'): 
            noise = np.array(image, copy=True)
            for x in np.arange(image.shape[0]):
                for y in np.arange(image.shape[1]):
                    rnd = np.random.random()
                    if rnd < prob:
                        rnd = np.random.random()
                        if rnd > 0.5:
                            noise[x,y] = 255
                        else:
                            noise[x,y] = 0 
            return noise

        self.gray_img = self.image_rgb
        imp_noise = impulsive_noise(self.gray_img, prob = 0.1) 
        # adding and clipping values below 0 or above 255
        img_imp_noise = self.gray_img +imp_noise
        self.new_image= img_imp_noise

        ######
        self.show_new_image()  
    def Uniform_noise(self):
        def uniform_noise(size, prob=0.4): 
            levels = int((prob * 255) // 2)
            noise = np.random.randint(-levels, levels, size) 
            return noise 
        self.gray_img = self.image_rgb
        uni_noise = uniform_noise(self.gray_img.shape, prob=0.1)
        img_uni = self.gray_img+uni_noise
        self.new_image=img_uni

        ######
        self.show_new_image() 
    def chain_code(self):
        image=self.image_rgb
        rows, cols = image.shape
        result = np.zeros_like(image)
        for x in range(rows):
            for y in range(cols):
                if image[x, y] >= 70:
                    result[x, y] = 0
                else:
                    result[x, y] = 1
         
        ## Discover the first point
        for i, row in enumerate(result):
            for j, value in enumerate(row):
                if value == 1:
                    start_point = (i, j)
                    #            print(start_point, value)
                    break
            else:
                continue
            break

        directions = [0, 1, 2,
                      7, 3,
                      6, 5, 4]
        dir2idx = dict(zip(directions, range(len(directions))))
        # print(dir2idx)
        change_j = [-1, 0, 1,  # x or columns
                    -1, 1,
                    -1, 0, 1]

        change_i = [-1, -1, -1,  # y or rows
                    0, 0,
                    1, 1, 1]

        border = []
        chain = []

        curr_point = start_point
        for direction in directions:
            idx = dir2idx[direction]
            print(idx)
            new_point = (start_point[0] + change_i[idx], start_point[1] + change_j[idx])
            print(new_point)
            if result[new_point] != 0:  # if is ROI
                border.append(new_point)
                chain.append(direction)
                curr_point = new_point
                break
            count = 0

        while curr_point != start_point:
            # figure direction to start search
            b_direction = (direction + 5) % 8
            dirs_1 = range(b_direction, 8)
            dirs_2 = range(0, b_direction)
            dirs = []
            dirs.extend(dirs_1)
            dirs.extend(dirs_2)
            for direction in dirs:
                idx = dir2idx[direction]
                new_point = (curr_point[0] + change_i[idx], curr_point[1] + change_j[idx])
                if result[new_point] != 0:  # if is ROI
                    border.append(new_point)
                    chain.append(direction)
                    curr_point = new_point
                    break
            if count == 1000: break
            count += 1
        ###
        fig, ax = plt.subplots(figsize=(9,5))
        ax.imshow(image, cmap='Greys')
        ax.plot([i[1] for i in border], [i[0] for i in border])
        self.canvas = FigureCanvasTkAgg(fig,master = self.canvas1)  
        # placing the canvas on the Tkinter window
        self.canvas.get_tk_widget().pack() 
        # creating the Matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas1)
        self.toolbar.update() 
        self.flag=1 
        #########################################################################################################################################################################
    def classification(self):
          
 

        dir = 'C:\\Users\star\\Downloads\\The IQ-OTHNCCD lung cancer dataset\\The IQ-OTHNCCD lung cancer dataset'
        classes = ['Bengin','Malignant','Normal']
        Data = []
        Lables = []
        for category in os.listdir(dir):
    
            newPath = os.path.join(dir,category)
            for img in os.listdir(newPath):
                img_path = os.path.join(newPath,img)
                if 'Thumbs.db' not in img_path:
                    print(img_path)
                    Data.append((image.img_to_array(ocv.resize(ocv.imread(img_path,0),(100,100)))))
                    Lables.append(classes.index(category))
        combined = list(zip(Data,Lables))
        shuffle(combined)
        Data[:],Lables[:] = zip(*combined)
        X_train = np.array(Data)
        Y_train = np.array(Lables)
        Y_train = np_utils.to_categorical(Y_train)
         
        #Data Augmentation
        dataGen = ImageDataGenerator(rotation_range=20,width_shift_range=0.01,height_shift_range=0.01,horizontal_flip=False,vertical_flip=False)
        dataGen.fit(X_train)
 
        global hist
        ############
        file_path = 'C:\\Users\\star\\Downloads\\The IQ-OTHNCCD lung cancer dataset\\my_model.h5'
        if os.path.exists(file_path):
             # using saved training model ..
            model=keras.models.load_model(file_path)
    
            # training will happen if the trained model isnot available
        else:
            model = Sequential()
            IMAGE_WIDTH=100
            IMAGE_HEIGHT=100
            IMAGE_CHANNELS=1
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS)))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(3, activation='softmax')) # 3 because we 3 classes
            # compile model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            #file_path = 'C:\\Users\\star\\Downloads\\The IQ-OTHNCCD lung cancer dataset\\my_model.h5'
            modelcheckpoint = ModelCheckpoint(file_path,monitor='val_loss',verbose=2,save_best_only=True,mode='min')
            callBackList = [modelcheckpoint]
            # Fit the model
            model=model.fit(X_train,Y_train,batch_size=32,epochs=15,validation_split=0.25,callbacks=callBackList)
     
     # Make prediction 
        self.new_image=self.image_rgb
        img = image.img_to_array( self.new_image)
        img = ocv.resize(img,(100,100))
        img = img.reshape(1,100,100,1)
        print(classes[(np.argmax(model.predict(img)))]) 
        predict = classes[(np.argmax(model.predict(img)))]
        acc = model.evaluate(X_train,Y_train)

        

        #messagebox.showinfo("Model Prediction", "Prediction is: [ "+predict+" ]\n\n"+"Accurecy of model : "+ str(acc[1]*100)+"%"+"\n\nLoss of model: "+str(acc[0]))
        

          
        #######
        self.label_class= Label(self.frame3, font=('Roboto',15) ,background='Deep Sky Blue3',fg='gray20' ,relief='solid',height=3 ,width=20)
        self.label_class.grid( column=1, row= 0)

        self.label_acc= Label(self.frame3, font=('Roboto',15) ,background='Deep Sky Blue3',fg='gray20',relief='solid',height=3 ,width=20)
        self.label_acc.grid( column=6, row= 0)

        self.label_loss= Label(self.frame3, font=('Roboto',15) ,background='Deep Sky Blue3',fg='gray20',relief='solid',height=3 ,width=20)
        self.label_loss.grid( column=12, row= 0)
        
         
         
        
        self.label_class.config(text="Prediction= "+ predict)
        self.label_acc.config(text="Accurecy= "+str(acc[1]*100)[:5]+"%")
        self.label_loss.config(text="Loss = "+str(acc[0]*100)[:5]+"%")
         
        #self.myload =circularloadbar.CircularLoadBar(root, 360, 180, 200, 150)
        
      
      #########################################################################################################################################################################
     
    ### GUI #####
    def create_widgets(self):
        self.configure(bg='gray12')   

        #Canvas
        self.canvas1 = tk.Canvas(self)
        self.canvas1.configure(width=900,height=450, bg='gray15',highlightbackground="Deep Sky Blue3",highlightthickness=1) 
        self.canvas1.grid(column=0, row=1)
        #self.canvas1.grid(padx=20, pady=20)
        
       #Frame1
        self.frame1_button = tk.Canvas(self)
        self.frame1_button.configure(width=640,height=50,background='gray20')#,highlightbackground="MediumPurple")
        self.frame1_button.grid(column=0,row=0)
        self.frame1_button.grid(padx=20, pady=30)
    
      #Frame2
        self.frame2_button = tk.Canvas(self)
        self.frame2_button.configure(width=640,height=50,background='gray20')
        self.frame2_button.grid(column=0,row=2)
        self.frame2_button.grid(padx=20, pady=30) 

       
       

        #File open and Load Image
        self.new_size=(40,40)
        self.img3 = Image.open('new.png')
        self.imag_resize3 = self.img3.resize(self.new_size)
        self.img_3 = ImageTk.PhotoImage(self.imag_resize3)
        self.button_open = tk.Button(self.frame2_button,image=self.img_3,compound = LEFT,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray20',fg='Deep Sky Blue3',font=('Roboto',15),relief='solid',height=50 ,width=200)
        self.button_open.configure(text = '    New Image  ' )
        self.button_open.grid(column=0, row=1)
        self.button_open.configure(command=self.loadImage)
         

        # Clear Button
        self.new_size=(30,30)
        self.img1 = Image.open('clear.png')
        self.imag_resize1 = self.img1.resize(self.new_size)
        self.img_1 = ImageTk.PhotoImage(self.imag_resize1)
        self.button_clear = tk.Button( self.frame2_button,image=self.img_1,compound =LEFT,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray20',fg='Deep Sky Blue3',font=('Roboto',15),relief='solid',height=50,width=200)
        self.button_clear.configure( text='   Clear Actions  ' )
        self.button_clear.grid( column=1, row=1 )
        self.button_clear.configure(command=self.clearImage) 

        # Save Button  
        self.new_size=(30,30)
        self.img2 = Image.open('floppy-disk (1).png')
        self.imag_resize2 = self.img2.resize(self.new_size)
        self.img_2 = ImageTk.PhotoImage(self.imag_resize2)
        self.button_save =tk.Button(self.frame2_button,image=self.img_2,compound = LEFT,text='Save as',activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray20',fg='Deep Sky Blue3',font=('Roboto',15),relief='solid',height=50 ,width=200)
        self.button_save.config( text='    Save as  ')
        self.button_save.grid( column=2, row=1 )
        self.button_save.configure(command = self.save_image)
          

        #classification Button 
         
        self.new_size=(30,30)
        self.img4 = Image.open('classification.png')
        self.imag_resize4 = self.img4.resize(self.new_size)
        self.img_4 = ImageTk.PhotoImage(self.imag_resize4)
        self.button_cls = tk.Button(self.frame2_button,image=self.img_4,compound = LEFT,text='Classification',activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray20',fg='Deep Sky Blue3',font=('Roboto',15),relief='solid',height=50 ,width=200)
        self.button_cls.config( text='  Classifiy   ')
        self.button_cls.grid( column=3, row=1 )
        self.button_cls.configure(command = self.classification)

        ###############################
       
         
        #self.label_acc= Label(self.frame3, font=('Roboto',15) ,background='Deep Sky Blue3',fg='gray20',relief='solid',height=3 ,width=20)
        #self.label_acc.grid( column=1, row= 0)

        #self.label_loss= Label(self.frame3, font=('Roboto',15) ,background='Deep Sky Blue3',fg='gray20',relief='solid',height=3 ,width=20)
        #self.label_loss.grid( column=4, row= 0)
        ##self.bar = ttk.Progressbar(self.frame3,  style="TProgressbar", length=500, mode="determinate")
        #self.label_class= Label(self.frame3, font=('Roboto',15) ,background='Deep Sky Blue3',fg='gray20' ,relief='solid',height=3 ,width=20)
        #self.label_class.grid( column=6, row= 0)

         

        #self.label_class.config(text="Prediction")
        #self.label_acc.config(text="Accurecy")
        #self.label_loss.config(text="Loss ")
        #self.myload =circularloadbar.CircularLoadBar(root, 360, 180, 200, 150)
        
        #self.bar.grid( column=1, row=2)
         
        ###########################

        ######## Spatial filters 
        self.menubtn1 =tk.Menubutton(  self.frame1_button, text='Spatial Filters ',activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray20',fg='Deep Sky Blue3',font=('Roboto',15),relief='solid',height=2,width=18)
        self.menubtn1.grid( column=2, row=0 )
        self.menubtn1.menu = Menu(self.menubtn1, tearoff=0)
        self.menubtn1["menu"] = self.menubtn1.menu 
        
        #self.menubtn1.configure(command = self.quit_app)
        # add a submenu of linear spatial filter :
        sub_menu = Menu(self.menubtn1, tearoff=0)
        sub_menu.add_command(label='Average Filter',command=self.Average_filter,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        sub_menu.add_command(label='Gaussian Filter',command=self.Gaussian_filter,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        sub_menu.add_command(label='Laplace Filter',command=self.lablacian_filter,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        # add a submenu of Intensity Transformation:
        sub_menu2 = Menu(self.menubtn1, tearoff=0)
        sub_menu2.add_command(label='Addition',command=self.addition,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        sub_menu2.add_command(label='Subtraction',command=self.Subtraction,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
         
        # add a submenu of Non Linear Spatial filter:
        sub_menu3 = Menu(self.menubtn1, tearoff=0)
        sub_menu3.add_command(label='Max Filter',command=self.Max_filter,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        sub_menu3.add_command(label='Min Filter',command=self.Min_filter,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        sub_menu3.add_command(label='Median Filter',command=self.Median_filter,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        # add the File menu to the menubar
        self.menubtn1.menu.add_cascade(
            label='Linear Spatial Filter',
            menu=sub_menu,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
         
        ##########
        self.menubtn1.menu.add_cascade(
            label='Intensity Transformation',
            menu=sub_menu2,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
         
        ######
        self.menubtn1.menu.add_cascade(
            label='Non Linear Spatial filter',
            menu=sub_menu3,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
       

        ######## Filters in Frequency Domain
        self.menubtn2 =tk.Menubutton( self.frame1_button, text='Frequency Domain Filters',activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray20',fg='Deep Sky Blue3',font=('Roboto',15),relief='solid',height=2,width=22)
        self.menubtn2.grid( column=3, row=0 )
        self.menubtn2.menu = Menu(self.menubtn2, tearoff=0)
        self.menubtn2["menu"] = self.menubtn2.menu 
        # add a submenu of Low Pass Filters  :
        sub_menu4 = Menu(self.menubtn2, tearoff=0)
        sub_menu4.add_command(label='Ideal Lowpass Filter',command=self.ideal_low_pass,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        sub_menu4.add_command(label='Butterworth Lowpass Filter',command=self.butterworth_low_pass,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        sub_menu4.add_command(label='Gaussian Lowpass Filter',command=self.gussion_low_pass,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        # add a submenu of High Pass Filters :
        sub_menu5 = Menu(self.menubtn2, tearoff=0)
        sub_menu5.add_command(label='Ideal Highpass Filter',command=self.ideal_high_pass,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        sub_menu5.add_command(label='Butterworth Highpass Filter',command=self.butterworth_high_pass,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        sub_menu5.add_command(label='Gaussian Highpass Filter',command=self.gussion_high_pass,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        #####
        self.menubtn2.menu.add_cascade(
            label='Smoothing Filters',
            menu=sub_menu4,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        
        ######
        self.menubtn2.menu.add_cascade(
            label='Sharpening Filters',
            menu=sub_menu5,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        
        #########Image Restoration and Reconstraction 
        self.menubtn3 =tk.Menubutton( self.frame1_button, text='Image Restoration ',activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray20',fg='Deep Sky Blue3',font=('Roboto',15),relief='solid',height=2,width=18)
        self.menubtn3.grid( column=4, row=0 )
        self.menubtn3.menu = Menu(self.menubtn3, tearoff=0)
        self.menubtn3["menu"] = self.menubtn3.menu 
        # add a submenu of Mean Filters  :
        sub_menu6 = Menu(self.menubtn3, tearoff=0)
        sub_menu6.add_command(label='Arithmatic Mean Filter',command=self.Arithmatic_Mean_Filter,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        sub_menu6.add_command(label='Geometric Mean Filter',command=self.Geometric_filter,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        sub_menu6.add_command(label='Harmonic Mean Filter',command=self.harmonic_Mean_Filter,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        sub_menu6.add_command(label='Contraharmonic Mean Filter',command=self.Contraharmonic_Filter,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        # add a submenu of Order-Statistic Filters :
        sub_menu7 = Menu(self.menubtn3, tearoff=0)
        sub_menu7.add_command(label='Median Filter',command=self.Median_filter,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        sub_menu7.add_command(label='Max Filter',command=self.Max_filter,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        sub_menu7.add_command(label='Min Filter',command=self.Min_filter,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        sub_menu7.add_command(label='Midpoint Filter',command=self.midpoint,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        sub_menu7.add_command(label='Alpha-trimmed Filter',command=self.AlphaTrimmedMeanFilter,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
         
        self.menubtn3.menu.add_cascade(
            label='Mean Filters',
            menu=sub_menu6,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
         ######
        self.menubtn3.menu.add_cascade(
            label='Order-Statistic Filters',
            menu=sub_menu7,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
         ###### 
       #####Image Segmantation
        self.menubtn4 =tk.Menubutton( self.frame1_button, text='Image Segmantation',activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray20',fg='Deep Sky Blue3',font=('Roboto',15),relief='solid',height=2,width=20)
        self.menubtn4.grid( column=5, row=0 )
        self.menubtn4.menu = Menu(self.menubtn4, tearoff=0)
        self.menubtn4["menu"] = self.menubtn4.menu 
        #self.menubtn4.grid(padx=20, pady=20)
        # add a submenu of thresholding Filters  :
        sub_menu9 = Menu(self.menubtn4, tearoff=0)
        sub_menu9.add_command(label='Global Thresholding',command=self.global_thresholding_binary,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        sub_menu9.add_command(label='Optimal Threshold\n(Otsu Method)',command=self.otsu_thresholding,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        sub_menu9.add_command(label='Adaptive \ Local threshold',command=self.adaptive_thresholding,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        ###
        self.menubtn4.menu.add_cascade(
            label='Thresholding based segmentation',
            menu=sub_menu9,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
   
        # add a submenu of Region -based Filters  :
        sub_menu10 = Menu(self.menubtn4, tearoff=0)
        sub_menu10.add_command(label='Region Growing',command=self.reigon_based,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
          ###
        self.menubtn4.menu.add_cascade(
            label='Region based segmentation',
            menu=sub_menu10 ,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
      
        # add a submenu of Edge-based Filters  :
        self.sub_menu11 = Menu(self.menubtn4, tearoff=0) 
        self.sub_menu11.add_command(label='Canny edge detection',command=self.Canny_Edge_Detection,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        ###
        sub_menu_s=Menu(self.menubtn4, tearoff=0)
        sub_menu_s.add_command(label='Sobel X',command=self.Sobel_X,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        sub_menu_s.add_command(label='Sobel Y',command=self.Sobel_Y,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        sub_menu_s.add_command(label='Sobel X Y',command=self.Sobel_XY,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        ###
        self.sub_menu11.add_cascade (
            label='Sobel Operator',
            menu=sub_menu_s,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14)) 
        #####
        self.menubtn4.menu.add_cascade(
            label='Edge based segmentation',
            menu=self.sub_menu11,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
       # add a submenu of cluster-based Filters  :
        sub_menu12 = Menu(self.menubtn4, tearoff=0)
        sub_menu12.add_command(label='K-Means Clustering Algorithm',command=self.K_Means_Algorithm,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        sub_menu12.add_command(label='Mean Shift Algorithm',command=self.Mean_Shift_Algorithm,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
      ###
        self.menubtn4.menu.add_cascade(
            label='Cluster based segmentation',
            menu=sub_menu12,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
    
       # add a submenu of other Filters  :
        sub_menu13 = Menu(self.menubtn4, tearoff=0)
        sub_menu13.add_command(label='Active Contours ( Snake )',command=self.snake,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        sub_menu13.add_command(label='Watershed segmentation',command=self.Watershed_Seg,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        sub_menu13.add_command(label='Level Set ',command=self.level_set,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        ###
        self.menubtn4.menu.add_cascade(
            label='Other segmentation Methods',
            menu=sub_menu13,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
         
         #########feature recognation
        self.menubtn5 =tk.Menubutton( self.frame1_button, text='Feature recognation & Classification',activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray20',fg='Deep Sky Blue3',font=('Roboto',15),relief='solid',height=2,width=30)
        self.menubtn5.grid( column=6, row=0 )
        self.menubtn5.menu = Menu(self.menubtn5, tearoff=0)
        self.menubtn5["menu"] = self.menubtn5.menu 
        self.menubtn5.menu.add_command(label='Connected components labeling algorithm',command=self.Connected_Labels,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        self.menubtn5.menu.add_command(label='Cain Code  ',command=self.chain_code,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))

         ######### Add Noise
        self.menubtn6 =tk.Menubutton( self.frame1_button, text='Add Noise',activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray20',fg='Deep Sky Blue3',font=('Roboto',15),relief='solid',height=2,width=17)
        self.menubtn6.grid( column=7, row=0 )
        self.menubtn6.menu = Menu(self.menubtn6, tearoff=0)
        self.menubtn6["menu"] = self.menubtn6.menu 
        self.menubtn6.menu.add_command(label='Gaussian noise',command=self.Gaussian_noise ,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        self.menubtn6.menu.add_command(label='Impulse(Salt&pepper)noise',command=self.Impulse_noise,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
        self.menubtn6.menu.add_command(label='Uniform noise',command=self.Uniform_noise,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray12',foreground='Deep Sky Blue3',font=('Robot',14))
         
     ##### Event Call Back###

    def loadImage(self): 
        self.filename= filedialog.askopenfilename() 
        self.image_bgr = cv2.imread(self.filename)
         
        self.height, self.width = self.image_bgr.shape[:2]
        print(self.height, self.width)
        if self.width > self.height:
            self.new_size = (700,480)
        else:
            self.new_size = (700,480) 
             

        self.image_bgr_resize = cv2.resize(self.image_bgr, self.new_size, interpolation=cv2.INTER_AREA)
        self.image_bgr_resize = cv2.normalize(self.image_bgr_resize, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) 
        self.image_rgb = cv2.cvtColor(self.image_bgr_resize, cv2.COLOR_BGR2GRAY) #Since imread is BGR, it is converted to RGB
        self.image_PIL = Image.fromarray(self.image_rgb) #Convert from RGB to PIL format
        self.image_tk = ImageTk.PhotoImage(self.image_PIL) #Convert to ImageTk format
        self.canvas1.create_image(450,240, image=self.image_tk)

    def clearImage(self):
        if self.flag==1: 
            self.canvas.get_tk_widget().pack_forget()
            self.toolbar.pack_forget()
            self.new_image = self.image_rgb
            self.show_new_image()
        else:
            self.new_image = self.image_rgb
            self.show_new_image() 

    def save_image(self):
         
        file_type = self.filename.split('.')[-1]
        filename = filedialog.asksaveasfilename()
        filename = filename + "." + file_type 
        save_image =  self.image_bgr_resize
        cv2.imwrite(filename,save_image)
        #tk.messagebox.showinfo("OK", "Image has been saved successsfully") 

    def quit_app(self):
        self.Msgbox = tk.messagebox.askquestion("Exit Applictaion", "Are you sure?", icon="warning")
        if self.Msgbox == "yes":
            self.master.destroy()
        
            
 
        

def main():
    root = tk.Tk()
    app = Application(master=root)#Inherit 
    app.mainloop()
if __name__ == "__main__":
    main()