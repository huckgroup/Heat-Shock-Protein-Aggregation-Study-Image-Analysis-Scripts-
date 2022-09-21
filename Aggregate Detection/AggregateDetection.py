
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 14:22:23 2021

@author: Bob van Sluijs
Quick analysis of aggregate sizes
"""

import numpy
import copy
import scipy 
import os
import cProfile
import pandas as pd
import numpy as np
import cv2 as cv2
import networkx as nx
import itertools as it
from PIL import Image, ImageEnhance
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial import distance
from scipy.spatial.ckdtree import cKDTree
from scipy import fftpack
from scipy import ndimage
from scipy.signal import argrelextrema

from skimage.feature import peak_local_max
from skimage.morphology import extrema
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage import io
from skimage import draw

from ImageAnalysisOperations import *
import warnings


class Spot:
    def __init__(self,coordinates,spots):
        self.coordinates = coordinates
        self.GPS         = GPS

class Aggregate:
    def __init__(self,z,centroid,area,call,label,image):
        self.z          = z
        self.centroid   = centroid
        self.size       = area
        self.coordinates = (self.z,self.centroid[0],self.centroid[1])
        self.call       = call
        self.label      = label
        self.centroid   = self.coordinates
        
        
        self.signal = []
        for r,c,z in label:
            self.signal.append(image[r,c])
        self.signalmean = numpy.mean(self.signal)
        """get equidistance"""
        x,y = zip(*self.call)
        min_x,max_x = min(x),max(x)
        min_y,max_y = min(y),max(y)
        columns = max_x - min_x
        rows    = max_y - min_y


        """calculate"""
        r = math.sqrt(self.size/math.pi)
        
        """create matrix"""
        zx,xc,yc = self.centroid
        emptyarr = numpy.zeros((max_x+50,max_y+50))
        arr = numpy.zeros((max_x+50,max_y+50))
        for x,y in self.call:
            arr[x,y] = 1
        
        zc,xc,yc = self.centroid
        c = circle(emptyarr,xc,yc,radius = r)
        inside = 0
        for x,y in call:
            if c[x,y] == 1:
                inside += 1
                
        """the circularity"""
        self.circularity = inside/float(self.size)
        
class Aggregate3D:
    def __init__(self,centroid,call,image):
        self.centroid = centroid
        self.label    = label  
        
        self.signal = []
        for x,y,z in label:
            self.signal.append(image[x,y,z])
        self.signalmean = numpy.mean(self.signal)
        self.size = len(self.label)
                
        """the circularity"""
        self.circularity = inside/float(self.area)

class Slice:
    def __init__(self,image,r,z,show = False,path = '',nbs = {},statistical_test = False,tol = 2,factor = 2.5,noise = 7):
        print(factor)
        from scipy.ndimage import gaussian_filter
        """the name and the file"""
        self.z      = z
        self.round  = r
        self.matrix = numpy.array(image)
        
        """get the row and column with pixel values sorted, the mean to follow"""
        self.pixels = {}
        for row in range(len(self.matrix)):
            for column in range(len(self.matrix[:,0])):
                self.pixels[(row,column)] = self.matrix[row,column]
        
        """get the mean and median of the pixelset"""
        self.median = numpy.median(numpy.array(list(self.pixels.values())))
        self.mean   = numpy.mean(numpy.array(list(self.pixels.values())))
               
        """map out the neighbors of the pixels"""
        if nbs == {}:
            self.neighbors = {}
            for i in self.pixels.keys():
                self.neighbors[i] = [n for n in neighbors(i)]
        else:
            self.neighbors = nbs

        """remove the median/mean from the matrix"""
        thrsh = []
        arr   = numpy.array(image/numpy.amax(image)*255,dtype = numpy.uint8)
        otsu  = cv2.threshold(arr, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        x,y = numpy.nonzero(otsu)
        self.beta = numpy.zeros(self.matrix.shape)
        for i in range(len(x)):
           self.beta[x[i],y[i]] = self.matrix[x[i],y[i]] 
        for i in range(len(self.beta)):
            for j in range(len(self.beta)):
                 if self.beta[i,j] < 0:
                    self.beta[i,j] = 0  
                    
        arr   = numpy.array(self.beta/numpy.amax(self.beta)*255,dtype = numpy.uint8)
        otsu  = cv2.threshold(arr, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]   
        x,y = numpy.nonzero(otsu)
        self.filtered_matrix = numpy.zeros(self.matrix.shape)
        for i in range(len(x)):
           self.filtered_matrix[x[i],y[i]] = self.matrix[x[i],y[i]]
        for i in range(len(self.filtered_matrix)):
            for j in range(len(self.filtered_matrix)):
                 if self.filtered_matrix[i,j] < 0:
                    self.filtered_matrix[i,j] = 0 
                                
        signal = []
        for row in self.filtered_matrix:
            for column in row:
                if column > 0:
                    signal.append(column)

        flattening = [i for i in self.filtered_matrix.flatten() if i != 0]
        if len(flattening) == 0:
            flattening.append(0)
        minimum    = min(flattening)
        self.M     = copy.copy(self.filtered_matrix)
        self.M[self.filtered_matrix < minimum*factor] = 0
        
        

        """sort the vector for interpolation"""
        sorted_vector = flattening
        array = self.M
        
        """average the bins to calculate the percentage"""
        collect = {int(i):[] for i in sorted_vector}
        for i in sorted_vector:
            collect[int(i)].append(True)
        count = [(k,len(v)) for k,v in collect.items()]
        x,y   = zip(*count)
        t     = numpy.linspace(0,2000,2000)
        IDI   = running_mean(numpy.interp(t,x,y),25)
        half  = float(numpy.max(IDI))/tol
        signal_threshold = numpy.argmin(abs(IDI-half))*2

        """signal thresholds"""
        thresholded = []
        for i in range(len(array)):
            for j in range(len(array)):
                if array[i,j] < signal_threshold:
                    array[i,j] = 0
                if array[i,j] > signal_threshold:
                    thresholded.append(array[i,j]) 
    
        # array = self.filtered_matrix
        """after thresholding the image we remove the noise by removing
        pixels with without neighboring pixel signals"""
        zero = []
        """neirest neighbor search"""
        for coordinate,n in self.neighbors.items():
            
            """get x,y"""
            x,y = coordinate
            
            """set zero count"""
            if array[x,y] > 0:
                zc = 0
                for j in n:
                    try:
                        xn,yn = j
                        if array[xn,yn] == 0:
                            zc += 1
                    except:
                        pass
                    
                """if the pixel is neighbored by 4 zeros i.e. half its set as noise"""
                if zc > noise:
                    zero.append(coordinate)  

                    
        """set this sucker to zero"""
        for x,y in zero:
            array[x,y] = 0
        self.filtered = array
        self.modified = numpy.zeros((512,512))
        x,y = numpy.nonzero(self.filtered)
        for i in range(len(x)):
           self.modified[x[i],y[i]] = 2   
        
        """get the local maxima in the filtered array, these are spots"""
        array = numpy.uint8(255*(array/numpy.max(array)))
        img = cv2.threshold(array, 0, 255, cv2.THRESH_BINARY)[1]  # ensure binary
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img)
        
        labels = {}
        row,column = numpy.nonzero(output)
        for i in range(len(row)):
            try:
               labels[output[row[i],column[i]]].append((row[i],column[i],z))
            except:
                labels[output[row[i],column[i]]] =  []
                labels[output[row[i],column[i]]].append((row[i],column[i],z))
        
        library = {}
        for i in range(len(output)):
            for j in range(len(output[i])):
                try:
                    library[output[i,j]].append((i,j))
                except:
                    library[output[i,j]] = []
                    library[output[i,j]].append((i,j))

        """ the aggregates """
        sizes = stats[1:, -1]
        self.aggregates = []
        for i in range(len(centroids)-1):
            centroid    = centroids[i+1]
            size        = sizes[i]
            call        = library[i+1]
            if size > 4 and size < 125*125:
                c = labels[i+1]
                self.aggregates.append(Aggregate(self.z,copy.copy((int(centroid[1]),int(centroid[0]))),copy.copy(size),copy.copy(call),c,self.matrix))

def ConvertLabellist(coordinates):
    labels = []
    for i in coordinates:
        s = []
        for xyz in i:
            z,x,y = tuple(xyz)
            s.append((x,y,z))
        labels.append(copy.copy(s))
    return labels
    
    
class Image:
    def __init__(self,arrayset,folder = "C:\\Users\\Bob\\Desktop\\Test_01\\", plot = False,tol = 2,mRNA = False,factor = 2.5,condition = '',noise = 7):
        if type(arrayset) == str:
            arrayset = OpenImage(arrayset)
        #go through a single image and get all the imformation out of it
        """the compounded image sets"""
        i = 0
        for n in arrayset:
            if i == 0:
                arr = copy.copy(n)
            else:
                arr += copy.copy(n)
            i += 1
                
        self.aggregateset = []
        """the stack and the single images within the stack"""
        self.stack,nbs = {},{}
        for i in range(len(arrayset)):
            array = Slice(arrayset[i],None,i,nbs = nbs,tol = tol,factor = factor,noise = noise)
            for n in range(len(array.aggregates)):
                self.aggregateset.append(array.aggregates[n])
            self.stack[i] = copy.copy(array)
            """this is a timeintensive step so best to only do it once
            for a single matrix then copy it into others"""
            nbs  = array.neighbors
            
        matrix_3D = numpy.zeros(numpy.array(arrayset).shape)
        for agg in self.aggregateset:
            z = agg.z
            for row,column,z in agg.label:
                matrix_3D[z,row,column] = 255      
                
        import cc3d
        labels     = []
        labels_out = cc3d.connected_components(matrix_3D.astype(numpy.uint8), connectivity=6)
        IDs = labels_out.flatten()
        for i in list(set(list(labels_out.flatten()))):
            if i != 0:
                labels.append(numpy.argwhere(labels_out==i))    
                
        """get the 3D data out"""
        labels    = ConvertLabellist(labels)
        stats     = cc3d.statistics(labels_out)
        narr      = numpy.array(arrayset)
        centroids = [stats['centroids'][i] for i in range(0,len(labels))]
        size      = [len(i) for i in labels]

        signalstrength = []
        for i in labels:
            signal = []
            for x,y,z in i:
                # print(x,y,z)
                signal.append(narr[z,x,y])
            signalstrength.append(numpy.mean(signal))

        """create the datasets"""
        self.nbs = nbs
        self.arrayset = arrayset
        self.data3D = {"Size 3D":[],"Signal 3D":[],'Centroid':[],"Label":[]}
        self.data2D = {"Size 2D":[],"Signal 2D":[],"Circularity":[],'Centroid':[],"Label":[],'Size':[]}
        
        for i in range(len(labels)):
            self.data3D['Size 3D'].append(size[i])
            self.data3D['Signal 3D'].append(signalstrength[i])        
            self.data3D['Centroid'].append(tuple(centroids[i]))   
            self.data3D['Label'].append(labels[i])            
         
        for i in self.aggregateset:
            self.data2D["Size 2D"].append(i.size)
            self.data2D["Signal 2D"].append(i.signalmean)
            self.data2D["Circularity"].append(i.circularity)
            self.data2D['Centroid'].append(i.centroid)
            self.data2D["Label"].append(i.label)
