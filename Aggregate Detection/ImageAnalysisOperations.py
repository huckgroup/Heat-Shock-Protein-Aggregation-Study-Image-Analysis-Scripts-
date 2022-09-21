# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 14:38:35 2021

@author: Bob
"""

from PIL import Image, ImageEnhance
import numpy
from scipy.spatial import distance
import os
from scipy import fftpack
from matplotlib.colors import LogNorm
from scipy import ndimage
from skimage.feature import peak_local_max
import itertools as it
from skimage.morphology import extrema
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import matplotlib.patches as mpatches
from shapely.geometry import Polygon
from scipy.signal import argrelextrema
import matplotlib.pylab as plt
import copy
import scipy
import numpy as np
from scipy.ndimage import gaussian_filter
import scipy as sp
import scipy.ndimage
from skimage.segmentation import watershed
import math
# import open3d as o3d

def OpenImage(path,stack = 250):
    """open the images"""
    image  = Image.open(path)
    """store the arrays"""
    arrays = []
    """get the z stacks out of the tiff file"""
    for n in range(0,stack,1):
        try:
            image.seek(n)
        except:
            break           
        """arrays with different z stacks"""
        arrays.append(copy.copy(numpy.array(image)))
    return arrays

def OpenSlicedImage(path,stack = 250):
    """open the images"""
    print(path)
    image  = Image.open(path)
    """store the arrays"""
    arrays = []
    """get the z stacks out of the tiff file"""
    for n in range(0,stack,1):
        try:
            image.seek(n)
        except:
            break           
        """arrays with different z stacks"""
        arrays.append(copy.copy(numpy.array(image)))
    """the arrays in the dataset"""
    arrays = numpy.array([arrays[int(len(arrays)/1.5)-1],arrays[int(len(arrays)/1.5)],arrays[int(len(arrays)/1.5) + 1]])
    return arrays

def neighbors(index):
    N = len(index)
    import itertools as it
    for relative_index in it.product((-1, 0, 1), repeat=N):
        if not all(i == 0 for i in relative_index):
            yield tuple(i + i_rel for i, i_rel in zip(index, relative_index))

def findpaths(folder):
    paths = [folder + '\\' + i for i in  os.listdir(folder)]
    return paths
  
def boxcentroid(box): 
    x = numpy.average(box[:,0])
    y = numpy.average(box[:,1])    
    return (x,y)

def distpoints(x_1 , x_2, y_1 , y_2):
    return math.sqrt((x_1 - x_2)**2 + (y_1 + y_2)**2)
    
#read the image
def get_filenames(folder):
    filenames = os.listdir(folder)
    paths = [] 
    for i in filenames:
        paths.append((i,folder+i))
    return paths,filenames
   
def plot_spectrum(im_fft):
    from matplotlib.colors import LogNorm
    plt.imshow(numpy.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()
 
def RemovedNeighbor(n):
    x,y = n
    """initial neighbors"""
    cns = neighbors((x,y))
    """neirest neighbors"""
    ngb = [i for i in cns]
    ngb.append((x,y))
    second = []
    for x,y in ngb:
        n = neighbors((x,y))
        for l in n:
            second.append(l)
    ngb += second
    return list(set(ngb))
    
def create_shifted_matrix(matrix,coordinates,coordinate):
    benchmark = numpy.zeros((len(matrix),len( matrix)))
    """the ymin and xmin coordinate"""
    xmin,ymin = coordinate
    """the coordinates for the shift"""
    shiftC = []
    for x,y in coordinates:
        xn = x + xmin
        yn = y + ymin
        shiftC.append((copy.copy(xn),copy.copy(yn)))
    shiftC = [(x,y) for x,y in shiftC if x < len(matrix) and x > -1 and y > -1 and y < len(matrix)] 
    for x,y in shiftC:
        benchmark[x,y] = 1
        for i in neighbors((x,y)):
            xi,yi=  i
            if xi < len(matrix) and xi > -1 and yi > -1 and yi < len(matrix):
                benchmark[xi,yi] = 1
    return benchmark

def running_mean(x, N):
    cumsum = numpy.cumsum(numpy.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

            
def fill_contours(arr):
    return numpy.maximum.accumulate(arr,1) & \
           numpy.maximum.accumulate(arr[:,::-1],1)[:,::-1]  
            
def ParsePath(path,channel):
    """ get the image """
    if '\\' in path:
        split = path.split('\\')
    else:
        split = path.split('$\$')
        
    information = {identifiers[i]:[] for i in range(len(identifiers))}
    """ get the information from the class """
    for ids in range(len(data)):
        for item in range(len(data[ids])):
            information[identifiers[ids]].append(data[ids][item])
    path = path.replace(split[-1],'')

def findpaths(folder):
    import os
    paths = [folder + '\\' + i for i in  os.listdir(folder)]
    return paths 

def NucleusCheck(c,label,reference):
    cytoplasm = 0        
    nucleus   = 0
    for i in label:
        x,y,z = i
        c = (x,y,z)
        if reference[c] == True:
            nucleus += 1
        else:
            cytoplasm += 1
            
    if cytoplasm >= nucleus:
        location = 'Cytoplasm'
    else:
        location = "Nucleus"  
    """The cytoplasm and nucleus (sum them and check, if 25% in one or other then accept)"""
    both = False
    if float(cytoplasm)/float((cytoplasm+nucleus)) > 0.25 and float(cytoplasm)/float((cytoplasm+nucleus)) < 0.75:
        both = True
    return location,cytoplasm,nucleus,both

def ImportRGBImage(arrayset):
    import numpy
    import cv2
    import skimage.io as skio
    import matplotlib.pylab as plt
    from skimage.transform import resize
    
    """tiny function that splits RGB image in relevant parts"""
    z = {i:[] for i in range(3)}
    for i in skio.imread(arrayset):
        for c in range(3):
            z[c].append(resize(i[:,:,c], (512, 512)))
    """return split image stacks"""  
    r,g,b = z[0],z[1],z[2]
    return r,g,b

def circle(arr,x,y,radius = 40):
    """the perimeter"""
    rr, cc = draw.circle_perimeter(int(x), int(y), radius=int(radius), shape=arr.shape)
    """bool activation of the cell"""
    arr[rr, cc] = 1
    """we have the perimiter now we still need to fill the perimiter"""
    arr = numpy.array(arr,dtype = 'int')
    """filled array"""
    arr = fill_contours(arr)    
    return arr