 # -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:16:55 2021

@author: bob van sluijs
Script to Analyse Aggregation, this is the runfile to analyse protein aggregations
sizes, intended to analyse the microscopic images from the paper...
If anyone attempts to use these scripts best contact me at bob.vansluijs@ru.nl or bob.vansluijs@gmail.com
"""

import matplotlib.pylab as plt
import numpy
import copy
import math
from skimage import draw
   
class Aggregate:
    def __init__(self,coordinates,area,path,ratio,label):
        self.c     = tuple([int(i) for i in coordinates])
        self.area  = area
        self.path  = path
        self.ratio = ratio
        self.label = label
        
        """get equidistance"""
        x,y,z = zip(*self.label)
        min_x,max_x = min(x),max(x)
        min_y,max_y = min(y),max(y)
        columns = max_x - min_x
        rows    = max_y - min_y

        """calculate"""
        r = math.sqrt(self.area/math.pi)
        
        """create matrix"""
        xc,yc,z = self.c
        emptyarr = numpy.zeros((max_x+50,max_y+50))
        arr = numpy.zeros((max_x+50,max_y+50))
        for x,y,z in self.label:
            arr[x,y] = 1
        
        xc,yc,z = self.c
        c = circle(emptyarr,xc,yc,radius = r)
        inside = 0
        for x,y,z in label:
            if c[x,y] == 1:
                inside += 1
                
        """the circularity"""
        self.circularity = inside/float(self.area)

    def NucleusCheck(self,reference):
        self.nuclear = False
        try:
            self.nuclear = reference[self.c]
        except:
            self.nuclear = False
       
        self.cytoplasm = 0        
        self.nucleus   = 0
        for i in self.label:
            try:
                test = reference[i]
                self.nucleus += 1
            except:
                self.cytoplasm += 1
                
        if self.cytoplasm >= self.nuclear:
            self.location = 'Cytoplasm'
        else:
            self.location = "Nucleus"
            
        
class ReconstructedDAPI:
    def __init__(self,path = [],array = []):
        if array == []:
            images = OpenImage(path)    
        else:
            images = array
            
        self.images = images
        """the limit of the image"""
        limit  = skimage.filters.threshold_otsu(numpy.array(images))
        """the distribution"""
        self.distribution = []
        """the coordinates"""
        self.signal  = []
        """new image set"""
        self.nuclei = numpy.zeros((len(images),len(images[0]),len(images[0][:,0])))
        for i in range(len(images)):
            for row in range(len(images[i])):
                for column in range(len(images[i][:,0])):
                    self.distribution.append(copy.copy(images[i][row][column]))
                    
        self.DAPI        = []
        self.coordinates = {}
        """simple segmentation of entire set"""
        for i in range(len(self.images)):
            image  = self.images[i]
            gray   = numpy.array(image/numpy.amax(image)*255,dtype = numpy.uint8)
            gray   = skimage.morphology.area_opening(gray)
            thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]          
            init_thresh = skimage.morphology.remove_small_objects(thresh,min_size=12)
            thresh = ndimage.binary_fill_holes(init_thresh)
            self.DAPI.append(copy.copy(thresh))

            for x in range(len(image)):
                for y in range(len(image[x])):
                    self.coordinates[copy.copy((x,y,i))] = False
                    
            x,y = thresh.nonzero()
            for n in range(len(x)):
                coordinate = (x[n],y[n],i)
                self.coordinates[copy.copy(coordinate)] = True

class RatioAnalysis:
    def __init__(self,ratio,aggregates):
        self.ratio      = ratio 
        self.aggregates = aggregates
        
        """the aggregate"""
        self.nuclear     = []
        self.cytoplasmic = []
        
        for i in aggregates:
            if i.area > 2:
                if i.nuclear:
                    self.nuclear.append(copy.copy(i.area))
                else:
                    self.cytoplasmic.append(copy.copy(i.area))
                    
        self.total = []
        for i in self.nuclear:
            self.total.append(i)
        for i in self.cytoplasmic:
            self.total.append(i)
            
        self.average = 0
        for i in self.total:
            self.average += i
        self.average = (numpy.average(numpy.array(self.average)),numpy.std(numpy.array(self.average)))
        
        self.average_N = 0
        for i in self.nuclear:
            self.average_N += i
        self.average_N = (numpy.average(numpy.array(self.average_N)),numpy.std(numpy.array(self.average_N)))
            
        self.average_C = 0
        for i in self.cytoplasmic:
            self.average_C += i
        self.average_C = (numpy.average(numpy.array(self.average_C)),numpy.std(numpy.array(self.average_C)))

        """calculate the roundness of pixels"""
                    

def LoadImages(paths,factor = 1,noise = 3):
    from AggregateDetection import Image
    """image folder paths"""
    aggregate_library = {}
    
    """data needed for analysis"""
    data3D = {"Size 3D":[],
              "Signal 3D":[],
              'Centroid':[],
              "Label":[],
              'Experiment':[],
              'Cytoplasm':[],
              'Nuclei':[],
              'Location':[],
              'Both':[]}
    
    data2D = {"Size 2D":[],
              "Signal 2D":[],
              "Circularity":[],
              'Centroid':[],
              "Label":[],
              'Size':[],
              'Experiment':[],
              'Cytoplasm':[],
              'Nuclei':[],
              'Location':[],
              'Both':[]}
    
    for i in paths: 
        images,dapi = [],[]
        """image paths"""
        image_paths = i
        """image path"""
        aggregates = []
        cname = copy.copy(i.split('\\')[-1])
        """Image paths and the set"""
        for n in image_paths:
            imagename = n.split('\\')[-1]
            if 'csv' not in n:
                split = n.split('\\')
                name  = split[-1].split('.')[0]
                b,r,g = ImportRGBImage(path)
                
                """the RGB channels"""
                r = [r[i] for i in range(8,len(r)-8,1)]
                g = [g[i] for i in range(8,len(g)-8,1)]
                b = [b[i] for i in range(8,len(b)-8,1)]
                
                """the images and the z factor"""
                image = Image(r,factor = factor,noise = noise)
                z     = len(image.arrayset)
                
                """get nuclei in the dataset"""
                dapistain = ReconstructedDAPI(array = b)

                coordinates  = dapistain.coordinates
                for keys,impdata in image.data2D.items():
                    for i in range(len(impdata)):
                        data2D[keys].append(impdata[i])
                        if keys == 'Label':
                            loc,cyt,nuc,both = NucleusCheck(image.data2D['Centroid'][i],image.data2D['Label'][i],coordinates)
                            data2D[keys].append(loc)
                            data2D['Experiment'].append(copy.copy(cname))
                            data2D['Location'].append(copy.copy(loc))
                            data2D['Cytoplasm'].append(copy.copy(cyt))
                            data2D['Nuclei'].append(copy.copy(nuc))
                            data2D['Both'].append(copy.copy(both))
                        
                for keys,impdata in image.data3D.items():
                    for i in range(len(impdata)):
                        data3D[keys].append(impdata[i])
                        if keys == 'Label':
                            loc,cyt,nuc,both = NucleusCheck(image.data3D['Centroid'][i],image.data3D['Label'][i],coordinates)
                            data3D[keys].append(loc)
                            data3D['Experiment'].append(copy.copy(cname))  
                            data3D['Location'].append(copy.copy(loc))
                            data3D['Cytoplasm'].append(copy.copy(cyt))
                            data3D['Nuclei'].append(copy.copy(nuc))
                            data3D['Both'].append(copy.copy(both))   
                            
    import pickle
    """store shifts in the main round folder"""
    storepath = '[INSERT PATH] dataset2D {0} {1}.pickle'.format(str(factor),str(noise))
    """pickle the data dict and store it"""
    with open(storepath, 'wb') as handle:
        pickle.dump(data2D, handle, protocol=pickle.HIGHEST_PROTOCOL)           
    """store shifts in the main round folder"""
    storepath = '[INSERT PATH] dataset3D {0} {1}.pickle'.format(str(factor),str(noise))
    """pickle the data dict and store it"""
    with open(storepath, 'wb') as handle:
        pickle.dump(data3D, handle, protocol=pickle.HIGHEST_PROTOCOL)  

"""insert the path where you with to store data in line 236, and 243,add paths in line 250 for all images, je can set the noies and factor/threshold parameter"""
from ImageAnalysisOperations import OpenImage
LoadImages(['[INSERT paths where images located]'],noise = 5,factor = 4)

















































###############################################################################
"""Data analysis"""
#takes dataset