'''
Adam Szatrowski, Hieu Phan, Dan Nelson
April 15, 2012
harris.py

tracking
'''

import time
import sys
import cv,cv2
import numpy
from scipy.ndimage import filters
from scipy.ndimage import interpolation
from scipy.spatial import distance
import imgutil
import pipeline
import source
import glob


class Tensor(pipeline.ProcessObject):
    """
    @image: a 2D grayscale image
    @sigmaD: differentiation scale
    @sigmaI: integration scale
    output: [Ix, Iy, Ixx, Iyy, Ixy]
    """
    def __init__(self, input=None, sigmaD=1.0, sigmaI=1.5):
        pipeline.ProcessObject.__init__(self, input, outputCount=5)
        self.sigmaD = sigmaD
        self.sigmaI = sigmaI
    def generateData(self):
        image = self.getInput(0).getData()
        
        image = image.astype(numpy.float32)
        rows, cols = image.shape[:2]
    
        # compute derivatives in x
        Ix = filters.gaussian_filter1d(image, self.sigmaD, 0, 0)
        Ix = filters.gaussian_filter1d(Ix, self.sigmaD, 1, 1)
    
        # compute derivatives in y
        Iy = filters.gaussian_filter1d(image, self.sigmaD, 1, 0)
        Iy = filters.gaussian_filter1d(Iy, self.sigmaD, 0, 1)
    
        # compute components of the structure tensor
        # 2nd derivative in x
        Ixx = filters.gaussian_filter(Ix**2, self.sigmaI, 0)
        # 2nd derivative in y
        Iyy = filters.gaussian_filter(Iy**2, self.sigmaI, 0)
        # IxIy
        Ixy = filters.gaussian_filter(Ix * Iy, self.sigmaI, 0)
        
        self.getOutput(0).setData(Ix)
        self.getOutput(1).setData(Iy)
        self.getOutput(2).setData(Ixx)
        self.getOutput(3).setData(Iyy)
        self.getOutput(4).setData(Ixy)

class HarrisCorners(pipeline.ProcessObject):
    '''
    Finds Harris corner features for an input image.
    input: [image, Ixx, Iyy, Ixy]
    @count: number of corners to detect
    @return: a maxPoints x 3 array, where each row contains the (x, y) 
    position, and Harris score of a feature point. The array is sorted
    by decreasing score.
    '''     
    
    def __init__(self, input=None, count=512, interval=-1):
        pipeline.ProcessObject.__init__(self, input, outputCount=2, inputCount=4)
        self.count = count
        self.interval = interval
        self.frame=-1

    def generateData(self):
        self.frame += 1
        if self.frame == 0 or (self.interval > 0 and self.frame%self.interval==0):
        
            image = self.getInput(0).getData()
            Ixx = self.getInput(1).getData()
            Iyy = self.getInput(2).getData()
            Ixy = self.getInput(3).getData()

            image = image.astype(numpy.float32)
            rows, cols = image.shape[:2]
            
        
            # compute Harris weights
            #hW = (Ixx * Iyy - Ixy**2) - 0.06*(Ixx + Iyy)
            hW = (Ixx * Iyy - Ixy**2) / (Ixx + Iyy + 1e-8)
        
            # exclude points near the image border
            hW[:16, :] = 0
            hW[-16:, :] = 0
            hW[:, :16] = 0
            hW[:, -16:] = 0
        
            # non-maximum suppression in 5x5 regions
            maxH = filters.maximum_filter(hW, (5,5))
            hW = hW * (hW == maxH)
        
            sortIdx = numpy.argsort(hW.flatten())[::-1]
            sortIdx = sortIdx[:self.count]
            yy = sortIdx / cols
            xx = sortIdx % cols
            
            # concatenate positions and values
            xyv = numpy.vstack((xx, yy, hW.flatten()[sortIdx])).transpose()
                
            self.getOutput(0).setData(hW)
            self.getOutput(1).setData(xyv)


class ShowFeatures(pipeline.ProcessObject):
    '''
    '''
    def __init__(self, input = None):
        pipeline.ProcessObject.__init__(self, input, inputCount=2, outputCount=1)
    
    def generateData(self):
        input1 = numpy.copy(self.getInput(0).getData())
        input2 = self.getInput(1).getData()
        #print input2[:5]
        r = 5
        print input2.shape
        for x,y in input2[:,:2]:
            cv2.rectangle(input1, (int(x-r), int(y-r)), (int(x+r), int(y+r)), (255,0,0), thickness=2)
            #cv2.rectangle(input1, (int(pxy[0]-5), int(pxy[1]-5)), (int(pxy[0]+5), int(pxy[1]+5)), (0, 0, 255))
        self.getOutput(0).setData(input1)
        
        
class OpticalFlow(pipeline.ProcessObject):
    '''
    inputs: [Image, Corners, Ix, Iy, Ixx, Iyy, Ixy]
    '''
    def __init__(self, input = None, sorted = None, radius = 5):
        pipeline.ProcessObject.__init__(self, input, inputCount=7)
        self.prevInpt = None
        self.features = []
        self.frame = 0
        self.radius = radius
        
    def generateData(self):
        inpt = self.getInput(0).getData()
        if self.frame == 0:
            self.prevInpt = inpt
            hc = self.getInput(1).getData()
            fts = numpy.ones((hc.shape[0],4), dtype=numpy.float32)
            fts[:,:3]=hc
            self.features = [fts]
            self.getOutput(0).setData(fts)
        else:
            newFeatures = numpy.copy(self.features[-1])
            r = self.radius
            Ix  = self.getInput(2).getData()
            Iy  = self.getInput(3).getData()
            Ixx = self.getInput(4).getData()
            Iyy = self.getInput(5).getData()
            Ixy = self.getInput(6).getData()
            It = inpt - self.prevInpt
            Ixt = Ix*It
            Iyt = Iy*It
            h, w = inpt.shape
            for i, (y,x, _, a) in enumerate(newFeatures):
                if a==0:
                    continue
                # g= imgutil.gaussian 
                # gg = dot(g.transpose, g)
                # r= g.size/2
                # iyy, ixx = numpy.mgrid[-r:r+1, -r, r+1]
                # ryy, rxx = y+iyy, x+ixx
                # patch0 = interpolation.map_coordinates(I, (ryy, rxx)) # maybe stack these into 2 by n^2 matrix
                # patch1 = interpolation.map_coordinates(I1, (ryy+u, rxx+v)) # maybe stack these into 2 by n^2 matrix
                # pIt = patch1-patch0 muliply by pIy and pIx
                # GIxIt = (patchIt*patchIx*gg.flaten()).sum()
                pIxx = Ixx[x,y]
                pIyy = Iyy[x,y]
                pIxy = Iyy[x,y]
                ATA = numpy.array([[pIxx, pIxy],[pIxy, pIyy]])
                ATb = -numpy.array([Ixt[y-r:y+r,x-r:x+r].sum(), Iyt[y-r:y+r,x-r:x+r].sum()])
                v = numpy.linalg.lstsq(ATA, ATb)[0]
                newFeatures[i][:2]+=v
                if newFeatures[i][0]>h or newFeatures[i][1]>w or newFeatures[i][0]<0 or newFeatures[i][1]<0:
                    newFeatures[i][3]=0.0
            self.features.append(newFeatures)
            self.prevInpt = inpt
            self.getOutput(0).setData(newFeatures)
        self.frame += 1
        
        
class Grayscale(pipeline.ProcessObject):
    
    def __init__(self, input = None):
        pipeline.ProcessObject.__init__(self, input)
        
    def generateData(self):
        input = self.getInput(0).getData()
        
        if input.ndim == 3 and input.shape[2]==3:       
            output = input[...,0]*0.114 + input[...,1]*0.587 + input[...,2]*0.229
        #output = input[...,1]
        self.getOutput(0).setData(output)


class Display(pipeline.ProcessObject):
    '''
    '''
    def __init__(self, input = None, name = "pipeline"):
        pipeline.ProcessObject.__init__(self, input)
        cv2.namedWindow(name, cv.CV_WINDOW_NORMAL)
        self.name = name
        
    def generateData(self):
        input = self.getInput(0).getData()
        
        # Convert back to OpenCV BGR from RGB
        if input.ndim == 3 and input.shape[2] == 3:
            input = input[..., ::-1]
        
        cv2.imshow(self.name, input.astype(numpy.uint8))
    
    
if __name__ == "__main__":
    
    #pipesource = source.CameraCV()
    #pipesource = source.FileReader("test.jpg")
    files = glob.glob("./images5/*.npy")
    pipesource = source.FileStackReader(files)
    pipesource.setLoop(True)
    
    grayscale = Grayscale(pipesource.getOutput())
    
    # find harris corners and sorted list'
    tensor = Tensor(grayscale.getOutput())
    hc = HarrisCorners()
    hc.setInput(grayscale.getOutput(0), 0)
    hc.setInput(tensor.getOutput(2), 1)
    hc.setInput(tensor.getOutput(3), 2)
    hc.setInput(tensor.getOutput(4), 3)
    
    of = OpticalFlow()
    of.setInput(grayscale.getOutput(0), 0)
    of.setInput(hc.getOutput(1), 1)
    of.setInput(tensor.getOutput(0), 2)
    of.setInput(tensor.getOutput(1), 3)
    of.setInput(tensor.getOutput(2), 4)
    of.setInput(tensor.getOutput(3), 5)
    of.setInput(tensor.getOutput(4), 6)
    
    
    
    # pass in original frame and list of points
    features = ShowFeatures(pipesource.getOutput())
    features.setInput(of.getOutput(0),1)
    
    # 2 displays
    display1 = Display(pipesource.getOutput(), "source")
    display2 = Display(features.getOutput(), "features")

    key = None
    frame = 0
    t0 = time.time()
    span = 30
    while key != 27:
        #pipesource.updatePlayMode()
        pipesource.increment()
        
        display1.update()
        display2.update()
        of.update()
        key = cv2.waitKey(100)
        if key >= 0:
            char = chr(key)
            print "Key: ", key, char
            
        frame += 1
        if frame % span == 0:
            t1 = time.time()
            print "{0:8.5f} fps".format(span / (t1 - t0))
            t0 = t1
