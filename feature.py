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


class HarrisCorners(pipeline.ProcessObject):
	'''
	Finds Harris corner features for an input image.
	@image: a 2D grayscale image
	@sigmaD: differentiation scale
	@sigmaI: integration scale
	@count: number of corners to detect
	@return: a maxPoints x 3 array, where each row contains the (x, y) 
	position, and Harris score of a feature point. The array is sorted
	by decreasing score.
	'''		
	
	def __init__(self, input=None, sigmaD=1.0, sigmaI=1.5, count=512, descending=True):
		pipeline.ProcessObject.__init__(self, input, outputCount=2)
		self.sigmaD = sigmaD
		self.sigmaI = sigmaI
		self.count = count
		self.descending = descending

	def generateData(self):
		input = self.getInput(0).getData()
		
		input = input.astype(numpy.float32)
		rows, cols = input.shape[:2]
		
		# compute derivatives in x
		Ix = filters.gaussian_filter1d(input, self.sigmaD, 0, 0)
		Ix = filters.gaussian_filter1d(Ix, self.sigmaD, 1, 1)
		
		# compute derivatives in y
		Iy = filters.gaussian_filter1d(input, self.sigmaD, 1, 0)
		Iy = filters.gaussian_filter1d(Iy, self.sigmaD, 0, 1)
		
		# compute components of the structure tensor
		# 2nd derivative in x
		Ixx = filters.gaussian_filter(Ix**2, self.sigmaI, 0)
		# 2nd derivative in y
		Iyy = filters.gaussian_filter(Iy**2, self.sigmaI, 0)
		# IxIy
		Ixy = filters.gaussian_filter(Ix * Iy, self.sigmaI, 0)
		
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
		
		if self.descending:
			sortIdx = numpy.argsort(input.flatten())[::-1]
		else:
			sortIdx = numpy.argsort(input.flatten())
		sortIdx = sortIdx[:self.count]
		yy = sortIdx / cols
		xx = sortIdx % cols
			
		# concatenate positions and values
		xyv = numpy.vstack((xx, yy, input.flatten()[sortIdx])).transpose()
				
		self.getOutput(0).setData(hW)
		self.getOutput(1).setData(xyv)


class ShowFeatures(pipeline.ProcessObject):
	'''
	'''
	def __init__(self, input = None, sorted = None):
		pipeline.ProcessObject.__init__(self, input, inputCount=2, outputCount=1)
	
	def generateData(self):
		input1 = self.getInput(0).getData()
		input2 = self.getInput(1).getData()
		
		for pxy in input2:
			cv2.rectangle(input1, (int(pxy[0]-5), int(pxy[1]-5)), (int(pxy[0]+5), int(pxy[1]+5)), (0, 0, 255))
		
		self.getOutput(0).setData(input1)
		
		
class OpticalFlow(pipeline.ProcessObject):
	'''
	'''
	def __init__(self, input = None, sorted = None):
		pipeline.ProcessObject.__init__(self, input)
		self.prev = None
		
	def generateData(self):
		if self.prev == None:
			self.prev = self.getInput(0).getData()
		else:
			pass
	
		
		
class Grayscale(pipeline.ProcessObject):
	
	def __init__(self, input = None):
		pipeline.ProcessObject.__init__(self, input)
		
	def generateData(self):
		input = self.getInput(0).getData()
		
		if input.ndim == 3 and input.shape[2]==3:		
			output = input[...,0]*0.114 + input[...,1]*0.587 + input[...,2]*0.229
		
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
	
	pipesource = source.CameraCV()
	grayscale = Grayscale(pipesource.getOutput())
	
	# find harris corners and sorted list
	hc = HarrisCorners(grayscale.getOutput())
	
	# pass in original frame and list of weights
	features = ShowFeatures(pipesource.getOutput())
	features.setInput(hc.getOutput(1),1)
	
	# 2 displays
	display1 = Display(features.getOutput(), "source")
	display2 = Display(features.getOutput(), "features")

	key = None
	frame = 0
	t0 = time.time()
	span = 30
	while key != 27:
		pipesource.updatePlayMode()
 
		display1.update()
		display2.update()
		key = cv2.waitKey(10)
		if key >= 0:
			char = chr(key)
			print "Key: ", key, char
			
		frame += 1
		if frame % span == 0:
			t1 = time.time()
			print "{0:8.5f} fps".format(span / (t1 - t0))
			t0 = t1
	
	

	
	
"""
def harris(image, sigmaD=1.0, sigmaI=1.5, count=512):
	'''
	Finds Harris corner features for an input image.  The input image
	should be a 2D numpy array.  The sigmaD and sigmaI parameters define
	the differentiation and integration scales (respectively) of the
	Harris feature detector---the defaults are reasonable for many images.
	
	Returns:
	a maxPoints x 3 array, where each row contains the (x, y) 
	position, and Harris score of a feature point.  The array is sorted
	by decreasing score.
	'''
	
	image = image.astype(numpy.float32)
	h, w = image.shape[0:2]
	
	# compute image derivatives
	Ix = filters.gaussian_filter1d(image, sigmaD, 0, 0)
	Ix = filters.gaussian_filter1d(Ix, sigmaD, 1, 1)
	Iy = filters.gaussian_filter1d(image, sigmaD, 1, 0)
	Iy = filters.gaussian_filter1d(Iy, sigmaD, 0, 1)
	
	# compute elements of the structure tensor
	Ixx = filters.gaussian_filter(Ix**2, sigmaI, 0)
	Iyy = filters.gaussian_filter(Iy**2, sigmaI, 0)
	Ixy = filters.gaussian_filter(Ix * Iy, sigmaI, 0)
	
	# compute Harris feature strength, avoiding divide by zero
	imgH = (Ixx * Iyy - Ixy**2) / (Ixx + Iyy + 1e-8)
		
	# exclude points near the image border
	imgH[:16, :] = 0
	imgH[-16:, :] = 0
	imgH[:, :16] = 0
	imgH[:, -16:] = 0
	
	# non-maximum suppression in 5x5 regions
	maxH = filters.maximum_filter(imgH, (5,5))
	imgH = imgH * (imgH == maxH)
	
	# sort points by strength and find their positions
	sortIdx = numpy.argsort(imgH.flatten())[::-1]
	sortIdx = sortIdx[:count]
	yy = sortIdx / w
	xx = sortIdx % w
		
	# concatenate positions and values
	xyv = numpy.vstack((xx, yy, imgH.flatten()[sortIdx])).transpose()
	
	return xyv
	
def showHarris(img, harrisCorners):
		'''
		Creates boxes around the harrisCorners of an image
		@img: a cvimg
		@harrisCorners: an array of harrisCorners
		'''
		for pxy in harrisCorners:
			cv2.rectangle(img, (int(pxy[0]-5), int(pxy[1]-5)), (int(pxy[0]+5), int(pxy[1]+5)), (0, 0, 255))
	
"""