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
import pylab


class Tensor(pipeline.ProcessObject):
	"""
	@image: a 2D grayscale image
	@sigmaD: differentiation scale
	@sigmaI: integration scale
	output: [Ix, Iy, Ixx, Iyy, Ixy]
	"""
	def __init__(self, inpt=None, sigmaD=1.0, sigmaI=2.0):
		pipeline.ProcessObject.__init__(self, inpt, outputCount=5)
		self.sigmaD = sigmaD
		self.sigmaI = sigmaI
		
	def generateData(self):
		image = self.getInput(0).getData()
		
		image = image.astype(numpy.float32)
	
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
	
	def __init__(self, inpt=None, count=512, interval=-1):
		pipeline.ProcessObject.__init__(self, inpt, outputCount=2, inputCount=4)
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
			hW = (Ixx * Iyy - Ixy**2) / (Ixx + Iyy + 1e-8)
		
			# exclude image boundaries
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
	Draws boxes around the features in an image
	'''
	def __init__(self, inpt = None):
		pipeline.ProcessObject.__init__(self, inpt, inputCount=2, outputCount=1)
	
	def generateData(self):
		input1 = numpy.copy(self.getInput(0).getData())
		input2 = self.getInput(1).getData()
		r = 5
		for x,y in input2[:,:2]:
			cv2.rectangle(input1, (int(x-r), int(y-r)), (int(x+r), int(y+r)), (255,0,0), thickness=2)
		self.getOutput(0).setData(input1)

		
class FeatureTracer(pipeline.ProcessObject):	
	'''
	Draws the paths in which features are moving
	'''
	
	def __init__(self, inpt = None):
		pipeline.ProcessObject.__init__(self, inpt, inputCount=2, outputCount=1)
		self.features = []
	
	def generateData(self):
		input1 = numpy.copy(self.getInput(0).getData()) # the image
		input2 = self.getInput(1).getData() # most recent feature positions
		self.features.append(input2)
		r = 5
		
		# creating feature paths
		for feature in self.features:
			for x,y in feature[:,:2]:
				cv2.circle(input1, (int(x), int(y)),2, (255,0,0), thickness=2)
		self.getOutput(0).setData(input1)

		
class OpticalFlow(pipeline.ProcessObject):
	'''
	inputs: [Image, Corners, Ix, Iy, Ixx, Iyy, Ixy]
	iterations: The maximum number of iterations to perform in calculating the velocity of a feature
	epsilon: The tolerance for a change in velocity.
	sigmaD: sigma to use for calculation of patch size.
	
	In calculating the velocity for a feature, we will stop after [iterations] number of iterations,
	or if the norm of the change in velocity is less than [epsilon]
	'''
	def __init__(self, inpt = None, iterations=10, epsilon=float('1.0e-3'), sigmaD=2.0):
		pipeline.ProcessObject.__init__(self, inpt, inputCount=7)
		self.prevInpt = None
		self.baseInpt = None
		self.patches = None
		self.frame = 0
		self.iterations = iterations
		self.epsilon = epsilon**2
		self.sigmaD = sigmaD
		
	def generateData(self):
		print self.frame
		inpt = self.getInput(0).getData()
		
		# if we are on the first frame
		if self.frame == 0:
			self.prevInpt = inpt
			
			hc = self.getInput(1).getData()
			# Create the features vector for t_0 storing (x,y,confidence, active)
			fts = numpy.ones((hc.shape[0],4), dtype=numpy.float32)
			fts[:,:2]=hc[:,:2] #Copy the x-y positions of features detected by hc to the features vector
			self.prevFeatures = fts # store it as the first entry in the features array which stores features over time
			self.getOutput(0).setData(fts)
			
			
			r = numpy.floor(self.sigmaD * 5.0/2)
			self.patches = numpy.zeros((hc.shape[0], (2*r+1)**2), dtype=numpy.float32)
			for i, (x,y, _, _) in enumerate(fts):
			    self.patches[i,...] = inpt[y-r:y+r+1, x-r:x+r+1].flatten()
		else:
			newFeatures = numpy.copy(self.prevFeatures)
			Ix  = self.getInput(2).getData()
			Iy  = self.getInput(3).getData()
			Ixx = self.getInput(4).getData()
			Iyy = self.getInput(5).getData()
			Ixy = self.getInput(6).getData()
			
			h, w = inpt.shape
			velocity = numpy.array([0.0,0.0]) # intial velocity values
			
			for i, (x,y, s, a) in enumerate(newFeatures):
				iterations = self.iterations
				
				# throw away inactive points
				if a==0:
					continue
					
				# CALCULATE ATA
				pIxx = Ixx[y,x]
				pIyy = Iyy[y,x]
				pIxy = Ixy[y,x]
				ATA = numpy.array([[pIxx, pIxy],[pIxy, pIyy]])                
				
				#change in velocity
				dv = numpy.array([100.0,100.0])
				
				g = imgutil.gaussian(self.sigmaD)[0]
				g = g[:,None]
				gg = numpy.dot(g, g.transpose()).flatten()
				r = g.size/2
				iyy, ixx = numpy.mgrid[-r:r+1, -r:r+1]
				ryy, rxx = y+iyy, x+ixx
				
				patchIx = interpolation.map_coordinates(Ix, numpy.array([ryy.flatten(), rxx.flatten()]))
				patchIy = interpolation.map_coordinates(Iy, numpy.array([ryy.flatten(), rxx.flatten()]))

				# ITERATE AND CALCULATE ATb
				while iterations > 0 and numpy.dot(dv, dv)>self.epsilon: 
					patch1 = interpolation.map_coordinates(inpt, numpy.array([ryy.flatten(), rxx.flatten()]))
					patch0 = interpolation.map_coordinates(self.prevInpt, numpy.array([(ryy-velocity[1]).flatten(), (rxx-velocity[0]).flatten()])) 

					#imgutil.imageShow(patch0.reshape((g.size,g.size)), "p0")
					#imgutil.imageShow(patch1.reshape((g.size,g.size)),"p1")
		
					patchIt = patch1-patch0
					pIxt = (patchIt*patchIx*gg).sum()
					pIyt = (patchIt*patchIy*gg).sum()
					
					# solve ATAv = -ATb
					ATb = -numpy.array([pIxt, pIyt])
					dv = numpy.linalg.lstsq(ATA, ATb)[0]
					# update velocity and iterations
					velocity += dv
					iterations -= 1
				
				#print "stopped after", (self.iterations-iterations), "with norm", numpy.dot(dv,dv)**.5
				
				# calculate new feature positions
				newFeatures[i][:2]+= velocity
				
				#Compute the similarity between our new feature location and the original feature location
				
				#imgutil.imageShow(patch1.reshape(g.size,g.size), "p0")
				#imgutil.imageShow(self.patches[i], "p1")
				newFeatures[i][2] = imgutil.ncc(patch1, self.patches[i,...])
				
				# set feature status (active or inactive)
				if newFeatures[i][0] > w or newFeatures[i][1] > h or newFeatures[i][0] < 0 or newFeatures[i][1] < 0:
					newFeatures[i][3] = 0
			
			self.prevFeatures = newFeatures
			self.prevInpt = inpt
			self.getOutput(0).setData(newFeatures)
			
		self.frame += 1
		
		
class Grayscale(pipeline.ProcessObject):
	'''
	Converts an image to grayscale if it has 3 channels
	'''
	def __init__(self, inpt = None):
		pipeline.ProcessObject.__init__(self, inpt)
		
	def generateData(self):
		inpt = self.getInput(0).getData()
		
		if inpt.ndim == 3 and inpt.shape[2]==3:       
			output = inpt[...,0]*0.114 + inpt[...,1]*0.587 + inpt[...,2]*0.229

		self.getOutput(0).setData(output)


class Display(pipeline.ProcessObject):
	'''
	'''
	def __init__(self, inpt = None, name = "pipeline"):
		pipeline.ProcessObject.__init__(self, inpt)
		cv2.namedWindow(name, cv.CV_WINDOW_NORMAL)
		self.name = name
		
	def generateData(self):
		inpt = self.getInput(0).getData()
		
		# Convert back to OpenCV BGR from RGB
		if inpt.ndim == 3 and inpt.shape[2] == 3:
			inpt = inpt[..., ::-1]
		
		cv2.imshow(self.name, inpt.astype(numpy.uint8))

class ScoreTracker(pipeline.ProcessObject):
	'''
	'''
	def __init__(self, inpt = None):
		pipeline.ProcessObject.__init__(self, inpt, outputCount=0)
		self.scores = []
		
	def generateData(self):
		inpt = self.getInput(0).getData()
		self.scores.append(inpt[:,2])
	
	def getScoreData(self):
	    return numpy.array(self.scores)
	
if __name__ == "__main__":
	
	#pipesource = source.CameraCV()
	#pipesource = source.FileReader("test.jpg")
	files = glob.glob("./images5/*.npy")
	pipesource = source.FileStackReader(files)
	pipesource.setLoop(False)
	
	# convert image to grayscale
	grayscale = Grayscale(pipesource.getOutput())
	
	# find image tensor
	tensor = Tensor(grayscale.getOutput(),sigmaD=2.0, sigmaI=3.0)
	
	# find harris corners and sorted list
	hc = HarrisCorners(count=15)
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
	
	# feature tracer
	tracer = FeatureTracer(pipesource.getOutput())
	tracer.setInput(of.getOutput(0),1)	
	
	scoretracker = ScoreTracker(of.getOutput(0))
	
	# 2 displays
	display1 = Display(pipesource.getOutput(), "source")
	display2 = Display(features.getOutput(), "features")
	display3 = Display(tracer.getOutput(), "tracer")

	key = None
	frame = 0
	t0 = time.time()
	span = 30
	while key != 27:
		#pipesource.updatePlayMode()
		pipesource.increment()
		
		display1.update()
		display2.update()
		display3.update()
		scoretracker.update()
		key = cv2.waitKey(100)
		if key >= 0:
			char = chr(key)
			print "Key: ", key, char
			
		frame += 1
		if frame % span == 0:
			t1 = time.time()
			print "{0:8.5f} fps".format(span / (t1 - t0))
			t0 = t1
		
		if pipesource.getIndex() == pipesource.getLength()-1:
		    break
	scores = scoretracker.getScoreData().transpose()
	import pickle
	import datetime
	pickle.dump(scores, file("score-data-%s.p"%datetime.datetime.now().strftime("%y-%m-%d-%H.%M.%S"), "w"))
	for i in range(scores.shape[0]):
	    plt, = pylab.plot(scores[i,...])
	pylab.show()
	    
