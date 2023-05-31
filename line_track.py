import cv2
import numpy as np
class LineTracker:
	def __init__(self):
		self.all_frame = np.zeros((1000,1000),dtype=np.uint8)
		cv2.line(self.all_frame, (100,100), (900,100), 255, 5)
		cv2.line(self.all_frame, (900,100), (900,500), 255, 5)
		cv2.line(self.all_frame, (900,500), (100,500), 255, 5)
		cv2.line(self.all_frame, (100,500), (100,900), 255, 5)
		cv2.line(self.all_frame, (100,900), (900,900), 255, 5)
		self.x = 50
		self.y = 50

	def get_frame(self):
		# get frame of desired position
		if self.y>950:
			self.y=950
		if self.x>950:
			self.x=950
			
		cutten_frame = self.all_frame[self.y:self.y+100,self.x:self.x+100]
		rescaled_frame = cv2.resize(cutten_frame, (500,500))
		return rescaled_frame

	def get_frame_related_pos(self):
		frame = cv2.merge([self.all_frame.copy(),self.all_frame.copy(),self.all_frame.copy()])
		cv2.rectangle(frame, (self.x,self.y), (self.x+100, self.y+100), (255,0,0),2)
		return cv2.resize(frame, (500,500))

tracker = LineTracker() # the window that follow the line

while cv2.waitKey(100) !=27:

	if tracker.x<850: # if it don't reaches the line end keep going left
		tracker.x+=10
	else: # if window reaches the right side go down  
		tracker.y+=10

	frame = tracker.get_frame() # get frame seen by ROV
	cv2.imshow("frame", frame)
	cv2.imshow("related_pos", tracker.get_frame_related_pos()) # show frame of the ROV location on the line map
