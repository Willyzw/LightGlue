"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import numpy as np 
from enum import Enum
import cv2

class Dataset(object):
    def __init__(self, path, name, fps=None, associations=None, start_frame_id=0):
        self.path = path 
        self.name = name 
        self.scale_viewer_3d = 1.0 
        self.is_ok = True
        self.fps = fps   
        if fps is not None:       
            self.Ts = 1./fps 
        else: 
            self.Ts = None 
          
        self.start_frame_id = start_frame_id
        self.timestamps = None 
        self._timestamp = None       # current timestamp if available [s]
        self._next_timestamp = None  # next timestamp if available otherwise an estimate [s]
        
    def isOk(self):
        return self.is_ok

    def getImage(self, frame_id):
        return None 

    def getImageRight(self, frame_id):
        return None

    def getDepth(self, frame_id):
        return None        

    # Adjust frame id with start frame id only here
    def getImageColor(self, frame_id):
        frame_id += self.start_frame_id
        try: 
            img = self.getImage(frame_id)
            return img[:,:,0]        
        except:
            img = None  
            #raise IOError('Cannot open dataset: ', self.name, ', path: ', self.path)        
            print(f'Cannot open dataset: {self.name}, path: {self.path}')
            return img    
        
    # Adjust frame id with start frame id only here
    def getImageColorRight(self, frame_id):
        frame_id += self.start_frame_id
        try: 
            img = self.getImageRight(frame_id)
            if img is not None and img.ndim == 2:
                return cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)     
            else:
                return img             
        except:
            img = None  
            #raise IOError('Cannot open dataset: ', self.name, ', path: ', self.path)        
            print(f'Cannot open dataset: {self.name}, path: {self.path}, right image')
            return img         
        
    def getTimestamp(self):
        return self._timestamp
    
    def getNextTimestamp(self):
        return self._next_timestamp

    def _read_timestamps(self, timestamps_file):
        timestamps = []
        try:
            with open(timestamps_file, 'r') as file:
                for line in file:
                    timestamp = int(float(line.strip()))
                    timestamps.append(timestamp)
        except FileNotFoundError:
            print('Timestamps file not found:', timestamps_file)
        return timestamps   


class VideoDataset(Dataset): 
    def __init__(self, path, name, associations=None, timestamps=None, start_frame_id=0): 
        super().__init__(path, name, None, associations, start_frame_id)    
        self.filename = path + '/' + name 
        #print('video: ', self.filename)
        self.cap = cv2.VideoCapture(self.filename)
        self.i = 0        
        self.timestamps = None
        if timestamps is not None:
            self.timestamps = self._read_timestamps(path + '/' + timestamps)
        if not self.cap.isOpened():
            raise IOError('Cannot open movie file: ', self.filename)
        else: 
            print('Processing Video Input')
            self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
            self.Ts = 1./self.fps 
            print('num frames: ', self.num_frames)  
            print('fps: ', self.fps)              
        self.is_init = False   
            
    def getImage(self, frame_id):
        # retrieve the first image if its id is > 0 
        if self.is_init is False and frame_id > 0:
            self.is_init = True 
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        self.is_init = True
        ret, image = self.cap.read()
        if self.timestamps is not None:
            # read timestamps from timestamps file
            self._timestamp = int(self.timestamps[self.i])
            self._next_timestamp = int(self.timestamps[self.i + 1])
            self.i += 1
        else:
            #self._timestamp = time.time()  # rough timestamp if nothing else is available 
            self._timestamp = float(self.cap.get(cv2.CAP_PROP_POS_MSEC)*1000)
            self._next_timestamp = self._timestamp + self.Ts 
        if ret is False:
            self.is_ok = False
        return image[:,:,0]
