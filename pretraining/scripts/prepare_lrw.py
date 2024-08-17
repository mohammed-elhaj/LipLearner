# encoding: utf-8
import cv2
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
import torch

import numpy as np
import csv
import glob
import time
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torch

jpeg = TurboJPEG()
def extract_opencv(filename):
    video = []
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):
        ret, frame = cap.read() # BGR
        if ret:
            # frame = frame[115:211, 79:175]
            frame = frame[99:227, 63:191]
            frame = jpeg.encode(frame)
            video.append(frame)
        else:
            break
    cap.release()
    return video        


target_dir = 'lrw_roi_63_99_191_227_size128_gray_jpeg'

if(not os.path.exists(target_dir)):
    os.makedirs(target_dir)    

class LRWDataset(Dataset):
    def __init__(self):

        with open('sorted_labels.txt') as myfile:
            self.labels = myfile.read().splitlines()            
        
        self.list = []

        for (i, label) in enumerate(self.labels):
            files = glob.glob(os.path.join('train', label, '*.mp4'))
            for file in files:
                savefile = file.replace('train', target_dir).replace('.mp4', '.pkl')
                savepath = os.path.split(savefile)[0]
                if(not os.path.exists(savepath)):
                    os.makedirs(savepath)
                
            files = sorted(files)
            

            self.list += [(file, i) for file in files]                                                                                
            
        
    def __getitem__(self, idx):
            
        inputs = extract_opencv(self.list[idx][0])
        result = {}        
         
        name = self.list[idx][0]
        duration = self.list[idx][0]            
        labels = self.list[idx][1]

                    
        result['video'] = inputs
        result['label'] = int(labels)
        result['duration'] = self.load_duration(duration.replace('.mp4', '.csv')).astype(np.bool)
        savename = self.list[idx][0].replace('train', target_dir).replace('.mp4', '.pkl')
        torch.save(result, savename)
        
        return result

    def __len__(self):
        return len(self.list)

    def load_duration(self, file, target_word):
        with open(file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header

            for row in reader:
                if row[3] == target_word:
                    end = float(row[1])  # Get 'end' value
                    start = float(row[2])  # Get 'start' value
                
                    duration = end - start  # Calculate duration
                    break  # Stop after finding the first match
        
        tensor = np.zeros(29)
        mid = 29 / 2
        start_idx = int(mid - duration / 2 * 25)
        end_idx = int(mid + duration / 2 * 25)
        tensor[start_idx:end_idx] = 1.0
    
        return tensor         

if(__name__ == '__main__'):
    loader = DataLoader(LRWDataset(),
            batch_size = 96, 
            num_workers = 2,   
            shuffle = False,         
            drop_last = False)
    
    import time
    tic = time.time()
    for i, batch in enumerate(loader):
        toc = time.time()
        eta = ((toc - tic) / (i + 1) * (len(loader) - i)) / 3600.0
        print(f'eta:{eta:.5f}')        
