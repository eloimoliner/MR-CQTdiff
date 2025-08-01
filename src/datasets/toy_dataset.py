import os
import numpy as np
import torch
import random
import glob
import soundfile as sf
import glob


class ToyDataset(torch.utils.data.IterableDataset):
    def __init__(self,
        fs,
        path,
        load_len,
        seed=42 
        ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        self.main_path=path

        #glob .wav files
        self.train_samples=glob.glob(os.path.join(self.main_path,"*.wav"))

        self.seg_len=int(load_len)

    def __iter__(self):
        while True:
            try:
                num=random.randint(0,len(self.train_samples)-1)

                file_path=self.train_samples[num]


                data, samplerate = sf.read(file_path)
                if samplerate>48000:
                    continue

                data_clean=data
                #Stereo to mono
                if len(data.shape)>1 :
                    data_clean=np.mean(data_clean,axis=1)

                if len(data_clean)<self.seg_len:
                    #circular padding with random offset
                    idx=0
                    segment=np.tile(data_clean,(self.seg_len//data_clean.shape[-1]+1))[...,idx:idx+self.seg_len]
                    #random offset (circular shift)
                    segment=np.roll(segment, np.random.randint(0,self.seg_len), axis=-1)
                else:
                    idx=np.random.randint(0,len(data_clean)-self.seg_len)
                    segment=data_clean[idx:idx+self.seg_len]

                segment=segment.astype('float32')
        
                yield  segment, samplerate

            except Exception as e:
                print("Error in ToyDataset:", e)
                continue
