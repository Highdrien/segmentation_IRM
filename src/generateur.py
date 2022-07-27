import os 
import sys
import numpy as np
from os.path import dirname as up

import torch
from torch.utils.data import Dataset

import variable as var
sys.path.append(os.path.join(up(up(os.path.abspath(__file__))), 'utils'))
from utils import load_nii
from resize import resize_for_unet

np.random.seed(0)
CLASSES_DISTRIBUTION=torch.tensor([0.9621471811176255, 0.012111862189784502, 0.013016226246835367, 0.01272473044575458])


class DataGenerator(Dataset):
    def __init__(self, list_IDs, data_path=var.DATASET_PATH, n_channels=var.NUMBER_OF_CHANNELS, shuffle=var.SHUFFLE_DATA):
        'Initialisation'
        self.classes = os.listdir(data_path)
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle  
        self.on_epoch_end()
        self.data_path=data_path

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs) 

    def __getitem__(self, index):
        'Generate one batch of data'
        X, y = self.__data_generation(self.list_IDs[self.indexes[index]])        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))#crée une liste de 0 à len(self.list_IDs)
        if self.shuffle: 
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *image_size, n_channels)

        path=list_IDs_temp[0]
        path_gt=list_IDs_temp[1]
        # print(path[19:22],path[40]) 
        image_size=np.shape(load_nii(path)[0])

        X1=resize_for_unet(load_nii(path)[0],image_size,var.DEEP_UNET)
        y1=resize_for_unet(load_nii(path_gt)[0],image_size,var.DEEP_UNET)

        image_size=(image_size[0]-image_size[0]%(2**var.DEEP_UNET),image_size[1]-image_size[1]%(2**var.DEEP_UNET),image_size[2])

        X = np.empty(( self.n_channels,*image_size)) 
        y = np.empty(image_size,dtype=np.int64)  

        X[0,:,:,:] = X1
        y[:,:,:] = y1

        return torch.tensor(X,dtype=torch.float),torch.tensor(y)
    


def create_generators(data_path=var.DATASET_PATH):
    'Returns three generators'
    image_paths = []
    for patient in os.listdir(data_path): #list des fichers dans le data_set
        if patient[-1]!='x' and patient[-1]!='y':
            frame1=[0,1]
            frame2=[0,1]
            for frame in os.listdir(os.path.join(data_path,patient)):
                if len(frame)>15:
                    if len(frame)==25 and frame[17]=='1':
                        frame1[0]=os.path.join(os.path.join(data_path,patient),frame) #image_paths.append(os.path.join(os.path.join(data_path,patient),frame))
                    elif len(frame)==28 and frame[17]=='1':
                        frame1[1]=os.path.join(os.path.join(data_path,patient),frame)
                    elif len(frame)==25 and frame[17]=='2':
                        frame2[0]=os.path.join(os.path.join(data_path,patient),frame)
                    elif len(frame)==28 and frame[17]=='2':
                        frame2[1]=os.path.join(os.path.join(data_path,patient),frame)
            image_paths.append(frame1+frame2)

    train_list, val_list, test_list = data_split(np.asarray(image_paths)) #répartie les images en 3
    train_list = coupe_en2(train_list)
    val_list = coupe_en2(val_list)
    test_list = coupe_en2(test_list)

    train_data_generator = DataGenerator(train_list)
    validation_data_generator = DataGenerator(val_list)
    test_data_generator = DataGenerator(test_list)

    return train_data_generator, validation_data_generator, test_data_generator


def data_split(paths_list):
    'Splits the paths list into three for train, val and test'
    split_1 = len(paths_list)-var.TEST-var.VAL 
    split_2 = len(paths_list)-var.TEST 
    # np.random.shuffle(paths_list)
    return paths_list[:split_1], paths_list[split_1:split_2], paths_list[split_2:]


def coupe_en2(L):
    "transforme une liste de shape (n,4) en (2n,2)"
    l=[]
    for i in range(len(L)):
        l.append(L[i][:2])
        l.append(L[i][2:])
    l=np.array(l)
    #np.random.shuffle(l)
    return l


    
###############################################################
# Weighting Function for Semantic Segmentation                #
###############################################################
def gen_weights(class_distribution, c = 1.02):
    return 1/torch.log(c + class_distribution)

if __name__ == "__main__":
    train_data_generator, validation_data_generator, test_data_generator=create_generators()
    X,y=test_data_generator.__getitem__(1)


    

    



