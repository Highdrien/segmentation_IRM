import os
from tqdm import tqdm
import numpy as np

from utils import load_nii
import variable as var


def list_target_path(data_path=var.DATASET_PATH):
    'Returns three generators'
    target_paths = []
    for patient in os.listdir(data_path): #liste des fichers dans le dataset
        if patient[-1]!='x' and patient[-1]!='y': # on enleve le .dox et le .py 
            for frame in os.listdir(os.path.join(data_path,patient)): #on parcours les fichier du dossier "patient_i"
                if len(frame)>15: # on enleve l'info doc
                    if len(frame)==28 and frame[17]=='1': #on prend la réponce de l'irm 1
                        target_paths.append(os.path.join(os.path.join(data_path,patient),frame))
                    elif len(frame)==28 and frame[17]=='2':#on prend la réponce de l'irm 2
                        target_paths.append(os.path.join(os.path.join(data_path,patient),frame))
    return target_paths

def classes_distrib(L):
    dist=[0,0,0,0]
    for path_gt in tqdm(L,desc = 'calcule de la distribution des classes'):
        target=np.array(load_nii(path_gt)[0])
        X,Y,Z=np.shape(target)
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    dist[target[x,y,z]]+=1
    somme=sum(dist)
    dist_normal=[dist[i]/somme for i in range(4)]
    return dist_normal


if __name__=="__main__":
    print(classes_distrib(list_target_path()))