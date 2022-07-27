import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import variable as var
from model import UNet
import os
import numpy as np
from generateur import create_generators, gen_weights, CLASSES_DISTRIBUTION
from sklearn.metrics import accuracy_score, jaccard_score


def train(epochs=var.EPOCHS,lr=var.LEARNING_RATE,batch_size=1,eval_every=1,checkpoint_path="trained_models"):

    dataset_train,dataset_val,_=create_generators()
    train_loader = DataLoader(dataset_train,batch_size = batch_size)
    val_loader = DataLoader(dataset_val,batch_size = batch_size)

    # Use gpu or cpu
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model = UNet(input_channels = 1, 
                 output_classes = 4, 
                 hidden_channels = 8)

    model.to(device)

    # Weighted Cross Entropy Loss & adam optimizer
    weight = gen_weights(CLASSES_DISTRIBUTION, c = 1.03)

    criterion = torch.nn.CrossEntropyLoss(reduction= 'mean', weight=weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ###############################################################
    # Start Training                                              #
    ###############################################################
    model.train()
    
    for epoch in range(1, epochs+1):
        training_loss = []
        print('epoch:'+str(epoch))
        for (image, target) in train_loader:
            
            image = image.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            
            logits = model(image)
            
            loss = criterion(logits, target)

            loss.backward()

            training_loss.append((loss.data*target.shape[0]).tolist())
            
            optimizer.step()
        
        print("Training loss was: " + str(sum(training_loss)/var.TRAIN))

        ###############################################################
        # Start Evaluation                                            #
        ###############################################################
        
        if epoch % eval_every == 0:
            model.eval()

            val_loss = []
            acc=0
            iou=0
            iou_par_classes=0
            
            with torch.no_grad():
                for (image, target) in val_loader:

                    image = image.to(device)
                    target = target.to(device)

                    logits = model(image)
                    
                    loss = criterion(logits, target)   

                    logits = torch.movedim(logits, (0,1,2,3), (0,3,1,2))
                    logits = logits.reshape((-1,4))
                    target = target.reshape(-1)

                    probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                    target = target.cpu().numpy()
                    
                    val_loss.append(loss.tolist())
                    y_pred = probs.argmax(1).tolist()
                    y_true = target.tolist()
                    
                    acc += accuracy_score(y_true, y_pred)
                    iou += jaccard_score(y_true, y_pred, average='micro')
                    iou_par_classes += jaccard_score(y_true, y_pred, average='macro')
                    

                ####################################################################
                # Save Scores to the .log file and visualize also with tensorboard #
                ####################################################################           
                  
                print("Test loss was: " + str(sum(val_loss)/var.VAL))
                print("METRIC: acc=",acc/var.VAL," iou=",iou/var.VAL," iou par classes=",iou_par_classes/var.VAL)

                model_dir = os.path.join(checkpoint_path, str(epoch))
                os.makedirs(model_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))


if __name__=="__main__":
    train()
            
        

    