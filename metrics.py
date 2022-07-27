from sklearn.metrics import accuracy_score, jaccard_score


# Evaluation for Pixel-level semantic segmentation
def Evaluation(y_predicted, y_true):
        
    subset_acc = accuracy_score(y_true, y_predicted)
    
    iou_acc = jaccard_score(y_true, y_predicted, average='macro')

    info = {
            "subsetAcc" : subset_acc,
            "IoU": iou_acc
            }
    
    return info
