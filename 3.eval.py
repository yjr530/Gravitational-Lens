import os

import numpy as np
import torch

from classification import (Classification, cvtColor, letterbox_image,
                            preprocess_input)
from utils.utils import letterbox_image
from utils.utils_metrics import evaluteTop1_5


test_annotation_path    = 'cls_val.txt'

metrics_out_path        = "metrics_out"

class Eval_Classification(Classification):
    def detect_image(self, image):        

        image       = cvtColor(image)

        image_data  = letterbox_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)

        image_data  = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

        with torch.no_grad():
            photo   = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                photo = photo.cuda()

            preds   = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()

        return preds

if __name__ == "__main__":
    if not os.path.exists(metrics_out_path):
        os.makedirs(metrics_out_path)
            
    classfication = Eval_Classification()
    
    with open("./cls_val.txt","r") as f:
        lines = f.readlines()
    top1, top5, Recall, Precision = evaluteTop1_5(classfication, lines, metrics_out_path)
    print("top-1 accuracy = %.2f%%" % (top1*100))
    print("mean Recall = %.2f%%" % (np.mean(Recall)*100))
    print("mean Precision = %.2f%%" % (np.mean(Precision)*100))
