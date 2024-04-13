import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from nets import get_model_from_name
from utils.utils import (cvtColor, get_classes, letterbox_image,
                         preprocess_input, show_config)


class Classification(object):
    _defaults = {

        "model_path"        : 'model_data/****.pth',
        "classes_path"      : 'model_data/cls_classes.txt',

        "input_shape"       : [224, 224],
        #--------------------------------------------------------------------#
        #   所用模型种类：
        #   mobilenetv2、
        #   resnet18、resnet34、resnet50、resnet101、resnet152
        #   vgg11、vgg13、vgg16、vgg11_bn、vgg13_bn、vgg16_bn、
        #   vit_b_16、
        #   swin_transformer_tiny、swin_transformer_small、swin_transformer_base
        #--------------------------------------------------------------------#
        "backbone"          : 'mobilenetv2',

        "letterbox_image"   : False,

        "cuda"              : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"


    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)


        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.generate()
        
        show_config(**self._defaults)


    def generate(self):

        if self.backbone not in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
            self.model  = get_model_from_name[self.backbone](num_classes = self.num_classes, pretrained = False)
        else:
            self.model  = get_model_from_name[self.backbone](input_shape = self.input_shape, num_classes = self.num_classes, pretrained = False)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.model  = self.model.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()


    def detect_image(self, image):

        image       = cvtColor(image)

        image_data  = letterbox_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)

        image_data  = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

        with torch.no_grad():
            photo   = torch.from_numpy(image_data)
            if self.cuda:
                photo = photo.cuda()

            preds   = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()

        class_name  = self.class_names[np.argmax(preds)]
        probability = np.max(preds)


        plt.subplot(1, 1, 1)
        plt.imshow(np.array(image))
        plt.title('Class:%s Probability:%.3f' %(class_name, probability))
        plt.show()
        return class_name
