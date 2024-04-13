import os
from os import getcwd

from utils.utils import get_classes


classes_path    = 'model_data/cls_classes.txt'

datasets_path   = 'datasets'

sets            = ["train", "val"]
classes, _      = get_classes(classes_path)

if __name__ == "__main__":
    for se in sets:
        list_file = open('cls_' + se + '.txt', 'w')

        datasets_path_t = os.path.join(datasets_path, se)
        types_name      = os.listdir(datasets_path_t)
        for type_name in types_name:
            if type_name not in classes:
                continue
            cls_id = classes.index(type_name)
            
            photos_path = os.path.join(datasets_path_t, type_name)
            photos_name = os.listdir(photos_path)
            for photo_name in photos_name:
                _, postfix = os.path.splitext(photo_name)
                if postfix not in ['.jpg', '.png', '.jpeg']:
                    continue
                list_file.write(str(cls_id) + ";" + '%s'%(os.path.join(photos_path, photo_name)))
                list_file.write('\n')
        list_file.close()

