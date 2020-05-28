import torch
import numpy as np
import os
import cv2
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class OmniLoader(Dataset):
    def __init__(self, root, validation = False):
        self.root = root
        self.class_image_dict = {}
        self.make_image_list()
        
        
        self.l_lim, self.h_lim = 0, 16
        self.char_mult = 16
        if validation:
            self.char_mult = 4
            self.l_lim, self.h_lim = 16, 20
        
    def make_image_list(self):
        alpha_list = os.listdir(self.root)
        class_counter = 0
        
        print("Gettin image list....")
        for alpha in alpha_list:
            char_list = os.listdir(os.path.join(self.root, alpha))
            
            for char in char_list:
                img_list = os.listdir(os.path.join(self.root, alpha, char))
                
                for img in img_list:
                    if class_counter not in self.class_image_dict:
                        self.class_image_dict[class_counter] = []
                    
                    self.class_image_dict[class_counter].append(os.path.join(self.root, alpha, char, img))
                class_counter += 1
        print(class_counter, "classes present")
            
    def preprocess_img(self, img):
        img = np.float32(img/255)
        img = cv2.resize(img, (64, 64))
        img = torch.Tensor(img).unsqueeze(0)
        return img
        
    def __len__(self):
        return len(self.class_image_dict)*self.char_mult
    
    def __getitem__(self, index):
        label1 = np.random.randint(len(self.class_image_dict))
        label2 = label1
        index1, index2 = np.random.randint(self.l_lim, self.h_lim, size = 2)
        
        #Coin flip to figure out whether to feed similar or dissimilar pairs
        take_similar = np.random.randint(0, 2)
        
        sim = 1
        if not take_similar:
            while label2 == label1:
                label2 = np.random.randint(len(self.class_image_dict))
            sim = 0

        img1 = cv2.imread(self.class_image_dict[label1][index1], 0)
        img2 = cv2.imread(self.class_image_dict[label2][index2], 0)
        img1 = self.preprocess_img(img1)
        img2 = self.preprocess_img(img2)

        return img1, img2, np.float32([sim])
    
    def get_oneshot_task(self, ways = 25):
        support_set = []
        query_set = []
        class_labels = []
        
        one_shot_classes = np.random.choice(len(self.class_image_dict), ways, replace=False)
        
        for k, i in enumerate(one_shot_classes):
            index = np.random.randint(0, 20)
            
            img = cv2.imread(self.class_image_dict[i][index], 0)
            img = self.preprocess_img(img).unsqueeze(0)
            support_set.append(img)
            
            for j in range(20):
                if j != index:
                    img = cv2.imread(self.class_image_dict[i][j], 0)
                    img = self.preprocess_img(img).unsqueeze(0)
                    query_set.append(img)
                    
                    class_labels.append(k)
              
        return torch.cat(support_set), torch.cat(query_set), class_labels
            