
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import taming.models.vqgan
import open_clip
import random
from PIL import Image
import torch
import math
import json
import torchvision.transforms as transforms
from glob import glob
import clip
from tqdm import tqdm

torch.manual_seed(0)
np.random.seed(0)

device = "cuda:0"
model, preprocess = clip.load("ViT-L/14@336px", device=device)

class test_custom_dataset(Dataset):
    
    def __init__(self, style: str = None):
        self.empty_context = np.load("assets/contexts/empty_context.npy")
        self.object=[
            "A chihuahua ",
            "A tabby cat ",
            "A portrait of chihuahua ",
            "An apple on the table ",
            "A banana on the table ",
            "A church on the street ",
            "A church in the mountain ",
            "A church in the field ",
            "A church on the beach ",
            "A chihuahua walking on the street ",
            "A tabby cat walking on the street ",
            "A portrait of tabby cat ",
            "An apple on the dish ", 
            "A banana on the dish ", 
            "A human walking on the street ", 
            "A temple on the street ",
            "A temple in the mountain ",
            "A temple in the field ",
            "A temple on the beach ",
            "A chihuahua walking in the forest ",
            "A tabby cat walking in the forest ",
            "A portrait of human face ",
            "An apple on the ground ",
            "A banana on the ground ",
            "A human walking in the forest ",
            "A cabin on the street ",
            "A cabin in the mountain ",
            "A cabin in the field ",
            "A cabin on the beach ",
        ]
        self.style = [
            "in 3d rendering style",
        ]
        if style is not None:
            self.style = [style]
        
    def __getitem__(self, index):
        prompt = self.object[index]+self.style[0]

        return prompt, prompt
    
    def __len__(self):
        return len(self.object)
    
    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v.clamp_(0., 1.)
        return v
    
    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_cc3m_val.npz'
    
    
class train_custom_dataset(Dataset):
    
    def __init__(self, transform = None, train_file: str=None, ):
        
        self.train_img = json.load(open(train_file, 'r'))
        self.path_preffix = "/".join(train_file.split("/")[:-1])
        self.prompt = []
        self.image = []
        self.style = []
        for im in self.train_img.keys():
            im_path = os.path.join(self.path_preffix, im)
            self.object = self.train_img[im][0]
            self.style = self.train_img[im][1]
            im_prompt = self.object +" "+self.style
            self.image.append(im_path)
            self.prompt.append(im_prompt)
        self.empty_context = np.load("assets/contexts/empty_context.npy")
        
        # if transform is None:
            
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
            
        # if transform is not None:
        #     print("int")
        #     self.transfrom = transform
        print("-----------------"*3)
        print("train dataset length: ", len(self.prompt))
        print("train dataset length: ", len(self.image))
        print(self.prompt[0])
        print(self.image[0])
        print("-----------------"*3)
    def __getitem__(self, index):
        prompt = self.prompt[0]
        image = Image.open(self.image[0]).convert("RGB")
        image = self.transform(image)
        
        return image,prompt
        # return dict(img=image_embedding, text=text_embedding)
    
    def __len__(self):
        return 24
    
    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v.clamp_(0., 1.)
        return v
    
    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_cc3m_val.npz'
    
class train_it_custom_dataset(Dataset):
    
    def __init__(self, transform = None, train_file: str=None, it_train_file: str=None, filter_sample = False):
        
        self.train_img = json.load(open(train_file, 'r'))
        # self.path_preffix = "/".join(train_file.split("/")[:-1])
        self.prompt = []
        image = glob(os.path.join(it_train_file, "*.png"))
        if filter_sample:
            self.image = filter_images(it_train_file, image)
        else:
            self.image = image
        self.style = list(self.train_img.values())[0][1]
        for im in self.image:
            self.prompt.append(os.path.basename(im).split(".")[0])
        # self.style = []
        # for im in self.train_img.keys():
        #     im_path = os.path.join(self.path_preffix, im)
        #     self.object = self.train_img[im][0]
        #     self.style = self.train_img[im][1]
        #     im_prompt = self.object +" "+self.style
        #     self.image.append(im_path)
        #     self.prompt.append(im_prompt)
        self.empty_context = np.load("assets/contexts/empty_context.npy")
        
        # if transform is None:
            
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
            
        # if transform is not None:
        #     print("int")
        #     self.transfrom = transform
        print("-----------------"*3)
        print("train dataset length: ", len(self.prompt))
        print("train dataset length: ", len(self.image))
        print(self.prompt[0])
        print(self.image[0])
        print("-----------------"*3)
    def __getitem__(self, index):
        prompt = self.prompt[index]
        image = Image.open(self.image[index]).convert("RGB")
        image = self.transform(image)
        
        return image,prompt
        # return dict(img=image_embedding, text=text_embedding)
    
    def __len__(self):
        return len(self.prompt)
    
    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v.clamp_(0., 1.)
        return v
    
    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_cc3m_val.npz'
    
        
    
    
    
class  Discriptor(Dataset):
    def __init__(self,style: str=None):
        self.object =[
            # "A parrot ",
            # "A bird ",
            # "A chihuahua in the snow",
            # "A towel ",
            # "A number '1' ",
            # "A number '2' ",
            # "A number '3' ",
            # "A number '6' ",
            # "A letter 'L' ",
            # "A letter 'Z' ",
            # "A letter 'D' ",
            # "A rabbit ",
            # "A train ",
            # "A table ",
            # "A dish ",
            # "A large boat ",
            # "A puppy ",
            # "A cup ",
            # "A watermelon ",
            # "An apple ",
            # "A banana ",
            # "A chair ",
            # "A Welsh Corgi ",
            # "A cat ",
            # "A house ",
            # "A flower ",
            # "A sunflower ",
            # "A car ",
            # "A jeep car ",
            # "A truck ",
            # "A Posche car ",
            # "A vase ",
            "A chihuahua ",
            "A tabby cat ",
            "A portrait of chihuahua ",
            "An apple on the table ",
            "A banana on the table ",
            "A human ",
            "A church on the street ",
            "A church in the mountain ",
            "A church in the field ",
            "A church on the beach ",
            "A chihuahua walking on the street ",
            "A tabby cat walking on the street ",
            "A portrait of tabby cat ",
            "An apple on the dish ", 
            "A banana on the dish ", 
            "A human walking on the street ", 
            "A temple on the street ",
            "A temple in the mountain ",
            "A temple in the field ",
            "A temple on the beach ",
            "A chihuahua walking in the forest ",
            "A tabby cat walking in the forest ",
            "A portrait of human face ",
            "An apple on the ground ",
            "A banana on the ground ",
            "A human walking in the forest ",
            "A cabin on the street ",
            "A cabin in the mountain ",
            "A cabin in the field ",
            "A cabin on the beach ",
            # "A letter 'A' ",
            # "A letter 'B' ",
            # "A letter 'C' ",
            # "A letter 'D' ",
            # "A letter 'E' ",
            # "A letter 'F' ",
            # "A letter 'G' ",
            # "A butterfly ",
            # " A baby penguin ",
            # "A bench ",
            # "A boat ",
            # "A cow ",
            # "A hat ",
            # "A piano ",
            # "A robot ",
            # "A christmas tree ",
            # "A dog ",
            # "A moose ",
        ]
        
        self.style =[
            "in 3d rendering style",
        ]
        if style is not None:
            self.style = [style]
        
    def __getitem__(self, index):
        prompt = self.object[index]+self.style[0]
        return prompt
    
    def __len__(self):
        return len(self.object)
    
    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v.clamp_(0., 1.)
        return v
    
    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_cc3m_val.npz'
    



class CLIPDataset(Dataset):
    def __init__(self, img_dir, transform = None, tokenizer = None):
        super().__init__()
        self.transform = transform
        self.tokenizer = tokenizer
        self.img_list = glob(os.path.join(img_dir, "*.png"))
        style_ref = self.img_list[0].split("/")[-3]
        data_dir = "./data/"
        style_ref_path = os.path.join(data_dir, style_ref+".jpg")
        self.ref_img = self._load_img(style_ref_path)
                             
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        prompt = os.path.basename(img_path).split(".")[0]
        img = self._load_img(img_path)
        text = self._load_txt(prompt)
        
        sample = dict(image=img, text=text, style=self.ref_img)
        return sample
        # return img, text, self.ref_img

    def _load_img(self, path):
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def _load_txt(self, data):
        if self.tokenizer is not None:
            data = self.tokenizer(data).squeeze()
        return data
    
@torch.no_grad()
def calculate_clip_score(dataloader, model, device):
    # text_score_acc = 0.
    style_score_acc = 0.
    sample_num = 0.
    logit_scale = model.logit_scale.exp()
    
    img_score_list = []
    for idx, batch_data in tqdm(enumerate(dataloader)):
        img = batch_data['image']
        img_features = model.encode_image(img.to(device))
        # text = batch_data['text']
        # text_features = model.encode_text(text.to(device))
        style = batch_data['style']
        style_features = model.encode_image(style.to(device))
        
        # normalize features
        img_features = clip_normalize(img_features)
        # text_features = clip_normalize(text_features)
        style_features = clip_normalize(style_features)
        
        # calculate scores
        # score = logit_scale * real_features @ fake_features.t()
        # score_acc += torch.diag(score).sum()
            
#         text_score = logit_scale * (img_features * text_features).sum()
#         text_score_acc += text_score
        style_score = logit_scale * (img_features * style_features).sum()
        style_score_acc += style_score
        sample_num += img.shape[0]
        img_score_list.append(style_score.cpu().item())
    
    average_score = style_score_acc / sample_num
    average_score_value = average_score.cpu().item()
    # print(img_score_list)
    # print(average_score_value)
    result_idx_list = [idx for idx, x in enumerate(img_score_list) if x > average_score_value]
    
    return result_idx_list
def clip_normalize(features):
    features = features / features.norm(dim=1, keepdim=True).to(torch.float32)
    return features

def filter_images(img_dir, img_list):
    # img_dir = "results/oriental_egret/HF_it_data_5/"
    style_name = img_dir.split("/")[-2]
    batch_size = 1
    dataset = CLIPDataset(img_dir = img_dir, transform=preprocess, tokenizer=clip.tokenize)
    dataloader = DataLoader(dataset, batch_size, pin_memory=True)

    result_idx_list = calculate_clip_score(dataloader, model, device)
    
    result_img_list = [img_list[x] for x in result_idx_list]
    
    return result_img_list
