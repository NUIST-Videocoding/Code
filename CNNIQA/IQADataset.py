
import torch.nn.functional as F
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from torchvision.transforms.transforms import ToPILImage
from PIL import Image
from scipy.signal import convolve2d
import numpy as np
import h5py


def default_loader(path):

    return Image.open(path)

def default_Saliency_loader(path):
    return Image.open(path).convert('L')


def LocalNormalization(patch, P=3, Q=3, C=1):

    kernel = np.ones((P, Q)) / (P * Q)

    patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')

    patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')

    patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C

    patch_ln = torch.from_numpy((patch - patch_mean) / patch_std).float().unsqueeze(0)

    return patch_ln


def NonOverlappingCropPatches(im, patch_size=32, stride=32):
    w, h = im.size

    patches = ()
    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):
            patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))
            for i in range(3):
                patch[i] = LocalNormalization(patch[i].numpy())
            patches = patches + (patch,)

    return patches

def NonOverlappingCropSaliencyPatches(im, patch_size=32, stride=32):
    w, h = im.size

    patches = ()
    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):

            patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))

            patches = patches + (patch,)

    return patches


def Norm(x):
    return (x-torch.min(x)) / (torch.max(x) - torch.min(x))


class IQADataset(Dataset):
    def __init__(self, conf, exp_id=0, status='train', loader=default_loader, saliency_loader=default_Saliency_loader):
        self.loader = loader
        self.saliency_loader = saliency_loader
        im_dir = conf['im_dir']
        self.patch_size = conf['patch_size']
        self.stride = conf['stride']
        datainfo = conf['datainfo']

        Info = h5py.File(datainfo)
        index = Info['index'][:, int(exp_id) % 1000]
        ref_ids = Info['ref_ids'][0, :]
        # ref_ids = ref_ids[0:169]
        # ref_ids = ref_ids[169:344]
        # ref_ids = ref_ids[344:489]
        # ref_ids = ref_ids[489:634]
        # ref_ids = ref_ids[634:779]

        test_ratio = conf['test_ratio']
        train_ratio = conf['train_ratio']
        trainindex = index[:int(train_ratio * len(index))]
        print('trainindex:'+str(trainindex))
        testindex = index[int((1-test_ratio) * len(index)):]
        train_index, val_index, test_index = [],[],[]
        for i in range(len(ref_ids)):
            train_index.append(i) if (ref_ids[i] in trainindex) else \
                test_index.append(i) if (ref_ids[i] in testindex) else \
                    val_index.append(i)
        if status == 'train':
            self.index = train_index
            print("# Train Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(trainindex)
        if status == 'test':
            self.index = test_index
            print("# Test Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(testindex)
        if status == 'val':
            self.index = val_index
            print("# Val Images: {}".format(len(self.index)))

        self.mos = Info['subjective_scores'][0, self.index]
        self.mos_std = Info['subjective_scoresSTD'][0, self.index]

        im_names = [Info[Info['im_names'][0, :][i]].value.tobytes()\
                        [::2].decode() for i in self.index]

        im_names_saliency = [Info[Info['im_names1'][0, :][i]].value.tobytes() \
                         [::2].decode() for i in self.index]
        self.patches_distorted = ()
        self.patchesD_ = ()
        self.patches_saliency = ()
        self.patchesS_ = ()
        self.label = []
        self.label_std = []

        for idx in range(len(self.index)):

            patchesD_ = ()
            patchesS_ = ()

            im = self.loader(os.path.join(im_dir, im_names[idx]))
            im_saliency = self.saliency_loader(os.path.join(im_dir, im_names_saliency[idx]))
            patches_distorted = NonOverlappingCropPatches(im, self.patch_size, self.stride)
            patches_saliency = NonOverlappingCropSaliencyPatches(im_saliency, self.patch_size, self.stride)
            remove_index = []
            for i in range(len(patches_distorted)):
                sal = torch.mean(patches_saliency[i])

                if sal <= 0.1:
                    remove_index.append(i)

            for i in range(len(patches_distorted)):
                if i not in remove_index:

                    patchesD_ = patchesD_ + (patches_distorted[i],)
                    patchesS_ = patchesS_ + (patches_saliency[i],)

            if status == 'train':

                self.patchesD_ = self.patchesD_ + patchesD_
                self.patchesS_ = self.patchesS_ + patchesS_
                self.patches_distorted = self.patchesD_
                self.patches_saliency = self.patchesS_

                for i in range(len(patchesD_)):
                    self.label.append(self.mos[idx])
                    self.label_std.append(self.mos_std[idx])
            else:
                self.patchesD_ = self.patchesD_ + (torch.stack(patchesD_), )
                self.patchesS_ = self.patchesS_ + (torch.stack(patchesS_), )
                self.patches_distorted = self.patchesD_
                self.patches_saliency = self.patchesS_

                self.label.append(self.mos[idx])
                self.label_std.append(self.mos_std[idx])


    def __len__(self):
        return len(self.patches_distorted)

    def __getitem__(self, idx):
        return (self.patchesD_[idx], self.patchesS_[idx]), (torch.Tensor([self.label[idx],]),
                torch.Tensor([self.label_std[idx],]))
