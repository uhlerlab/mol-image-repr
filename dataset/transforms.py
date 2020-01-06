import torch
from torchvision.transforms import ToTensor
from gulpio.transforms import ComposeVideo, RandHorFlipVideo, RandVerFlipVideo, RandomCropVideo, CenterCrop

class VideoToTensor(object):
    '''Converts a list of images to a list of Torch.FloatTensor objects'''
    def __init__(self):
        self.transform = ToTensor()

    def __call__(self, imgs):
        return [self.transform(img) for img in imgs]

def get_train_transform():

    img_transforms = []
    video_transforms = [RandomCropVideo(512), RandHorFlipVideo(), RandVerFlipVideo(), VideoToTensor(),]

    return ComposeVideo(img_transforms=img_transforms, video_transforms=video_transforms)

def get_test_transform():
    
    img_transforms = [CenterCrop(512),]
    video_transforms = [VideoToTensor(),]

    return ComposeVideo(img_transforms=img_transforms, video_transforms=video_transforms)