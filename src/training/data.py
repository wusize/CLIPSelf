import logging
import os
import random
from dataclasses import dataclass
from multiprocessing import Value
import numpy as np
from training.utils import mask2box
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from open_clip.transform import get_scale
from pycocotools.coco import COCO
from training.coco_api import COCOPanoptic
from panopticapi import utils
# import mmcv
import io
# from mmengine.fileio import get
try:
    from petrel_client.client import Client
except:
    Client = None
from open_clip.transform import ResizeLongest


try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

# import image transforms
from torchvision.transforms import RandomHorizontalFlip, Compose
from training.custom_transforms import CustomRandomResize, CustomRandomCrop


class ProposalDistillDataset(Dataset):
    def __init__(self, input_filename, transforms, image_root,
                 crop_size=224,
                 tokenizer=None, args=None):
        logging.debug(f'Loading coco style data from {input_filename}.')
        self.coco = COCO(input_filename)
        logging.debug('Done loading data.')
        self.transforms = transforms
        self.tokenize = tokenizer
        self.image_root = image_root
        self.image_ids = list(self.coco.imgs.keys())
        self.max_anns = 20
        if not isinstance(crop_size, (tuple, list)):
            crop_size = [crop_size, crop_size]
        self.crop_size = crop_size
        self.args = args

        self.min_size = args.min_size
        self.max_size = args.max_size

        self.ceph_root = args.train_ceph_root
        self.use_ceph = (self.ceph_root != "")
        self.FILE_CLIENT = None

    def read_image(self, image_name):
        if self.use_ceph:
            image_path = os.path.join(self.ceph_root, image_name)
            if self.FILE_CLIENT is None:
                self.FILE_CLIENT = Client()
            try:
                img_bytes = self.FILE_CLIENT.get(image_path)
                buff = io.BytesIO(img_bytes)
                image = Image.open(buff)
            except:
                print(f"Cannot load {image_path}", flush=True)
                return None
        else:
            image_path = os.path.join(self.image_root, image_name)
            try:
                image = Image.open(image_path)
            except:
                print(f"Cannot load {image_path}", flush=True)
                return None

        width, height = image.size
        if width < 10 or height < 10:
            print(f"Invalid image, size {image.size}", flush=True)
            return None

        return image

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]
        if 'file_name' in image_info:
            image_name = image_info['file_name']
        else:
            assert 'coco_url' in image_info
            coco_url = image_info['coco_url'].split('/')
            image_name = os.path.join(coco_url[-2], coco_url[-1])

        old_image = self.read_image(image_name)
        if old_image is None:
            next_id = random.choice(range(self.__len__()))
            return self.__getitem__(next_id)
        img_w, img_h = old_image.width, old_image.height
        new_image = self.transforms[0](old_image)

        scale = get_scale(old_image, new_image)
        anns = self.coco.imgToAnns[image_id]
        boxes_template = torch.zeros(self.max_anns, 4 + 1)    # xyxy s
        image_crops = torch.zeros(self.max_anns, 3, *self.crop_size)

        indices = list(range(len(anns)))
        random.shuffle(indices)
        num_valid_boxes = 0
        for i, ann_id in enumerate(indices[:self.max_anns]):
            ann = anns[ann_id]
            x, y, w, h = ann['bbox']
            if w*h < (self.min_size ** 2) or w*h > (self.max_size ** 2):
                continue
            num_valid_boxes += 1
            cx, cy = x + w*0.5, y + h*0.5
            x0, y0, x1, y1 = \
                max(cx - w*0.75, 0), max(cy - h*0.75, 0), min(cx + w*0.75, img_w), min(cy + h*0.75, img_h)
            image_crops[i] = self.transforms[1](old_image.crop((x0, y0, x1, y1)))   # image crops
            box_info = torch.tensor([x, y, x + w, y + h, 1.0])    # x, y, x + w, y + h
            boxes_template[i] = box_info

        if num_valid_boxes == 0:
            boxes_template[0] = torch.tensor([0, 0, img_w / 4, img_h / 4, 1.0])    # avoid empty
            image_crops[0] = self.transforms[1](old_image.crop((0, 0, img_w // 4, img_h // 4)))

        _, h, w = new_image.shape

        boxes_template[:, :4] *= scale
        boxes_template[:, [0, 2]] /= w
        boxes_template[:, [1, 3]] /= h

        return new_image, boxes_template, image_crops


class GridDistillDataset(Dataset):
    def __init__(self,
                 input_filename, transforms, image_root,
                 max_split=16,
                 crop_size=224,
                 pre_transforms=False,
                 ceph_root="", args=None):
        self._init_choices(max_split)
        logging.debug(f'Loading coco caption style data from {input_filename}.')
        self.coco = COCO(input_filename)
        logging.debug('Done loading data.')
        self.transforms = transforms
        self.image_root = image_root
        self.args = args
        image_ids = list(self.coco.imgs.keys())
        train_ratio = args.train_ratio
        if train_ratio < 1.0:
            num_images = int(len(image_ids) * train_ratio)
            random.shuffle(image_ids)
            image_ids = image_ids[:num_images]
        self.image_ids = image_ids
        self.max_anns = args.max_boxes
        if not isinstance(crop_size, (tuple, list)):
            crop_size = [crop_size, crop_size]
        self.crop_size = crop_size
        self._init_boxes()
        self.ceph_root = ceph_root
        self.use_ceph = (ceph_root != "")
        self.FILE_CLIENT = None
        if pre_transforms:
            self.pre_transforms = Compose([
                CustomRandomResize(scale=(0.5, 2.0)),
                CustomRandomCrop(size=self.transforms[0].transforms[0].max_size),
                RandomHorizontalFlip()])
        else:
            self.pre_transforms = None

    def read_image(self, image_name):
        if self.use_ceph:
            image_path = os.path.join(self.ceph_root, image_name)
            if self.FILE_CLIENT is None:
                self.FILE_CLIENT = Client()
            try:
                img_bytes = self.FILE_CLIENT.get(image_path)
                buff = io.BytesIO(img_bytes)
                image = Image.open(buff)
            except:
                print(f"Cannot load {image_path}", flush=True)
                return None
        else:
            image_path = os.path.join(self.image_root, image_name)
            try:
                image = Image.open(image_path)
            except:
                print(f"Cannot load {image_path}", flush=True)
                return None

        width, height = image.size
        if width < 10 or height < 10:
            print(f"Invalid image, size {image.size}", flush=True)
            return None

        return image


    def _init_choices(self, M=16):
        choices = []
        for m in range(1, M+1):
            for n in range((m + 1)//2, min(m*2 + 1, M+1)):
                choices.append((m, n))
        self.choices = choices

    def __len__(self):
        return len(self.image_ids)

    def _init_boxes(self, ):
        box_templates = {}
        for choice in self.choices:
            M, N = choice
            grid_x, grid_y = torch.meshgrid(torch.linspace(0, 1, N + 1), torch.linspace(0, 1, M + 1),
                                            indexing='xy')
            x0y0s = torch.stack([grid_x[:M, :N], grid_y[:M, :N]], dim=-1)
            x1y1s = torch.stack([grid_x[1:, 1:], grid_y[1:, 1:]], dim=-1)
            pseudo_boxes = torch.cat([x0y0s, x1y1s],
                                     dim=-1).view(-1, 4)

            assert pseudo_boxes.shape[0] == M*N
            box_templates[choice] = pseudo_boxes

        self.box_templates = box_templates

    def _obtain_image_crops(self, image, choice):
        image_crops = []
        img_w, img_h = image.size
        normed_boxes = self.box_templates[choice]
        indices = list(range(len(normed_boxes)))
        random.shuffle(indices)
        indices = indices[:self.max_anns]
        boxes = normed_boxes * torch.tensor([img_w, img_h, img_w, img_h])
        for idx in indices:
            box = boxes[idx]
            x0, y0, x1, y1 = box.tolist()    # todo expand
            if self.args.crop_scale > 1.0:
                box_w, box_h = x1 - x0, y1 - y0
                cx, cy = (x1 + x0)/2, (y1 + y0)/2
                delta_factor = 0.5 * self.args.crop_scale
                x0, y0, x1, y1 = max(cx - box_w * delta_factor, 0), max(cy - box_h * delta_factor, 0), \
                    min(cx + box_w * delta_factor, img_w), min(cy + box_h * delta_factor, img_h)
            image_crops.append(self.transforms[1](image.crop((x0, y0, x1, y1))))

        return torch.stack(image_crops), boxes[indices]

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]
        if 'file_name' in image_info:
            image_name = image_info['file_name']
        else:
            assert 'coco_url' in image_info
            coco_url = image_info['coco_url'].split('/')
            image_name = os.path.join(coco_url[-2], coco_url[-1])
        # image_path = os.path.join(self.image_root, image_name)
        # old_image = Image.open(image_path)
        old_image = self.read_image(image_name)
        if old_image is None:
            next_id = random.choice(range(self.__len__()))
            return self.__getitem__(next_id)
        new_image = self.transforms[0](old_image)

        scale = get_scale(old_image, new_image)
        boxes_template = torch.zeros(self.max_anns, 4 + 1)    # xyxy s
        image_crops_template = torch.zeros(self.max_anns, 3, *self.crop_size)
        image_crops, boxes = self._obtain_image_crops(old_image,
                                                      random.choice(self.choices))
        assert image_crops.shape[0] == boxes.shape[0]
        _, h, w = new_image.shape

        boxes[:, :4] *= scale
        boxes[:, [0, 2]] /= w
        boxes[:, [1, 3]] /= h

        boxes_template[:boxes.shape[0], :4] = boxes
        boxes_template[:boxes.shape[0], 4] = 1.0

        image_crops_template[:boxes.shape[0]] = image_crops

        return new_image, boxes_template, image_crops_template


class COCOPanopticDataset(Dataset):
    def __init__(self, input_filename, transforms, image_root, embed_path,
                 segm_root,
                 crop_size=224,
                 tokenizer=None,
                 downsample_factor=16,
                 min_size=8, max_size=1024):
        logging.debug(f'Loading coco caption style data from {input_filename}.')
        self.coco = COCOPanoptic(input_filename)
        logging.debug('Done loading data.')
        self.transforms = transforms
        self.tokenize = tokenizer
        self.image_root = image_root
        self.embeddings = np.load(embed_path)
        self.image_ids = list(self.coco.imgs.keys())
        num_annos = [len(anns) for anns in self.coco.imgToAnns.values()]
        self.max_anns = min(max(num_annos), 100)
        if not isinstance(crop_size, (tuple, list)):
            crop_size = [crop_size, crop_size]
        self.crop_size = crop_size
        self.min_size = 8  # fix for val
        self.max_size = 1024
        self.segm_root = segm_root
        self.downsample_factor = downsample_factor
        self.segm_transform = ResizeLongest(max_size=self.transforms[0].transforms[0].max_size // downsample_factor,
                                            fill=0)       # downsample to the output size of image encoder

        cat_ids = sorted([cat['id'] for cat in self.coco.cats.values()])

        self.cat_id2label = {cat_id: label for label, cat_id in enumerate(cat_ids)}

    def __len__(self):
        return len(self.image_ids)

    @staticmethod
    def _load_segm(segm_path):
        segmentation = np.array(
            Image.open(segm_path),
            dtype=np.uint8
        )
        # img_bytes = get(segm_path)
        # pan_png = mmcv.imfrombytes(
        #     img_bytes, flag='color', channel_order='rgb').squeeze()
        segm_map = utils.rgb2id(segmentation)

        return segm_map

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]
        image_name = image_info['file_name']
        segm_file = image_info['segm_file']
        image_path = os.path.join(self.image_root, image_name)
        segm_path = os.path.join(self.segm_root, segm_file)
        segm_map = self._load_segm(segm_path)

        old_image = Image.open(image_path)
        img_w, img_h = old_image.width, old_image.height
        new_image = self.transforms[0](old_image)

        scale = get_scale(old_image, new_image)
        anns = self.coco.imgToAnns[image_id]
        boxes_template = torch.zeros(self.max_anns, 4 + 2 + 1 + 1)    # xyxy c valid size, isthing
        image_crops = torch.zeros(self.max_anns, 3, *self.crop_size)
        gt_masks = torch.zeros(self.max_anns, self.segm_transform.max_size,
                               self.segm_transform.max_size)
        masked_image_crops = torch.zeros(self.max_anns, 3, *self.crop_size)

        for i, ann in enumerate(anns):
            if i == self.max_anns:
                break
            cat_id = ann['category_id']
            is_thing = self.coco.cats[cat_id]['isthing']
            if is_thing > 0:
                x, y, w, h = ann['bbox']
                cx, cy = x + w*0.5, y + h*0.5
                x0, y0, x1, y1 = \
                    max(cx - w*0.75, 0), max(cy - h*0.75, 0), min(cx + w*0.75, img_w), min(cy + h*0.75, img_h)
            else:
                x0, y0, x1, y1 = mask2box(segm_map == ann['id'])
                x, y, w, h = x0, y0, x1 - x0, y1 - y0
            if w * h < (self.min_size ** 2) or w * h > (self.max_size ** 2):
                continue
            image_crops[i] = self.transforms[1](old_image.crop((x0, y0, x1, y1)))   # image crops
            # masked image crop
            np_old_image = np.asarray(old_image.copy())
            np_old_image[segm_map != ann['id']] = 114
            masked_old_image = Image.fromarray(np_old_image)
            masked_image_crops[i] = self.transforms[1](masked_old_image.crop((x0, y0, x1, y1)))   # image crops

            gt_mask = torch.from_numpy(segm_map == ann['id']).float()
            gt_mask = self.segm_transform(gt_mask[None]) > 0.0
            cls_label = self.cat_id2label[cat_id]
            box_info = torch.tensor([x, y, x + w, y + h, cls_label, 1.0, w * h, is_thing])    # x, y, x + w, y + h
            boxes_template[i] = box_info
            gt_masks[i] = gt_mask[0]

        _, h, w = new_image.shape

        boxes_template[:, :4] *= scale
        boxes_template[:, [0, 2]] /= w
        boxes_template[:, [1, 3]] /= h

        return new_image, boxes_template, image_crops, gt_masks, masked_image_crops


class COCORegionCLIPDataset(Dataset):
    def __init__(self, input_filename, transforms, image_root, args):
        logging.debug(f'Loading coco caption style data from {input_filename}.')
        self.coco = COCO(input_filename)
        logging.debug('Done loading data.')
        self.transforms = transforms
        self.image_root = image_root
        image_ids = list(self.coco.imgToAnns.keys())    # only use images that have anns
        train_ratio = args.train_ratio
        if train_ratio < 1.0:
            num_images = int(len(image_ids) * train_ratio)
            random.shuffle(image_ids)
            image_ids = image_ids[:num_images]
        self.image_ids = image_ids

        num_annos = [len(anns) for anns in self.coco.imgToAnns.values()]
        self.max_anns = min(max(num_annos), 20)
        self.args = args
        self.ceph_root = args.train_ceph_root
        self.use_ceph = (self.ceph_root != "")
        self.FILE_CLIENT = None
        cat_ids = sorted([cat['id'] for cat in self.coco.cats.values()])

        self.cat_id2label = {cat_id: label for label, cat_id in enumerate(cat_ids)}

    def __len__(self):
        return len(self.image_ids)

    def read_image(self, image_name):
        if self.use_ceph:
            image_path = os.path.join(self.ceph_root, image_name)
            if self.FILE_CLIENT is None:
                self.FILE_CLIENT = Client()
            img_bytes = self.FILE_CLIENT.get(image_path)
            buff = io.BytesIO(img_bytes)
            image = Image.open(buff)
        else:
            image_path = os.path.join(self.image_root, image_name)
            image = Image.open(image_path)
        return image

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]
        image_name = image_info['file_name']
        # image_path = os.path.join(self.image_root, image_name)
        # old_image = Image.open(image_path)
        old_image = self.read_image(image_name)
        new_image = self.transforms[0](old_image)

        scale = get_scale(old_image, new_image)
        anns = self.coco.imgToAnns[image_id]
        boxes_template = torch.zeros(self.max_anns, 4 + 2)    # xyxy cls valid

        for i, ann in enumerate(anns):
            if i == self.max_anns:
                break
            cat_id = ann['category_id']
            x, y, w, h = ann['bbox']
            cls_label = self.cat_id2label[cat_id]
            box_info = torch.tensor([x, y, x + w, y + h, cls_label, 1.0])    # x, y, x + w, y + h
            boxes_template[i] = box_info

        _, h, w = new_image.shape

        boxes_template[:, :4] *= scale
        boxes_template[:, [0, 2]] /= w
        boxes_template[:, [1, 3]] /= h

        return new_image, boxes_template


def get_coco_panoptic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = COCOPanopticDataset(
        input_filename,
        preprocess_fn,
        segm_root=args.val_segm_root,
        image_root=args.val_image_root,
        embed_path=args.embed_path,
        tokenizer=tokenizer,
        crop_size=args.input_size,
        min_size=args.min_size,
        max_size=args.max_size,
        downsample_factor=args.downsample_factor
    )
    num_samples = len(dataset)
    # TODO: distributed for test
    sampler = DistributedSampler(dataset) if args.distributed else None  #  and is_train else None
    shuffle = is_train and sampler is None
    if is_train:
        batch_size = args.batch_size
    else:
        batch_size = min(args.batch_size, 1)     # only support bs = 1 for inference
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_proposal_distill_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    assert is_train
    input_filename = args.train_data # if is_train else args.val_data
    assert input_filename
    dataset = ProposalDistillDataset(
        input_filename,
        preprocess_fn,
        image_root=args.train_image_root,
        tokenizer=tokenizer,
        crop_size=args.input_size,
        args=args
    )
    num_samples = len(dataset)
    # TODO: distributed for test
    sampler = DistributedSampler(dataset) if args.distributed else None  #  and is_train else None
    shuffle = is_train and sampler is None
    batch_size = args.batch_size
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_grid_distill_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    assert is_train
    input_filename = args.train_data
    assert input_filename
    datasetname = GridDistillDatasetWithImageNet if args.imagenet_length > 0 else GridDistillDataset
    dataset = datasetname(
        input_filename=input_filename,
        transforms=preprocess_fn,
        image_root=args.train_image_root,
        crop_size=args.input_size,
        max_split=args.max_split,
        ceph_root=args.train_ceph_root,
        pre_transforms=args.pre_transforms,
        args=args
    )
    num_samples = len(dataset)
    # TODO: distributed for test
    sampler = DistributedSampler(dataset) if args.distributed else None  #  and is_train else None
    shuffle = is_train and sampler is None
    batch_size = args.batch_size
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_region_clip_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    assert is_train
    input_filename = args.train_data
    assert input_filename
    dataset = COCORegionCLIPDataset(
        input_filename=input_filename,
        transforms=preprocess_fn,
        image_root=args.train_image_root,
        args=args,
    )
    num_samples = len(dataset)
    # TODO: distributed for test
    sampler = DistributedSampler(dataset) if args.distributed else None  #  and is_train else None
    shuffle = is_train and sampler is None
    batch_size = args.batch_size
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)



class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == 'coco_panoptic':
        return get_coco_panoptic_dataset
    elif dataset_type == 'proposals_distill':
        return get_proposal_distill_dataset
    elif dataset_type == 'grid_distill':
        return get_grid_distill_dataset
    elif dataset_type == 'region_clip':
        return get_region_clip_dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data:
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, dataset_type=args.test_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    return data
