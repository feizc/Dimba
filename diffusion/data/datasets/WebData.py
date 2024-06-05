import webdataset as wds
import logging
import torch
import random
import math
import json,os,re
import io

from PIL import Image
from torchvision import transforms
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample
from functools import partial

"""
Set hyper-parameter for wds.shuffle
"""

_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

class WebdatasetFilter:
    def __init__(self, min_size=1024, max_pwatermark=0.5):
        self.min_size = min_size
        self.max_pwatermark = max_pwatermark

    def __call__(self, x):
        try:
            if "json" in x:
                x_json = json.loads(x["json"])
                filter_size = (x_json.get("original_width", 0.0) or 0.0) >= self.min_size and x_json.get(
                    "original_height", 0
                ) >= self.min_size
                filter_watermark = (x_json.get("pwatermark", 1.0) or 1.0) <= self.max_pwatermark
                return filter_size and filter_watermark
            else:
                return False
        except Exception:
            return False


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image

def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True

def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample

def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)

    return samples

class wds_process:
    def __init__(self, transform=None):
        if transform == None: 
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        else:
            self.transform = transform
        
    def __call__(self, sample): 
        base64_str = sample['jpg']
        img = Image.open(io.BytesIO(base64_str)).convert("RGB") 
        img = self.transform(img)
        # img.save('1.png')
        json_line = sample['json']
        text = json.loads(json_line)['caption_long']
        return img, text

class Webdataset:
    def __init__(
        self,
        anna_path,
        transform,
        world_size: int,
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        tar_list = os.listdir(anna_path)
        urls = [os.path.join(anna_path, f) for f in tar_list]
        process = wds_process(transform)
        
        self.dataset = wds.DataPipeline(
            wds.SimpleShardList(urls),
            # at this point we have an iterator over all the shards
            wds.shuffle(len(urls)),
            # add wds.split_by_node here if you are using multiple nodes
            wds.split_by_worker,
            wds.split_by_node,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(),
            # this shuffles the samples in memory
            wds.shuffle(1000),
            # this decodes the images and json
            wds.map(process),
            wds.shuffle(1000),
            wds.batched(int(per_gpu_batch_size),)
        ) 

        num_worker_batches = math.ceil(num_train_examples / (global_batch_size * num_workers))  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size

        self.dataloader = wds.WebLoader(
            self.dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        # add meta-data to dataloader instance for convenience
        self.dataloader.num_batches = num_batches
        self.dataloader.num_train_images = num_samples

    @property
    def train_dataset(self):
        return self.dataset

    @property
    def train_dataloader(self):
        return self.dataloader



from PIL import Image
from torch.utils.data import Dataset
class SimpleDataset(Dataset): 
    def __init__(self, path, transform): 
        with open(path, 'r') as f: 
            self.data_list = json.load(f) 
        self.transform = transform 
        print('data size: ', len(self.data_list))

    def __len__(self): 
        return len(self.data_list)

    def __getitem__(self, index): 
        img_path = self.data_list[index]['image']
        img = Image.open(img_path).convert("RGB") 
        img = self.transform(img)
        txt = self.data_list[index]['caption'] 
        return img, txt 
