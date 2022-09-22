import ast
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from multiprocessing import Value

import braceexpand
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop

from srt.srt.utils.nerf import get_extrinsic, transform_points



try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import tokenize

class MsnDataset(IterableDataset):
    def __init__(self, path, num_samples_in_clip_data, points_per_item=8192 * 3, canonical_view=True,
                 full_scale=False):
        super(MsnDataset).__init__()
        self.num_target_pixels = points_per_item
        self.msn_path = path
        self.canonical = canonical_view
        self.full_scale = full_scale
        mean = (0.48145466, 0.4578275, 0.40821073)  # OpenAI dataset mean
        std = (0.26862954, 0.26130258, 0.27577711)  # OpenAI dataset std
        clip_image_size = 224
        self.resize = Resize(clip_image_size)
        self.normalize = Normalize(mean=mean, std=std)
        self.num_scenes = 100000
        self.num_samples_in_clip_data = num_samples_in_clip_data
    
    def __iter__(self):
        for i in range(self.num_samples_in_clip_data):
            yield self.prep_item(i % self.num_scenes)

    def prep_item(self, msn_pickle_number):
        #get MSN data
        msn_pickle_filename = "MSN_" + str(msn_pickle_number) + ".pkl"
        msn_pickle_path = os.path.join(self.msn_path,msn_pickle_filename)
        scenes = pd.read_pickle(msn_pickle_path)
        msn_idx = 0
        data = scenes[msn_idx]


        #arrange MSN data
        input_views = np.random.choice(np.arange(10), size=1, replace=False)
        target_views = np.array(list(set(range(10)) - set(input_views)))


        #get input data
        input_images = np.transpose(data['color_image'][input_views], (0, 3, 1, 2))
        input_rays = data['ray_directions'][input_views]
        input_camera_pos = data['ray_origins'][input_views][:, 0, 0]

        if self.canonical:
            canonical_extrinsic = get_extrinsic(input_camera_pos[0], input_rays[0]).astype(np.float32)
            input_rays = transform_points(input_rays, canonical_extrinsic, translate=False)
            nput_camera_pos = transform_points(input_camera_pos, canonical_extrinsic)

        target_pixels = np.reshape(data['color_image'][target_views], (-1, 3))
        target_rays = np.reshape(data['ray_directions'][target_views], (-1, 3))
        target_camera_pos = np.reshape(data['ray_origins'][target_views], (-1, 3))

        num_pixels = target_pixels.shape[0]

        if not self.full_scale:
            sampled_idxs = np.random.choice(np.arange(num_pixels),
                                            size=(self.num_target_pixels,),
                                            replace=False)

            target_pixels = target_pixels[sampled_idxs]
            target_rays = target_rays[sampled_idxs]
            target_camera_pos = target_camera_pos[sampled_idxs]

        if self.canonical:
            target_rays = transform_points(target_rays, canonical_extrinsic, translate=False)
            target_camera_pos = transform_points(target_camera_pos, canonical_extrinsic)


        input_images = torch.from_numpy(input_images)
        input_camera_pos = torch.from_numpy(input_camera_pos)
        input_rays = torch.from_numpy(input_rays)
        target_pixels = torch.from_numpy(target_pixels)
        target_camera_pos = torch.from_numpy(target_camera_pos)
        target_rays = torch.from_numpy(target_rays)
            

        return input_images, input_camera_pos,\
                 input_rays, target_pixels, target_camera_pos, \
                     target_rays


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t"):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        try:
            images = self.transforms(Image.open(str(self.images[idx])))
            texts = tokenize([str(self.captions[idx])])[0]
        except:
            idx += 1
            images = self.transforms(Image.open(str(self.images[idx])))
            texts = tokenize([str(self.captions[idx])])[0]
        return images, texts


class CsvMsnDataset(Dataset):
    def __init__(self, input_filename, msn_path, mode, transforms, img_key, caption_key, sep="\t", points_per_item=8192, canonical_view=True,
                 full_scale=False):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        
        self.msn_path = msn_path
        self.mode = mode
        self.msn_scenes_number = 100000
        self.scenes_per_pickle = 100
        self.num_target_pixels = points_per_item
        self.canonical = canonical_view
        self.full_scale = full_scale
        mean = (0.48145466, 0.4578275, 0.40821073)  # OpenAI dataset mean
        std = (0.26862954, 0.26130258, 0.27577711)  # OpenAI dataset std
        clip_image_size = 224
        self.resize = Resize(clip_image_size)
        self.normalize = Normalize(mean=mean, std=std)
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        if self.mode == "val":
            try:
                images = self.transforms(Image.open(str(self.images[idx])))
                texts = tokenize([str(self.captions[idx])])[0]
            except:
                idx += 1
                images = self.transforms(Image.open(str(self.images[idx])))
                texts = tokenize([str(self.captions[idx])])[0]
            return images, texts
        else:
            #CLIP original Data
            try:
                images = self.transforms(Image.open(str(self.images[idx])))
                texts = tokenize([str(self.captions[idx])])[0]
            except:
                idx += 1
                images = self.transforms(Image.open(str(self.images[idx])))
                texts = tokenize([str(self.captions[idx])])[0]


            #get MSN data
            msn_idx = idx % self.msn_scenes_number
            msn_pickle_number = int(msn_idx/self.scenes_per_pickle)
            msn_pickle_filename = "MSN_" + str(msn_pickle_number) + ".pkl"
            msn_pickle_path = os.path.join(self.msn_path,msn_pickle_filename)
            scenes = pd.read_pickle(msn_pickle_path)
            msn_idx = 0
            data = scenes[msn_idx]


            #arrange MSN data
            input_views = np.random.choice(np.arange(10), size=1, replace=False)
            target_views = np.array(list(set(range(10)) - set(input_views)))


            #transform images from unit8 to float
            data['color_image'] = data['color_image'].astype(np.float32) / 255.

            #resize images to fit CLIP size and normalize
            color_images = data['color_image']
            color_images = torch.from_numpy(color_images)
            color_images = torch.permute(color_images,(0,3,1,2))
            color_images = self.resize(color_images)
            color_images = self.normalize(color_images)
            color_images = torch.permute(color_images,(0,2,3,1))
            data['color_image'] = color_images.numpy()

            #resize ray directions
            ray_directions = data['ray_directions']
            ray_directions = torch.from_numpy(ray_directions)
            ray_directions = torch.permute(ray_directions,(0,3,1,2))
            ray_directions = self.resize(ray_directions)
            ray_directions = torch.permute(ray_directions,(0,2,3,1))
            data['ray_directions'] = ray_directions.numpy()

            #resize ray origins
            ray_origins = data['ray_origins']
            ray_origins = torch.from_numpy(ray_origins)
            ray_origins = torch.permute(ray_origins,(0,3,1,2))
            ray_origins = self.resize(ray_origins)
            ray_origins = torch.permute(ray_origins,(0,2,3,1))
            data['ray_origins'] = ray_origins.numpy()

            #get input data
            input_images = np.transpose(data['color_image'][input_views], (0, 3, 1, 2))
            input_rays = data['ray_directions'][input_views]
            input_camera_pos = data['ray_origins'][input_views][:, 0, 0]

            if self.canonical:
                canonical_extrinsic = get_extrinsic(input_camera_pos[0], input_rays[0]).astype(np.float32)
                input_rays = transform_points(input_rays, canonical_extrinsic, translate=False)
                input_camera_pos = transform_points(input_camera_pos, canonical_extrinsic)

            target_pixels = np.reshape(data['color_image'][target_views], (-1, 3))
            target_rays = np.reshape(data['ray_directions'][target_views], (-1, 3))
            target_camera_pos = np.reshape(data['ray_origins'][target_views], (-1, 3))

            num_pixels = target_pixels.shape[0]

            if not self.full_scale:
                sampled_idxs = np.random.choice(np.arange(num_pixels),
                                            size=(self.num_target_pixels,),
                                            replace=False)

                target_pixels = target_pixels[sampled_idxs]
                target_rays = target_rays[sampled_idxs]
                target_camera_pos = target_camera_pos[sampled_idxs]

            if self.canonical:
                target_rays = transform_points(target_rays, canonical_extrinsic, translate=False)
                target_camera_pos = transform_points(target_camera_pos, canonical_extrinsic)

            sceneid = int(data['scene_name'][6:])

            input_images = torch.from_numpy(input_images)#.clone()
            input_camera_pos = torch.from_numpy(input_camera_pos)#.clone()
            input_rays = torch.from_numpy(input_rays)#.clone()
            target_pixels = torch.from_numpy(target_pixels)#.clone()
            target_camera_pos = torch.from_numpy(target_camera_pos)#.clone()
            target_rays = torch.from_numpy(target_rays)#.clone()
            

            return images, texts, input_images, input_camera_pos,\
                 input_rays, target_pixels, target_camera_pos, \
                     target_rays
            

        


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


def preprocess_txt(text):
    return tokenize([str(text)])[0]


def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards)
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption(sample):
    return 'txt' in sample


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
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


def pytorch_worker_seed():
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour the seed already created for pytorch dataloader workers if it exists
        return worker_info.seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            seed = pytorch_worker_seed() + epoch
        else:
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = wds.shardlists.expand_urls(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = pytorch_worker_seed if worker_seed is None else worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic, worker seed should be deterministic due to arg.seed
            self.rng.seed(self.worker_seed() + epoch)
        for _ in range(self.nshards):
            yield dict(url=self.rng.choice(self.urls))


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png", text="txt"),
        wds.map_dict(image=preprocess_img, text=preprocess_txt),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train),
    ])

    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        persistent_workers = True
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_csv_msn_dataset(args, preprocess_fn, is_train, epoch=0):
    input_filename = args.train_data if is_train else args.val_data
    msn_path = args.msn_path
    mode = "train" if is_train else "val"
    assert input_filename
    dataset = CsvMsnDataset(
        input_filename,
        msn_path,
        mode,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "csv-msn":
        return get_csv_msn_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extention {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0):
    preprocess_train, preprocess_val = preprocess_fns 
    data = {}

    if args.train_data:
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data
