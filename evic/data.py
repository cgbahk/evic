# TODO Assert image is channel 3, but where?
# TODO What happens if grey scale image sourced?
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class DistinctLabelImageContextDataset(Dataset):
    """
    A dataset that returns a context (group) of images, where each image
    in the group has a different label.

    Iteration order is based on the original list of image paths: for each
    index, the first image in the returned context is the image at that
    position, and the remaining images are sampled from other labels.

    Context formation is based on uniform sampling over labels first, then
    image, so for unbalanced data the distribution of contexts may be uneven
    over images.

    Dividing train & valid split might be pointless for this dataset, as
    images would be mixed in context randomly. To create "true" valid split,
    one should split on filepath level before createing this dataset.

    Args:
        transform: Transform for each image (not for context of images)
    """

    # NOTE This dataset maybe slow on CPU. There are multiple solutions:
    # - Preprocess data and load them
    # - Profile which part is bottleneck
    # - Run heavy task on GPU
    # - Run on large number of CPUs and big memory (e.g. request more cpus for slurm)

    def __init__(self, img_paths, labels, context_size: int, transform):
        assert len(img_paths) == len(labels)
        assert context_size >= 2
        for path in img_paths:
            assert Path(path).is_file()

        self.context_size = context_size
        self._transform = transform

        self._df = pd.DataFrame({
            "imgpath": [str(path) for path in img_paths],
            "label": labels,
        })
        assert self._df["label"].value_counts().nunique() == 1

        self._label_to_imgpaths = {
            label: group["imgpath"].reset_index(drop=True)
            for label, group in self._df.groupby("label")
        }
        self._labels = list(self._label_to_imgpaths.keys())
        assert len(self._labels) >= context_size

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        if torch.utils.data.get_worker_info() is None:
            # In case of `num_workers == 0`, this is required for reproducibility.
            torch.manual_seed(42 + idx)

        # In case of `num_workers >= 1`:
        #
        # By documentation, if using random funtionalities from torch, seeds are set reasonable,
        # i.e. `base_seed + worker_id`. So same seed by main process will generate same behavior,
        # and each worker behave differently from each others.
        #
        #   https://github.com/pytorch/pytorch/blob/v2.9.1/docs/source/data.md?plain=1#L362

        first_path = self._df.iloc[idx]["imgpath"]
        first_label = self._df.iloc[idx]["label"]

        while True:
            perm = torch.randperm(len(self._labels))
            chosen_labels = [
                self._labels[i.item()] for i in perm[: self.context_size - 1]
            ]

            if first_label not in chosen_labels:
                break

        context_paths = [first_path]
        for label in chosen_labels:
            paths = self._label_to_imgpaths[label]
            rand_index = torch.randint(len(paths), (1,)).item()
            rand_path = paths.iloc[rand_index]
            context_paths.append(rand_path)

        images = []
        for path in context_paths:
            img = Image.open(path).convert("RGB")
            img = self._transform(img)
            images.append(img)

        return torch.stack(images, dim=0)
