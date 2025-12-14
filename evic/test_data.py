import tempfile
from pathlib import Path
import multiprocessing as mp
import random

import pytest
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

from .data import (
    DistinctLabelImageContextDataset,
    PredefinedImageContextDataset,
)


def create_random_images(root: Path, num_labels, imgs_per_label, size_wh):
    """
    Create synthetic RGB images for testing under root directory.
    Returns (img_paths, labels)
    """
    img_paths = []
    labels = []
    for label in range(num_labels):
        label_dir = root / f"label_{label}"
        label_dir.mkdir(parents=True, exist_ok=True)
        for i in range(imgs_per_label):
            img_path = label_dir / f"img_{i}.png"

            Image.fromarray(
                np.random.randint(0, 256, (size_wh[1], size_wh[0], 3), dtype=np.uint8)
            ).save(img_path)

            img_paths.append(img_path)
            labels.append(label)

    return img_paths, labels


def create_predefined_dataset(img_paths, *, context_count, context_size):
    assert (context_count * context_size) % len(img_paths) == 0
    duplication = (context_count * context_size) // len(img_paths)

    contexts = []

    indices = list(range(len(img_paths))) * duplication
    random.shuffle(indices)

    for i in range(context_count):
        idxs_for_context = indices[i * context_size : (i + 1) * context_size]
        ctx = [img_paths[j] for j in idxs_for_context]
        contexts.append(ctx)

    return PredefinedImageContextDataset(
        contexts=contexts,
        context_size=context_size,
        transform=T.ToTensor(),
    )


def make_loader(seed, **kwargs):
    assert "multiprocessing_context" not in kwargs

    if kwargs.get("num_workers", 0) >= 1:
        # Otherwise, gets unsafe 'fork' warning
        #
        # 'spawn' might be much slower than 'fork'
        kwargs["multiprocessing_context"] = mp.get_context("spawn")

    return torch.utils.data.DataLoader(
        generator=torch.Generator().manual_seed(seed),
        **kwargs,
    )


def test_positive():
    context_size = 2
    W, H = 16, 32

    with tempfile.TemporaryDirectory() as tmpdir:
        img_paths, labels = create_random_images(
            Path(tmpdir),
            num_labels=3,
            imgs_per_label=4,
            size_wh=(W, H),
        )

        dataset_distinct = DistinctLabelImageContextDataset(
            img_paths=img_paths,
            labels=labels,
            context_size=context_size,
            transform=T.ToTensor(),
        )

        dataset_predefined = create_predefined_dataset(
            img_paths,
            context_count=len(img_paths),  # Don't need to be, but to pass length check
            context_size=context_size,
        )

        for dataset in (dataset_distinct, dataset_predefined):
            assert len(dataset) == len(img_paths)

            sample = dataset[0]
            assert isinstance(sample, torch.Tensor)
            assert sample.shape[0] == context_size
            assert sample.shape[1] == 3  # channels: RGB
            assert sample.shape[2] == H
            assert sample.shape[3] == W

        # TODO For DistinctLabelImageContextDataset, check that all labels in
        # the context are distinct. The dataset need to return paths and labels
        # consisting the context for this. Dataset users might ignore this
        # information.


def test_same_seed_same_result():
    batch_size = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        img_paths, labels = create_random_images(
            Path(tmpdir),
            num_labels=7,
            imgs_per_label=5,
            size_wh=(11, 13),
        )

        dataset = DistinctLabelImageContextDataset(
            img_paths=img_paths,
            labels=labels,
            context_size=3,
            transform=T.ToTensor(),
        )

        def _test_same(seed, **kwargs):
            loader1 = make_loader(
                seed, dataset=dataset, batch_size=batch_size, **kwargs
            )
            loader2 = make_loader(
                seed, dataset=dataset, batch_size=batch_size, **kwargs
            )

            for batch1, batch2 in zip(loader1, loader2):
                assert torch.equal(batch1, batch2)

        _test_same(42, shuffle=True, num_workers=0)
        _test_same(43, shuffle=False, num_workers=0)
        _test_same(44, shuffle=True, num_workers=1)
        _test_same(45, shuffle=False, num_workers=1)
        _test_same(46, shuffle=True, num_workers=2)
        _test_same(47, shuffle=False, num_workers=2)


# TODO Check `PredefinedImageContextDataset`
# - same seed same result
# - diff setting diff result

def test_diff_workers_same_result():
    batch_size = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        img_paths, labels = create_random_images(
            Path(tmpdir),
            num_labels=7,
            imgs_per_label=5,
            size_wh=(11, 13),
        )

        dataset = create_predefined_dataset(
            img_paths,
            context_count=len(img_paths),
            context_size=3,
        )

        def _test_same_for_diff_workers(seed, num_workers_pair: tuple[int, int], **kwargs):
            nw1, nw2 = num_workers_pair

            assert nw1 != nw2
            assert "num_workers" not in kwargs

            loader1 = make_loader(seed, num_workers=nw1, dataset=dataset, batch_size=batch_size, **kwargs)
            loader2 = make_loader(seed, num_workers=nw2, dataset=dataset, batch_size=batch_size, **kwargs)

            for batch1, batch2 in zip(loader1, loader2):
                assert torch.equal(batch1, batch2)

        _test_same_for_diff_workers(42, num_workers_pair=(0, 1), shuffle=True)
        _test_same_for_diff_workers(43, num_workers_pair=(0, 1), shuffle=False)
        _test_same_for_diff_workers(44, num_workers_pair=(1, 2), shuffle=True)
        _test_same_for_diff_workers(45, num_workers_pair=(1, 2), shuffle=False)
        _test_same_for_diff_workers(46, num_workers_pair=(0, 2), shuffle=True)
        _test_same_for_diff_workers(47, num_workers_pair=(0, 2), shuffle=False)


def test_diff_setting_diff_result():
    batch_size = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        img_paths, labels = create_random_images(
            Path(tmpdir),
            num_labels=5,
            imgs_per_label=7,
            size_wh=(11, 13),
        )

        dataset = DistinctLabelImageContextDataset(
            img_paths=img_paths,
            labels=labels,
            context_size=3,
            transform=T.ToTensor(),
        )

        def _are_diff_for_diff_seed(**kwargs):
            loader1 = make_loader(42, dataset=dataset, batch_size=batch_size, **kwargs)
            loader2 = make_loader(23, dataset=dataset, batch_size=batch_size, **kwargs)

            for batch1, batch2 in zip(loader1, loader2):
                if not torch.equal(batch1, batch2):
                    return True

            return False

        assert _are_diff_for_diff_seed(shuffle=True, num_workers=0)

        # This is known limitation: same results are generated even with
        # different seeds when there is no worker, i.e. when dataloader runs on
        # main process.
        #
        # This might be okay because:
        # - On training, we usually use workers,
        # - On validation, we usually want reproducibility from same seeds.
        assert not _are_diff_for_diff_seed(shuffle=False, num_workers=0)

        assert _are_diff_for_diff_seed(shuffle=True, num_workers=1)
        assert _are_diff_for_diff_seed(shuffle=False, num_workers=1)
        assert _are_diff_for_diff_seed(shuffle=True, num_workers=2)
        assert _are_diff_for_diff_seed(shuffle=False, num_workers=2)

        def _are_diff_for_diff_workers(seed, num_workers_pair: tuple[int, int], **kwargs):
            nw1, nw2 = num_workers_pair

            assert nw1 != nw2
            assert "num_workers" not in kwargs

            loader1 = make_loader(seed, num_workers=nw1, dataset=dataset, batch_size=batch_size, **kwargs)
            loader2 = make_loader(seed, num_workers=nw2, dataset=dataset, batch_size=batch_size, **kwargs)

            for batch1, batch2 in zip(loader1, loader2):
                if not torch.equal(batch1, batch2):
                    return True

            return False

        assert _are_diff_for_diff_workers(42, num_workers_pair=(0, 1), shuffle=True)
        assert _are_diff_for_diff_workers(43, num_workers_pair=(0, 1), shuffle=False)
        assert _are_diff_for_diff_workers(44, num_workers_pair=(1, 2), shuffle=True)
        assert _are_diff_for_diff_workers(45, num_workers_pair=(1, 2), shuffle=False)
        assert _are_diff_for_diff_workers(46, num_workers_pair=(0, 2), shuffle=True)
        assert _are_diff_for_diff_workers(47, num_workers_pair=(0, 2), shuffle=False)


def test_label_vs_context_size():
    def _run(num_labels, context_size):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_paths, labels = create_random_images(
                Path(tmpdir),
                num_labels=num_labels,
                imgs_per_label=4,
                size_wh=(16, 32),
            )

            DistinctLabelImageContextDataset(
                img_paths=img_paths,
                labels=labels,
                context_size=context_size,
                transform=T.ToTensor(),
            )

    with pytest.raises(AssertionError):
        _run(num_labels=3, context_size=4)

    _run(num_labels=4, context_size=3)  # No raise


def test_small_context_size():
    context_size = 1

    with pytest.raises(AssertionError):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_paths, labels = create_random_images(
                Path(tmpdir),
                num_labels=3,
                imgs_per_label=4,
                size_wh=(16, 32),
            )

            DistinctLabelImageContextDataset(
                img_paths=img_paths,
                labels=labels,
                context_size=context_size,
                transform=T.ToTensor(),
            )
