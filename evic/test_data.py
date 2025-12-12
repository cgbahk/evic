import tempfile
from pathlib import Path

import pytest
import torch
import torchvision.transforms as T
from PIL import Image

from data import DistinctLabelImageContextDataset


def create_dummy_images(root: Path, num_labels, imgs_per_label, size_wh):
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
            img = Image.new("RGB", size_wh, color=(label * 40, i * 20, 100))
            img.save(img_path)

            assert isinstance(img_path, Path)
            img_paths.append(img_path)

            labels.append(label)
    return img_paths, labels


def test_positive():
    context_size = 2
    W, H = 16, 32

    with tempfile.TemporaryDirectory() as tmpdir:
        img_paths, labels = create_dummy_images(
            Path(tmpdir),
            num_labels=3,
            imgs_per_label=4,
            size_wh=(W, H),
        )

        dataset = DistinctLabelImageContextDataset(
            img_paths=img_paths,
            labels=labels,
            context_size=context_size,
            transform=T.ToTensor(),
        )

        assert len(dataset) == len(img_paths)

        sample = dataset[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.shape[0] == context_size
        assert sample.shape[1] == 3  # channels: RGB
        assert sample.shape[2] == H
        assert sample.shape[3] == W

        # TODO Check that all labels in the context are distinct
        # The dataset need to return paths and labels consisting the context for this
        # Dataloader might ignore this information


# TODO Test reproducibility by seed


def test_label_vs_context_size():
    def _run(num_labels, context_size):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_paths, labels = create_dummy_images(
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
            img_paths, labels = create_dummy_images(
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
