from pathlib import Path

Label = str


# TODO Search for imagenet test data
# TODO Rename to imagenet "validation"
def get_imagenet_paths_labels(root_dir: Path) -> tuple[list[Path], list[Label]]:
    # NOTE This function is very hard-coded with ImageNet, directory structure and filename convention
    assert Path(root_dir).is_dir()

    paths = sorted(Path(root_dir).rglob("*.JPEG"), key=lambda p: p.name)
    assert len(paths) == 50_000

    labels = []
    for path in paths:
        label = path.parent.name
        assert len(label) == 9
        assert label.startswith("n")

        labels.append(label)

    assert len(labels) == 50_000

    return paths, labels
