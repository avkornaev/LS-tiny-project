import torch

from label_smoothing.data import filter_and_map, stratified_indices


def test_filter_keeps_three_and_seven_and_maps_labels() -> None:
    images = torch.arange(5 * 28 * 28, dtype=torch.uint8).reshape(5, 28, 28)
    labels = torch.tensor([1, 3, 7, 3, 9])
    dataset = filter_and_map(images, labels)
    filtered_images, mapped_labels = dataset.tensors

    assert filtered_images.shape == (3, 784)
    assert mapped_labels.tolist() == [0, 1, 0]
    assert 0 <= filtered_images.min() <= filtered_images.max() <= 1


def test_stratified_split_is_fixed_and_disjoint() -> None:
    labels = torch.tensor([0] * 10 + [1] * 10)
    train_a, validation_a = stratified_indices(labels)
    train_b, validation_b = stratified_indices(labels)

    assert train_a.tolist() == train_b.tolist()
    assert validation_a.tolist() == validation_b.tolist()
    assert set(train_a).isdisjoint(validation_a)
    assert labels[validation_a].bincount().tolist() == [2, 2]

