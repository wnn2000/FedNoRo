import numpy as np
import torch
import torchvision.transforms as transforms

from .all_datasets import isic2019, ICH

from utils.sampling import iid_sampling, non_iid_dirichlet_sampling


def get_dataset(args):
    if args.dataset == "isic2019":
        root = "your path"
        args.n_classes = 8

        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = isic2019(root, "train", train_transform)
        test_dataset = isic2019(root, "test", val_transform)


    elif args.dataset == "ICH":
        root = "your path"
        args.n_classes = 5

        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = ICH(root, "train", train_transform)
        test_dataset = ICH(root, "test", val_transform)

    else:
        exit("Error: unrecognized dataset")

    n_train = len(train_dataset)
    y_train = np.array(train_dataset.targets)
    assert n_train == len(y_train)

    if args.iid:
        dict_users = iid_sampling(n_train, args.n_clients, args.seed)
    else:
        dict_users = non_iid_dirichlet_sampling(y_train, args.n_classes, args.non_iid_prob_class, args.n_clients, seed=100, alpha_dirichlet=args.alpha_dirichlet)

    return train_dataset, test_dataset, dict_users



