#!/usr/bin/env python3
import argparse
import os
import datetime
import re
import numpy as np
from random import randrange
import torch
from torch import nn
import torch.nn.functional as F
from data.dataset import Dataset 
from data.trajnet_loader import TrajnetLoader

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=8, type=int, help="Number of epochs.")
parser.add_argument("--dataset_type", default='trajnet++', type=str, help="The dataset type.")
parser.add_argument("--dataset_name", default='orca_synth_train', type=str, help="The dataset name.")
parser.add_argument("--seed", default=21, type=int, help="Random seed.")
parser.add_argument("--device", default="cpu", type=str, help="Device.")
parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--n_interaction_size", default=512, type=int, help="Number of hidden channels.")
parser.add_argument("--n_hidden_sizes", default=[1024], type=int, nargs='+', help="List of hidden channel sizes.")

def main(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # create logdir
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # load data
    if args.dataset_type == "trajnet++":
        loader = TrajnetLoader()
    else:
        "unknown dataset"

    # Load the data. `data.train` contains the training inputs and outputs, and `data.test` contains the testing input
    # and the correposing (possible) outputs
    Dataset(loader, 
    data = ARCDataset(args.dataset, args.task_id)

    def prepare_example(input, output):
        def pad_to_max_dim(x):
            x_one_hot = F.one_hot(x, num_classes=ARCDataset.N_CELL_TYPES)
            x_height, x_width = x.shape[:2]
            x_padding = (0, 1, 0, ARCDataset.MAX_WIDTH - x_width, 0, ARCDataset.MAX_HEIGHT - x_height)
            x_padded = F.pad(x_one_hot, x_padding)
            x_padded[x_height:, :, ARCDataset.N_CELL_TYPES] = 1.0
            x_padded[:, x_width:, ARCDataset.N_CELL_TYPES] = 1.0
            return x_padded
        input_preprocessed = pad_to_max_dim(torch.tensor(input))
        input_preprocessed = F.pad(input_preprocessed, (0, args.n_hidden_channels))
        output_preprocessed = pad_to_max_dim(torch.tensor(output))
        return input_preprocessed, output_preprocessed

    train = data.train.transform(prepare_example)
    test = data.test.transform(prepare_example)
        
    def prepare_batch(data):
        inputs, outputs = zip(*data)
        inputs = torch.stack(inputs)
        outputs = torch.stack(outputs)
        return inputs.float(), outputs

    train = torch.utils.data.DataLoader(train, batch_size=args.batch_size, collate_fn=prepare_batch,
                                        sampler=torch.utils.data.RandomSampler(train, replacement=True, num_samples=args.batch_size))
    test = torch.utils.data.DataLoader(test, batch_size=len(test), collate_fn=prepare_batch)

    if args.model == 'nca':
        from nca import Model
    elif args.model == 'unet':
        from unet import Model
    model = Model(args, ARCDataset.N_CELL_TYPES + 1) 

    model.configure(
        optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
        metrics={'accuracy': Accuracy(ARCDataset.N_CELL_TYPES + 1)},
        device=args.device,
        logdir=args.logdir
    )

    def visualization_callback(model, epoch, logs):
        if epoch % 50 == 0 and epoch > 0:
            train_id = randrange(len(train.dataset))
            train_x = train.dataset[train_id][0]
            train_output = model.predict([train_x.unsqueeze(0).float()], as_numpy=True)[0]
            train_reference = train.dataset[train_id][1].numpy()

            test_id = randrange(len(test.dataset))
            test_x = test.dataset[test_id][0]
            test_output = model.predict([test_x.unsqueeze(0).float()], as_numpy=True)[0]
            test_reference = test.dataset[test_id][1].numpy()

            fig = visualize_state_evolution([train_x.numpy(), train_output, train_reference, np.zeros_like(train_output),
                                             test_x.numpy(), test_output, test_reference], 
                                            ARCDataset.N_CELL_TYPES + 1, ARCDataset.COLOR_LIST + ['#FFFFFF'])
            fig.savefig(args.logdir + f'/state_evolution_epoch_{epoch}.png')

    logs = model.fit(train, dev=test, epochs=args.epochs, callbacks=[visualization_callback])
    # model.save_weights('reading_comprehension.weights')

if __name__ == "__main__":
    main(parser.parse_args([] if "__file__" not in globals() else None))