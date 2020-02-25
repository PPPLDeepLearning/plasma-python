from torch.nn.utils import weight_norm
import torch.optim as opt
import torch.nn as nn
import torch

from keras.utils.generic_utils import Progbar
from plasma.utils.downloading import makedirs_process_safe
from plasma.utils.performance import PerformanceAnalyzer
from plasma.utils.evaluation import get_loss_from_list
from plasma.models.torch_runner import (
    # make_predictions_and_evaluate_gpu,
    # make_predictions,
    get_signal_dimensions,
    calculate_conv_output_size,
)

from functools import partial
import os
import numpy as np
import logging
import random
# import tqdm

model_filename = "torch_model.pt"
LOGGER = logging.getLogger("plasma.transformer.runner")

global device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# if torch.cuda.is_available():
#    torch.cuda.set_device(device_id)
#    device = torch.device("cuda", index=device_id)
# else:
#    device = torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = "0"


class TransformerNet(nn.Module):
    def __init__(
        self,
        n_scalars,
        n_profiles,
        profile_size,
        layer_sizes_spatial,
        kernel_size_spatial,
        linear_size,
        dropout=0.1,
    ):
        super(TransformerNet, self).__init__()
        self.input_layer = InputBlock(
            n_scalars,
            n_profiles,
            profile_size,
            layer_sizes_spatial,
            kernel_size_spatial,
            linear_size,
            dropout,
        )
        self.temporal_encoder = TransformerSequenceEncoder()
        self.model = nn.Sequential(self.input_layer, self.temporal_encoder)

    def forward(self, x):
        return self.model(x)


class InputBlock(nn.Module):
    def __init__(
        self,
        n_scalars,
        n_profiles,
        profile_size,
        layer_sizes,
        kernel_size,
        linear_size,
        dropout=0.2,
    ):
        super(InputBlock, self).__init__()
        self.pooling_size = 2
        self.n_scalars = n_scalars
        self.n_profiles = n_profiles
        self.profile_size = profile_size
        self.conv_output_size = profile_size
        if self.n_profiles == 0:
            self.net = None
            self.conv_output_size = 0
        else:
            self.layers = []
            for (i, layer_size) in enumerate(layer_sizes):
                if i == 0:
                    input_size = n_profiles
                else:
                    input_size = layer_sizes[i - 1]
                self.layers.append(
                    weight_norm(nn.Conv1d(input_size, layer_size, kernel_size))
                )
                self.layers.append(nn.ReLU())
                self.conv_output_size = calculate_conv_output_size(
                    self.conv_output_size, 0, 1, 1, kernel_size
                )
                self.layers.append(nn.MaxPool1d(kernel_size=self.pooling_size))
                self.conv_output_size = calculate_conv_output_size(
                    self.conv_output_size, 0, 1, self.pooling_size,
                    self.pooling_size
                )
                self.layers.append(nn.Dropout2d(dropout))
            self.net = nn.Sequential(*self.layers)
            self.conv_output_size = self.conv_output_size * layer_sizes[-1]
        self.linear_layers = []

        print("Final feature size = {}".format(self.n_scalars
                                               + self.conv_output_size))
        self.linear_layers.append(
            nn.Linear(self.conv_output_size + self.n_scalars, linear_size)
        )
        self.linear_layers.append(nn.ReLU())
        self.linear_layers.append(nn.Linear(linear_size, linear_size))
        self.linear_layers.append(nn.ReLU())
        print("Final output size = {}".format(linear_size))
        self.linear_net = nn.Sequential(*self.linear_layers)

    def forward(self, x):
        if self.n_profiles == 0:
            full_features = x  # x_scalars
        else:
            if self.n_scalars == 0:
                x_profiles = x
            else:
                x_scalars = x[:, : self.n_scalars]
                x_profiles = x[:, self.n_scalars:]
            x_profiles = x_profiles.contiguous().view(
                x.size(0), self.n_profiles, self.profile_size
            )
            profile_features = self.net(x_profiles).view(x.size(0), -1)
            if self.n_scalars == 0:
                full_features = profile_features
            else:
                full_features = torch.cat([x_scalars, profile_features], dim=1)

        # FIXME do not use linear layers
        # out = self.linear_net(full_features)
        out = full_features
        return out


class TransformerSequenceEncoder(nn.Module):
    def __init__(
        self,
        max_seq_length=2048,
        d_model=11,
        num_layers=6,
        dim_feedforward=1024,
        nhead=11,
        dropout=0.1,
    ):
        super(TransformerSequenceEncoder, self).__init__()

        self.__transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.__max_seq_length = max_seq_length
        self.__d_model = d_model
        # FIXME
        self.__positional_encodings = nn.Embedding(
            max_seq_length, d_model).float()

    def forward(self, x):
        """
        Shape of x is sequence, length
        """
        # Force-pad
        mask = (
            torch.arange(x.shape[1], device=device)
            .unsqueeze(0)
            .lt(torch.tensor([self.__max_seq_length],
                             device=device).unsqueeze(-1))
        )
        transformer_input = x * mask.unsqueeze(-1).float()  # B x max_len x D

        positional_encodings = self.__positional_encodings(
            torch.arange(x.shape[1], dtype=torch.int64, device=device)
        ).unsqueeze(0)
        transformer_input = (transformer_input
                             + positional_encodings)  # B x max_len x D

        out = self.__transformer_encoder(
            transformer_input  # .transpose(0, 1), src_key_padding_mask=~mask
        )
        return out


def build_torch_model(conf):

    dropout = conf["model"]["dropout_prob"]
    n_scalars, n_profiles, profile_size = get_signal_dimensions(conf)
    # output_size = 1
    layer_sizes_spatial = [6, 3, 3]
    kernel_size_spatial = 3
    linear_size = 5  # FIXME Alexeys there will be no linear layers

    model = TransformerNet(
        n_scalars,
        n_profiles,
        profile_size,
        layer_sizes_spatial,
        kernel_size_spatial,
        linear_size,
        dropout,
    )
    model.to(device)

    return model


def get_model_path(conf):
    return (
        conf["paths"]["model_save_path"] + "torch/" + model_filename
    )  # save_prepath + model_filename


def train_epoch(model, data_gen, optimizer, scheduler, loss_fn):

    loss = 0
    total_loss = 0

    step = 0
    while True:
        x_, y_, num_so_far, num_total, _ = next(data_gen)

        x = torch.from_numpy(x_).float().to(device)
        y = torch.from_numpy(y_).float().to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1

        LOGGER.info(f"[{step}]  [{num_so_far}/{num_total}] loss: {loss.item()}, ave_loss: {total_loss / step}")  # noqa
        if num_so_far >= num_total:
            break

    return (step, loss.item(), total_loss, num_so_far,
            1.0 * num_so_far / num_total)


def train(conf, shot_list_train, shot_list_validate, loader):
    # set random seed
    set_seed(0)
    num_epochs = conf["training"]["num_epochs"]
    # patience = conf["callbacks"]["patience"]
    lr_decay = conf["model"]["lr_decay"]
    # batch_size = conf['training']['batch_size']
    lr = conf["model"]["lr"]
    # clipnorm = conf['model']['clipnorm']
    e = 0

    loader.set_inference_mode(False)
    train_data_gen = partial(
        loader.simple_batch_generator,
        shot_list=shot_list_train,
    )()
    valid_data_generator = partial(  # noqa
        loader.simple_batch_generator,
        shot_list=shot_list_validate,
        inference=True
    )()
    LOGGER.info(f"validate: {len(shot_list_validate)} shots, {shot_list_validate.num_disruptive()} disruptive")  # noqa
    LOGGER.info(f"training: {len(shot_list_train)} shots, {shot_list_train.num_disruptive()} disruptive")  # noqa

    loss_fn = nn.MSELoss(size_average=True)
    train_model = build_torch_model(conf)

    optimizer = opt.Adam(train_model.parameters(), lr=lr)
    scheduler = opt.lr_scheduler.ExponentialLR(optimizer, lr_decay)

    model_path = get_model_path(conf)
    makedirs_process_safe(os.path.dirname(model_path))

    train_model.train()
    LOGGER.info(f"{num_epochs - 1 - e} epochs left to go")
    while e < num_epochs - 1:
        LOGGER.info(f"Epoch {e}/{num_epochs}")
        (step, ave_loss, curr_loss, num_so_far,
         effective_epochs) = train_epoch(
            train_model, train_data_gen, optimizer, scheduler, loss_fn
        )

        e = effective_epochs
        torch.save(train_model.state_dict(), model_path)
        # FIXME no validation for now as OOM
        # _, _, _, roc_area, loss = make_predictions_and_evaluate_gpu(
        #    conf, shot_list_validate, valid_data_generator
        # )

        # # stop_training = False
        # print("=========Summary======== for epoch{}".format(step))
        # print("Training Loss numpy: {:.3e}".format(ave_loss))
        # print("Validation Loss: {:.3e}".format(loss))
        # print("Validation ROC: {:.4f}".format(roc_area))


def apply_model_to_np(model, x):
    return model(torch.from_numpy(x).float()).data.numpy()


# FIXME Alexeys change
def make_predictions(conf, shot_list, generator, custom_path=None):
    # generator = loader.inference_batch_generator_full_shot(shot_list)
    inference_model = build_torch_model(conf)

    if custom_path is None:
        model_path = get_model_path(conf)
    else:
        model_path = custom_path
    inference_model.load_state_dict(torch.load(model_path))
    # shot_list = shot_list.random_sublist(10)

    y_prime = []
    y_gold = []
    disruptive = []
    num_shots = len(shot_list)

    pbar = Progbar(num_shots)
    while True:
        x_, y_, num_so_far, num_total, disr = next(generator)

        x = torch.from_numpy(x_).float().to(device)
        y = torch.from_numpy(y_).float().to(device)
        # output = apply_model_to_np(inference_model, x)
        output = inference_model(x)

        for batch_idx in range(x.shape[0]):
            # curr_length = lengths[batch_idx]
            y_prime += [output[batch_idx, :, 0]]
            y_gold += [y[batch_idx, :, 0]]
            disruptive += [disr[batch_idx]]
            pbar.add(1.0)
        if len(disruptive) >= num_shots:
            y_prime = y_prime[:num_shots]
            y_gold = y_gold[:num_shots]
            disruptive = disruptive[:num_shots]
            break
    return y_prime, y_gold, disruptive


# FIXME ALexeys change loader --> generator
def make_predictions_and_evaluate_gpu(conf, shot_list, generator,
                                      custom_path=None):
    y_prime, y_gold, disruptive = make_predictions(
        conf, shot_list, generator, custom_path)
    analyzer = PerformanceAnalyzer(conf=conf)
    roc_area = analyzer.get_roc_area(y_prime, y_gold, disruptive)
    loss = get_loss_from_list(y_prime, y_gold, conf['data']['target'])
    return y_prime, y_gold, disruptive, roc_area, loss
