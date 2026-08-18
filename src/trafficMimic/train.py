"""Optional traffic-BERT retraining entry point.

The minimal reproduction uses the checked-in checkpoint and does not invoke
this module.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from trafficMimic.dataset.dataset import FlowDataset
from trafficMimic.dataset.vocab import SizeVocab, TimeVocab
from trafficMimic.model.architecture import BERT, BERTLM
from trafficMimic.utils import (feature_extract, read_flow_pkl, read_yaml,
                                recursive_namespace, resolve_repo_path,
                                set_device, set_seed)


class BertTrainer:
    def __init__(self, args, bert, timevocab, sizevocab):
        self.args = args
        self.device = set_device(args.trainer.device)
        self.model = BERTLM(bert, len(timevocab), len(sizevocab)).to(self.device)
        checkpoint = args.trainer.load_save_pth
        if checkpoint and Path(checkpoint).is_file():
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device),
                                       strict=True)
            print("loaded {}".format(checkpoint))

        self.train_dataloader = DataLoader(
            FlowDataset(args, timevocab, sizevocab, train=True),
            batch_size=args.trainer.batch_size, shuffle=True, drop_last=False)
        self.test_dataloader = DataLoader(
            FlowDataset(args, timevocab, sizevocab, train=False),
            batch_size=args.trainer.batch_size, shuffle=False, drop_last=False)
        optimizer = args.trainer.optimizer
        self.optim = torch.optim.AdamW(
            self.model.parameters(), lr=optimizer.lr,
            betas=tuple(optimizer.beta), weight_decay=optimizer.weight_decay)
        self.criterion = nn.NLLLoss(ignore_index=0)
        self.best_ipd_acc = self.best_size_acc = 0.0

    def run_epoch(self, epoch, dataloader, training):
        self.model.train(training)
        totals = {"loss": 0.0, "ipd_correct": 0, "ipd_count": 0,
                  "size_correct": 0, "size_count": 0}
        iterator = tqdm(dataloader, desc=("train" if training else "test") +
                        " epoch {}".format(epoch))
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in iterator:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                ipd_pred, size_pred = self.model(batch["flow_ipd"], batch["flow_size"])
                ipd_loss = self.criterion(ipd_pred.transpose(1, 2),
                                          batch["flow_ipd_label"])
                size_loss = self.criterion(size_pred.transpose(1, 2),
                                            batch["flow_size_label"])
                loss = ipd_loss + size_loss
                if training:
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                totals["loss"] += loss.item()
                for prefix, prediction in (("ipd", ipd_pred), ("size", size_pred)):
                    label = batch["flow_{}_label".format(prefix)]
                    mask = label.ne(0)
                    totals[prefix + "_correct"] += (
                        prediction.argmax(-1).eq(label) & mask).sum().item()
                    totals[prefix + "_count"] += mask.sum().item()
        metrics = {
            "ipd_accuracy": totals["ipd_correct"] / max(1, totals["ipd_count"]),
            "size_accuracy": totals["size_correct"] / max(1, totals["size_count"]),
        }
        print(metrics)
        return metrics

    def save(self, suffix):
        output = Path(self.args.trainer.model_save_pth + "_" + suffix + ".pth")
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), output)
        return output


def _resolve_paths(args):
    for name in ("train_data_pth", "test_data_pth"):
        value = getattr(args.trainer, name)
        if isinstance(value, list):
            value = [str(resolve_repo_path(path)) for path in value]
        else:
            value = str(resolve_repo_path(value))
        setattr(args.trainer, name, value)
    for name in ("timevocab_pth", "sizevocab_pth", "model_save_pth"):
        setattr(args.trainer, name,
                str(resolve_repo_path(getattr(args.trainer, name))))
    if args.trainer.load_save_pth:
        args.trainer.load_save_pth = str(resolve_repo_path(args.trainer.load_save_pth))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/trafficMimic/config/bert.yaml")
    cli = parser.parse_args(argv)
    args = recursive_namespace(read_yaml(cli.config))
    _resolve_paths(args)
    set_seed(args.trainer.seed)

    train_data = None
    if not Path(args.trainer.timevocab_pth).is_file() or not Path(
            args.trainer.sizevocab_pth).is_file():
        train_data = feature_extract(read_flow_pkl(args.trainer.train_data_pth))
    timevocab = (TimeVocab.load_vocab(args.trainer.timevocab_pth)
                 if Path(args.trainer.timevocab_pth).is_file()
                 else TimeVocab(train_data[0]))
    sizevocab = (SizeVocab.load_vocab(args.trainer.sizevocab_pth)
                 if Path(args.trainer.sizevocab_pth).is_file()
                 else SizeVocab(train_data[1]))
    if not Path(args.trainer.timevocab_pth).is_file():
        Path(args.trainer.timevocab_pth).parent.mkdir(parents=True, exist_ok=True)
        timevocab.save_vocab(args.trainer.timevocab_pth)
    if not Path(args.trainer.sizevocab_pth).is_file():
        Path(args.trainer.sizevocab_pth).parent.mkdir(parents=True, exist_ok=True)
        sizevocab.save_vocab(args.trainer.sizevocab_pth)

    bert = BERT(len(timevocab), len(sizevocab), args.model.hidden,
                args.model.d_ff, args.model.n_layers, args.model.attn_heads,
                args.model.dropout)
    trainer = BertTrainer(args, bert, timevocab, sizevocab)
    for epoch in range(args.trainer.epochs):
        trainer.run_epoch(epoch, trainer.train_dataloader, True)
        trainer.save("ep{}".format(epoch))
        metrics = trainer.run_epoch(epoch, trainer.test_dataloader, False)
        if metrics["ipd_accuracy"] > trainer.best_ipd_acc:
            trainer.best_ipd_acc = metrics["ipd_accuracy"]
            trainer.save("ipdbest")
        if metrics["size_accuracy"] > trainer.best_size_acc:
            trainer.best_size_acc = metrics["size_accuracy"]
            trainer.save("sizebest")


if __name__ == "__main__":
    main()
