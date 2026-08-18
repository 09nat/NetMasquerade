"""Train and evaluate the released NetBeacon target classifier."""

import argparse
import json
from pathlib import Path
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from advGenerate.environments.NetBeacon.netbeacon import NetBeacon
from trafficMimic.utils import (feature_extract, read_flow_pkl, read_yaml,
                                recursive_namespace, resolve_repo_path,
                                set_seed, valid_flows)


def _labeled(flows, label):
    ipd, size = feature_extract(flows)
    return [(ipd[index], size[index], label) for index in range(len(ipd))]


class NetBeaconTrainer:
    def __init__(self, model, train_dataset, test_dataset):
        self.model = model
        self.train_data = train_dataset
        self.test_data = test_dataset

    def fit(self, top_features=10):
        labels = pd.DataFrame(
            [(len(item[0]), item[2]) for item in self.train_data],
            columns=["length", "label"])
        for phase_count, phase in enumerate(self.model.phases):
            phase_labels = labels[labels.length >= phase]["label"].astype(int)
            if set(phase_labels.tolist()) != {0, 1}:
                raise ValueError("phase {} does not contain both classes".format(phase))
            features = self.model.get_phase_features(self.train_data, phase_count)
            self.model.models[phase_count].fit(features, phase_labels)
            order = np.argsort(
                self.model.models[phase_count].feature_importances_)[::-1]
            self.model.features[phase_count] = [
                features.columns[index] for index in order[:top_features]]
        for phase_count, phase in enumerate(self.model.phases):
            phase_labels = labels[labels.length >= phase]["label"].astype(int)
            features = self.model.get_phase_features(self.train_data, phase_count)
            self.model.models[phase_count].fit(features, phase_labels)

    def evaluate(self):
        flow_predictions, flow_labels = [], []
        packet_correct = packet_count = 0
        for item in self.test_data:
            predictions = self.model.predict_packets(item)
            label = int(item[2])
            flow_predictions.append(predictions[-1])
            flow_labels.append(label)
            packet_count += len(predictions)
            packet_correct += predictions.count(label)
        metrics = {
            "packet_accuracy": packet_correct / packet_count,
            "flow_accuracy": float(np.mean(
                np.asarray(flow_predictions) == np.asarray(flow_labels))),
            "flow_f1": f1_score(flow_labels, flow_predictions),
            "flow_auc": roc_auc_score(flow_labels, flow_predictions),
            "test_flows": len(flow_labels),
        }
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return metrics

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self.model, handle, protocol=4)
        print("saved target model to {}".format(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",
                        default="src/advGenerate/environments/NetBeacon/config/netbeacon.yaml")
    cli = parser.parse_args(argv)
    args = recursive_namespace(read_yaml(cli.config))
    set_seed(args.trainer.seed)

    paths = {}
    for name in ("benign_train", "benign_test", "malicious_train", "malicious_test"):
        paths[name] = resolve_repo_path(getattr(args.data, name))
    benign_train = valid_flows(read_flow_pkl(paths["benign_train"]))
    benign_test = valid_flows(read_flow_pkl(paths["benign_test"]))
    malicious_train = valid_flows(read_flow_pkl(paths["malicious_train"]))
    malicious_test = valid_flows(read_flow_pkl(paths["malicious_test"]))
    rng = random.Random(args.trainer.seed)
    rng.shuffle(benign_train)
    rng.shuffle(malicious_train)
    limit = args.trainer.train_per_class
    if min(len(benign_train), len(malicious_train)) < limit:
        raise ValueError("not enough training flows for train_per_class={}".format(limit))
    train_data = (_labeled(benign_train[:limit], 0) +
                  _labeled(malicious_train[:limit], 1))
    test_data = _labeled(benign_test, 0) + _labeled(malicious_test, 1)

    model = NetBeacon(args.model.phases, args.model.depth, args.trainer.seed)
    trainer = NetBeaconTrainer(model, train_data, test_data)
    trainer.fit(args.model.top_features)
    metrics = trainer.evaluate()
    model_path = resolve_repo_path(args.trainer.model_path)
    trainer.save(model_path)
    report_path = resolve_repo_path(args.trainer.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
