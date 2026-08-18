"""Shared configuration, flow-loading, and feature helpers."""

import os
from pathlib import Path
import pickle
import random
from types import SimpleNamespace

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def recursive_namespace(data):
    if isinstance(data, dict):
        return SimpleNamespace(**{key: recursive_namespace(value)
                                  for key, value in data.items()})
    if isinstance(data, list):
        return [recursive_namespace(value) for value in data]
    return data


def read_yaml(yaml_path):
    with open(yaml_path) as handle:
        return yaml.safe_load(handle)


def resolve_repo_path(path):
    """Resolve config paths consistently against the repository root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def set_device(device):
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; use --device cpu")
    return resolved


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _FlowUnpickler(pickle.Unpickler):
    """Map historical ``Flow.Flow`` records to the packaged Flow class."""

    def find_class(self, module, name):
        if module == "Flow" and name == "Flow":
            from PacketProcessing.Flow import Flow
            return Flow
        return super().find_class(module, name)


def _flow_paths(source):
    if isinstance(source, (str, os.PathLike)):
        source = Path(source)
        if source.is_dir():
            return sorted(source.rglob("*.pkl"))
        return [source]
    return [Path(path) for path in source]


def read_flow_pkl(source):
    """Load trusted Flow pickle files with compatibility for legacy modules."""
    flows = []
    for path in _flow_paths(source):
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            data = _FlowUnpickler(handle).load()
        if not isinstance(data, list):
            raise TypeError("{} does not contain list[Flow]".format(path))
        flows.extend(data)
    return flows


def valid_flows(flow_list):
    return [flow for flow in flow_list if len(flow.timestp) > 1]


def timestamps_to_ipd(timestamps):
    timestamps = list(timestamps)
    if not timestamps:
        return []
    return [0.0] + [timestamps[index] - timestamps[index - 1]
                    for index in range(1, len(timestamps))]


def feature_extract(flow_list):
    """Return copied IPD and packet-size lists without mutating Flow objects."""
    flows = valid_flows(flow_list)
    return ([timestamps_to_ipd(flow.timestp) for flow in flows],
            [list(flow.total_len) for flow in flows])


def _addresses(flow):
    first_ip, second_ip, first_port, second_port, _protocol = flow.key.split("-")
    source_ip = [first_ip if direction == 1 else second_ip
                 for direction in flow.direction]
    destination_ip = [second_ip if direction == 1 else first_ip
                      for direction in flow.direction]
    source_port = [first_port if direction == 1 else second_port
                   for direction in flow.direction]
    destination_port = [second_port if direction == 1 else first_port
                        for direction in flow.direction]
    return source_ip, destination_ip, source_port, destination_port


def feature_extract_full(flow_list):
    flows = valid_flows(flow_list)
    ipd, size = feature_extract(flows)
    addresses = [_addresses(flow) for flow in flows]
    if not addresses:
        return ipd, size, [], [], [], []
    source, destination, source_port, destination_port = zip(*addresses)
    return ipd, size, list(source), list(destination), list(source_port), list(destination_port)
