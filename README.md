# NetMasquerade

![Licence](https://img.shields.io/github/license/09nat/NetMasquerade)
![Last commit](https://img.shields.io/github/last-commit/09nat/NetMasquerade)
![Language](https://img.shields.io/github/languages/count/09nat/NetMasquerade)

An implementation of NetMasquerade, introduced in:

_[A Hard-Label Black-Box Evasion Attack against ML-based Malicious Traffic Detection Systems](https://www.ndss-symposium.org/ndss-paper/a-hard-label-black-box-evasion-attack-against-ml-based-malicious-traffic-detection-systems/)_<br>
In the 33rd Network and Distributed System Security Symposium ([NDSS'26](https://www.ndss-symposium.org/ndss2026/)).<br>
Zixuan Liu, Yi Zhao, Zhuotao Liu, Qi Li, Chuanpu Fu, Guangmeng Zhou, and Ke Xu.

NetMasquerade uses reinforcement learning to reshape malicious network flows
until a target classifier no longer recognizes them. The agent chooses where
to insert a packet or change an inter-packet delay, while Traffic-BERT supplies
the new delay and packet-size values.

## Quick Start

The reference environment uses Python 3.8, PyTorch 1.13.1, and CUDA 11.7.

```bash
conda create -n netmasquerade python=3.8 -y
conda activate netmasquerade
pip install -r requirements.txt

# Train the NetBeacon target classifier.
bash scripts/train_target_model.sh

# Train and evaluate NetMasquerade.
bash scripts/run_netmasquerade.sh
```

These commands run the demo directly: NetBeacon is trained and
evaluated first, then NetMasquerade is trained and evaluated against it.

The second command uses `cuda:0` by default. Use `--device cuda:1` to select
another GPU, or `--device cpu` to run on the CPU. Results are written to
`outputs/netbeacon/` and `outputs/netmasquerade/`.

## Code and data

```text
scripts/                              experiment entry points
src/PacketProcessing/                 pcap -> CSV -> flow pickle
src/trafficMimic/                     Traffic-BERT and vocabularies
src/advGenerate/                      SAC agent, editing environment, NetBeacon
traffic_data/fuzzing/                 demo flows
pretrained_models/traffic_bert/       pretrained Traffic-BERT checkpoint
```

The demo data in `traffic_data/fuzzing/` is derived from the fuzzing trace
released by the [Kitsune project](https://github.com/ymirsky/Kitsune-py).

## Preparing another capture

The preprocessing scripts are adapted from
[PeerShark](https://github.com/pratiknarang/peershark).

The demo does not need this step. To process another `pcap` or
`pcapng`, install `tshark` and run from the repository root:

```bash
export PYTHONPATH="$PWD/src"

python -m PacketProcessing.FilterPackets \
  --pcap /path/to/captures \
  --out /path/to/packet_csv \
  --workers 4

python -m PacketProcessing.GenerateFlows \
  --csv /path/to/packet_csv \
  --out /path/to/flow_pickles \
  --timegap 300 \
  --threshold 510
```

`--pcap` may point to one capture, a directory, or a text file containing
capture paths. The second command groups packets into bidirectional flows,
starts a new flow after 300 seconds of inactivity, and caps each flow at 510
packets to fit Traffic-BERT's 512-token input.

## License

NetMasquerade is available under the [MIT License](LICENSE).

## Reference

```bibtex
@inproceedings{NDSS26-NetMasquerade,
  author    = {Zixuan Liu and Yi Zhao and Zhuotao Liu and Qi Li and
               Chuanpu Fu and Guangmeng Zhou and Ke Xu},
  title     = {A Hard-Label Black-Box Evasion Attack against ML-based
               Malicious Traffic Detection Systems},
  booktitle = {Network and Distributed System Security Symposium},
  publisher = {Internet Society},
  year      = {2026},
  doi       = {10.14722/ndss.2026.240916}
}
```
