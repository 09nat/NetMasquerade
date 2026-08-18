"""Flow record and deterministic packet-to-flow aggregation."""

from pathlib import Path
import pickle


# 510 packets + start/end tokens exactly fills traffic-BERT's 512 positions.
DEFAULT_PACKET_THRESHOLD = 510


class Flow:
    def __init__(self, firstpacket):
        self.key = firstpacket.key
        self.timestp = [firstpacket.timestamp]
        self.total_len = [firstpacket.total_len]
        self.diffserv = [firstpacket.diffserv]
        self.ttl = [firstpacket.ttl]
        self.tcp_dataOffset = [firstpacket.tcp_dataOffset]
        self.tcp_window = [firstpacket.tcp_window]
        self.udp_length = [firstpacket.udp_length]
        self.ipd = [0]
        self.direction = [1 if firstpacket.source < firstpacket.dest else 2]

    def addPacket(self, packet):
        if packet.key != self.key:
            raise ValueError("packet does not belong to flow {}".format(self.key))
        self.timestp.append(packet.timestamp)
        self.total_len.append(packet.total_len)
        self.diffserv.append(packet.diffserv)
        self.ttl.append(packet.ttl)
        self.tcp_dataOffset.append(packet.tcp_dataOffset)
        self.tcp_window.append(packet.tcp_window)
        self.udp_length.append(packet.udp_length)
        self.ipd.append(self.timestp[-1] - self.timestp[-2])
        self.direction.append(1 if packet.source < packet.dest else 2)

    def getEnd(self):
        return self.timestp[-1]

    def __len__(self):
        return len(self.timestp)


def packetsToFlows_and_write(packets, timegap, filename,
                             threshold=DEFAULT_PACKET_THRESHOLD):
    """Aggregate packets and write a ``list[Flow]`` pickle."""
    if timegap < 0:
        raise ValueError("timegap must be non-negative")
    if threshold < 2:
        raise ValueError("threshold must be at least 2")

    packets.sort(key=lambda packet: (packet.key, packet.timestamp))
    flows = []
    current = None
    for packet in packets:
        if (current is None
                or current.key != packet.key
                or packet.timestamp - current.getEnd() > timegap
                or len(current) >= threshold):
            if current is not None and len(current) > 1:
                flows.append(current)
            current = Flow(packet)
        else:
            current.addPacket(packet)
    if current is not None and len(current) > 1:
        flows.append(current)

    output = Path(filename)
    if output.suffix == ".csv":
        output = output.with_suffix(".pkl")
    elif output.suffix != ".pkl":
        output = Path(str(output) + ".pkl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(flows, handle, protocol=4)
    print("{}: {} flows".format(output, len(flows)))
    return str(output)
