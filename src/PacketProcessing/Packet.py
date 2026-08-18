"""Packet record parsed from FilterPackets' 13-column CSV schema."""


_LAST_PORTS = {}


class Packet:
    def __init__(self, fields):
        self.fields = fields
        if len(fields) != 13:
            raise ValueError("Packet expects 13 fields, got {}".format(len(fields)))

        self.source = fields[0]
        self.dest = fields[1]
        self.timestamp = float(fields[2])
        self.ihl = int(fields[3])
        self.diffserv = int(fields[4])
        self.ttl = int(fields[5])
        self.total_len = int(fields[6])
        self.proto = int(fields[7])
        self.tcp_dataOffset = int(fields[8])
        self.tcp_window = int(fields[9])
        self.udp_length = int(fields[10])
        self.src_port, self.dst_port = self._ports()

        if self.source < self.dest:
            endpoints = (self.source, self.dest, self.src_port, self.dst_port)
        else:
            endpoints = (self.dest, self.source, self.dst_port, self.src_port)
        self.key = "{}-{}-{}-{}-{}".format(*endpoints, self.proto)

    def _ports(self):
        cache_key = (self.source, self.dest, self.proto)
        src_text, dst_text = self.fields[11].strip(), self.fields[12].strip()
        if src_text and dst_text:
            ports = (int(float(src_text)), int(float(dst_text)))
            _LAST_PORTS[cache_key] = ports
            _LAST_PORTS[(self.dest, self.source, self.proto)] = ports[::-1]
            return ports
        return _LAST_PORTS.get(cache_key, (0, 0))
