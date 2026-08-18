"""Target-evaluator adapter used by the NetMasquerade environment."""

import pickle

class NetBeaconEvaluator:
    def __init__(self, model_path):
        with open(model_path, "rb") as handle:
            self.model = pickle.load(handle)
        self.query_count = 0

    def evaluate(self, ipd, size):
        """Return 1 for malicious and 0 for benign."""
        self.query_count += 1
        return self.model.predict((ipd, size))

    def reset_query_count(self):
        self.query_count = 0
