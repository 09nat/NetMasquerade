import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


DEFAULT_PHASES = [2, 4, 8]


class NetBeacon:
    def __init__(self, phases=None, depth=9, random_state=512):
        self.phases = list(phases or DEFAULT_PHASES)
        if not self.phases or self.phases != sorted(set(self.phases)):
            raise ValueError("phases must be a non-empty strictly increasing list")
        self.depth = int(depth)
        self.random_state = int(random_state)
        self.features = ["all" for _ in self.phases]
        self.models = [
            RandomForestClassifier(n_estimators=1, max_depth=self.depth,
                                   random_state=self.random_state + index)
            for index, _phase in enumerate(self.phases)
        ]

    def get_phase_features(self, data, phase_count):
        """Build size and IPD features for flows long enough for one phase.

        Each item is ``(ipd, size[, label])``. Unlike the historical experiment
        script, size is sent to byte bins and IPD is sent to delay bins.
        """
        phase = self.phases[phase_count]
        size_dist = self._distribution(data, phase, "len", 1,
                                       step=16, minimum=0, maximum=1504, start=0)
        ipd_dist = self._distribution(data, phase, "ipd", 0,
                                      step=2e-5, minimum=1e-5,
                                      maximum=1.1e-3, start=1)
        size_stats = self._statistics(data, phase, "len", 1, start=0)
        ipd_stats = self._statistics(data, phase, "ipd", 0, start=1)
        frame = pd.concat([size_dist, ipd_dist, size_stats, ipd_stats], axis=1)
        selected = self.features[phase_count]
        return frame if selected == "all" else frame[selected]

    @staticmethod
    def _distribution(data, phase, prompt, feature_index, step, minimum,
                      maximum, start):
        names = ["{}_less_{}".format(prompt, minimum)]
        current = minimum
        while current < maximum:
            names.append("{}_in_{}_{}".format(prompt, current, current + step))
            current += step
        names.append("{}_more_{}".format(prompt, maximum))
        rows = []
        for item in data:
            if len(item[feature_index]) < phase:
                continue
            values = [0] * len(names)
            for value in item[feature_index][start:phase]:
                if value < minimum:
                    values[0] += 1
                elif value >= maximum:
                    values[-1] += 1
                else:
                    values[int((value - minimum) / step) + 1] += 1
            rows.append(values)
        return pd.DataFrame(rows, columns=names)

    @staticmethod
    def _statistics(data, phase, prompt, feature_index, start):
        names = ["min_{}".format(prompt), "max_{}".format(prompt),
                 "avg_{}".format(prompt), "var_{}".format(prompt)]
        rows = []
        for item in data:
            if len(item[feature_index]) < phase:
                continue
            values = item[feature_index][start:phase]
            rows.append([np.min(values), np.max(values), np.mean(values),
                         np.var(values)])
        return pd.DataFrame(rows, columns=names)

    def predict_packets(self, flow_data):
        if len(flow_data[0]) < self.phases[0]:
            raise ValueError("flow is shorter than the first NetBeacon phase")
        predictions = [0] * (self.phases[0] - 1)
        for phase_count, phase in enumerate(self.phases):
            if len(flow_data[0]) < phase:
                break
            features = self.get_phase_features([flow_data], phase_count)
            prediction = int(self.models[phase_count].predict(features)[0])
            upper = (self.phases[phase_count + 1] - 1
                     if phase_count + 1 < len(self.phases)
                     else len(flow_data[0]))
            predictions.extend([prediction] * max(0, upper - (phase - 1)))
        return predictions[:len(flow_data[0])]

    def predict(self, flow_data):
        return self.predict_packets(flow_data)[-1]
