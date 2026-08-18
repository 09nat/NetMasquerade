import argparse
import json
import time

import numpy as np
from tqdm import tqdm

from advGenerate.environments.environment import make_env
from advGenerate.rl_utils import ReplayBuffer
from advGenerate.sac import SAC
from trafficMimic.dataset.vocab import SizeVocab, TimeVocab
from trafficMimic.utils import (read_yaml, recursive_namespace,
                                resolve_repo_path, set_device, set_seed)


class Trainer:
    def __init__(self, rl_args, bert_args):
        self.rl_args = rl_args
        self.bert_args = bert_args
        self.device = set_device(rl_args.trainer.device)
        timevocab = TimeVocab.load_vocab(resolve_repo_path(
            bert_args.trainer.timevocab_pth))
        sizevocab = SizeVocab.load_vocab(resolve_repo_path(
            bert_args.trainer.sizevocab_pth))
        self.replay_buffer = ReplayBuffer(rl_args.trainer.buffer_size)
        model = rl_args.model
        self.agent = SAC(
            len(timevocab), len(sizevocab), model.embedding_dim,
            model.state_dim, model.hidden_dim, model.actor_lr,
            model.critic_lr, model.alpha_lr, model.target_entropy,
            model.tau, model.gamma, self.device)

    @staticmethod
    def _next_detected(env):
        """Return the next initially detected flow, excluding trivial evasions."""
        for _attempt in range(len(env.dataset)):
            state = env.reset()
            if env.get_res() == 1:
                env.evaluator.reset_query_count()
                return state
        raise RuntimeError("target model detects no flows in this dataset")

    def train(self):
        env = make_env(self.rl_args, self.bert_args, "train")
        episodes = int(self.rl_args.trainer.episodes)
        successes, returns = [], []
        started = time.perf_counter()
        progress = tqdm(range(episodes), desc="training SAC", unit="flow")
        for _ in progress:
            state = self._next_detected(env)
            done = False
            episode_return = 0.0
            evaded = False
            while not done:
                action = self.agent.take_action(state)
                next_state, reward, done, evaded = env.step(action)
                self.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                ready = max(int(self.rl_args.trainer.minimal_size),
                            int(self.rl_args.trainer.batch_size))
                if len(self.replay_buffer) >= ready:
                    batch = self.replay_buffer.sample(
                        self.rl_args.trainer.batch_size)
                    transition = dict(zip(
                        ("states", "actions", "rewards", "next_states", "dones"),
                        batch))
                    self.agent.update(transition)
            successes.append(int(evaded))
            returns.append(episode_return)
            window = min(100, len(successes))
            rolling_asr = float(np.mean(successes[-window:]))
            progress.set_postfix(asr="{:.3f}".format(rolling_asr),
                                 replay=len(self.replay_buffer))

        self.agent.save_model(resolve_repo_path(
            self.rl_args.trainer.model_save_pth))
        return {
            "episodes": episodes,
            "training_asr": float(np.mean(successes)),
            "mean_return": float(np.mean(returns)),
            "elapsed_seconds": time.perf_counter() - started,
        }

    def evaluate(self, eval_episodes, feedback):
        """Evaluate conditional ASR on flows initially detected as malicious."""
        env = make_env(self.rl_args, self.bert_args, "test")
        attempted = initially_detected = successes = 0
        edits, attack_queries = [], []
        started = time.perf_counter()
        progress = tqdm(range(int(eval_episodes)),
                        desc="evaluation ({})".format(
                            "feedback" if feedback else "no feedback"),
                        unit="flow")
        for _ in progress:
            state = env.reset()
            attempted += 1
            baseline = env.get_res()
            env.evaluator.reset_query_count()
            if baseline == 0:
                continue
            initially_detected += 1
            evaded = False
            if feedback:
                done = False
                while not done:
                    action = self.agent.take_deterministic_action(state)
                    state, _reward, done, evaded = env.step(action)
            else:
                for _step in range(int(self.rl_args.trainer.max_stop_step)):
                    action = self.agent.take_deterministic_action(state)
                    should_stop = self.agent.eval_stop(
                        state, self.rl_args.trainer.stop_threshold, action)
                    state = env.apply_action(action)
                    if should_stop:
                        break

                evaded = env.get_res() == 0
            successes += int(evaded)
            edits.append(env.step_num)
            attack_queries.append(env.evaluator.query_count)
            progress.set_postfix(asr="{:.3f}".format(
                successes / initially_detected))

        elapsed = time.perf_counter() - started
        if initially_detected == 0:
            raise RuntimeError("target detected none of the evaluation flows")
        return {
            "feedback": bool(feedback),
            "sampled_flows": attempted,
            "initially_detected": initially_detected,
            "initial_detection_rate": initially_detected / attempted,
            "successful_evasions": successes,
            "conditional_asr": successes / initially_detected,
            "mean_edits": float(np.mean(edits)),
            "mean_attack_queries": float(np.mean(attack_queries)),
            "elapsed_seconds": elapsed,
            "milliseconds_per_detected_flow": 1000.0 * elapsed / initially_detected,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-config",
                        default="src/advGenerate/config/sac.yaml")
    parser.add_argument("--bert-config",
                        default="src/trafficMimic/config/bert.yaml")
    parser.add_argument("--device", help="override cpu/cuda:N from config")
    parser.add_argument("--skip-train", action="store_true")
    cli = parser.parse_args(argv)
    rl_args = recursive_namespace(read_yaml(cli.rl_config))
    bert_args = recursive_namespace(read_yaml(cli.bert_config))
    if cli.device:
        rl_args.trainer.device = cli.device
    set_seed(rl_args.trainer.seed)

    target_path = resolve_repo_path(rl_args.trainer.target_model_pth)
    if not target_path.is_file():
        raise FileNotFoundError(
            "target checkpoint is missing; run scripts/train_target_model.sh first: {}"
            .format(target_path))
    trainer = Trainer(rl_args, bert_args)
    agent_path = resolve_repo_path(rl_args.trainer.model_save_pth)
    training = None
    if not cli.skip_train:
        training = trainer.train()
    elif not (agent_path / "actor.pth").is_file():
        raise FileNotFoundError("no saved SAC agent under {}".format(agent_path))
    trainer.agent.load_model(agent_path)

    evaluation_count = int(rl_args.trainer.test_episodes)
    with_feedback = trainer.evaluate(evaluation_count, feedback=True)
    without_feedback = trainer.evaluate(evaluation_count, feedback=False)
    report_path = resolve_repo_path(rl_args.trainer.report_path)
    report = {
        "seed": rl_args.trainer.seed,
        "device": str(trainer.device),
        "edit_budget": int(rl_args.trainer.max_stop_step),
        "training": training,
        "with_feedback": with_feedback,
        "without_feedback": without_feedback,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("report written to {}".format(report_path))
    return report


if __name__ == "__main__":
    main()
