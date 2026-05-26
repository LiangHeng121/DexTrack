"""
Minimal WandbAlgoObserver compatible with rl_games' AlgoObserver interface.
Mirrors the standard IsaacGymEnvs upstream implementation.
"""
import os

import wandb
from omegaconf import OmegaConf
from rl_games.common.algo_observer import AlgoObserver


class WandbAlgoObserver(AlgoObserver):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def before_init(self, base_name, config, experiment_name):
        # Unique id: derive from log_path (which already encodes seq + timestamp).
        # rl_games' experiment_name="." when we force flat layout, so it can't be used.
        cfg = self.cfg
        log_path = config.get("log_path", "") or config.get("train_dir", "")
        if log_path and log_path not in (".", "./"):
            cleaned = log_path.strip().rstrip("/")
            if cleaned.startswith("./"):
                cleaned = cleaned[2:]
            unique_token = cleaned.replace("/", "_") or experiment_name
        else:
            unique_token = experiment_name
        wandb_unique_id = f"uid_{unique_token}"
        print(f"Wandb using unique id {wandb_unique_id}")

        # Display name: <seq>_<TS> from the last two segments of log_path
        # (log_path layout: ./logs/<script_stem>/<seq>/<TS>).
        run_display_name = cfg.wandb_name or "policy"
        if log_path:
            parts = [p for p in log_path.rstrip("/").split("/") if p not in (".", "")]
            if len(parts) >= 2:
                run_display_name = f"{parts[-2]}_{parts[-1]}"
            elif parts:
                run_display_name = parts[-1]

        wandb_kwargs = dict(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity or None,
            group=cfg.wandb_group or None,
            tags=list(cfg.wandb_tags) if cfg.wandb_tags else None,
            sync_tensorboard=True,
            id=wandb_unique_id,
            name=run_display_name,
            resume="allow",
            monitor_gym=True,
        )
        if cfg.wandb_logcode_dir:
            wandb_kwargs["settings"] = wandb.Settings(code_dir=cfg.wandb_logcode_dir)

        wandb.init(**wandb_kwargs)
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True),
                            allow_val_change=True)

        # Make `info/epochs` the default x-axis for all rl_games scalars.
        # Without this, `sync_tensorboard=True` collapses every TB step onto a
        # single `_step`, so `rewards/iter` (TB step=epoch) and `rewards/step`
        # (TB step=frame) end up on the same axis and look wrong.
        wandb.define_metric("info/epochs")
        for pattern in (
            "rewards/*", "shaped_rewards/*", "episode_lengths/*",
            "losses/*", "performance/*",
            "info/lr", "info/last_lr", "info/kl", "info/lr_mul", "info/e_clip",
        ):
            wandb.define_metric(pattern, step_metric="info/epochs")

    def after_init(self, algo):
        pass

    def process_infos(self, infos, done_indices):
        pass

    def after_steps(self):
        pass

    def after_print_stats(self, frame, epoch_num, total_time):
        pass
