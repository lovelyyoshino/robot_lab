"""验证 CLI/Hydra 续训优先级，防止昂贵训练静默丢弃 checkpoint。

@author pony
@date 2026-09-17
@version v0.1.0
@last_modified 2026-09-17
@changelog
    - v0.1.0 (2026-09-17): 回归未指定 CLI 覆盖时保留配置，显式续训仍有最高优先级。
"""

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("configured_resume", [False, True])
@pytest.mark.parametrize("explicit_resume", [False, True])
def test_resume_override_only_when_cli_explicit(configured_resume, explicit_resume):
    path = Path(__file__).parents[3] / "scripts/reinforcement_learning/rsl_rl/cli_args.py"
    spec = importlib.util.spec_from_file_location("rsl_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = argparse.ArgumentParser()
    module.add_rsl_rl_args(parser)
    # parse_known_args 与真实 train.py 相同，Hydra 的配置项留给后续装饰器处理。
    tokens = ["agent.resume=true", "agent.load_checkpoint=/tmp/source.pt"]
    if explicit_resume:
        tokens += ["--resume", "--checkpoint", "/tmp/explicit.pt"]
    args, unknown = parser.parse_known_args(tokens)
    assert unknown == tokens[:2]
    cfg = SimpleNamespace(resume=configured_resume, load_checkpoint="/tmp/source.pt", logger="tensorboard")
    module.update_rsl_rl_cfg(cfg, args)
    assert cfg.resume is (configured_resume or explicit_resume)
    assert cfg.load_checkpoint == ("/tmp/explicit.pt" if explicit_resume else "/tmp/source.pt")
