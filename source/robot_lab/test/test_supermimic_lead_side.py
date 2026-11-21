"""
test_supermimic_lead_side.py — 先抬侧标记（r21）：观测/镜像/critic/回放/导出的接口约束（CPU，不启动 Isaac）。

@author pony
@date 2026-11-16
@version v1.0.0
@last_modified 2026-11-16

@changelog
    - v1.0.0 (2026-11-16): 初始创建
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robot_lab"))

from supermimic import kinematics, state, symmetry, terrain_map  # noqa: E402
from supermimic.learner import SuperMimicCfg, SuperMimicLearner  # noqa: E402
from supermimic.networks import Actor, Critic, LeadSideActor  # noqa: E402
from supermimic.preview_task import PreviewRewardCfg, PreviewTask  # noqa: E402

KIN = kinematics.RoverKinematics()
LIMITS = torch.tensor([[-0.58, 0.58]] * 4 + [[-1.3, 1.3]] * 4)


def _pattern():
    # 关于 y=0 对称的小图样（镜像需要成对的点）
    return torch.tensor([[x, y, 20.0] for x in (0.0, 0.3) for y in (-0.2, 0.0, 0.2)])


def _task(lead_obs: bool) -> PreviewTask:
    xs = torch.arange(0.0, 4.0 + 1e-9, 0.02)
    heights = torch.tensor([[0.12 if float(x) >= 1.2 else 0.0] * 101 for x in xs])
    hmap = terrain_map.HeightMap(heights, (0.0, -1.0), 0.02)
    return PreviewTask(hmap, KIN, _pattern(), LIMITS, PreviewRewardCfg(), lead_obs=lead_obs,
                       lane_center_y=torch.tensor([0.0]))


def _states(n):
    joints = torch.zeros(n, 12)
    joints[:, :4] = (torch.rand(n, 4) - 0.5) * 0.6
    pos = torch.stack((torch.rand(n) * 0.8 + 0.2, torch.zeros(n), torch.full((n,), 0.37)), -1)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, 4)
    return state.make_state(pos, quat, torch.randn(n, 3) * 0.1, torch.randn(n, 3) * 0.1, joints, torch.randn(n, 12) * 0.1)


def test_lead_obs_appends_command_fourth_dim_and_off_is_unchanged():
    on, off = _task(True), _task(False)
    assert (on.obs_dim, on.command_dim) == (off.obs_dim + 1, 4) and off.command_dim == 3
    s, prev = _states(5), torch.zeros(5, 12)
    cmd = torch.tensor([[0.3, 0.0, 0.0, 1.0]] * 3 + [[0.3, 0.0, 0.0, -1.0]] * 2)
    obs = on.policy_obs(s, cmd, prev)
    assert obs.shape[-1] == on.obs_dim and torch.equal(obs[:, -1], cmd[:, 3])
    assert torch.equal(obs[:, :-1], off.policy_obs(s, cmd[:, :3], prev))
    with pytest.raises(ValueError, match="4 维命令"):
        on.policy_obs(s, cmd[:, :3], prev)
    # 奖励只看前 3 维：标记取反不改变任何奖励项
    flipped = cmd.clone()
    flipped[:, 3] *= -1
    a, b = on.reward_terms(s, s, cmd, prev, prev), on.reward_terms(s, s, flipped, prev, prev)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_mirror_flips_lead_flag_and_stays_an_involution():
    task = _task(True)
    mirror = symmetry.RoverMirror.for_task(task, "cpu", symmetry.CONTEXT_DIM, terrain_map.RELIEF_DIM)
    x = torch.randn(4, 3, task.obs_dim)
    m = mirror.obs(x)
    assert torch.equal(m[..., -1], -x[..., -1]) and torch.equal(mirror.obs(m), x)
    c = torch.randn(4, 4)
    assert torch.equal(mirror.command(mirror.command(c)), c) and torch.equal(mirror.command(c)[:, 3], -c[:, 3])
    # 镜像后的观测 = 镜像状态（镜像世界）+ 取反标记：前 232 维与不带标记的镜像一致
    plain = symmetry.RoverMirror("cpu", task.obs_dim - 1, task.pattern)
    assert torch.equal(m[..., :-1], plain.obs(x[..., :-1]))
    with pytest.raises(ValueError, match="不一致"):  # 标记维未声明时拒绝构造
        symmetry.RoverMirror("cpu", task.obs_dim, task.pattern)


def test_critic_ignores_lead_flag():
    critic = Critic((16,), symmetry.CONTEXT_DIM)
    s = _states(6)
    cmd = torch.tensor([[0.3, 0.0, 0.0, 1.0]]).expand(6, 4)
    flipped = cmd * torch.tensor([1.0, 1.0, 1.0, -1.0])
    prev, ctx = torch.zeros(6, 12), torch.zeros(6, symmetry.CONTEXT_DIM)
    assert torch.equal(critic(s, cmd, prev, ctx), critic(s, flipped, prev, ctx))
    assert torch.equal(critic(s, cmd, prev, ctx), critic(s, cmd[:, :3], prev, ctx))


def test_lead_side_actor_is_scriptable_and_fixes_the_flag(tmp_path):
    actor = Actor(10, (8,)).eval()
    wrapped = LeadSideActor(actor, 1.0).eval()
    obs = torch.randn(4, 9)
    expect = actor(torch.cat((obs, torch.ones(4, 1)), -1))
    torch.jit.script(wrapped).save(str(tmp_path / "p.pt"))
    loaded = torch.jit.load(str(tmp_path / "p.pt"))
    assert torch.allclose(loaded(obs), expect) and torch.allclose(wrapped(obs[0]), expect[0])
    with pytest.raises(ValueError):
        LeadSideActor(actor, 0.5)


def test_learner_with_lead_task_trains_and_exports_232_interface(tmp_path):
    torch.manual_seed(0)
    task = _task(True)
    cfg = SuperMimicCfg(num_envs=8, buffer_len=40, wm_hidden=(32,), actor_hidden=(32,), critic_hidden=(32,),
                        wm_window=4, wm_batch=16, wm_updates=2, horizon=4, imag_batch=8, warmup_iterations=0,
                        context_dim=symmetry.CONTEXT_DIM, aux_dim=0, mirror_aug=True, mirror_loss_coef=1.0)
    hmap = task.hmap
    learner = SuperMimicLearner(cfg, "cpu", extra_fn=lambda s: terrain_map.relief_features(hmap, KIN, s),
                                extra_dim=terrain_map.RELIEF_DIM, task=task)
    assert learner.replay.command.shape[-1] == 4 and learner.actor.actor[0].in_features == task.obs_dim
    lead = torch.tensor([1.0, -1.0] * 4)
    prev = torch.zeros(8, 12)
    for _ in range(12):
        s = _states(8)
        cmd = torch.cat((torch.tensor([[0.3, 0.0, 0.0]]).expand(8, 3), lead[:, None]), -1)
        action = learner.act(task.policy_obs(s, cmd, prev))
        learner.observe(s, action, prev, cmd, torch.zeros(8, dtype=torch.bool), torch.zeros(8, symmetry.CONTEXT_DIM))
        prev = action
    logs = learner.update()
    assert math.isfinite(logs["actor_loss"]) and math.isfinite(logs["actor_mirror_loss"])
    learner.export_actor(tmp_path / "policy.pt")
    exported = torch.jit.load(str(tmp_path / "policy.pt"))
    obs = task.policy_obs(_states(3), torch.tensor([[0.3, 0.0, 0.0, 1.0]]).expand(3, 4), torch.zeros(3, 12))
    source = learner.actor_ema if learner.actor_ema is not None else learner.actor
    assert torch.allclose(exported(obs[:, :-1]), source(obs), atol=1e-6)
