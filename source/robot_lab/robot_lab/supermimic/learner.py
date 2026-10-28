"""
learner.py — SuperMimic 训练器：世界模型多步拟合 + 想象 BPTT actor + λ-return critic。

与论文/supermimic_vibed 的对应关系：
- 世界模型在 k 步自回归窗口上拟合（Eq.1），误差在增量空间按 delta_norm 归一化；
  输入含动力学上下文 ξ（论文 sim context），另有 aux 头回归 env 的关节加速度惩罚量；
- actor 从回放状态出发想象 H 步，奖励/观测/存活全部解析可导，梯度穿过世界模型（Eq.2）；
- critic 默认拟合想象轨迹的 λ-return（Eq.3），目标用 EMA 目标网络，避免自举发散；
  critic_real_returns=True 时改为拟合回放里的真实 λ-return（越障任务默认，偏离论文：
  只在想象上拟合的 critic 会把"卡在台阶前"估得和"爬上去"一样高，v1 因此退化）；
- anchor_coef>0 时 actor 额外受热启动 Teacher 的信赖域约束（论文没有）。
世界模型参数在 actor 反传时 requires_grad=False：梯度流经它，但只更新 actor。

@author pony
@date 2026-10-26
@version v0.6.0
@last_modified 2026-10-27
@changelog
    - v0.6.0 (2026-10-27): 锚定损失支持 task.anchor_weight 按状态加权；文档补充与论文的差异。
    - v0.5.0 (2026-10-26): critic 可改用真实回报（critic_real_returns）；actor 信赖域 anchor_coef。
    - v0.4.1 (2026-10-26): 新增 critic_warmup_iterations（热启动时先只训 critic），记录 actor 梯度范数。
    - v0.4.0 (2026-10-26): 观测/奖励/存活改由 task 对象提供（平地 FlatTask / 越障 PreviewTask），critic 可读地形。
    - v0.3.0 (2026-10-26): 新增 extra_fn（随状态变化的世界模型输入，用于地形 relief），wm_step 统一调用入口。
    - v0.2.0 (2026-10-26): 接入 ξ 与 aux 头；新增 actor 更新间隔与奖励偏差诊断（wm_reward_bias）。
    - v0.1.0 (2026-10-26): 初始创建
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field

import torch

from . import rover_task
from .networks import Actor, Critic, WorldModel
from .replay import ReplayBuffer
from .state import compute_delta, state_error


@dataclass
class SuperMimicCfg:
    num_envs: int = 4096
    buffer_len: int = 1000
    wm_hidden: tuple[int, ...] = (1024, 1024, 1024)
    actor_hidden: tuple[int, ...] = (512, 256, 128)
    critic_hidden: tuple[int, ...] = (512, 256, 128)
    context_dim: int = 0
    aux_dim: int = 0
    wm_window: int = 8
    wm_batch: int = 2048
    wm_updates: int = 10
    wm_lr: float = 3e-4
    aux_loss_coef: float = 1.0
    horizon: int = 32
    imag_batch: int = 4096
    actor_lr: float = 2e-4
    critic_lr: float = 3e-4
    critic_epochs: int = 2
    # actor 每 N 轮更新一次：给世界模型时间跟上策略分布的变化，抑制"钻模型空子"。
    actor_update_every: int = 1
    gamma: float = 0.99
    lam: float = 0.95
    target_tau: float = 0.02
    grad_clip: float = 1.0
    actor_elem_clip: float = 0.1
    action_noise: float = 0.1
    action_bound: float = 6.2831853  # 见 networks.Actor：对齐 env 轮速目标 clip
    action_squash: str = "tanh"
    # 前若干轮只拟合世界模型：模型尚未成形时 BPTT 的梯度是噪声，会把 actor 推坏。
    warmup_iterations: int = 50
    # 热启动（actor 已会走）时：世界模型预热后再只训 critic 若干轮。随机 critic 的自举值会让第一批
    # BPTT 梯度把已训好的 actor 推坏；从零训练时设 0。
    critic_warmup_iterations: int = 0
    # critic 用真实回报拟合（回放窗口上的 λ-return，env 奖励），代替想象轨迹：想象视界只有 H 步，
    # 越台阶这种长时序成败只能靠 critic 承接；只用想象数据拟合的 critic 会把卡台阶也评得很高（实测）。
    critic_real_returns: bool = False
    real_return_window: int = 32
    real_return_batch: int = 2048
    # 信赖域：actor 动作向冻结的参考策略（热启动来源）回归，系数为 0 时关闭。
    anchor_coef: float = 0.0
    # 预热期 actor 近似输出 0（静止），只靠 0.1 噪声采不到行驶数据；预热期加大探索。
    warmup_action_noise: float = 0.5
    reward: rover_task.RoverRewardCfg = field(default_factory=rover_task.RoverRewardCfg)


class SuperMimicLearner:
    def __init__(self, cfg: SuperMimicCfg, device, extra_fn=None, extra_dim: int = 0, task=None):
        """extra_fn(state) → [..., extra_dim]：随状态变化的世界模型输入（如地形 relief_features），
        每一步都从当前（预测）状态重新计算并拼在 ξ 之后；梯度可穿过它回到位姿。
        task：观测/奖励/存活/critic 额外输入的提供者，缺省为平地 rover_task.FlatTask。"""
        self.cfg, self.device = cfg, torch.device(device)
        self.task = task if task is not None else rover_task.FlatTask(cfg.reward)
        self.extra_fn, self.extra_dim = extra_fn, extra_dim
        self.world_model = WorldModel(cfg.wm_hidden, cfg.context_dim + extra_dim, cfg.aux_dim).to(self.device)
        self.actor = Actor(self.task.obs_dim, cfg.actor_hidden, cfg.action_bound, cfg.action_squash).to(self.device)
        self.critic = Critic(cfg.critic_hidden, cfg.context_dim + self.task.critic_extra_dim).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.wm_opt = torch.optim.Adam(self.world_model.parameters(), lr=cfg.wm_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, betas=(0.9, 0.95))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.replay = ReplayBuffer(cfg.buffer_len, cfg.num_envs, self.device,
                                   context_dim=cfg.context_dim, aux_dim=cfg.aux_dim)
        self.iteration = 0
        self.skipped_actor_updates = 0
        self.anchor_actor = None

    def set_anchor(self, actor) -> None:
        """冻结一份参考 actor（通常是热启动时的 Teacher），用于 anchor_coef 信赖域。"""
        self.anchor_actor = copy.deepcopy(actor).requires_grad_(False).eval()

    # ---------------- 真实环境交互 ----------------
    @torch.no_grad()
    def act(self, obs: torch.Tensor, explore: bool = True) -> torch.Tensor:
        action = self.actor(obs)
        if explore:
            warm = self.iteration < self.cfg.warmup_iterations
            noise = self.cfg.warmup_action_noise if warm else self.cfg.action_noise
            action = action + noise * torch.randn_like(action)
        return action

    def observe(self, state, action, prev_action, command, done, context=None, aux=None, reward=None,
                terminated=None) -> None:
        self.replay.add(state, action, prev_action, command, done, context, aux, reward, terminated)

    def wm_context(self, state, context):
        """世界模型实际吃的上下文 = 固定 ξ ⊕ extra_fn(state)。"""
        if self.extra_fn is None:
            return context
        return torch.cat((context, self.extra_fn(state)), dim=-1)

    def wm_step(self, state, action, context):
        return self.world_model(state, action, self.wm_context(state, context))

    def _acc_sq(self, state, action, context):
        return self.world_model.aux(state, action, self.wm_context(state, context)) if self.cfg.aux_dim else None

    # ---------------- 世界模型 ----------------
    def update_world_model(self) -> dict[str, float]:
        cfg, wm = self.cfg, self.world_model
        logs = {"wm_loss": 0.0, "wm_err_1step": 0.0, "wm_err_last": 0.0, "wm_aux_loss": 0.0}
        for _ in range(cfg.wm_updates):
            batch = self.replay.sample_windows(cfg.wm_batch, cfg.wm_window)
            states, actions, ctx, valid = batch["state"], batch["action"], batch["context"], batch["valid"]
            # 归一化统计只取单步有效转移，跨回合样本的"增量"是 reset 跳变，会撑大方差。
            mask0 = valid[:, 0] > 0
            if mask0.any():
                wm.update_normalizers(states[mask0, 0], actions[mask0, 0],
                                      compute_delta(states[mask0, 0], states[mask0, 1]),
                                      self.wm_context(states[mask0, 0], ctx[mask0, 0]), batch["aux"][mask0, 0])
            pred = states[:, 0]
            losses = []
            for j in range(cfg.wm_window):
                pred = self.wm_step(pred, actions[:, j], ctx[:, j])
                err = state_error(pred, states[:, j + 1]) / wm.delta_norm.std
                losses.append((err.square().mean(-1) * valid[:, j]).sum() / valid[:, j].sum().clamp_min(1.0))
            loss = torch.stack(losses).mean()
            aux_loss = torch.zeros((), device=self.device)
            if cfg.aux_dim:
                # aux 在真实状态上监督（逐步、无自回归）：它是本步观测量，不参与状态推进。
                m = valid.reshape(-1) > 0
                rows = valid.numel()
                # 显式给出行数：context_dim=0 时张量为空，reshape(-1, 0) 无法推断。
                flat = lambda x: x.reshape(rows, x.shape[-1])[m]  # noqa: E731
                s_flat = flat(states[:, :-1])
                pred_aux = wm.aux_normalized(s_flat, flat(actions), self.wm_context(s_flat, flat(ctx)))
                aux_loss = (pred_aux - wm.aux_norm.normalize(flat(batch["aux"]))).square().mean()
            total = loss + cfg.aux_loss_coef * aux_loss
            self.wm_opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), cfg.grad_clip)
            self.wm_opt.step()
            logs["wm_loss"] += loss.item() / cfg.wm_updates
            logs["wm_err_1step"] += losses[0].item() / cfg.wm_updates
            logs["wm_err_last"] += losses[-1].item() / cfg.wm_updates
            logs["wm_aux_loss"] += aux_loss.item() / cfg.wm_updates
        logs.update(self._reward_gap(batch))
        return logs

    @torch.no_grad()
    def _reward_gap(self, batch) -> dict[str, float]:
        """用真实动作开环推 k 步，比较模型状态与真实状态上的解析奖励。

        wm_reward_bias > 0 表示模型系统性高估奖励——actor 钻空子时首先体现在这里。
        真实侧的关节加速度项用 env 真值（batch aux），模型侧用 aux 头。
        """
        cfg, wm = self.cfg, self.world_model
        states, actions, prev, cmd, ctx, valid = (batch[k] for k in
                                                  ("state", "action", "prev_action", "command", "context", "valid"))
        pred, diff, absd, n = states[:, 0], 0.0, 0.0, valid.sum().clamp_min(1.0)
        for j in range(cfg.wm_window):
            nxt = self.wm_step(pred, actions[:, j], ctx[:, j])
            real_acc = batch["aux"][:, j] if cfg.aux_dim else None
            r_model = self.task.reward(pred, nxt, cmd[:, j], actions[:, j], prev[:, j],
                                        self._acc_sq(pred, actions[:, j], ctx[:, j]))
            r_real = self.task.reward(states[:, j], states[:, j + 1], cmd[:, j], actions[:, j], prev[:, j], real_acc)
            d = (r_model - r_real) * valid[:, j]
            diff, absd = diff + d.sum(), absd + d.abs().sum()
            pred = nxt
        return {"wm_reward_bias": (diff / n).item(), "wm_reward_abs_gap": (absd / n).item()}

    # ---------------- 想象 ----------------
    def imagine(self, start: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """从回放起点想象 H 步；命令与 ξ 在整段 rollout 内不变（ξ 只在 reset 时改变）。"""
        cfg = self.cfg
        state, prev_action, command, ctx = start["state"], start["prev_action"], start["command"], start["context"]
        states, prev_actions, rewards, survivals = [state], [prev_action], [], []
        for _ in range(cfg.horizon):
            action = self.actor(self.task.policy_obs(state, command, prev_action))
            next_state = self.wm_step(state, action, ctx)
            rewards.append(self.task.reward(state, next_state, command, action, prev_action,
                                            self._acc_sq(state, action, ctx)))
            survivals.append(self.task.survival(next_state))
            state, prev_action = next_state, action
            states.append(state)
            prev_actions.append(prev_action)
        return {
            "state": torch.stack(states),  # [H+1, B, S]
            "prev_action": torch.stack(prev_actions),
            "command": command,
            "context": ctx,
            "reward": torch.stack(rewards),  # [H, B]
            "survival": torch.stack(survivals),
        }

    def lambda_returns(self, reward, survival, values) -> torch.Tensor:
        """values: [H+1, B]；survival 作为软折扣，翻倒后的价值按存活概率衰减。"""
        cfg = self.cfg
        returns = [values[-1]]
        for t in reversed(range(reward.shape[0])):
            cont = cfg.gamma * survival[t]
            returns.append(reward[t] + cont * ((1 - cfg.lam) * values[t + 1] + cfg.lam * returns[-1]))
        return torch.stack(returns[::-1][:-1])  # [H, B]

    def critic_context(self, state, context):
        if not self.task.critic_extra_dim:
            return context
        return torch.cat((context, self.task.critic_extra(state)), dim=-1)

    def _values(self, critic, traj) -> torch.Tensor:
        steps = traj["state"].shape[0]
        cmd = traj["command"].expand(steps, -1, -1)
        ctx = self.critic_context(traj["state"], traj["context"].expand(steps, -1, -1))
        return critic(traj["state"], cmd, traj["prev_action"], ctx)

    def update_actor_critic(self, update_actor: bool = True) -> dict[str, float]:
        cfg = self.cfg
        start = self.replay.sample_states(cfg.imag_batch)
        with torch.no_grad():
            self.critic.norm.update(self.critic.inputs(start["state"], start["command"], start["prev_action"],
                                                       self.critic_context(start["state"], start["context"])))

        # actor：梯度穿过世界模型与 critic，但只更新 actor 参数。
        self.world_model.requires_grad_(False)
        self.critic.requires_grad_(False)
        try:
            traj = self.imagine(start)
            returns = self.lambda_returns(traj["reward"], traj["survival"], self._values(self.critic, traj))
            actor_loss = -returns[0].mean()
            anchor_mse = torch.zeros((), device=self.device)
            if cfg.anchor_coef > 0 and self.anchor_actor is not None:
                obs0 = self.task.policy_obs(start["state"], start["command"], start["prev_action"])
                with torch.no_grad():
                    ref = self.anchor_actor(obs0)
                per = (self.actor(obs0) - ref).square().sum(-1)
                # 任务可按状态加权（如后轮贴台阶立面时），只放大该处对 Teacher 的约束
                weight = self.task.anchor_weight(start["state"]) if hasattr(self.task, "anchor_weight") else None
                anchor_mse = (per * weight).mean() if weight is not None else per.mean()
                actor_loss = actor_loss + cfg.anchor_coef * anchor_mse
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
        finally:
            self.world_model.requires_grad_(True)
            self.critic.requires_grad_(True)
        grads = [p.grad for p in self.actor.parameters() if p.grad is not None]
        if not all(torch.isfinite(g).all() for g in grads):
            # 一次 NaN 更新会污染 actor，进而把 NaN 动作送进仿真和回放；跳过并计数，不静默吞掉。
            self.skipped_actor_updates += 1
            return {"actor_skipped": float(self.skipped_actor_updates)}
        for p in self.actor.parameters():
            if p.grad is not None:
                p.grad.clamp_(-cfg.actor_elem_clip, cfg.actor_elem_clip)
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip).item()
        if update_actor:
            self.actor_opt.step()

        traj = {k: v.detach() for k, v in traj.items()}
        if cfg.critic_real_returns:
            critic_loss = self._fit_critic_real()
        else:
            critic_loss = self._fit_critic_imagined(traj)
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                tp.lerp_(p, cfg.target_tau)
            self.target_critic.norm.load_state_dict(self.critic.norm.state_dict())
        return {
            "actor_loss": actor_loss.item(),
            "anchor_mse": anchor_mse.item(),
            "actor_grad_norm": actor_grad_norm,
            "critic_loss": critic_loss,
            "imag_reward": traj["reward"].mean().item(),
            "imag_survival": traj["survival"][-1].mean().item(),
        }

    def _fit_critic_imagined(self, traj) -> float:
        """在截断梯度的想象轨迹上拟合目标网络给出的 λ-return。"""
        cfg = self.cfg
        with torch.no_grad():
            target = self.lambda_returns(traj["reward"], traj["survival"], self._values(self.target_critic, traj))
            # 已大概率翻倒的后续步价值无意义，按累计存活加权。
            weight = torch.cat((torch.ones_like(traj["survival"][:1]), traj["survival"][:-1])).cumprod(0)
        for _ in range(cfg.critic_epochs):
            value = self._values(self.critic, traj)[:-1]
            critic_loss = ((value - target).square() * weight).mean()
            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
            self.critic_opt.step()
        return critic_loss.item()

    def real_lambda_returns(self, batch, values) -> torch.Tensor:
        """回放窗口上的 λ-return。values: [B, L+1]（目标 critic）。

        真终止：不自举；超时/成功截断：回放里没有真实后继，用本步价值近似自举；
        窗口末端用 V(s_L) 自举。返回 [B, L]。
        """
        cfg = self.cfg
        r, done, term = batch["reward"], batch["done"], batch["terminated"]
        L = r.shape[1]
        G = values[:, L]
        out = []
        for t in reversed(range(L)):
            cont = ((1 - cfg.lam) * values[:, t + 1] + cfg.lam * G)
            g = r[:, t] + cfg.gamma * cont
            g = torch.where(done[:, t], r[:, t] + cfg.gamma * values[:, t] * (~term[:, t]).float(), g)
            out.append(g)
            G = g
        return torch.stack(out[::-1], dim=1)

    def _fit_critic_real(self) -> float:
        cfg = self.cfg
        L = cfg.real_return_window
        batch = self.replay.sample_windows(cfg.real_return_batch, L + 1)
        S = batch["state"][:, : L + 1]
        cmd, prev, ctx = batch["command"], batch["prev_action"], batch["context"]
        with torch.no_grad():
            values = self.target_critic(S, cmd, prev, self.critic_context(S, ctx))
            target = self.real_lambda_returns({k: batch[k][:, :L] for k in ("reward", "done", "terminated")}, values)
            done = batch["done"][:, :L]
            # 第 t 步之前没有 done 才属于同一回合；done 那一步本身的奖励仍有效。
            alive = torch.cat((torch.ones_like(done[:, :1]), ~done[:, :-1]), dim=1).float().cumprod(1)
        for _ in range(cfg.critic_epochs):
            value = self.critic(S[:, :L], cmd[:, :L], prev[:, :L], self.critic_context(S[:, :L], ctx[:, :L]))
            critic_loss = ((value - target).square() * alive).sum() / alive.sum().clamp_min(1.0)
            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
            self.critic_opt.step()
        return critic_loss.item()

    def update(self) -> dict[str, float]:
        logs = self.update_world_model()
        cfg = self.cfg
        warmed = self.iteration >= cfg.warmup_iterations
        if warmed and (self.iteration - cfg.warmup_iterations) % cfg.actor_update_every == 0:
            critic_only = self.iteration < cfg.warmup_iterations + cfg.critic_warmup_iterations
            logs.update(self.update_actor_critic(update_actor=not critic_only))
        self.iteration += 1
        return logs

    # ---------------- 持久化 ----------------
    def state_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "cfg": asdict(self.cfg),
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "wm_opt": self.wm_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }

    def load_state_dict(self, data: dict, load_optimizers: bool = True) -> None:
        self.iteration = data["iteration"]
        for name in ("world_model", "actor", "critic", "target_critic"):
            getattr(self, name).load_state_dict(data[name])
        if load_optimizers:
            for name in ("wm_opt", "actor_opt", "critic_opt"):
                getattr(self, name).load_state_dict(data[name])

    def export_actor(self, path) -> None:
        """TorchScript：输入 task.obs_dim 维观测（平地 45、越障 232），输出 12 维原始动作（含 tanh/clamp 界）。"""
        actor = copy.deepcopy(self.actor).cpu().eval()
        torch.jit.script(actor).save(str(path))
