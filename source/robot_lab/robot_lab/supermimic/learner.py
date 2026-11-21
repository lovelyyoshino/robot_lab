"""
learner.py — SuperMimic 训练器：世界模型多步拟合 + 想象 BPTT actor + λ-return critic。

@author pony
@date 2026-10-26
@version v0.15.0
@last_modified 2026-11-18

与论文/supermimic_vibed 的对应关系：
- 世界模型在 k 步自回归窗口上拟合（Eq.1），误差在增量空间按 delta_norm 归一化；
  输入含动力学上下文 ξ（论文 sim context），另有 aux 头回归 env 的关节加速度惩罚量；
- actor 从回放状态出发想象 H 步，奖励/观测/存活全部解析可导，梯度穿过世界模型（Eq.2）；
- critic 默认拟合想象轨迹的 λ-return（Eq.3），目标用 EMA 目标网络，避免自举发散；
  critic_real_returns=True 时改为拟合回放里的真实 λ-return（偏离论文：只在想象上拟合的 critic
  会把"卡在台阶前"估得和"爬上去"一样高）；
- mirror_aug：想象起点与世界模型窗口加入左右镜像副本（symmetry.py），mirror_loss_coef 为 actor 镜像一致性。
世界模型参数在 actor 反传时 requires_grad=False：梯度流经它，但只更新 actor。

提速：diagnostics_interval 让 reward_gap 诊断只在记日志的轮次计算；compile_wm / compile_imagination
用 torch.compile 融合世界模型多步拟合与想象的单步计算（射线投射经 torch.compiler.disable 隔离，失败自动退回 eager）。

@changelog
    - v0.15.0 (2026-11-18): cfg.action_bound 可给 12 个逐维上界（networks v0.5.0）；动作饱和界按各维自己的上界换算。
    - v0.14.0 (2026-11-16): 回放命令维数取 task.command_dim（先抬侧标记时 4 维）；task.lead_obs 时导出 LeadSideActor
      （标记固定 +1，部署接口不变）；policy.onnx 与 policy.pt 同用 EMA 权重（原来 onnx 导出未平均的 actor）。
    - v0.13.0 (2026-11-10): actor_ema_decay（actor 权重 EMA，存档存 actor_ema，导出用 EMA）；warm_start_reset（热启动后
      warmup / critic_warmup / 学习率衰减从载入轮数重新起算，存档记 start_iteration）。默认均为原行为。
    - v0.12.0 (2026-11-09): action_saturation_coef / action_limits：腿关节动作超出软限位的 tanh 前原始输出计平方代价
      （第十二轮 it16000–24000 平地静止时前臂动作 −6.27，界 ±6.28：关节早已顶在限位上，世界模型梯度与 tanh 导数都≈0，
      平地中立代价拉不回来）；镜像一致性与饱和代价共用起点观测；日志 actor_saturation_loss / actor_saturated_fraction。
    - v0.11.0 (2026-11-05): 纯 SuperMimic：删除 Teacher 信赖域锚定（anchor_coef/set_anchor/anchor_actor）、
      物理动作参考 BC（motion_reference_*）、世界模型离地几何监督（wm_clearance_coef）、接触轨迹保留回放
      （wm_rehearsal_*）及旧存档价值梯度解析（resolve_actor_value_grad*）；新增 diagnostics_interval、compile_wm/compile_imagination、
      mirror_aug/mirror_loss_coef、actor 学习率余弦衰减（actor_lr_final/actor_lr_decay_iterations）。
    - (2026-11-04): 梯度审计独立测量动作参考梯度，保留采样RNG，避免把混合优化误报为纯模型贡献。
    - (2026-11-04): 可用真实协调片段辅助actor BPTT，单独记录参考损失并随存档保留。
    - (2026-11-04): 在线混入独立原场景的接触轨迹，避免模型遗忘校准后的卸载/协调动作。
    - (2026-11-04): 共享不更新统计的多步模型损失用于校准；价值梯度可显式恢复并独立限幅。
    - (2026-11-04): 可对过度偏离的模型训练轨迹从真实后继重新展开，仍监督偏离当前步并记录比例。
    - (2026-11-04): 梯度范数用float64累计再裁剪，避免有限大梯度被float32溢出误判；记录模型梯度。
    - (2026-11-03): Actor可用完整折扣回报，critic保留lambda；模型可监督真实净空并拒绝非有限更新。
    - (2026-11-03): 可按平地/越障分层采样想象起点，维持模型事件采样并记录实际平地占比。
    - (2026-11-03): 恢复优化器动量后应用当前配置学习率，避免旧存档静默覆盖调参。
    - v0.9.0 (2026-11-02): 审计参考约束与模型目标的梯度关系；记录并保存非有限actor梯度累计跳过次数。
    - v0.8.0 (2026-11-02): critic真实回报预热跳过无用BPTT；记录actor模型/锚定损失及越障/平地模型回报误差。
    - v0.7.0 (2026-10-28): priority_fraction（事件优先回放），observe 透传 priority。
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
import math
import warnings
from dataclasses import asdict, dataclass, field

import torch

from . import rover_task
from .networks import Actor, Critic, WorldModel
from .replay import ReplayBuffer
from .state import NUM_ACTIONS, compute_delta, state_error


@torch.no_grad()
def stable_clip_grad_norm_(parameters, max_norm):
    """保持全局L2裁剪方向；范数累计不在float32中平方，真实NaN/Inf仍由调用方拒绝。"""
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        return torch.tensor(0.0)
    norm = torch.linalg.vector_norm(torch.stack([g.double().norm() for g in gradients]))
    if torch.isfinite(norm):
        scale = (max_norm / (norm + 1.e-6)).clamp_max(1.0)
        for gradient in gradients:
            gradient.mul_(scale.to(gradient.dtype))
    return norm


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
    wm_reanchor_error: float | None = None
    aux_loss_coef: float = 1.0
    horizon: int = 32
    imag_batch: int = 4096
    actor_lr: float = 2e-4
    # 余弦衰减：actor 开始更新（warmup+critic_warmup）后 actor_lr_decay_iterations 轮内从 actor_lr 降到 actor_lr_final；
    # actor_lr_final=None 或衰减轮数 0 时恒定（旧行为）。
    actor_lr_final: float | None = None
    actor_lr_decay_iterations: int = 0
    critic_lr: float = 3e-4
    critic_epochs: int = 2
    # actor 每 N 轮更新一次：给世界模型时间跟上策略分布的变化，抑制"钻模型空子"。
    actor_update_every: int = 1
    actor_value_grad: bool = True
    actor_value_grad_scale: float = 1.0
    actor_return_mode: str = "lambda"
    gamma: float = 0.99
    lam: float = 0.95
    target_tau: float = 0.02
    grad_clip: float = 1.0
    actor_elem_clip: float = 0.1
    action_noise: float = 0.1
    action_bound: float | tuple = 6.2831853  # 见 networks.Actor：对齐 env 轮速目标 clip；12 元组 = 逐维上界
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
    # 世界模型窗口 / 想象起点中按事件优先级抽取的比例（0 = 均匀；从零越障训练用 0.25，见 replay._pick）
    priority_fraction: float = 0.0
    imag_flat_fraction: float | None = None
    # 预热期 actor 近似输出 0（静止），只靠 0.1 噪声采不到行驶数据；预热期加大探索。
    warmup_action_noise: float = 0.5
    # reward_gap / 样本离地比例等纯诊断只在 iteration % diagnostics_interval == 0 时计算（不影响训练与 RNG）；
    # 1 = 每轮（旧行为），train_terrain 应设为 --log_interval。
    diagnostics_interval: int = 1
    # torch.compile 世界模型多步拟合的单步 / 想象单步；失败时警告并退回 eager。想象单步编译在 Isaac 越障上实测
    # 会让部分 actor 梯度非有限（前向一致），出现即自动永久退回 eager；默认关闭，越障训练只开 compile_wm。
    compile_wm: bool = False
    compile_imagination: bool = False
    # 左右镜像增强：想象起点与世界模型窗口各加一份镜像副本；mirror_loss_coef × ‖π(M o) − M π(o)‖²。
    mirror_aug: bool = False
    mirror_loss_coef: float = 0.0
    # 动作饱和代价：action_limits 为 12 个 (lo, hi)（tanh 之后的动作单位，None = 该维不限），换算成 tanh 之前的原始输出
    # 界 u_lim；对想象起点（含镜像观测）计 coef·Σ relu(|u| − u_lim)²。腿关节动作 = 默认位 + 0.25·a，摆臂软限位 ±0.58 rad
    # 只对应 |a| ≤ 2.3，界 ±2π 的其余区间只是把关节顶在限位上：状态不变（世界模型梯度≈0）、tanh 在界附近导数≈1/300。
    action_saturation_coef: float = 0.0
    action_limits: tuple | None = None
    # actor 权重指数滑动平均：每次 actor 更新后 ema ← ema + (1 − decay)·(θ − ema)；0 关闭。导出/验收用 EMA 权重，
    # 采样仍用当前 actor。梯度范数常年是裁剪阈值的 ~10 倍、每步满幅更新，存档之间振荡大（第十三轮楼梯 12 cm 92%→56%）。
    actor_ema_decay: float = 0.0
    # 热启动重新计时：载入存档后 warmup / critic_warmup / 学习率余弦衰减都从载入时的轮数重新起算（默认 False = 续训）。
    # 改了奖励的热启动需要先让世界模型适应新回放、critic 适应新奖励，再更新 actor。
    warm_start_reset: bool = False
    reward: rover_task.RoverRewardCfg = field(default_factory=rover_task.RoverRewardCfg)


class SuperMimicLearner:
    def __init__(self, cfg: SuperMimicCfg, device, extra_fn=None, extra_dim: int = 0, task=None):
        """extra_fn(state) → [..., extra_dim]：随状态变化的世界模型输入（如地形 relief_features），
        每一步都从当前（预测）状态重新计算并拼在 ξ 之后；梯度可穿过它回到位姿。
        task：观测/奖励/存活/critic 额外输入的提供者，缺省为平地 rover_task.FlatTask。"""
        self.cfg, self.device = cfg, torch.device(device)
        if cfg.actor_return_mode not in ("lambda", "discounted"):
            raise ValueError("actor_return_mode 必须为lambda或discounted")
        if not 0.0 <= cfg.actor_value_grad_scale <= 1.0:
            raise ValueError("actor_value_grad_scale 必须在[0,1]内")
        if cfg.diagnostics_interval < 1:
            raise ValueError("diagnostics_interval 必须 ≥ 1")
        if not math.isfinite(cfg.mirror_loss_coef) or cfg.mirror_loss_coef < 0:
            raise ValueError("mirror_loss_coef 必须有限非负")
        if not math.isfinite(cfg.action_saturation_coef) or cfg.action_saturation_coef < 0:
            raise ValueError("action_saturation_coef 必须有限非负")
        if cfg.action_saturation_coef and (cfg.action_limits is None or len(cfg.action_limits) != NUM_ACTIONS):
            raise ValueError(f"action_saturation_coef > 0 需要 {NUM_ACTIONS} 个 action_limits")
        if cfg.actor_lr_final is not None and (not math.isfinite(cfg.actor_lr_final) or cfg.actor_lr_final <= 0
                                               or cfg.actor_lr_decay_iterations < 0):
            raise ValueError("actor_lr_final 必须为有限正数，衰减轮数非负")
        if cfg.wm_reanchor_error is not None and (not torch.isfinite(torch.tensor(cfg.wm_reanchor_error))
                                                  or cfg.wm_reanchor_error <= 0):
            raise ValueError("wm_reanchor_error 必须为有限正数")
        self.task = task if task is not None else rover_task.FlatTask(cfg.reward)
        if cfg.imag_flat_fraction is not None:
            if not 0.0 <= cfg.imag_flat_fraction <= 1.0 or not hasattr(self.task, "imagination_flat_mask"):
                raise ValueError("平地想象比例须在[0,1]内，且任务须提供平地判别")
        self.extra_fn, self.extra_dim = extra_fn, extra_dim
        self.world_model = WorldModel(cfg.wm_hidden, cfg.context_dim + extra_dim, cfg.aux_dim).to(self.device)
        self.actor = Actor(self.task.obs_dim, cfg.actor_hidden, cfg.action_bound, cfg.action_squash).to(self.device)
        self.critic = Critic(cfg.critic_hidden, cfg.context_dim + self.task.critic_extra_dim).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.wm_opt = torch.optim.Adam(self.world_model.parameters(), lr=cfg.wm_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, betas=(0.9, 0.95))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.replay = ReplayBuffer(cfg.buffer_len, cfg.num_envs, self.device,
                                   command_dim=getattr(self.task, "command_dim", 3),
                                   context_dim=cfg.context_dim, aux_dim=cfg.aux_dim)
        if not 0.0 <= cfg.actor_ema_decay < 1.0:
            raise ValueError("actor_ema_decay 须在 [0, 1)")
        self.actor_ema = copy.deepcopy(self.actor).requires_grad_(False) if cfg.actor_ema_decay else None
        self.iteration = 0
        self.start_iteration = 0  # warm_start_reset 时为载入存档的轮数
        self.skipped_actor_updates = 0
        self.skipped_wm_updates = 0
        self.mirror = None
        if cfg.mirror_aug or cfg.mirror_loss_coef:
            from .symmetry import RoverMirror
            self.mirror = RoverMirror.for_task(self.task, self.device, cfg.context_dim, extra_dim)
        self._mirror_wrappers = {}
        self._raw_lo, self._raw_hi, self._raw_limited = self._raw_action_limits()
        self._compiled = {}
        self._compile_failed = set()
        self._eager_imagination = False
        if cfg.compile_wm or cfg.compile_imagination:
            # 每个 (地形特征, 批量, 梯度模式) 组合各一份图；默认 8 份上限不够镜像/诊断等组合。
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            torch._dynamo.config.accumulated_cache_size_limit = max(
                torch._dynamo.config.accumulated_cache_size_limit, 512)
        if cfg.compile_imagination:
            # Warp 射线投射不能进图：任务上所有 exact_*_fn（扫描/立面射线）包一层 compiler.disable，eager 调用不受影响。
            for name in [n for n in vars(self.task) if n.startswith("exact_") and n.endswith("_fn")]:
                fn = getattr(self.task, name)
                if callable(fn):
                    setattr(self.task, name, torch.compiler.disable(fn))

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
                terminated=None, priority=None) -> None:
        self.replay.add(state, action, prev_action, command, done, context, aux, reward, terminated, priority)

    def wm_context(self, state, context):
        """世界模型实际吃的上下文 = 固定 ξ ⊕ extra_fn(state)。"""
        if self.extra_fn is None:
            return context
        return torch.cat((context, self.extra_fn(state)), dim=-1)

    def wm_step(self, state, action, context):
        return self.world_model(state, action, self.wm_context(state, context))

    def _acc_sq(self, state, action, context):
        return self.world_model.aux(state, action, self.wm_context(state, context)) if self.cfg.aux_dim else None

    # ---------------- torch.compile ----------------
    def _maybe_compiled(self, enabled: bool, key, build, *args):
        """按 key 缓存编译后的单步函数；编译/运行失败时警告一次并对该 key 永久退回 eager。"""
        if not enabled or key in self._compile_failed:
            return build()(*args)
        fn = self._compiled.get(key)
        if fn is None:
            fn = self._compiled[key] = torch.compile(build(), dynamic=False)
        try:
            return fn(*args)
        except Exception as error:  # noqa: BLE001 — 编译器异常类型繁多，统一回退
            self._compile_failed.add(key)
            warnings.warn(f"torch.compile 失败，{key[0]} 退回 eager：{type(error).__name__}: {error}")
            return build()(*args)

    @staticmethod
    def _fn_key(fn):
        # 绑定方法每次取属性都是新对象，按 (实例, 函数) 区分。
        if fn is None:
            return None
        return (id(fn.__self__), fn.__func__) if hasattr(fn, "__self__") and hasattr(fn, "__func__") else id(fn)

    # ---------------- 世界模型 ----------------
    def _wm_rollout_step(self, extra_fn, reanchor: bool):
        """多步拟合的一步：有效掩码 → 模型推进 → 归一化误差 → 偏离过大时从真实后继重启。"""
        cfg, wm = self.cfg, self.world_model

        def step(pred, action, ctx, target, valid, previous):
            pred = torch.where(valid[:, None] > 0, pred, previous)
            if extra_fn is None:
                pred = self.wm_step(pred, action, ctx)
            else:
                # 镜像窗口：地形特征由调用方给出（镜像行映回真实地形查询）。
                pred = wm(pred, action, torch.cat((ctx, extra_fn(pred)), dim=-1))
            error = state_error(pred, target) / wm.delta_norm.std
            squared = torch.where(valid > 0, error.square().mean(-1), 0.0)
            loss = squared.sum() / valid.sum().clamp_min(1.0)
            restart_fraction = None
            if reanchor:
                restart = (error.detach().abs().amax(-1) > cfg.wm_reanchor_error) & (valid > 0)
                restart_fraction = restart.float().mean()
                pred = torch.where(restart[:, None], target, pred)
            return pred, loss, restart_fraction

        return step

    def model_rollout_loss(self, batch, *, reanchor=True, extra_fn=None):
        """k 步自回归多步监督；不修改normalizer、回放或优化器。extra_fn=None 时用 learner 自己的地形特征。"""
        cfg = self.cfg
        states, actions, context, valid = (batch[key] for key in ("state", "action", "context", "valid"))
        reanchor = reanchor and cfg.wm_reanchor_error is not None
        key = ("wm_rollout", self._fn_key(extra_fn), reanchor)

        def build():
            return self._wm_rollout_step(extra_fn, reanchor)

        step_fn = build() if not cfg.compile_wm else None
        pred, losses, restarts = states[:, 0], [], []
        for step in range(actions.shape[1]):
            args = (pred, actions[:, step], context[:, step], states[:, step + 1], valid[:, step], states[:, step])
            if step_fn is not None:
                pred, loss_t, restart_t = step_fn(*args)
            else:
                pred, loss_t, restart_t = self._maybe_compiled(True, key, build, *args)
            losses.append(loss_t)
            if reanchor:
                restarts.append(restart_t)
        loss = torch.stack(losses).mean()
        return loss, {
            "wm_loss": loss.item(), "wm_err_1step": losses[0].item(), "wm_err_last": losses[-1].item(),
            "wm_reanchor_fraction": torch.stack(restarts).mean().item() if restarts else 0.0,
        }

    def update_world_model(self) -> dict[str, float]:
        cfg, wm = self.cfg, self.world_model
        logs = {"wm_loss": 0.0, "wm_err_1step": 0.0, "wm_err_last": 0.0, "wm_aux_loss": 0.0,
                "wm_grad_norm": 0.0, "wm_reanchor_fraction": 0.0}
        for _ in range(cfg.wm_updates):
            batch = self.replay.sample_windows(cfg.wm_batch, cfg.wm_window, cfg.priority_fraction)
            train, extra_fn = batch, None
            if cfg.mirror_aug:
                # 镜像副本按"镜像世界"解释：地形特征把镜像行映回真实地形计算（symmetry.py）。
                mirrored = self.mirror.windows(batch)
                train = {key: torch.cat((batch[key], value), dim=0) for key, value in mirrored.items()}
                extra_fn = self._half_mirrored_extra(batch["state"].shape[0])
            states, actions, ctx, valid = train["state"], train["action"], train["context"], train["valid"]
            # 归一化统计只取单步有效转移，跨回合样本的"增量"是 reset 跳变，会撑大方差。
            mask0 = valid[:, 0] > 0
            if mask0.any():
                if extra_fn is not None:
                    wm_ctx0 = self._context_with(extra_fn, states[:, 0], ctx[:, 0])[mask0]
                else:
                    wm_ctx0 = self.wm_context(states[mask0, 0], ctx[mask0, 0])
                wm.update_normalizers(states[mask0, 0], actions[mask0, 0],
                                      compute_delta(states[mask0, 0], states[mask0, 1]), wm_ctx0, train["aux"][mask0, 0])
            loss, model_logs = self.model_rollout_loss(train, extra_fn=extra_fn)
            aux_loss = torch.zeros((), device=self.device)
            if cfg.aux_dim:
                # aux 在真实状态上监督（逐步、无自回归）：它是本步观测量，不参与状态推进。
                m = valid.reshape(-1) > 0
                rows = valid.numel()
                # 显式给出行数：context_dim=0 时张量为空，reshape(-1, 0) 无法推断。
                flat = lambda x: x.reshape(rows, x.shape[-1])[m]  # noqa: E731
                s_flat = flat(states[:, :-1])
                if extra_fn is not None:
                    aux_ctx = flat(self._context_with(extra_fn, states[:, :-1], ctx))
                else:
                    aux_ctx = self.wm_context(s_flat, flat(ctx))
                pred_aux = wm.aux_normalized(s_flat, flat(actions), aux_ctx)
                aux_loss = (pred_aux - wm.aux_norm.normalize(flat(train["aux"]))).square().mean()
            total = loss + cfg.aux_loss_coef * aux_loss
            self.wm_opt.zero_grad(set_to_none=True)
            if not torch.isfinite(total):
                self.skipped_wm_updates += 1
                raise FloatingPointError("世界模型loss非有限，已拒绝参数更新")
            total.backward()
            gradient_norm = stable_clip_grad_norm_(wm.parameters(), cfg.grad_clip)
            if not torch.isfinite(gradient_norm):
                self.wm_opt.zero_grad(set_to_none=True)
                self.skipped_wm_updates += 1
                raise FloatingPointError("世界模型梯度非有限，已拒绝参数更新")
            self.wm_opt.step()
            logs["wm_grad_norm"] += gradient_norm.item() / cfg.wm_updates
            for key, value in model_logs.items():
                logs[key] += value / cfg.wm_updates
            logs["wm_aux_loss"] += aux_loss.item() / cfg.wm_updates
        if self.iteration % cfg.diagnostics_interval == 0:
            # 纯诊断（无梯度、无 RNG）：只在记日志的轮次计算，不改变训练结果。
            if hasattr(self.task, "wheel_ground_clearance"):
                with torch.no_grad():
                    clearance = self.task.wheel_ground_clearance(batch["state"][:, 1:])
                    airborne = (clearance[..., [0, 2]] >= 0.005).any(-1).float()
                    valid0 = batch["valid"]
                    logs["wm_sample_airborne_fraction"] = ((airborne * valid0).sum() / valid0.sum().clamp_min(1)).item()
            logs.update(self._reward_gap(batch))
        return logs

    def _context_with(self, extra_fn, state, context):
        return context if extra_fn is None else torch.cat((context, extra_fn(state)), dim=-1)

    def _half_mirrored_extra(self, half: int):
        """后一半是镜像世界样本时的地形特征；按批量缓存同一函数对象，编译图可复用。无地形特征时为 None。"""
        if self.extra_fn is None:
            return None
        if half not in self._mirror_wrappers:
            from .symmetry import half_mirrored_extra_fn
            self._mirror_wrappers[half] = half_mirrored_extra_fn(self.extra_fn, self.mirror, half)
        return self._mirror_wrappers[half]

    @torch.no_grad()
    def _reward_gap(self, batch) -> dict[str, float]:
        """用真实动作开环推 k 步，比较模型状态与真实状态上的解析奖励。

        wm_reward_bias > 0 表示模型系统性高估奖励——actor 钻空子时首先体现在这里。
        真实侧的关节加速度项用 env 真值（batch aux），模型侧用 aux 头。
        """
        cfg = self.cfg
        states, actions, prev, cmd, ctx, valid = (batch[k] for k in
                                                  ("state", "action", "prev_action", "command", "context", "valid"))
        pred, diff, absd, n = states[:, 0], 0.0, 0.0, valid.sum().clamp_min(1.0)
        segment_abs, segment_n = {}, {}
        for j in range(cfg.wm_window):
            nxt = self.wm_step(pred, actions[:, j], ctx[:, j])
            real_acc = batch["aux"][:, j] if cfg.aux_dim else None
            r_model = self.task.reward(pred, nxt, cmd[:, j], actions[:, j], prev[:, j],
                                        self._acc_sq(pred, actions[:, j], ctx[:, j]))
            r_real = self.task.reward(states[:, j], states[:, j + 1], cmd[:, j], actions[:, j], prev[:, j], real_acc)
            d = (r_model - r_real) * valid[:, j]
            diff, absd = diff + d.sum(), absd + d.abs().sum()
            if hasattr(self.task, "event_priority"):
                event = self.task.event_priority(states[:, j]) > 0
                for name, mask in (("event", event), ("flat", ~event)):
                    segment_abs[name] = segment_abs.get(name, 0.0) + (d.abs() * mask).sum()
                    segment_n[name] = segment_n.get(name, 0.0) + (valid[:, j] * mask).sum()
            pred = nxt
        return {"wm_reward_bias": (diff / n).item(), "wm_reward_abs_gap": (absd / n).item(),
                **{f"wm_{name}_reward_abs_gap": (segment_abs[name] / count.clamp_min(1.0)).item()
                   for name, count in segment_n.items()}}

    # ---------------- 想象 ----------------
    def _imagined_action(self, obs, mirror_half):
        """mirror_half=B：后半批是镜像起点，用共轭策略 M∘π∘M 在真实地形上推进（等价于 π 在镜像世界里行驶）。"""
        if mirror_half is None:
            return self.actor(obs)
        raw = self.actor(torch.cat((obs[:mirror_half], self.mirror.obs(obs[mirror_half:])), dim=0))
        return torch.cat((raw[:mirror_half], self.mirror.action(raw[mirror_half:])), dim=0)

    def _imagine_step(self, detach_dynamics: bool, mirror_half):
        def step(state, prev_action, command, ctx):
            action = self._imagined_action(self.task.policy_obs(state, command, prev_action), mirror_half)
            next_state = self.wm_step(state, action, ctx)
            if detach_dynamics:
                next_state = next_state.detach()
            reward = self.task.reward(state, next_state, command, action, prev_action, self._acc_sq(state, action, ctx))
            return action, next_state, reward, self.task.survival(next_state)
        return step

    def imagine(self, start: dict[str, torch.Tensor], detach_dynamics: bool = False,
                mirror_half: int | None = None) -> dict[str, torch.Tensor]:
        """从回放起点想象 H 步；命令与 ξ 在整段 rollout 内不变（ξ 只在 reset 时改变）。"""
        cfg = self.cfg
        state, prev_action, command, ctx = start["state"], start["prev_action"], start["command"], start["context"]
        states, prev_actions, rewards, survivals = [state], [prev_action], [], []
        key = ("imagine", detach_dynamics, mirror_half, torch.is_grad_enabled())

        def build():
            return self._imagine_step(detach_dynamics, mirror_half)

        step_fn = build() if not cfg.compile_imagination or self._eager_imagination else None
        for _ in range(cfg.horizon):
            if step_fn is not None:
                action, next_state, reward, survival = step_fn(state, prev_action, command, ctx)
            else:
                action, next_state, reward, survival = self._maybe_compiled(True, key, build, state, prev_action,
                                                                            command, ctx)
            rewards.append(reward)
            survivals.append(survival)
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

    def model_gradient_diagnostics(self, start) -> dict[str, float]:
        """不更新权重，比较完整actor梯度与截断动力学梯度；范数比不是学习贡献百分比。"""
        parameters = tuple(self.actor.parameters())

        def gradient(detach, detach_value=None):
            trajectory = self.imagine(start, detach_dynamics=detach)
            loss = -self._actor_returns(trajectory, detach_value=detach_value)[0].mean()
            grads = torch.autograd.grad(loss, parameters, allow_unused=True) if loss.requires_grad else (None,) * len(parameters)
            vector = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                                for p, g in zip(parameters, grads)])
            return loss.detach(), vector.detach()

        full_loss, full = gradient(False)
        detached_loss, direct = gradient(True)
        value_detached_loss, no_value = gradient(False, detach_value=True)
        through_model = full - direct
        full_norm, model_norm = full.norm(), through_model.norm()
        return {"full_gradient_norm": full_norm.item(), "detached_dynamics_gradient_norm": direct.norm().item(),
                "through_dynamics_gradient_norm": model_norm.item(),
                "through_dynamics_to_full_norm_ratio": (model_norm / full_norm.clamp_min(1e-12)).item(),
                "through_dynamics_full_cosine": (torch.dot(through_model, full)
                                                 / (model_norm * full_norm).clamp_min(1e-12)).item(),
                "forward_loss_abs_difference": (full_loss - detached_loss).abs().item(),
                "without_value_gradient_norm": no_value.norm().item(),
                "through_value_gradient_norm": (full - no_value).norm().item(),
                "value_to_full_norm_ratio": ((full - no_value).norm() / full_norm.clamp_min(1e-12)).item(),
                "forward_value_loss_abs_difference": (full_loss - value_detached_loss).abs().item()}

    def _raw_action_limits(self):
        """action_limits（tanh 之后的动作单位）→ tanh 之前原始输出的 (lo, hi, 受限掩码)，各 [12]；None 项为 ±inf。"""
        cfg = self.cfg
        lo = torch.full((NUM_ACTIONS,), -math.inf, device=self.device)
        hi = torch.full((NUM_ACTIONS,), math.inf, device=self.device)
        if cfg.action_limits is not None:
            bounds = ([float(cfg.action_bound)] * NUM_ACTIONS if isinstance(cfg.action_bound, (int, float))
                      else [float(b) for b in cfg.action_bound])

            def raw(value, bound):
                if cfg.action_squash == "tanh":
                    return bound * math.atanh(max(-0.999, min(0.999, value / bound)))
                return value

            for i, limit in enumerate(cfg.action_limits):
                if limit is None:
                    continue
                low, high = float(limit[0]), float(limit[1])
                if not low < high:
                    raise ValueError(f"action_limits[{i}] 须满足 lo < hi")
                lo[i], hi[i] = raw(low, bounds[i]), raw(high, bounds[i])
        return lo, hi, torch.isfinite(lo) | torch.isfinite(hi)

    def _start_obs(self, start) -> torch.Tensor:
        with torch.no_grad():
            return self.task.policy_obs(start["state"], start["command"], start["prev_action"])

    def _saturation(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        """(Σ relu(|u| − u_lim)² 的样本均值, 受限维里超界的比例)；u = actor 的 tanh 前原始输出，观测不求导。"""
        raw = self.actor.actor(obs)
        excess = (raw - self._raw_hi).clamp_min(0.0) + (self._raw_lo - raw).clamp_min(0.0)
        limited = self._raw_limited.to(raw.dtype)
        fraction = ((excess > 0).to(raw.dtype) * limited).sum(-1).mean() / limited.sum().clamp_min(1.0)
        return excess.square().sum(-1).mean(), fraction.detach()

    def _mirror_consistency(self, start, obs=None) -> torch.Tensor:
        """‖π(M o) − M π(o)‖²（逐样本对动作维求和后取均值）；两支都反传到 actor，观测不求导。"""
        if obs is None:
            obs = self._start_obs(start)
        out = self.actor(torch.cat((obs, self.mirror.obs(obs)), dim=0))
        action, mirrored = out[:obs.shape[0]], out[obs.shape[0]:]
        return (mirrored - self.mirror.action(action)).square().sum(-1).mean()

    def _actor_lr(self) -> float:
        cfg = self.cfg
        if cfg.actor_lr_final is None or cfg.actor_lr_decay_iterations <= 0:
            return cfg.actor_lr
        begin = self.start_iteration + cfg.warmup_iterations + cfg.critic_warmup_iterations
        progress = min(max(self.iteration - begin, 0) / cfg.actor_lr_decay_iterations, 1.0)
        return cfg.actor_lr_final + 0.5 * (cfg.actor_lr - cfg.actor_lr_final) * (1.0 + math.cos(math.pi * progress))

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

    def _actor_returns(self, traj, detach_value: bool | None = None):
        """可选阻断critic输入梯度以诊断价值捷径；奖励仍通过世界模型反传。"""
        values = self._values(self.critic, traj)
        stop_value = not self.cfg.actor_value_grad if detach_value is None else detach_value
        if stop_value:
            values = values.detach()
        else:
            # 前向价值保持原值；独立缩放末值/中间值的输入梯度，避免新critic压过模型奖励梯度。
            scale = self.cfg.actor_value_grad_scale
            values = values.detach() + scale * (values - values.detach())
        if self.cfg.actor_return_mode == "lambda":
            return self.lambda_returns(traj["reward"], traj["survival"], values)
        # Actor优化整段奖励，避免第32步的起抬收益被lambda额外压到约0.15；critic仍独立用lambda。
        result, value = [], values[-1]
        for t in reversed(range(traj["reward"].shape[0])):
            value = traj["reward"][t] + self.cfg.gamma * traj["survival"][t] * value
            result.append(value)
        return torch.stack(result[::-1])

    def _sample_imagination_start(self):
        cfg = self.cfg
        if cfg.imag_flat_fraction is None:
            return self.replay.sample_states(cfg.imag_batch, cfg.priority_fraction)
        pool = self.replay.sample_states(max(128, 4 * cfg.imag_batch), cfg.priority_fraction)
        with torch.no_grad():
            flat = self.task.imagination_flat_mask(pool["state"])
        groups = (flat.nonzero().flatten(), (~flat).nonzero().flatten())
        n_flat = round(cfg.imag_batch * cfg.imag_flat_fraction)
        indices = []
        for group, count in zip(groups, (n_flat, cfg.imag_batch - n_flat)):
            if count == 0:
                continue
            # 初始采样还没有台阶时，使用可用状态并如实记录占比，不能伪造越障状态。
            if group.numel() == 0:
                group = torch.arange(len(flat), device=self.device)
            indices.append(group[torch.randint(group.numel(), (count,), device=self.device)])
        selected = torch.cat(indices)
        return {key: value[selected] for key, value in pool.items()}

    def _actor_backward(self, start):
        """想象 → 回报 → actor 损失反传；梯度穿过世界模型与 critic，但只写入 actor 的 .grad。"""
        cfg = self.cfg
        self.world_model.requires_grad_(False)
        self.critic.requires_grad_(False)
        try:
            if cfg.mirror_aug:
                # 起点加一份镜像副本（共轭策略在真实地形上推进，见 _imagined_action）。
                half = start["state"].shape[0]
                traj = self.imagine({key: torch.cat((start[key], start[key]), dim=0)
                                     for key in ("state", "prev_action", "command", "context")}, mirror_half=half)
            else:
                traj = self.imagine(start)
            actor_loss = -self._actor_returns(traj)[0].mean()
            model_loss = actor_loss
            mirror_loss = actor_loss.new_zeros(())
            saturation = {}
            obs = self._start_obs(start) if cfg.mirror_loss_coef or cfg.action_saturation_coef else None
            if cfg.mirror_loss_coef:
                mirror_loss = self._mirror_consistency(start, obs)
                actor_loss = actor_loss + cfg.mirror_loss_coef * mirror_loss
            if cfg.action_saturation_coef:
                # 镜像起点上的策略输入同样要在界内（共轭策略 M∘π∘M 推进时 actor 看到的是镜像观测）。
                sat_obs = torch.cat((obs, self.mirror.obs(obs)), dim=0) if self.mirror is not None else obs
                saturation_loss, saturated = self._saturation(sat_obs)
                actor_loss = actor_loss + cfg.action_saturation_coef * saturation_loss
                saturation = {"actor_saturation_loss": saturation_loss.item(), "actor_saturated_fraction": saturated.item()}
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
        finally:
            self.world_model.requires_grad_(True)
            self.critic.requires_grad_(True)
        return traj, actor_loss, model_loss, mirror_loss, saturation

    def update_actor_critic(self, update_actor: bool = True) -> dict[str, float]:
        cfg = self.cfg
        start = self._sample_imagination_start()
        sampling_logs = ({"imag_flat_start_fraction": self.task.imagination_flat_mask(start["state"]).float().mean().item()}
                         if cfg.imag_flat_fraction is not None else {})
        with torch.no_grad():
            self.critic.norm.update(self.critic.inputs(start["state"], start["command"], start["prev_action"],
                                                       self.critic_context(start["state"], start["context"])))
        if not update_actor and cfg.critic_real_returns:
            # 真实回报预热不依赖想象；避免白做整段BPTT，也避免actor坏梯度阻断critic预热。
            critic_loss = self._fit_critic_real()
            self._update_target_critic()
            return {"critic_loss": critic_loss, "actor_update_applied": 0.0, **sampling_logs}

        traj, actor_loss, model_loss, mirror_loss, saturation = self._actor_backward(start)
        finite = all(torch.isfinite(p.grad).all() for p in self.actor.parameters() if p.grad is not None)
        if not finite and cfg.compile_imagination and not self._eager_imagination:
            # 实测（Isaac 越障 A/B）：编译后的想象单步前向一致，但反传会在部分台阶状态出 NaN/Inf，而 eager 有限。
            # 首次出现即永久退回 eager 并重算本轮（无 RNG 消耗），不计入跳过次数。
            warnings.warn("torch.compile 想象单步产生非有限 actor 梯度，永久退回 eager 想象并重算本轮")
            self._eager_imagination = True
            traj, actor_loss, model_loss, mirror_loss, saturation = self._actor_backward(start)
            finite = all(torch.isfinite(p.grad).all() for p in self.actor.parameters() if p.grad is not None)
        if not finite:
            # 一次 NaN 更新会污染 actor，进而把 NaN 动作送进仿真和回放；跳过并计数，不静默吞掉。
            self.skipped_actor_updates += 1
            return {"actor_skipped": float(self.skipped_actor_updates)}
        for p in self.actor.parameters():
            if p.grad is not None:
                p.grad.clamp_(-cfg.actor_elem_clip, cfg.actor_elem_clip)
        actor_grad_norm = stable_clip_grad_norm_(self.actor.parameters(), cfg.grad_clip).item()
        actor_lr = self._actor_lr()
        if update_actor:
            for group in self.actor_opt.param_groups:
                group["lr"] = actor_lr
            self.actor_opt.step()
            if self.actor_ema is not None:
                with torch.no_grad():
                    for p, ema in zip(self.actor.parameters(), self.actor_ema.parameters()):
                        ema.lerp_(p, 1.0 - cfg.actor_ema_decay)
                    for b, ema_b in zip(self.actor.buffers(), self.actor_ema.buffers()):
                        ema_b.copy_(b)

        traj = {k: v.detach() for k, v in traj.items()}
        if cfg.critic_real_returns:
            critic_loss = self._fit_critic_real()
        else:
            critic_loss = self._fit_critic_imagined(traj)
        self._update_target_critic()
        return {
            **sampling_logs,
            **saturation,
            "actor_loss": actor_loss.item(),
            "actor_model_loss": model_loss.item(),
            "actor_mirror_loss": mirror_loss.item(),
            "actor_lr": actor_lr,
            "actor_update_applied": float(update_actor),
            "actor_grad_norm": actor_grad_norm,
            "critic_loss": critic_loss,
            "imag_reward": traj["reward"].mean().item(),
            "imag_survival": traj["survival"][-1].mean().item(),
        }

    @torch.no_grad()
    def _update_target_critic(self) -> None:
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.lerp_(p, self.cfg.target_tau)
        self.target_critic.norm.load_state_dict(self.critic.norm.state_dict())

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
            norm = stable_clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
            if not torch.isfinite(norm):
                self.critic_opt.zero_grad(set_to_none=True)
                raise FloatingPointError("critic梯度非有限，已拒绝参数更新")
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
            norm = stable_clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
            if not torch.isfinite(norm):
                self.critic_opt.zero_grad(set_to_none=True)
                raise FloatingPointError("critic梯度非有限，已拒绝参数更新")
            self.critic_opt.step()
        return critic_loss.item()

    def update(self) -> dict[str, float]:
        logs = self.update_world_model()
        cfg = self.cfg
        rel = self.iteration - self.start_iteration
        warmed = rel >= cfg.warmup_iterations
        actor_due = (rel - cfg.warmup_iterations) % cfg.actor_update_every == 0
        if warmed and (cfg.critic_real_returns or actor_due):
            critic_only = rel < cfg.warmup_iterations + cfg.critic_warmup_iterations
            # 降低actor频率不能同时削减真实回报critic的预热和拟合次数。
            logs.update(self.update_actor_critic(update_actor=actor_due and not critic_only))
        self.iteration += 1
        logs["actor_skipped_total"] = float(self.skipped_actor_updates)
        return logs

    # ---------------- 持久化 ----------------
    def state_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "skipped_actor_updates": self.skipped_actor_updates,
            "skipped_wm_updates": self.skipped_wm_updates,
            "cfg": asdict(self.cfg),
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            **({"actor_ema": self.actor_ema.state_dict()} if self.actor_ema is not None else {}),
            "start_iteration": self.start_iteration,
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "wm_opt": self.wm_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }

    def load_state_dict(self, data: dict, load_optimizers: bool = True) -> None:
        """旧存档里的 anchor_actor / motion_reference* / wm_rehearsal_rng 字段（v0.10 及以前）直接忽略。"""
        self.iteration = data["iteration"]
        # 热启动重新计时：从载入轮数起重新 warmup；续训同一个热启动 run 时沿用存档里记录的起点。
        self.start_iteration = (data["iteration"] if self.cfg.warm_start_reset and not data.get("start_iteration")
                                else data.get("start_iteration", 0))
        self.skipped_actor_updates = data.get("skipped_actor_updates", 0)
        self.skipped_wm_updates = data.get("skipped_wm_updates", 0)
        for name in ("world_model", "actor", "critic", "target_critic"):
            getattr(self, name).load_state_dict(data[name])
        if self.actor_ema is not None:
            # 旧存档没有 EMA：从载入的 actor 开始平均
            self.actor_ema.load_state_dict(data.get("actor_ema", data["actor"]))
        if load_optimizers:
            for name, rate in (("wm_opt", self.cfg.wm_lr), ("actor_opt", self._actor_lr()),
                               ("critic_opt", self.cfg.critic_lr)):
                optimizer = getattr(self, name)
                optimizer.load_state_dict(data[name])
                for group in optimizer.param_groups:
                    group["lr"] = rate

    def export_actor(self, path) -> None:
        """TorchScript：输入 task.obs_dim 维观测（平地 45、越障 232），输出 12 维原始动作（含 tanh/clamp 界）。"""
        source = self.actor_ema if self.actor_ema is not None else self.actor  # 开 EMA 时导出平均权重
        actor = self.deploy_actor(source)
        torch.jit.script(actor).save(str(path))

    def deploy_actor(self, source=None):
        """部署用 actor（CPU、eval）：默认与 policy.pt 相同（开 EMA 时用平均权重）；task.lead_obs 时包一层
        LeadSideActor，先抬侧固定 +1，输入不含标记。"""
        if source is None:
            source = self.actor_ema if self.actor_ema is not None else self.actor
        actor = copy.deepcopy(source).cpu().eval()
        if getattr(self.task, "lead_obs", False):
            from .networks import LeadSideActor
            actor = LeadSideActor(actor, 1.0).eval()
        return actor
