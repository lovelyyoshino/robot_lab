"""
preview_task.py — SuperMimic 越障任务的可导版本：232 维观测（45 本体 + 187 高度扫描）、统一的解析奖励、软存活。

@author pony
@date 2026-10-26
@version v0.23.0
@last_modified 2026-11-19

越障目标（真实回报与想象回报共用同一函数，每项权重见 PreviewRewardCfg）：
  1 速度跟踪（立面前 slow_band_m 内允许减速到 min(cmd, slow_band_speed)）+ 世界系前进 + 停滞代价；
  2 翻车终止 −20（倾角 0.8 rad）；
  3 统一姿态：−w_roll·g_y² − w_roll_excess·relu(|g_y| − sin roll_excess_deg)² − w_pitch·relu(|θ−θ_ref|−tol)²；
    θ_geo 为前后轮下地形决定的几何俯仰，θ_ref = sign(θ_geo)·max(|θ_geo| − deadband, 0)（无梯度）；
  4 统一平顺：角速度、摆臂速度、动作变化、竖直速度；
  5 航向/直行：sin²ψ + sinψ 的 Huber（小偏航也有梯度）、偏航角速度、世界系 v_y 的 Huber、转向角；楼梯区内航向目标
    为 |ψ| = stairs_slant_deg（左右任取，4WS 蟹行沿路线前进），让同一轴两轮先后碰立面；车道项 −w·y² 用训练特权横向偏移
    （lane_center_y，与 critic 车道特征同源），actor 观测仍为 232 维；
  6 平地（受地形门控）：摆臂回零 + 四轮承重；同轴两轮反向打角（toe，只刮地）、转向角平方和、四轮轮缘速度偏离均值
    （wheel_sync，不经世界模型的车速预测）与轮速打滑（轮缘速度 − 轮心沿轮向速度）——平地直行的唯一正解是转向归零、
    四轮同速；零速命令时偏航角速度的 Huber（actor 看不到航向，ω_z² 在小转速处没有梯度）；
  7 越障（物理约束见 docs/supermimic.md "预抬的物理边界"）：
    - 预抬只能单轮进行：两前轮同时离地时质心在后轴前 ≥27 cm，必倒；单轮离地靠其余三轮支撑，静稳定裕度
      在横滚 ≤5° 时最多约 2.2 cm，只动被抬轮摆臂时约 0（质心落在对角线上）。support_margin 惩罚离地轮及
      单台阶立面前 prelift_shift_zone 内该抬的轮对应三角形的质心裕度不足（离地轮门槛 support_margin_m，该抬的轮门槛
      prelift_margin_m：进窗口之前先移好质心；楼梯前轴只在预抬窗口内、门槛 support_margin_m），multi_lift 惩罚两轮同时悬空；
      立面上正在翻越的轮、刚越过下行边沿的轮（下台阶）不算悬空；
    - 单台阶（isolated step）的所有轮、以及任何立面前的前轴，在立面前 prelift_window 内跟踪离地净空；接触后跟踪翻越
      净空（所有立面）。每根轴一次只要求
      一个轮：同一轴两轮都在跟踪时取两轮误差的较小者——一个轮先抬/先翻，另一个轮等它上去后再翻；窄台阶上面对立面
      的那个轮必须抬。跟踪量 = 实际净空与"支撑净空"（其余三轮着地假设下的运动学净空，不经世界模型的车身预测）误差
      的平均：世界模型没见过单轮离地时学到"收臂车角下沉"，会把实际净空对摆臂的梯度抵消掉。单台阶接触前的支撑净空只在
      质心确实落在其余三轮三角形内时才可信：裕度从 0 到 prelift_margin_m 之间，误差由"两者取大"平滑过渡到"两者平均"
      （第十一轮：质心压在对角线上时收臂只让车身绕对角线倾倒，验收窗口内摆臂抬 4 cm、净空 0.2 mm，支撑净空却给足了分）；
    - 楼梯：踏面 26–30 cm < 4 个轮半径之和，每级让四个轮都离地预抬在几何上不可能。前轴每级一个轮先抬（车头抬起、
      质心后移，裕度比平地大）；后轴轻碰快上——接触前的速度超过 riser_safe_speed 计代价，接触后按上一条翻越。
      后轴两轮同时顶住整宽立面时只有两前轮的摩擦力往上推（车头抬起后前轴载荷更小），物理探针：直行 8 cm 需 μ≥0.86、
      12 cm 在 μ≤1.2 内都上不去；车身斜 12° 让两后轮先后碰立面，8 cm 降到 μ≥0.6、12 cm 降到 μ≥0.86；
    - 整宽单台阶（同一轴两轮都在单台阶区内）航向目标 |ψ| = single_slant_deg：同轴两轮先后到达立面，前轴先到的那个轮
      （领先轮）有确定的身份去预抬；正对立面时两轮 lead 只差毫米，镜像一致的策略分不出该抬哪个；
    - prelift_front：单台阶前轴领先轮（lead 较小者）的额外净空跟踪（窗口内到接触后）。μ 固定 1.0 后直接撞上去也能
      爬过单台阶，原来 prelift_track 每轴取两轮误差较小者（多半是离得远、目标还没升起的那个轮），撞台阶几乎没有代价；
    - 准备姿态：立面前 ready_zone_m 内，摆臂预留的上抬行程不足立面高度时计代价；
    立面距离由网格射线每个状态只算一次并缓存；早抬代价与预抬同一批轮；
  8 关节软限位。
地形几何（轮位、立面距离、θ_ref、平地门、支撑裕度）由 terrain_context 每个状态计算一次，所有奖励项共享。

@changelog
    - v0.23.0 (2026-11-19): post_climb_relief / post_climb_window_m（默认 0 = 原行为）：后轴轮心后方 window 内地面比轮下低
      min_step（刚翻上去）时，平地滚动四项（toe / flat_steer / wheel_sync / wheel_slip）乘 (1 − post_climb_relief)。
    - v0.22.0 (2026-11-19): 校验 tilt_temperature > 0（训练脚本开放 --tilt_temperature）。
    - v0.21.0 (2026-11-18): stall_linear_cap（默认 1 = 原行为）：线性停滞代价 (gate − v)/gate 的上限，>1 时倒车继续加重。
      第二十三轮第 22000 轮单台阶/楼梯驻车：前蹭 0.10 m/s 约 3 s 再快速倒回，净位移 0；原来的停滞项对速度有界（sigmoid、
      线性项封顶 1），71–75% 的时间速度在门槛 0.075 m/s 以上不计停滞，周期平均 −2.7/步，老实停住是 −12～−20/步。
      奖励对速度是凹函数时来回抖的时间平均不会好于净速度对应的值（Jensen），所以把线性项放开到倒车区间。
    - v0.20.0 (2026-11-16): 楼梯左右轮替（默认关 = 原行为）。stairs_climb_trail_weight < 1：楼梯区同一轴两轮都在翻越时，
      翻越跟踪按"已经更高的轮（领先轮）全权重、落后轮 × trail_weight"加权（领先轮按两轮净空软判、不求导，等高时两轮各半，
      左右对称）。原来 stairs_climb_pair_sum 两轮误差相加，Huber 是凸的，同样的总抬升量两轮平分时误差和最小——等于要求左右
      同时上，与第二十轮第 30000 轮后轴两轮对称顶死第一级一致。stairs_alt_lift（≤0）：同一轴两轮都在翻越时，两轮净空差
      |Δ| 不足 stairs_alt_lift_frac × 立面高时计代价 1 − |Δ|/(frac·立面高)（|Δ| 用平滑绝对值，对净空可导）；|Δ| 对左右互换
      不变，与镜像增广/镜像一致性兼容。不加观测维度（取代第二十一轮的先抬侧标记）。stairs_contact_climb_m > 0：楼梯立面前
      lead ≤ 该值的轮也算翻越中（三轮支撑/悬空、翻越跟踪、同轴判定共用 _climb_window）。探针（第二十轮第 30000 轮 16 cm
      楼梯死锁姿态）：两后轮 lead +0.001/+0.005 不算翻越、跟踪为 0；单抬后轮 1–2 cm 支撑裕度代价 −18～−36/步，原地顶死
      只有 −15.7/步——卡死是奖励的局部最优，不只是镜像对称造成的。
    - v0.19.0 (2026-11-16): lead_obs：actor 观测末尾追加先抬侧标记 command[..., 3]（±1，每回合随机，镜像取反），
      232 → 233 维；命令 4 维时奖励只用前 3 维。默认关闭，行为与 v0.18.0 相同。
    - v0.18.0 (2026-11-13): 楼梯区车道约束可逐步收紧：set_stairs_lane_scale(s)（0–1，张量缓冲，torch.compile 不重编译）
      把楼梯区（stairs_zone）的车道项 lane 与越障区横向速度的额外倍数 (obstacle_lateral_vel_scale − 1) 乘以 s；
      默认 stairs_lane_scale = 1（原行为）。先爬上去、训练后期再加车道。
    - v0.17.0 (2026-11-13): stall_climb_min_clearance / stall_climb_lead_m（贴立面停滞折扣只给真离地的轮，对净空可导）
      与 riser_spin（贴立面着地轮空转代价），默认关 = 原行为。第十六轮 it18000 回放左侧 12 cm 箱：命令 0.23–0.36 m/s，
      左后轮着地顶立面 13 s（≈50 N、轮速 2.5 rad/s、车速 ≈0）直到回合重置。探针（同一姿态、本轮参数）：顶着不动每步
      −10.5（停滞只按 20% 计）；左后臂抬 6 cm 不动 −12.9——轮心后移出 −r < lead ≤ 0 窗口，停滞折扣消失，抬轮反而更差。
    - v0.16.0 (2026-11-12): stairs_climb_pair_sum 与 stall_linear（默认关 = 原行为）。第十五轮 it26000 在 12 cm×40 cm 楼梯
      1/4 回合死锁：左前轮搭在第二级台阶角上（轮心离边 3 cm、翻越中误差 0.04），右前轮顶着第二级立面，四轮转速 0 直到超时。
      探针（同一姿态、本轮奖励参数）：原地不动每步 −5.9；右前臂抬到 −0.15/−0.58 反而 −9.2/−8.6——前轴两轮都在翻越时
      跟踪取两轮误差较小者，左轮误差 0.04 吃掉右轮的 1.57；右轮一抬，左轮的支撑净空（其余三轮着地假设）坏掉，误差升到
      1.2–2.1。停滞 sigmoid 在车速 0 处饱和（0.9994），没有梯度。楼梯区同一轴两轮都在翻越时改为两轮误差相加、只用实际净空；
      低于停滞门槛的速度差另计线性代价。
    - v0.15.0 (2026-11-11): obstacle_lateral_vel_scale（默认 1 = 原行为）：越障区（非平地门 1 − flat）世界系横向速度代价放大。
      第十四轮全宽 14/17.5 cm 单台阶检查点间 47%↔100% 的失败几乎都是"冲出"：翻过台阶但横偏 0.15–0.53 m；轨迹（it24000）
      前轴先抬的一侧决定偏向，后轴爬 0.40 m 短块、前轴已下块时以 v_y≈0.1 m/s 侧滑约 3 s，下块后 actor 看不到横向偏移、
      不会修回。
    - v0.14.0 (2026-11-11): trailing_contact_climb（默认关 = 原行为）：同轴另一个轮已站上更高一级（轮下地面高出 ≥ min_step）
      时，本轮不再是接触前预抬对象（预抬跟踪、支撑裕度"该抬的轮"、早抬），只在碰到立面后按翻越跟踪——"一个轮先抬、
      另一个碰到后再翻"。第十四轮 it40000 在 12 cm×40 cm 楼梯：左前轮上第二级后，右前轮单独留在预抬窗口里被要求也先抬
      （一个前轮已在台阶上时另一个前轮离地静态不可行，同时被支撑裕度代价顶住），停在立面前 4 cm、四轮转速 0，直到超时。
    - v0.13.0 (2026-11-11): 楼梯区单独的 stairs_stall_climb_scale / stairs_riser_contact_scale / stairs_riser_approach_m
      （默认 = 与单台阶相同，原行为）。第十四轮 it38000：12–17.5 cm 楼梯 64/64 回合"停住超时"、贴立面最长 7.3 s，
      16 cm 以上后轮碰立面 0%（第一级都没上去）；贴立面时停滞只罚 20%、轻碰 −400 / 15 cm 又压掉了上楼梯靠的冲劲。
    - v0.12.0 (2026-11-10): stall_climb_scale（贴立面翻越时停滞代价缩放）与 drop_relief（下行边沿减免竖直速度/角速度/
      俯仰/摆臂速度/两轮离地/支撑裕度代价），默认均为原行为。第十三轮 it20000 轨迹：整宽 14 cm 单台阶两前轮在接触瞬间
      同时翻上、后轴逐轮翻上，随后后轮停在块的下行边沿前直到停滞超时；立面接触速度所有档位 0.32–0.54 m/s。
    - v0.11.0 (2026-11-09): 第十二轮诊断：单台阶严格预抬到 0.34–0.88（it16000），楼梯课程却比第十/十一轮低 2–3 cm、
      12 cm 以上楼梯 0–5%。原因之一是 v0.10.0 的"先移质心"（prelift_shift_zone 0.25 m、门槛 prelift_margin_m、接触前支撑
      净空按裕度门控）对楼梯前轴同样生效，踏面 26–40 cm 时前轴几乎一直在下一级立面 0.25 m 内，爬楼全程都被要求侧倾移质心。
      改为只对单台阶生效；楼梯前轴回到第十一轮口径（预抬窗口内、门槛 support_margin_m、支撑净空与实际净空平均）。
    - v0.10.0 (2026-11-08): 第十一轮第 10000 轮诊断：验收预抬窗口内两前臂都抬 3–5 cm，轮离地却只有 0.2 mm（车头下沉 / 绕
      fr–rl 对角线倾倒，窄台阶面对立面的轮载荷降到 0 仍贴地），支撑净空照样给分；平地 0.30 m/s 横偏 P95 0.52 m，探针前轮内八
      约 20°、后轮外八约 13°、后轮比车速快 35%、转向 ±14° 以 0.57 Hz 摆动，toe −2 / 打滑 −5 每步只罚约 1.3。改为：
      该抬的轮的裕度门槛 prelift_margin_m 0.8 cm、从立面前 prelift_shift_zone 0.25 m 起就计（先移质心）；接触前支撑净空按
      裕度门控；平地 toe −2→−20、打滑 −5→−20，新增平地转向平方 flat_steer、四轮同速 wheel_sync。
    - v0.9.0 (2026-11-08): 第十轮（μ=1.0）诊断：单台阶预抬事件 0%（第七轮随机摩擦下右侧窄台阶 95%），前轮离立面 0–0.5 cm 才
      离地；平地探针 0.30 m/s 后轮左右各打约 ±43° 八字、前轮转速为车速的两倍（打滑），零速时前轮正转、后轮反转，轴间对顶
      造成横漂（P95 0.31 m）、零速自转（P95 21°）和 2 Hz 以上的俯仰振动。新增 prelift_front（单台阶前轴领先轮）、
      single_slant_deg（整宽单台阶斜行）、平地 toe / wheel_slip、零速 stand_yaw_rate。
    - v0.8.0 (2026-11-07): 楼梯区航向目标 |ψ|=12°（后轴两轮先后碰立面，见上）；停滞 −2→−5、sigmoid 温度 0.02→0.01（整宽单台阶探针：停在块上
      每步 −5.3，往前下块 −5.9、后轮上块 −6.4，策略前后轮反向对拉停住）；刚越过下行边沿的轮不计两轮离地/支撑裕度
      （成功回合下块时支撑裕度每步 −2.8）；偏航角速度 −2·ω_z²（零速命令时 actor 看不到航向，第五轮原地转 15–32°）。
    - v0.7.0 (2026-11-07): 前轴在楼梯每级立面前也预抬（第四轮 16–17.5 cm 楼梯前轮翻过第一级后卡在第二、三级立面，
      课程停在 7 cm；同一高度的单台阶前轮预抬 80%、通行 44–95%）；早抬代价与预抬同一批轮。航向加 sinψ 的 Huber 项
      −5，v_y Huber −5→−15，平地中立 −1→−5（第四轮平地 0.30 m/s 横偏 P95 0.4 m、平地俯仰 5–6°）。
    - v0.6.0 (2026-11-06): 净空跟踪加入支撑净空（其余三轮着地时的运动学净空，与实际净空误差各占一半）：上一轮
      第 2000 轮验收前轮在窗口内摆臂只收 6 mm、净空 0.3 mm——实际净空经世界模型车身预测的梯度被"收臂车角下沉"
      抵消，策略从不尝试抬轮；翻越跟踪也改为每轴一次一个轮（整宽立面高于轮半径时两前轮不能同时抬，楼梯 ≥12 cm 全卡）。
    - v0.5.0 (2026-11-06): 支撑裕度门槛 1.5→0.5 cm，并计入预抬窗口内该抬的轮（原来只罚已离地轮，不抬就逃掉
      代价，策略从不预抬）；立面上翻越中的轮不计两轮离地/支撑裕度（原来整宽立面两前轮同时翻越被顶住，
      楼梯课程停在 5.5 cm）；净空跟踪权重 3→5。
    - v0.4.0 (2026-11-06): 删除立面接近代价 riser_collision（整宽立面在物理上必须被接触，罚不掉的代价把爬升
      推向侧倾）；新增 support_margin / multi_lift / riser_contact_speed / ready_stroke / roll_excess；
      预抬净空跟踪只在单台阶立面前、每根轴一个轮，接触后的翻越跟踪对所有立面；预抬窗口与早抬前沿 0.12 m；减速带改为
      min(cmd, slow_band_speed)，不再要求轮已离地。
    - v0.3.0 (2026-11-05): 车道特权惩罚 lane、v_y 改 Huber；立面碰撞速度因子加命令项；预抬窗口/早抬前沿 0.14 m；
      减速带要求面向立面的轮已离地；θ_ref 扣除 pitch_ref_deadband_deg。
    - v0.2.0 (2026-11-05): 重写为统一越障目标；删除锚定、三版旧预抬目标、准备势、平地校准、车道观测等旧路径。
    - v0.1.0 (2026-10-26): 初始创建
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import terrain_map
from .quat import projected_gravity, quat_apply, yaw_quat
from .rover_task import ANG_VEL_SCALE, CONTROL_DT, JOINT_VEL_SCALE, OBS_CLIP, STEERING, WHEELS
from .state import ANG, JPOS, JVEL, LIN, NUM_LEG_JOINTS, POS, QUAT

PROPRIO_DIM = 45
WHEEL_RADIUS = 0.1085  # URDF 碰撞圆柱半径
PROFILE_STEP = 0.025  # 轮心沿航向的地形剖面采样间距 (m)
PROFILE_SAMPLES = 30  # 0–0.75 m
POST_CLIMB_SAMPLES = 16  # 登顶后回正窗口：后轴轮心后方的地面采样数
SINGLE_PROBE = 0.60  # 单障碍块长 0.40 m；立面后 0.60 m（留出偏航与立面插值余量）回落到原地面即判为单障碍；
# 楼梯踏面 ≤ 0.40 m，0.60 m 处已在第二级上，不会误判
TOP_PROBE = 0.03  # 立面后取顶面高度的距离（地图分辨率 0.02 m）
RIM_SAMPLES = 9  # 轮廓-地形最小间隙：轮心前后 ±r 内沿航向的采样数
LIFT_CLEARANCE = 0.012  # 轮廓离开地形（含立面角/顶面）超过该值算离地
LIFT_TEMPERATURE = 0.0015
LEADER_TEMPERATURE = 0.005  # 楼梯轮替：领先轮软判的净空差尺度 (m)
ALT_LIFT_EPS = 0.002  # 楼梯轮替：平滑绝对值 √(Δ²+ε²)−ε 的 ε (m)
LEFT_WHEELS, RIGHT_WHEELS = (0, 1), (2, 3)  # 关节顺序 (fl, rl, fr, rr)：同一轴的左右轮 = (fl, fr)、(rl, rr)
REAR_WHEELS = (1, 3)  # rl, rr
FRONT_MASK = (True, False, True, False)  # 前轴：每个立面前都预抬（楼梯上车头抬起、质心后移，单轮离地裕度更大）
SUPPORT_OTHERS = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))  # 第 i 个轮离地时的三个支撑轮
# 下台阶时按 drop_relief 减免的代价（落下必然有竖直速度、俯仰角速度、两轮短暂悬空）；翻车、横滚、跟踪不减免
DROP_RELIEF_TERMS = ("lin_vel_z", "ang_vel", "pitch", "arm_vel", "multi_lift", "support_margin")


@dataclass
class PreviewRewardCfg:
    # 1 速度跟踪 / 前进 / 停滞
    track_lin_vel: float = 5.0
    track_lin_vel_std: float = 0.5
    slow_band_m: float = 0.30
    slow_band_speed: float = 0.12  # 立面前 slow_band_m 内允许 [min(cmd, 该值), cmd]；停住仍在带外
    progress: float = 1.0  # × clamp(v_x世界 / cmd, −1, 1)
    # × sigmoid((stall_speed_ratio·cmd − v_x)/0.01)；须明显重于越障途中的代价，否则停在障碍前更划算。温度 0.01 让
    # 减速带下限 0.12 m/s 处只剩约 1%（0.02 时约 10%，乘 −5 后每步 −0.47）
    stall: float = -5.0
    stall_speed_ratio: float = 0.25
    # 任一轮正贴着上行立面翻越（−r < lead ≤ 0）时停滞代价乘该系数（1 = 原行为）。轻碰要求接触前 ≤0.15 m/s、贴面后
    # 靠摆臂翻越，此时车速≈0 被判停滞（命令 0.4 m/s 时门槛 0.10 m/s）每步 −5，比立面轻碰代价大一个量级，策略只能带速撞上去。
    stall_climb_scale: float = 1.0
    # 楼梯区（stairs_zone）的贴立面停滞系数；< 0 = 与 stall_climb_scale 相同（原行为）。楼梯每级都贴立面，单台阶的放松
    # 让"顶着第一级不动"几乎不扣分（第十四轮 12–17.5 cm 楼梯全部停住超时）
    stairs_stall_climb_scale: float = -1.0
    # 车速低于停滞门槛 stall_speed_ratio·cmd 的部分按比例计代价（× clamp((门槛 − v_x)/门槛, 0, 1)，与停滞同样乘贴立面系数；
    # 0 = 原行为）。停滞 sigmoid 在车速 0 处饱和、对动作没有梯度，停住后策略没有"重新起步"的方向（第十五轮楼梯死锁）。
    stall_linear: float = 0.0
    stall_linear_cap: float = 1.0  # 线性停滞代价 (gate − v)/gate 的上限；>1 时倒车（v<0）继续线性加重
    # ≥ 0：贴立面停滞折扣只给"立面前 stall_climb_lead_m 内到贴面、且离地的轮"，净空 0.5×该值→2×该值线性过渡（< 0 =
    # 原行为：只看水平 −r < lead ≤ 0，与是否抬轮无关）。第十六轮 it18000 左侧 12 cm 箱：左后轮着地顶住立面 13 s
    # （≈50 N、轮速 2.5 rad/s、车速 0）——顶住时停滞只扣 20%；后摆臂一抬轮心后移出 lead 窗口，折扣消失，原地抬轮
    # 每步 −12.9 比顶着 −10.5 更差。
    stall_climb_min_clearance: float = -1.0
    stall_climb_lead_m: float = 0.0
    # 堵转：贴立面（−r < lead ≤ riser_spin_lead_m）且着地（软门 净空 < 1 cm）的轮，× Σ smoothL1(relu(ω·r − v_∥); δ)
    # （δ = wheel_slip_delta；0 = 关）。轮子顶着立面空转不上不是翻越，真翻越时轮很快离地、门自动关闭。
    riser_spin: float = 0.0
    riser_spin_lead_m: float = 0.005
    # 下行边沿（轮心前 r 内或后 r 内的地面比轮下低/高出 min_step）时，下列代价乘 (1 − drop_relief)（0 = 原行为）：
    # 竖直速度、角速度、俯仰、摆臂速度、两轮离地、三轮支撑裕度。第十三轮整宽 14 cm 单台阶 77–92% 回合停在后轴即将下块处
    # 直到停滞超时（第八轮探针：停在块上每步 −5.3、往前下块 −5.9）；基于世界模型梯度的 actor 只看局部梯度，越不过这段
    # "先变贵再变便宜"的代价壁垒。
    drop_relief: float = 0.0
    # 登顶后回正窗口：后轴任一轮轮心后方 post_climb_window_m 内地面比轮下低出 min_step（刚翻上一级）时，平地滚动四项
    # （toe / flat_steer / wheel_sync / wheel_slip）乘 (1 − post_climb_relief)（0 = 原行为）。第二十二轮楼梯登顶时航向
    # 还有 6–10°（左右轮先后上台阶扭出来的），平台上这四项每步 −7～−11，航向 + 车道只有 −1.5～−3，回正要 1–2 m，
    # 横偏累计 0.2–0.3 m（验收门 0.15 m）。平地直行（这四项要保护的场景）身后没有立面，不受影响。
    post_climb_relief: float = 0.0
    post_climb_window_m: float = 1.0
    # 2 翻车
    terminated: float = -20.0
    tilt_limit: float = 0.8
    tilt_temperature: float = 0.05
    # 3 姿态
    roll: float = -50.0  # × g_y²
    roll_excess: float = -2000.0  # × relu(|g_y| − sin roll_excess_deg)²：10° 时约 −18/步，压住侧倾爬升
    roll_excess_deg: float = 4.5
    pitch: float = -50.0  # × relu(|θ − θ_ref| − tol)²
    pitch_tol_deg: float = 3.0
    pitch_ref_window: float = 0.15  # 每轮前后 ±window 的地面高度取平均
    # θ_ref = sign(θ_geo)·max(|θ_geo| − deadband, 0)：摆臂可补偿约 0.2 m 高差（轴距 0.694 m 上约 16°），
    # 扣 6° 使 12 cm 单台阶（θ_geo≈9.8°）的参考落在"单台阶俯仰 ≤8°"门内。
    pitch_ref_deadband_deg: float = 6.0
    # 4 平顺
    ang_vel: float = -2.0  # × (ωx² + 0.25 ωy²)
    arm_vel: float = -0.5  # × Σ q̇_arm²
    action_rate: float = -0.005
    lin_vel_z: float = -0.5
    # 5 航向 / 直行 / 车道
    heading: float = -30.0  # × sin²ψ
    # × smoothL1(sinψ; δ)：sin² 在小偏航处几乎没有梯度，而转向代价 −0.1·|δ| 会压住小角度修正；actor 看不到横向位置，
    # 平地横偏只能靠航向保持（第四轮平地 0.30 m/s 横偏 P95 0.4 m、偏航 P95 2.5–7°）
    heading_huber: float = -5.0
    heading_huber_delta: float = 0.02
    yaw_rate: float = -2.0  # × ω_z²：零速命令时 route_command 为 0，actor 看不到航向，只能直接压住转速
    # 楼梯区（任一轮前方 stairs_slant_zone_m 内有非单台阶立面）的航向目标 |ψ| = stairs_slant_deg，左右任取
    stairs_slant_deg: float = 12.0
    stairs_slant_zone_m: float = 0.60
    # 整宽单台阶（同一轴两轮都在单台阶区 stairs_slant_zone_m 内）的航向目标 |ψ|；0 关闭（窄台阶不受影响）
    single_slant_deg: float = 12.0
    lateral_vel: float = -15.0  # × smoothL1(v_y世界; δ)：|v_y|<δ 为 v_y²/(2δ)，否则 |v_y| − δ/2
    lateral_vel_delta: float = 0.02
    # 越障区（1 − 平地门）内横向速度代价的倍数：系数 = 1 + (scale − 1)·(1 − flat)。actor 看不到车道横偏，越障时侧滑多少
    # 就偏多少（第十四轮全宽单台阶后轴爬块时 v_y≈0.1 m/s 持续约 3 s，横偏到 0.5 m）
    obstacle_lateral_vel_scale: float = 1.0
    lane: float = -30.0  # × y²，y = 车道横向偏移（训练特权，需要 lane_center_y；actor 不观测）
    # 楼梯区车道约束系数（lane 与越障横向速度额外倍数在楼梯区 × 该值）；训练循环用 set_stairs_lane_scale 逐步收紧
    stairs_lane_scale: float = 1.0
    steering: float = -0.1  # × Σ|转向角|
    # 6 平地中立（摆臂回零 + 四轮承重）
    flat_neutral: float = -5.0  # 平地摆臂偏离中立 → 车身俯仰（第四轮平地俯仰 P95 5–6°）
    rest_clearance: float = 0.005
    flat_lookahead: float = 0.60
    # 平地滚动：× Σ_axle (δ_左 − δ_右)²（rad²）；同轴两轮反向打角不产生横移或转向，只刮地。
    # 第十一轮探针前 20° / 后 13° 八字约 −3.3/步（原 −2 只有 −0.33，把 ±43° 压到这里就不再起作用）
    toe: float = -20.0
    # 平地转向：× Σδ²（rad²）。第十一轮 0.30 m/s 四轮转向 rms 12°（0.57 Hz 摆动）约 −1.8/步；3° 小修正 −0.03/步
    flat_steer: float = -10.0
    # 平地四轮同速：× Σ smoothL1(ω_i·r − mean_j ω_j·r; δ)。只用关节速度，不依赖世界模型预测的车速；
    # 第十一轮后轮比前轮快 0.07 m/s 约 −2.6/步，零速前后轴对顶更重
    wheel_sync: float = -50.0
    wheel_sync_delta: float = 0.05
    # 平地着地轮打滑：× Σ smoothL1(ω·r − v_∥; δ)，v_∥ = 轮心速度沿轮向分量（摆臂速度不计）。
    # 第十一轮后轮快转 35% 约 −4/步
    wheel_slip: float = -20.0
    wheel_slip_delta: float = 0.05
    # 零速命令：× smoothL1(ω_z; δ)。0.04 rad/s 约 −0.15/步（原 ω_z² × 2 只有 −0.003）
    stand_yaw_rate: float = -5.0
    stand_yaw_rate_delta: float = 0.02
    # 7 越障
    riser_min_height: float = 0.03  # 立面顶高出轮底至少该值才算障碍
    riser_contact_gap: float = 0.005  # 统计/课程：轮廓到立面边 ≤ 该值记一次碰撞
    # 轻碰：轮廓距立面 < riser_approach_m 时按接近度 × relu(v − riser_safe_speed)² 计代价（0.3 m/s 两轮贴面约 −9/步）
    riser_contact_speed: float = -200.0
    riser_safe_speed: float = 0.15
    riser_approach_m: float = 0.05
    # 楼梯区的轻碰代价系数与接近距离（0 = 与 riser_approach_m 相同）；默认 = 原行为
    stairs_riser_contact_scale: float = 1.0
    stairs_riser_approach_m: float = 0.0
    # 同轴另一个轮已站上更高一级时本轮不要求接触前预抬，碰到立面后再翻越（>0.5 开启）
    trailing_contact_climb: float = 0.0
    # > 0.5：楼梯区同一轴两轮都在翻越（−r < lead ≤ 0，立面角托住）时，翻越跟踪两轮误差相加、只用实际净空（0 = 原行为：
    # 取较小者、实际与支撑净空平均）。一个轮搭上台阶角后，另一个轮的翻越误差不再被吃掉；支撑净空假设其余三轮着地，
    # 另一个轮贴着立面翻越时不成立。
    stairs_climb_pair_sum: float = 0.0
    # < 1：楼梯区同一轴两轮都在翻越时，领先轮（净空更高，软判、不求导）全权重、落后轮 × 该值（1 = 原行为）
    stairs_climb_trail_weight: float = 1.0
    # ≤ 0：楼梯区同一轴两轮都在翻越且净空差 < stairs_alt_lift_frac × 立面高时的代价系数（0 = 关）
    stairs_alt_lift: float = 0.0
    stairs_alt_lift_frac: float = 0.5
    # > 0：楼梯立面前 lead ≤ 该值的轮也算"翻越中"（不计悬空离地/三轮支撑裕度、跟踪翻越净空、参与同轴两轮判定）。顶着立面的轮
    # lead ≈ +0.001–0.005，用摆臂一抬轮心后移到 0.02–0.05，原来 lead ≤ 0 的窗口把它当悬空轮：车头上仰时质心在后，
    # 单抬后轮的支撑裕度 −3.4 cm，每步 −18～−36，比原地顶死（−15.7）还差（0 = 原行为）
    stairs_contact_climb_m: float = 0.0
    # 净空跟踪：单台阶立面前 prelift_window 内要求离地（与验收 3–12 cm 起抬对齐）；接触后（lead ≤ 0）所有立面
    # 都跟踪轮廓翻越立面所需的净空
    prelift_track: float = 5.0
    prelift_window: float = 0.12
    # 单台阶前轴领先轮（两前轮都在跟踪时取 lead 较小者）的额外净空跟踪，误差与 prelift_track 同一函数
    prelift_front: float = 15.0
    early_lift: float = -2.0  # 仅单障碍：lead > early_lift_lead 时离地的代价（缩短三轮支撑时间）
    early_lift_lead: float = 0.12
    # 三轮支撑：质心投影到支撑三角形的裕度不足门槛的 Huber 代价（单位 1 cm）。计入的轮 = 离地轮
    # （轮廓间隙 > 1.2 cm，权重不求导，门槛 support_margin_m）+ 立面前 prelift_shift_zone 内该抬的轮（不论是否已抬：
    # 先移质心再抬，不抬也逃不掉；门槛 prelift_margin_m）；立面上正在翻越的轮（−r < lead ≤ 0）不计。
    # 横滚 ≤5° 时最佳裕度约 2.2 cm，只动被抬轮摆臂时约 0。
    support_margin: float = -10.0
    support_margin_m: float = 0.005
    # 该抬的轮（单台阶所有轮 + 任何立面前的前轴，0 < lead ≤ prelift_shift_zone）的裕度门槛：进预抬窗口前就把质心移进
    # 其余三轮的三角形。0.8 cm ≈ 横滚 3° + 车头抬 3°（横滚 5° + 车头抬 3° 约 1.2 cm）。同一裕度也门控接触前的支撑净空。
    prelift_margin_m: float = 0.008
    prelift_shift_zone: float = 0.25
    multi_lift: float = -5.0  # × relu(离地轮数 − 1)²：两轮同时离地时只剩一条支撑线（翻越中的轮不计）
    # 准备姿态：立面前 ready_zone_m 内、轮仍着地时，摆臂剩余上抬行程 < 立面高度 + 1 cm 的 Huber 代价（单位 3 cm）
    ready_stroke: float = -1.0
    ready_zone_m: float = 0.30
    min_step: float = 0.03
    # 8 关节软限位
    joint_pos_limits: float = -5.0


def route_command_body(state: torch.Tensor, command_w: torch.Tensor) -> torch.Tensor:
    """preview_route_command：世界系 (vx, vy) 按车体航向转到车体系，yaw 分量原样。"""
    q = state[..., QUAT]
    yaw = torch.atan2(2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]), 1 - 2 * (q[..., 2].square() + q[..., 3].square()))
    c, s = yaw.cos(), yaw.sin()
    return torch.stack((c * command_w[..., 0] + s * command_w[..., 1],
                        -s * command_w[..., 0] + c * command_w[..., 1], command_w[..., 2]), dim=-1)


def proprio_observation(state, command_w, last_action) -> torch.Tensor:
    joint_pos = state[..., JPOS].clone()
    joint_pos[..., NUM_LEG_JOINTS:] = 0.0
    return torch.cat((
        state[..., ANG] * ANG_VEL_SCALE,
        projected_gravity(state[..., QUAT]),
        route_command_body(state, command_w),
        joint_pos,
        state[..., JVEL] * JOINT_VEL_SCALE,
        last_action,
    ), dim=-1).clamp(-OBS_CLIP, OBS_CLIP)


def _centered_tracking(command_xy, velocity_xy, std):
    # vision_preview.centered_velocity_tracking 的可导版本（不含 nan 处理）
    command_sq = command_xy.square().sum(-1)
    score = torch.exp(-(command_xy - velocity_xy).square().sum(-1) / std**2)
    rest = torch.exp(-command_sq / std**2)
    centered = ((score - rest) / (1.0 - rest).clamp_min(0.01)).clamp(-1.0, 1.0)
    return torch.where(command_sq >= 0.1**2, centered, score)


def _alignment(q):
    return (1.0 - 2.0 * (q[..., 2].square() + q[..., 3].square())).clamp(0.0, 1.0)


def first_crossing(values: torch.Tensor, offsets: torch.Tensor, threshold) -> torch.Tensor:
    """values [..., K] 沿 offsets [K] 采样；返回首次 ≥ threshold 处的线性插值距离，没有则 inf。"""
    threshold = torch.as_tensor(threshold, dtype=values.dtype, device=values.device).expand(values.shape[:-1])
    crossed = values >= threshold[..., None]
    index = crossed.to(torch.int64).argmax(-1, keepdim=True)
    lower_index = (index - 1).clamp_min(0)
    upper, lower = values.gather(-1, index).squeeze(-1), values.gather(-1, lower_index).squeeze(-1)
    x_hi, x_lo = offsets[index.squeeze(-1)], offsets[lower_index.squeeze(-1)]
    fraction = ((threshold - lower) / (upper - lower).clamp_min(1e-6)).clamp(0.0, 1.0)
    distance = torch.where(index.squeeze(-1) > 0, x_lo + fraction * (x_hi - x_lo), x_hi)
    return torch.where(crossed.any(-1), distance, torch.inf)


def _huber(error, beta):
    return torch.nn.functional.smooth_l1_loss(error, torch.zeros_like(error), reduction="none", beta=beta)


def _smoothstep(x):
    x = x.clamp(0.0, 1.0)
    return x.square() * (3.0 - 2.0 * x)


class PreviewTask:
    """learner 的任务接口：policy_obs / reward_terms / reward / survival / critic_extra / event_priority。"""

    obs_dim = PROPRIO_DIM + 187

    def __init__(self, hmap, kin, scan_pattern, joint_soft_limits, cfg: PreviewRewardCfg | None = None,
                 exact_scan_fn=None, exact_riser_fn=None, lane_center_y: torch.Tensor | None = None,
                 critic_route_features: bool = False, critic_lane: bool | None = None, lead_obs: bool = False):
        """joint_soft_limits: [8, 2] 腿关节软限位（相对默认位）。
        exact_scan_fn(state) → [..., 187]：真实射线扫描，直通前向、梯度走双线性插值。
        exact_riser_fn(starts [..., 3], directions [..., 3]) → 水平射线到地形网格的距离（未命中为 inf）；
          省略时用高度图剖面近似（仅数值测试）。
        lane_center_y：各车道中心的世界 y（训练特权信息，不进 actor），供车道惩罚与 critic 车道特征使用。
        critic_lane：critic 是否额外看横向偏移；None 时等价于"提供了 lane_center_y"。"""
        self.hmap, self.kin, self.pattern = hmap, kin, scan_pattern
        self.limits = joint_soft_limits
        self.cfg = cfg or PreviewRewardCfg()
        if self.cfg.lane and lane_center_y is None:
            raise ValueError("车道惩罚 lane≠0 需要 lane_center_y（训练特权车道中心）")
        if self.cfg.lateral_vel_delta <= 0:
            raise ValueError("lateral_vel_delta 必须为正")
        if self.cfg.stall_linear_cap < 1.0:
            raise ValueError("stall_linear_cap 须 ≥ 1")
        if self.cfg.tilt_temperature <= 0:
            raise ValueError("tilt_temperature 必须为正")
        if not 0.0 <= self.cfg.post_climb_relief <= 1.0:
            raise ValueError("post_climb_relief 须在 [0, 1]")
        if self.cfg.post_climb_window_m <= 0:
            raise ValueError("post_climb_window_m 必须为正")
        self.exact_scan_fn = exact_scan_fn
        self.exact_riser_fn = exact_riser_fn
        self.lane_center_y = lane_center_y
        self.critic_lane = lane_center_y is not None if critic_lane is None else critic_lane
        if self.critic_lane and lane_center_y is None:
            raise ValueError("critic_lane 需要 lane_center_y")
        self.critic_route_features = critic_route_features
        self._stairs_lane_scale = None  # set_stairs_lane_scale 写入的张量缓冲；None 时用 cfg.stairs_lane_scale
        self.critic_extra_dim = (terrain_map.RELIEF_DIM + (1 if self.critic_lane else 0)
                                 + (2 if critic_route_features else 0))
        # 先抬侧标记（command 第 4 维，±1）：左右对称的卡死状态下让 actor 能打破对称、先抬一侧，见 networks.LeadSideActor
        self.lead_obs = bool(lead_obs)
        self.command_dim = 4 if self.lead_obs else 3
        self.obs_dim = PROPRIO_DIM + scan_pattern.shape[0] + int(self.lead_obs)
        # 摆臂软限位两端的轮心高度（base 系）：较高一端即"上抬极限"。前腿 θ 负向、后腿 θ 正向抬轮，由 FK 判定。
        arm_limits = joint_soft_limits[:4].to(kin.wheel_center_link)
        ends = torch.stack([kin.wheel_centers_base(arm_limits[:, k], torch.zeros_like(arm_limits[:, k]))[..., 2]
                            for k in (0, 1)])
        self.wheel_top_z = ends.amax(0)
        self.max_stroke = self.wheel_top_z - ends.amin(0)
        self._others = torch.tensor(SUPPORT_OTHERS)
        self._cache = None

    # ---------------- 观测 ----------------
    def scan(self, state):
        smooth = terrain_map.height_scan(self.hmap, state, self.pattern)
        if self.exact_scan_fn is None:
            return smooth
        with torch.no_grad():
            exact = self.exact_scan_fn(state.reshape(-1, state.shape[-1])).reshape(smooth.shape)
        # 直通：前向用真实射线（台阶边不再是斜坡），反向用双线性插值的梯度。
        return smooth + (exact - smooth).detach()

    def policy_obs(self, state, command, last_action):
        parts = [proprio_observation(state, command, last_action), self.scan(state)]
        if self.lead_obs:
            if command.shape[-1] != 4:
                raise ValueError(f"lead_obs 需要 4 维命令（第 4 维为先抬侧 ±1），实际 {command.shape[-1]} 维")
            parts.append(command[..., 3:4])
        return torch.cat(parts, dim=-1)

    def critic_extra(self, state):
        parts = [terrain_map.relief_features(self.hmap, self.kin, state)]
        if self.critic_lane:
            parts.append(self.lateral_offset(state)[..., None])
        if self.critic_route_features:
            # 同一本体速度沿世界路线或横着走有不同回报；航向不变特征无法区分这两种价值。
            q = yaw_quat(state[..., QUAT])
            forward = quat_apply(q, state.new_tensor([1.0, 0.0, 0.0]).expand(q.shape[:-1] + (3,)))
            parts.append(forward[..., :2])
        return torch.cat(parts, dim=-1)

    def set_stairs_lane_scale(self, value: float, device=None) -> None:
        """原地写入楼梯区车道约束系数（0–1）；用张量而不是改 cfg，避免编译图按常量重编译。"""
        if not 0.0 <= value <= 1.0:
            raise ValueError("stairs_lane_scale 须在 [0, 1]")
        if self._stairs_lane_scale is None:
            self._stairs_lane_scale = torch.tensor(float(value), device=device)
        else:
            self._stairs_lane_scale.fill_(float(value))

    def _stairs_lane_factor(self, ctx) -> torch.Tensor:
        """楼梯区内为当前车道约束系数，区外为 1。"""
        zone = ctx["stairs_zone"]
        scale = (zone.new_tensor(self.cfg.stairs_lane_scale, dtype=torch.float32) if self._stairs_lane_scale is None
                 else self._stairs_lane_scale.to(zone.device))
        return torch.where(zone, scale, torch.ones_like(scale))

    def lateral_offset(self, state) -> torch.Tensor:
        """车体到最近车道中心线的带符号横向距离（块间距 8 m）。"""
        y = state[..., POS][..., 1]
        centers = self.lane_center_y.to(y.device)
        idx = torch.bucketize(y.detach().contiguous(), centers).clamp(1, centers.numel() - 1)
        lo, hi = centers[idx - 1], centers[idx]
        nearest = torch.where((y - lo).abs() <= (hi - y).abs(), lo, hi)
        return y - nearest

    # ---------------- 地形几何（每个状态一次） ----------------
    def terrain_context(self, state) -> dict[str, torch.Tensor]:
        """同一状态对象只计算一次（奖励、优先级、统计共用）；立面射线只在这里发出。"""
        grad = torch.is_grad_enabled()
        cached = self._cache
        # 无梯度算出的几何不能给需要梯度的调用复用。
        if cached is not None and cached[0] is state and cached[1] == state._version and (cached[2] or not grad):
            return cached[3]
        context = self._compute_context(state)
        self._cache = (state, state._version, grad, context)
        return context

    def _compute_context(self, state) -> dict[str, torch.Tensor]:
        cfg, radius = self.cfg, WHEEL_RADIUS
        jp = state[..., JPOS]
        local = self.kin.wheel_centers_base(jp[..., :4], jp[..., 4:8])
        q4 = state[..., QUAT].unsqueeze(-2).expand(local.shape[:-1] + (4,))
        wheels = state[..., POS].unsqueeze(-2) + quat_apply(q4, local)
        com = state[..., POS] + quat_apply(state[..., QUAT], self.kin.com_base(local))
        yq = yaw_quat(state[..., QUAT]).detach()
        fwd = quat_apply(yq, state.new_tensor([1.0, 0.0, 0.0]).expand(yq.shape[:-1] + (3,)))[..., :2]
        xy = wheels[..., :2]
        ground = self.hmap.query(xy)
        bottom = wheels[..., 2] - radius
        clearance = bottom - ground
        with torch.no_grad():
            offsets = torch.arange(PROFILE_SAMPLES + 1, device=state.device, dtype=state.dtype) * PROFILE_STEP
            base_xy = xy.detach()
            along = lambda distance: base_xy + distance[..., None] * fwd[..., None, :]  # noqa: E731
            profile = self.hmap.query(base_xy[..., None, :] + offsets[:, None] * fwd[..., None, None, :])
            ground_here = profile[..., 0]
            rises = profile - ground_here[..., None]
            # 相对轮下地面的第一个立面（与是否抬轮无关）：减速带、单障碍判别、净空跟踪共用。
            edge = first_crossing(rises[..., 1:], offsets[1:], cfg.min_step)
            lead = edge - radius
            edge_safe = edge.nan_to_num(posinf=0.0)
            riser_rise = self.hmap.query(along(edge_safe + TOP_PROBE)) - ground_here
            beyond = self.hmap.query(along(edge_safe + SINGLE_PROBE)) - ground_here
            single = torch.isfinite(edge) & (beyond < 0.5 * riser_rise)
            relief = rises.abs().masked_fill(offsets > cfg.flat_lookahead, 0.0).amax(-1).amax(-1)
            ground_range = ground_here.amax(-1) - ground_here.amin(-1)
            flat = (torch.sigmoid((cfg.min_step - relief) / 0.005)
                    * torch.sigmoid((cfg.min_step - ground_range) / 0.005))
            window = torch.linspace(-cfg.pitch_ref_window, cfg.pitch_ref_window, 5, device=state.device)
            heights = self.hmap.query(base_xy[..., None, :] + window[:, None] * fwd[..., None, None, :]).mean(-1)
            span = ((base_xy[..., [0, 2], :].mean(-2) - base_xy[..., [1, 3], :].mean(-2)) * fwd).sum(-1)
            pitch_geo = torch.atan2(heights[..., [0, 2]].mean(-1) - heights[..., [1, 3]].mean(-1), span.clamp_min(0.2))
            pitch_ref = pitch_geo.sign() * (pitch_geo.abs() - math.radians(cfg.pitch_ref_deadband_deg)).clamp_min(0.0)
            # 立面碰撞：在轮底 + min_height 处沿航向的水平射线，命中即为高出轮底 ≥ min_height 的立面。
            start_z = bottom.detach() + cfg.riser_min_height
            if self.exact_riser_fn is not None:
                starts = torch.cat((base_xy, start_z[..., None]), dim=-1)
                directions = torch.cat((fwd, torch.zeros_like(fwd[..., :1])), dim=-1)[..., None, :].expand_as(starts)
                face = self.exact_riser_fn(starts, directions)
            else:
                face = first_crossing(profile, offsets, start_z)
            face_valid = torch.isfinite(face)
            face_point = along(face.nan_to_num(posinf=0.0))
            riser_top = self.hmap.query(face_point + TOP_PROBE * fwd[..., None, :])
        # 立面距离对轮心水平位置、碰撞深度对轮底高度可导；立面点与顶高本身不求导。
        face_distance = ((face_point - xy) * fwd[..., None, :]).sum(-1)
        depth = riser_top - bottom
        reach = depth.clamp(0.005, radius)
        extent = (radius**2 - (radius - reach).square()).clamp_min(1e-8).sqrt()
        # 轮廓-地形最小间隙（轮心前后 ±r，含立面角与台阶顶面），对轮心位置可导；> LIFT_CLEARANCE 视为离地。
        rim_x = torch.linspace(-radius, radius, RIM_SAMPLES, device=state.device, dtype=state.dtype)
        rim_drop = (radius**2 - rim_x.square()).clamp_min(0.0).sqrt()
        rim_points = xy[..., None, :] + rim_x[:, None] * fwd[..., None, None, :]
        rim_clear = (wheels[..., 2:3] - rim_drop - self.hmap.query(rim_points)).amin(-1)
        lifted = torch.sigmoid((rim_clear - LIFT_CLEARANCE) / LIFT_TEMPERATURE)
        with torch.no_grad():
            # 下台阶：轮心后方 r 内的地面比轮心下高出 min_step（轮刚越过下行边沿、正在落下），不算抬轮离地
            rim_ground = self.hmap.query(rim_points.detach())
            behind = rim_ground[..., : RIM_SAMPLES // 2].amax(-1)
            dropping = (behind - ground.detach()) > cfg.min_step
            # 即将下台阶：轮心前方 r 内的地面比轮下低出 min_step（轮停在下行边沿前）
            drop_ahead = (ground.detach() - rim_ground[..., RIM_SAMPLES // 2 + 1:].amin(-1)) > cfg.min_step
            # 楼梯区：任一轮前方 stairs_slant_zone_m 内有非单台阶立面（含正在翻越的）
            stairs_zone = ((lead <= cfg.stairs_slant_zone_m) & ~single).any(-1)
            # 整宽单台阶区：同一轴两轮都在单台阶区内（窄台阶只有一侧的轮看到立面，不进入）
            near_single = single & (lead <= cfg.stairs_slant_zone_m)
            wide_single_zone = ((near_single[..., LEFT_WHEELS[0]] & near_single[..., RIGHT_WHEELS[0]])
                                | (near_single[..., LEFT_WHEELS[1]] & near_single[..., RIGHT_WHEELS[1]]))
            # 登顶后：后轴任一轮轮心后方 post_climb_window_m 内的地面比轮下低出 min_step（刚翻上去）
            if cfg.post_climb_relief:
                back = torch.linspace(PROFILE_STEP, cfg.post_climb_window_m, POST_CLIMB_SAMPLES,
                                      device=state.device, dtype=state.dtype)
                rear_xy = base_xy[..., list(REAR_WHEELS), :]
                behind_ground = self.hmap.query(rear_xy[..., None, :] - back[:, None] * fwd[..., None, None, :])
                post_climb = ((ground_here[..., list(REAR_WHEELS)][..., None] - behind_ground).amax(-1)
                              > cfg.min_step).any(-1)
            else:
                post_climb = torch.zeros_like(flat, dtype=torch.bool)
        # 第 i 个轮离地时，质心水平投影到其余三轮支撑三角形各边的最小带符号距离（内侧为正），对轮位/姿态可导。
        tri = xy[..., self._others.to(state.device), :]
        start, end, apex = tri, tri.roll(-1, dims=-2), tri.roll(-2, dims=-2)
        edge_dir = end - start
        normal = torch.stack((-edge_dir[..., 1], edge_dir[..., 0]), dim=-1)
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        inward = torch.sign(((apex - start) * normal).sum(-1)).detach()
        support_margin = (((com[..., None, None, :2] - start) * normal).sum(-1) * inward).amin(-1)
        # 支撑净空：假设其余三轮落在各自地形上（轮廓间隙为 0）时第 i 个轮的轮底净空 = 自身净空 − 三轮轮廓间隙在该轮
        # 水平位置的仿射外推（重心坐标权重，不求导）。车身整体平移/小角度转动对四个轮是同一个仿射函数，被外推精确抵消，
        # 梯度只剩摆臂运动学——世界模型从没见过单轮离地时，它学到的"收一条臂车角就跟着下沉"会把实际净空的梯度抵消掉。
        with torch.no_grad():
            a, b, c = xy.detach()[..., self._others.to(state.device), :].unbind(-2)
            e1, e2, p = b - a, c - a, xy.detach() - a
            cross = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]  # noqa: E731
            den = cross(e1, e2)
            den = torch.where(den.abs() > 1e-6, den, torch.full_like(den, 1e-6))
            s_b, s_c = cross(p, e2) / den, cross(e1, p) / den
            bary = torch.stack((1.0 - s_b - s_c, s_b, s_c), dim=-1)
        # 支撑轮离开支撑面的高度：轮廓间隙（被立面角托住时约 0）；贴住高于轮心的立面时竖直轮廓间隙为负是采样伪影，
        # 这时取轮下地面净空（着地约 0）。平地着地/悬空/轻微穿透时两者相等，车身位姿照常被抵消。
        support_gap = torch.maximum(rim_clear, clearance.clamp_max(0.0))
        support_clearance = clearance - (bary * support_gap[..., self._others.to(state.device)]).sum(-1)
        return {"wheels": wheels, "fwd": fwd, "ground": ground, "clearance": clearance, "lead": lead,
                "riser_rise": riser_rise, "single": single, "flat": flat, "ground_range": ground_range,
                "pitch_geo": pitch_geo, "pitch_ref": pitch_ref, "face_valid": face_valid, "gap": face_distance - extent,
                "depth": depth, "rim_clear": rim_clear, "lifted": lifted, "support_margin": support_margin,
                "support_clearance": support_clearance, "dropping": dropping, "drop_ahead": drop_ahead,
                "post_climb": post_climb,
                "stairs_zone": stairs_zone,
                "wide_single_zone": wide_single_zone,
                # 摆臂剩余上抬行程：base 系轮心到上抬极限的高度差，对摆臂角可导
                "stroke": self.wheel_top_z.to(local) - local[..., 2]}

    def riser_contacts(self, state) -> torch.Tensor:
        """[..., 4]：轮胎轮廓到立面边的水平距离 ≤ riser_contact_gap（四轮同一判据，统计/课程用）。"""
        ctx = self.terrain_context(state)
        return (ctx["face_valid"] & (ctx["gap"].detach() <= self.cfg.riser_contact_gap)
                & (ctx["depth"].detach() >= self.cfg.riser_min_height))

    def pitch(self, state) -> torch.Tensor:
        """车头抬起为正的俯仰角；asin 夹在 ±0.999 内，梯度有限。"""
        return torch.asin((-projected_gravity(state[..., QUAT])[..., 0]).clamp(-0.999, 0.999))

    @torch.no_grad()
    def posture_errors(self, state) -> tuple[torch.Tensor, torch.Tensor]:
        """(|roll|, |θ − θ_ref|)，弧度；统计/课程用。"""
        roll = torch.asin(projected_gravity(state[..., QUAT])[..., 1].abs().clamp(max=1.0))
        return roll, (self.pitch(state) - self.terrain_context(state)["pitch_ref"]).abs()

    @torch.no_grad()
    def progressing(self, state, command) -> torch.Tensor:
        """未下发前进命令，或世界系前进速度 ≥ stall_speed_ratio·cmd。"""
        v = quat_apply(state[..., QUAT], state[..., LIN])[..., 0]
        return (command[..., 0] < 0.05) | (v >= self.cfg.stall_speed_ratio * command[..., 0])

    @torch.no_grad()
    def event_priority(self, state) -> torch.Tensor:
        """[B]：接近立面、跨阶、轮离地为 1，轮贴立面再加 1；供回放优先采样。"""
        ctx = self.terrain_context(state)
        near = ((ctx["lead"] <= self.cfg.flat_lookahead).any(-1) | (ctx["ground_range"] > self.cfg.min_step)
                | (ctx["clearance"] > 0.02).any(-1))
        return near.float() + self.riser_contacts(state).any(-1).float()

    @torch.no_grad()
    def imagination_flat_mask(self, state):
        return self.terrain_context(state)["flat"] > 0.95

    def wheel_ground_clearance(self, state):
        jp = state[..., JPOS]
        wheels = self.kin.wheel_centers_world(state[..., POS], state[..., QUAT], jp[..., :4], jp[..., 4:8])
        return wheels[..., 2] - WHEEL_RADIUS - self.hmap.query(wheels[..., :2])

    # ---------------- 奖励 ----------------
    def reward_terms(self, state, next_state, command, action, last_action, acc_sq=None) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        ctx = self.terrain_context(next_state)
        q = next_state[..., QUAT]
        gravity = projected_gravity(q)
        upright = (-gravity[..., 2]).clamp(0.0, 0.7) / 0.7
        lin_b, ang_b = next_state[..., LIN], next_state[..., ANG]
        lin_w = quat_apply(q, lin_b)
        cmd_x = command[..., 0]
        commanded = (cmd_x >= 0.05).float()
        # 1 跟踪：立面前允许 [min(cmd, slow_band_speed), cmd] 内任意速度（轻碰需要减速）；停住不在带内，
        # 顶着立面不上仍有跟踪损失与停滞代价。
        near = (ctx["lead"] <= cfg.slow_band_m).any(-1) & (cmd_x > 0.0)
        band = torch.minimum(torch.maximum(lin_w[..., 0].detach(), cmd_x.clamp(max=cfg.slow_band_speed)), cmd_x)
        target = torch.stack((torch.where(near, band, cmd_x), torch.zeros_like(cmd_x)), dim=-1)
        align = _alignment(q)
        track = _centered_tracking(target, lin_w[..., :2], cfg.track_lin_vel_std)
        track = (track.clamp_min(0.0) * align + track.clamp_max(0.0) - 0.25 * (1.0 - align)) * upright
        progress = commanded * (lin_w[..., 0] / cmd_x.clamp_min(0.1)).clamp(-1.0, 1.0)
        stall = commanded * torch.sigmoid((cfg.stall_speed_ratio * cmd_x - lin_w[..., 0]) / 0.01)
        stall_gate = cfg.stall_speed_ratio * cmd_x
        stall_lin = commanded * ((stall_gate - lin_w[..., 0]) / stall_gate.clamp_min(1e-3)).clamp(0.0, cfg.stall_linear_cap)
        stairs_climb_scale = cfg.stall_climb_scale if cfg.stairs_stall_climb_scale < 0 else cfg.stairs_stall_climb_scale
        if cfg.stall_climb_scale != 1.0 or stairs_climb_scale != 1.0:
            lead_now = ctx["lead"].nan_to_num(posinf=1.0)
            climb_scale = torch.where(ctx["stairs_zone"], stall.new_tensor(stairs_climb_scale),
                                      stall.new_tensor(cfg.stall_climb_scale))
            if cfg.stall_climb_min_clearance < 0.0:
                climbing_any = ((lead_now > -WHEEL_RADIUS + 0.002) & (lead_now <= 0.0)).any(-1)
                stall_scale = torch.where(climbing_any, climb_scale, stall.new_tensor(1.0))
            else:
                # 只有立面前/贴立面的轮真的离地才算翻越：净空 0.5·c → 2·c 线性过渡（c = stall_climb_min_clearance），
                # 着地（净空 ≈0）不打折；对净空可导，抬轮本身就降低停滞代价
                near_riser = ((lead_now > -WHEEL_RADIUS + 0.002) & (lead_now <= cfg.stall_climb_lead_m)).float()
                floor = 0.5 * cfg.stall_climb_min_clearance
                lifted = ((ctx["clearance"] - floor) / (1.5 * cfg.stall_climb_min_clearance + 1e-6)).clamp(0.0, 1.0)
                climb_w = (near_riser * lifted).amax(-1)
                stall_scale = 1.0 - (1.0 - climb_scale) * climb_w
            stall = stall * stall_scale
            stall_lin = stall_lin * stall_scale
        # 3 姿态
        pitch_error = ((self.pitch(next_state) - ctx["pitch_ref"]).abs() - math.radians(cfg.pitch_tol_deg)).clamp_min(0.0)
        # 5 航向：sin(ψ) 由前向投影得到，避免 atan2 在 ψ=0 的梯度问题
        forward = quat_apply(q, q.new_tensor([1.0, 0.0, 0.0]).expand(q.shape[:-1] + (3,)))
        sin_yaw = forward[..., 1] / (forward[..., :2].square().sum(-1) + 1e-6).sqrt()
        # 楼梯区 / 整宽单台阶区：|sinψ| 向 sin(斜行角) 收敛（左右对称，镜像增广不变）；其余地方与原来的 sinψ 完全相同
        sin_stairs = sin_yaw.new_tensor(math.sin(math.radians(cfg.stairs_slant_deg)))
        sin_single = sin_yaw.new_tensor(math.sin(math.radians(cfg.single_slant_deg)))
        slant = torch.where(ctx["stairs_zone"], sin_stairs,
                            torch.where(ctx["wide_single_zone"], sin_single, sin_yaw.new_zeros(())))
        heading_error = torch.where(slant > 0, (sin_yaw.square() + 1e-6).sqrt() - slant, sin_yaw)
        # 6 平地中立
        arm = next_state[..., JPOS][..., :4]
        support = ((ctx["clearance"] - cfg.rest_clearance).clamp_min(0.0) / 0.02).square().mean(-1)
        # 7 越障。轻碰：着地轮的轮廓接近高出轮底 ≥ riser_min_height 的立面时，超过安全速度的部分计代价。
        v_forward = (lin_w[..., :2] * ctx["fwd"]).sum(-1)
        facing_riser = ctx["face_valid"] & (ctx["depth"].detach() >= cfg.riser_min_height)
        stairs_zone = ctx["stairs_zone"]
        approach = torch.where(stairs_zone, v_forward.new_tensor(cfg.stairs_riser_approach_m or cfg.riser_approach_m),
                               v_forward.new_tensor(cfg.riser_approach_m))[..., None]
        closeness = torch.minimum((approach - ctx["gap"]).clamp_min(0.0), approach) / approach
        impact = (closeness * facing_riser).sum(-1) * (v_forward - cfg.riser_safe_speed).clamp_min(0.0).square()
        if cfg.stairs_riser_contact_scale != 1.0:
            impact = impact * torch.where(stairs_zone, impact.new_tensor(cfg.stairs_riser_contact_scale),
                                          impact.new_tensor(1.0))
        # 三轮支撑：计入权重不求导（不能靠放下轮子逃避裕度代价）；两轮同时离地的代价对离地程度求导。
        support_weight, airborne, margin_target = self._support_weights(ctx)
        margin_deficit = (margin_target - ctx["support_margin"]).clamp_min(0.0) / 0.01
        support_cost = (support_weight * _huber(margin_deficit, 1.0)).sum(-1)
        multi_lift = (airborne.sum(-1) - 1.0).clamp_min(0.0).square()
        leg = next_state[..., JPOS][..., :NUM_LEG_JOINTS]
        out_of_limits = (self.limits[:, 0] - leg).clamp_min(0.0) + (leg - self.limits[:, 1]).clamp_min(0.0)
        lane_factor = self._stairs_lane_factor(ctx)
        terms = {
            "track_lin_vel": cfg.track_lin_vel * track,
            "progress": cfg.progress * progress,
            "stall": cfg.stall * stall,
            "terminated": cfg.terminated * (1.0 - self.survival(next_state)),
            "roll": cfg.roll * gravity[..., 1].square(),
            "roll_excess": cfg.roll_excess * (gravity[..., 1].abs() - math.sin(math.radians(cfg.roll_excess_deg))
                                              ).clamp_min(0.0).square(),
            "pitch": cfg.pitch * pitch_error.square(),
            "ang_vel": cfg.ang_vel * (ang_b[..., 0].square() + 0.25 * ang_b[..., 1].square()),
            "arm_vel": cfg.arm_vel * next_state[..., JVEL][..., :4].square().sum(-1),
            "action_rate": cfg.action_rate * (action - last_action).square().sum(-1),
            "lin_vel_z": cfg.lin_vel_z * lin_b[..., 2].square(),
            "heading": cfg.heading * heading_error.square(),
            "heading_huber": cfg.heading_huber * torch.nn.functional.smooth_l1_loss(
                heading_error, torch.zeros_like(heading_error), reduction="none", beta=cfg.heading_huber_delta),
            "yaw_rate": cfg.yaw_rate * ang_b[..., 2].square(),
            "lateral_vel": cfg.lateral_vel * (1.0 + (cfg.obstacle_lateral_vel_scale - 1.0) * (1.0 - ctx["flat"].detach())
                                              * lane_factor)
            * torch.nn.functional.smooth_l1_loss(lin_w[..., 1], torch.zeros_like(lin_w[..., 1]), reduction="none",
                                                 beta=cfg.lateral_vel_delta),
            "steering": cfg.steering * next_state[..., JPOS][..., STEERING].abs().sum(-1),
            "flat_neutral": cfg.flat_neutral * ctx["flat"] * (arm.square().sum(-1) + support),
            "riser_contact_speed": cfg.riser_contact_speed * impact,
            "support_margin": cfg.support_margin * support_cost,
            "multi_lift": cfg.multi_lift * multi_lift,
            "joint_pos_limits": cfg.joint_pos_limits * out_of_limits.sum(-1),
        }
        if cfg.stall_linear:
            terms["stall_linear"] = cfg.stall_linear * stall_lin
        if cfg.riser_spin:
            terms["riser_spin"] = cfg.riser_spin * self._riser_spin(next_state, ctx)
        if cfg.lane:
            terms["lane"] = cfg.lane * lane_factor * self.lateral_offset(next_state).square()
        if cfg.prelift_track:
            terms["prelift_track"] = cfg.prelift_track * self._prelift_tracking(ctx)
        if cfg.early_lift:
            lead = ctx["lead"]
            early = self._prelift_eligible(ctx) & (lead > cfg.early_lift_lead) & (lead <= cfg.flat_lookahead)
            cost = ((ctx["clearance"] - 0.003).clamp_min(0.0) / 0.02).square().clamp_max(4.0)
            terms["early_lift"] = cfg.early_lift * (cost * early).sum(-1)
        if cfg.ready_stroke:
            terms["ready_stroke"] = cfg.ready_stroke * self._ready_stroke(ctx)
        if cfg.stairs_alt_lift:
            terms["stairs_alt_lift"] = cfg.stairs_alt_lift * self._stairs_alt_lift(ctx)
        if cfg.prelift_front:
            terms["prelift_front"] = cfg.prelift_front * self._prelift_front(ctx)
        if cfg.toe or cfg.wheel_slip or cfg.flat_steer or cfg.wheel_sync:
            rolling = self._flat_rolling(next_state, ctx)
            gate = ctx["flat"]
            if cfg.post_climb_relief:
                gate = gate * (1.0 - cfg.post_climb_relief * ctx["post_climb"].to(gate.dtype))
            for name in ("toe", "flat_steer", "wheel_sync", "wheel_slip"):
                if getattr(cfg, name):
                    terms[name] = getattr(cfg, name) * gate * rolling[name]
        if cfg.stand_yaw_rate:
            terms["stand_yaw_rate"] = (cfg.stand_yaw_rate * (1.0 - commanded)
                                       * _huber(ang_b[..., 2], cfg.stand_yaw_rate_delta))
        if cfg.drop_relief:
            descending = (ctx["dropping"] | ctx["drop_ahead"]).any(-1)
            relief = 1.0 - cfg.drop_relief * descending.to(cmd_x.dtype)
            for name in DROP_RELIEF_TERMS:
                if name in terms:
                    terms[name] = terms[name] * relief
        return terms

    def _flat_rolling(self, state, ctx) -> dict[str, torch.Tensor]:
        """平地滚动的四个代价 [...]：toe = 同轴两轮反向打角的平方和；flat_steer = 转向角平方和；
        wheel_sync = 四轮轮缘速度偏离四轮均值的 Huber 和；wheel_slip = 着地轮"轮缘速度 − 轮心沿轮向速度"的 Huber 和。

        左右转向关节同一符号约定，四轮同号 = 蟹行，同轴异号 = toe（只刮地）。转向角正方向 = 轮头朝右（第十轮平地探针：
        四轮同号转向与世界系 v_y 的相关系数 −0.67）；轮速正方向 = 前进。轮心速度 = 车体线速度 + ω × 轮心（base 系，
        摆臂角速度不计）；转向角小时 v_∥ 主要是 v_x·cosδ，符号约定的误差只影响 v_y·sinδ 这一小项。"""
        jp, jv = state[..., JPOS], state[..., JVEL]
        steer = jp[..., STEERING]
        left, right = list(LEFT_WHEELS), list(RIGHT_WHEELS)
        toe = (steer[..., left] - steer[..., right]).square().sum(-1)
        local = self.kin.wheel_centers_base(jp[..., :4], steer)
        lin_b, ang_b = state[..., LIN], state[..., ANG]
        v_wheel = lin_b.unsqueeze(-2) + torch.linalg.cross(ang_b.unsqueeze(-2).expand_as(local), local, dim=-1)
        v_along = v_wheel[..., 0] * steer.cos() - v_wheel[..., 1] * steer.sin()
        rim = jv[..., WHEELS] * WHEEL_RADIUS
        slip = rim - v_along
        grounded = (ctx["clearance"].detach() < 0.01).to(slip.dtype)
        return {"toe": toe, "flat_steer": steer.square().sum(-1),
                "wheel_sync": _huber(rim - rim.mean(-1, keepdim=True), self.cfg.wheel_sync_delta).sum(-1),
                "wheel_slip": (_huber(slip, self.cfg.wheel_slip_delta) * grounded).sum(-1)}

    def _riser_spin(self, state, ctx) -> torch.Tensor:
        """[...] 贴着上行立面、仍着地的轮"轮缘速度超出轮心沿轮向速度"的 Huber 和（堵转空转）。"""
        jp, jv = state[..., JPOS], state[..., JVEL]
        steer = jp[..., STEERING]
        local = self.kin.wheel_centers_base(jp[..., :4], steer)
        lin_b, ang_b = state[..., LIN], state[..., ANG]
        v_wheel = lin_b.unsqueeze(-2) + torch.linalg.cross(ang_b.unsqueeze(-2).expand_as(local), local, dim=-1)
        v_along = v_wheel[..., 0] * steer.cos() - v_wheel[..., 1] * steer.sin()
        spin = (jv[..., WHEELS] * WHEEL_RADIUS - v_along).clamp_min(0.0)
        lead = ctx["lead"].detach().nan_to_num(posinf=1.0)
        pressed = ((lead > -WHEEL_RADIUS + 0.002) & (lead <= self.cfg.riser_spin_lead_m)).to(spin.dtype)
        grounded = torch.sigmoid((0.01 - ctx["clearance"].detach()) / 0.003)
        return (_huber(spin, self.cfg.wheel_slip_delta) * pressed * grounded).sum(-1)

    def _support_weights(self, ctx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(支撑裕度计入权重 [..., 4]（不求导）, 离地程度 [..., 4]（翻越中的轮置零）, 裕度门槛 [..., 4] (m))。

        立面上正在翻越的轮（−r < lead ≤ 0）由立面角托住、翻越跟踪要求它沿轮廓上抬，不算悬空离地——
        否则整宽立面（楼梯、整宽台阶）两前轮同时翻越会被两轮离地/支撑裕度代价顶住。刚越过下行边沿的轮（下台阶）
        同理不算：整宽块后沿两前轮必然同时落下，这段代价让策略停在块上不走。
        单台阶立面前 prelift_shift_zone（不短于预抬窗口）内该抬的轮不论是否已抬都计裕度，门槛 prelift_margin_m：进窗口
        之前就把质心移进其余三轮的三角形（质心压在对角线上时收臂只会让车身倾倒）；楼梯前轴只在预抬窗口内计、门槛
        support_margin_m。同一轴两轮都在区内时只计裕度代价较小的那个（与预抬跟踪"每轴一个轮"一致），窄台阶上计
        面对立面的轮。其余离地轮的门槛仍是 support_margin_m。"""
        cfg, radius = self.cfg, WHEEL_RADIUS
        lead = ctx["lead"].nan_to_num(posinf=1.0)
        climbing = self._climb_window(ctx)
        airborne = ctx["lifted"] * (~(climbing | ctx["dropping"])).to(ctx["lifted"].dtype)
        with torch.no_grad():
            # 单台阶：从 prelift_shift_zone 起、门槛 prelift_margin_m（先移质心）；楼梯前轴：只在预抬窗口内、门槛
            # support_margin_m（第十一轮的口径）。楼梯踏面 26–40 cm，前轴几乎一直在下一级立面 0.25 m 内，第十二轮把
            # 移质心要求推广到楼梯后，楼梯课程比第十/十一轮低 2–3 cm（验收不要求楼梯预抬，只要求轻碰快上）。
            single = ctx["single"]
            zone = torch.where(single, lead.new_tensor(max(cfg.prelift_window, cfg.prelift_shift_zone)),
                               lead.new_tensor(cfg.prelift_window))
            need = torch.where(single, airborne.new_tensor(cfg.prelift_margin_m), airborne.new_tensor(cfg.support_margin_m))
            deficit = (need - ctx["support_margin"]).clamp_min(0.0)
            expect = self._prelift_eligible(ctx) & (lead > 0.0) & (lead <= zone)
            left, right = list(LEFT_WHEELS), list(RIGHT_WHEELS)
            pair = expect[..., left] & expect[..., right]
            pick_left = deficit[..., left] <= deficit[..., right]
            chosen = torch.cat((expect[..., left] & (~pair | pick_left), expect[..., right] & (~pair | ~pick_left)), -1)
            weight = torch.maximum(airborne, chosen.to(airborne.dtype))
            target = torch.where(chosen, need, torch.full_like(need, cfg.support_margin_m))
        return weight, airborne, target

    def _prelift_eligible(self, ctx) -> torch.Tensor:
        """[..., 4] 接触前要求预抬的轮：单台阶的所有轮 + 任何立面前的前轴（楼梯也要求前轮先抬）。

        楼梯每级让四个轮都预抬在几何上不可能（踏面 < 4 个轮半径之和），但前轴每级一个轮先抬可行：车头已抬起、
        质心后移，前轮单轮离地的三轮支撑裕度比平地更大。后轴在楼梯上仍靠接触后翻越（质心后移，后轮单抬裕度为负）。"""
        eligible = ctx["single"] | ctx["single"].new_tensor(FRONT_MASK)
        if self.cfg.trailing_contact_climb > 0.5:
            # 同轴另一个轮（fl↔fr、rl↔rr）脚下已经高出一级：它先上去了，本轮碰到立面后再翻（翻越跟踪不受影响）
            ground = ctx["ground"].detach()
            partner = ground[..., [RIGHT_WHEELS[0], RIGHT_WHEELS[1], LEFT_WHEELS[0], LEFT_WHEELS[1]]]
            eligible = eligible & ~((partner - ground) >= self.cfg.min_step)
        return eligible

    def _ready_stroke(self, ctx) -> torch.Tensor:
        """立面前 ready_zone_m 内仍着地的轮：摆臂剩余上抬行程不足"立面高度 + 1 cm"（以最大行程为上限）的代价。"""
        lead = ctx["lead"].nan_to_num(posinf=1.0)
        zone = (lead >= 0.0) & (lead <= self.cfg.ready_zone_m) & (ctx["lifted"].detach() < 0.5)
        need = torch.minimum(ctx["riser_rise"] + 0.01, (self.max_stroke.to(lead) - 0.01).clamp_min(0.0))
        deficit = (need - ctx["stroke"]).clamp_min(0.0) / 0.03
        return (_huber(deficit, 1.0) * zone).sum(-1)

    def _prelift_tracking(self, ctx) -> torch.Tensor:
        """把轮底净空拉到越过立面圆周包络所需的高度（≤0）。

        接触前（0 < lead ≤ prelift_window）对单台阶所有轮和任何立面前的前轴预抬；接触后（−r < lead ≤ 0）对所有立面跟踪翻越。
        每根轴一次只要求一个轮：同一轴两轮都在跟踪（窗口内或翻越中）时取两轮误差的较小者——一个轮先抬/先翻，
        另一个轮等它上去后单独面对立面再翻（整宽立面高于轮半径时两前轮无法同时被抬起：前部重量没有支撑）。
        误差 = 实际净空与支撑净空（其余三轮着地假设下的运动学净空，见 _compute_context）两者误差的平均：
        支撑净空给出不经世界模型车身预测的抬轮梯度，实际净空保证真的离地。接触前的轮见 _prelift_eligible。"""
        error, active, _ = self._prelift_errors(ctx)
        left, right = list(LEFT_WHEELS), list(RIGHT_WHEELS)
        pair = active[..., left] & active[..., right]
        pair_climb = self._stairs_pair_climb(ctx)
        if pair_climb is not None:
            pair = pair & ~pair_climb[..., left]
        per_axle = torch.where(pair, torch.minimum(error[..., left], error[..., right]),
                               error[..., left] + error[..., right])
        if self.cfg.stairs_climb_trail_weight < 1.0:
            both = self._pair_climb_mask(ctx)[..., left]
            k = self.cfg.stairs_climb_trail_weight
            with torch.no_grad():
                clr = ctx["clearance"]
                lead_left = torch.sigmoid((clr[..., left] - clr[..., right]) / LEADER_TEMPERATURE)
            weighted = ((lead_left + (1.0 - lead_left) * k) * error[..., left]
                        + ((1.0 - lead_left) + lead_left * k) * error[..., right])
            per_axle = torch.where(both, weighted, per_axle)
        return -per_axle.sum(-1)

    def _stairs_alt_lift(self, ctx) -> torch.Tensor:
        """[...] 楼梯区同一轴两轮都在翻越时两轮等高的程度（每轴 0–1 之和）：1 − |Δ净空| / (frac × 立面高)，下限 0。"""
        left, right = list(LEFT_WHEELS), list(RIGHT_WHEELS)
        both = self._pair_climb_mask(ctx)[..., left]
        clr = ctx["clearance"]
        diff = clr[..., left] - clr[..., right]
        gap = (diff.square() + ALT_LIFT_EPS**2).sqrt() - ALT_LIFT_EPS
        with torch.no_grad():
            rise = torch.maximum(ctx["riser_rise"][..., left], ctx["riser_rise"][..., right]).nan_to_num(0.0)
            scale = (self.cfg.stairs_alt_lift_frac * rise).clamp_min(0.01)
        return ((1.0 - gap / scale).clamp_min(0.0) * both).sum(-1)

    def _prelift_errors(self, ctx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(每轮净空跟踪误差 [..., 4]（未激活为 0）, 激活掩码 [..., 4], lead [..., 4])。

        激活 = 接触前预抬窗口内（见 _prelift_eligible）或接触后翻越中；目标 = 越过立面圆周包络所需的轮底净空。
        误差 = trust·平均(实际, 支撑) + (1 − trust)·max(实际, 支撑)。单台阶接触前 trust = smoothstep(裕度 / prelift_margin_m)
        （不求导）：质心不在其余三轮三角形内时，支撑净空假设的"其余三轮着地、车身不动"不成立，收臂只会让车身绕对角线
        倾倒，必须实际净空也达标才算；楼梯前轴接触前与翻越中（立面角托住）trust = 1，与第十一轮相同。"""
        cfg, radius = self.cfg, WHEEL_RADIUS
        lead = ctx["lead"].nan_to_num(posinf=1.0)
        prelift = self._prelift_eligible(ctx) & (lead > 0.0) & (lead <= cfg.prelift_window)
        climb = self._climb_window(ctx)
        phase = _smoothstep((cfg.prelift_window - lead) / 0.03)
        height = ctx["riser_rise"].clamp(cfg.min_step, 0.25)
        rim = (radius**2 - (lead + radius).clamp(0.0, radius).square()).clamp_min(0.0).sqrt()
        target = phase * (height - radius + rim + 0.005).clamp_min(0.03)

        def miss(lift):
            return (_huber((target - lift).clamp_min(0.0) / 0.03, 0.5)
                    + 0.25 * _huber((lift - target - 0.015).clamp_min(0.0) / 0.03, 0.5))

        active = prelift | climb
        actual, support = miss(ctx["clearance"]), miss(ctx["support_clearance"])
        with torch.no_grad():
            trust = torch.where(prelift & ctx["single"], _smoothstep(ctx["support_margin"] / cfg.prelift_margin_m), 1.0)
        error = trust * 0.5 * (actual + support) + (1.0 - trust) * torch.maximum(actual, support)
        pair_climb = self._stairs_pair_climb(ctx)
        if pair_climb is not None:
            error = torch.where(pair_climb, actual, error)
        return error * active, active, lead

    def _stairs_pair_climb(self, ctx) -> torch.Tensor | None:
        """[..., 4] 楼梯区同一轴两轮都在翻越（−r < lead ≤ 0）的轮；stairs_climb_pair_sum 关闭时为 None。"""
        if self.cfg.stairs_climb_pair_sum <= 0.5:
            return None
        return self._pair_climb_mask(ctx)

    def _climb_window(self, ctx) -> torch.Tensor:
        """[..., 4] 翻越中的轮：−r < lead ≤ 0；stairs_contact_climb_m > 0 时楼梯立面前上限放宽到该值。"""
        lead = ctx["lead"].nan_to_num(posinf=1.0)
        upper = lead.new_zeros(())
        if self.cfg.stairs_contact_climb_m > 0.0:
            stairs_riser = ctx["stairs_zone"][..., None] & ~ctx["single"]
            upper = torch.where(stairs_riser, lead.new_tensor(self.cfg.stairs_contact_climb_m), upper)
        return (lead > -WHEEL_RADIUS + 0.002) & (lead <= upper)

    def _pair_climb_mask(self, ctx) -> torch.Tensor:
        """[..., 4] 楼梯区同一轴两轮都在翻越（见 _climb_window）的轮（不看开关）。"""
        climb = self._climb_window(ctx)
        left, right = list(LEFT_WHEELS), list(RIGHT_WHEELS)
        both = climb[..., left] & climb[..., right] & ctx["stairs_zone"][..., None]
        return torch.cat((both, both), -1)

    def _prelift_front(self, ctx) -> torch.Tensor:
        """单台阶前轴领先轮的净空跟踪（≤0）：两前轮都在跟踪时只算 lead 较小（先到立面）的那个轮，否则算在跟踪的那个。

        与 _prelift_tracking 的"取误差较小者"不同：那里误差较小的多半是离得远、目标还没升起的轮，先到的轮撞上立面
        几乎没有代价（第十轮 μ=1.0 下单台阶预抬事件 0%）。同轴另一个轮仍可接触后滚上去（验收只要求一个前轮预抬）。"""
        error, active, lead = self._prelift_errors(ctx)
        fl, fr = LEFT_WHEELS[0], RIGHT_WHEELS[0]
        on = active & ctx["single"]
        both = on[..., fl] & on[..., fr]
        leader = torch.where(lead[..., fl] <= lead[..., fr], error[..., fl], error[..., fr])
        alone = error[..., fl] * on[..., fl] + error[..., fr] * on[..., fr]
        return -torch.where(both, leader, alone)

    def reward(self, state, next_state, command, action, last_action, acc_sq=None) -> torch.Tensor:
        terms = self.reward_terms(state, next_state, command, action, last_action)
        return torch.stack(list(terms.values()), dim=-1).sum(-1) * CONTROL_DT

    def survival(self, state) -> torch.Tensor:
        cos_tilt = -projected_gravity(state[..., QUAT])[..., 2]
        return torch.sigmoid((cos_tilt - math.cos(self.cfg.tilt_limit))
                             / (self.cfg.tilt_temperature * math.sin(self.cfg.tilt_limit)))
