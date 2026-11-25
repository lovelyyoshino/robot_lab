#!/usr/bin/env bash
# SuperMimic 越障训练：从零（无 PPO Teacher / 无监督初始化 / 无锚定），nohup + setsid 后台运行。
# 用法：bash scripts/supermimic/run_train.sh [run_name] [num_envs] [iterations] [其他 train_terrain 参数…]
# 日志：logs/supermimic_terrain/<run_name>.log 与 .pid；每 1000 轮存 ckpt 与 exported/policy.pt。
# 续训：RESUME=1 bash scripts/supermimic/run_train.sh <run_name> [num_envs] [iterations]（同名 run 的最新存档，日志追加）
# 热启动：FROM_RUN=<旧 run> FROM_CKPT=ckpt_XXXXXX.pt bash scripts/supermimic/run_train.sh <新 run> ...
#   （learner 全部状态、课程等级与轮数从旧存档接着走，日志和存档写到新 run；iterations 是总轮数，含旧存档已跑的）
# @author pony
# @date 2026-11-05
# @version v2.21.0
# @last_modified 2026-11-23
# @changelog
#   - v2.21.0 (2026-11-23): 第三十轮（从零，种子 42）= 第二十九轮 − 短踏面列 + 翻车代价（第二十五轮的值）。第二十九轮第 8000 轮
#     验收 ≥12 cm 楼梯全部 0%（第二十八轮同期 12×40 86%、16×40 55%、17.5×40 42%），训练里楼梯回合 97% 以翻车收尾，
#     楼梯课程一直停在 8 cm 下限、降级率 0.31–0.41，想象存活率 0.90–0.94（第二十八轮 0.9998）。训练环境探针（r29@8k，
#     1500 步 × 512 环境）：普通楼梯 952 回合 945 翻车（8 cm 一档 375/382），不加探索噪声同样，全部在第一级立面前车头
#     朝上后翻（x 中位 1.11 m、约 5 s、倾角到 46° 限位）；短踏面列 133/133 翻车；同一环境 r28@8k 楼梯 144/153 到终点、
#     0 翻车（短踏面列 26/27 横着开出车道）。原因：翻车与开出车道只记一次 −20×dt，之后价值为 0；卡住加码让贴着立面
#     每步多付到 −20；短踏面列（14–17.5 cm × 26–30 cm，按第十四轮静态可行性大多爬不上）从第 0 轮起每回合都卡住，
#     策略学成在立面前主动后翻，并带到所有楼梯上。想象回报对存活的导数 ≈ |terminated|·dt − γ|V|，立面前 V −7～−16：
#     −20 时为负（越接近翻倒越划算），−10000 时约 +190。
#     改动：--stairs_short_frac 0.05 → 0（去掉短踏面列；16/17.5×28 先单独核几何可行性）；--terminated −20 → −10000、
#     --tilt_temperature 0.05 → 0.02（第二十五轮的值；test_supermimic_r25：28° 以内每步 > −0.1，40° 时 < −40）。
#     保留第二十九轮 F 卡住加码（r29@8k 严格预抬单台阶 14–73%，第二十八轮同期 0–19%）。其余全部参数不变。
#   - v2.20.0 (2026-11-23): 第二十九轮（从零，种子 42）= 第二十八轮 + 卡住两项（r28_plan.md 4.3）。第二十八轮最好 it24000
#     （楼梯 12/16/17.5×40 严格 100/95/91%，12×28 91%，满宽单台阶 98/95/94%），2.6 万轮以后满宽 ≥14 cm 单台阶与 ≥16 cm
#     楼梯在 26k/32k/36k/38k 塌到 2–27%：90% 以上回合以 0.42 m/s 撞上立面后贴着不动 13–15 s、摆臂 0.25 rad/s（成功回合
#     0.7）、车头最多抬 5°，停滞超时；训练里 36k 停滞收尾占比单台阶 0.39、楼梯 0.42。好段与坏段的每步账：停住省下的越障
#     代价（预抬跟踪、预留行程、摆臂速度、楼梯轮替、碰立面速度）约 +1.0，多付的停滞与线性停滞约 −1.0，几乎打平，
#     课程到顶后策略在"爬"和"顶着不动"之间来回。16/17.5×28 所有轮次 0–5%，训练里从未出现（踏面规则 ≥15 cm → ≥36 cm）。
#     E 短踏面楼梯列 --stairs_short_frac 0.05 --stairs_short_riser 0.14 0.175 --stairs_short_tread 0.26 0.30
#       （楼梯 9 列里拿 1 列，常驻探索：随机等级、不改课程、不进楼梯课程统计，日志 curriculum/stairs_short_*、end/stairs_short_*）；
#     F 卡住加码 --stall_hold -20 --stall_hold_grace_s 1 --stall_hold_ramp_s 2 --stall_hold_gain_m 0.01（train_terrain
#       v0.24.0，只进真实回报 → critic：命令前进时本回合最远前进 x 1 s 没有再前进 1 cm 后，2 s 内线性加到 −20/步；按位置
#       计时，倒车、原地、前后蹭都不清零，退一步重来只在真的没前进时才计时）。
#     不变：第二十八轮 A–D、停滞 −5 / 线性停滞 −3、单台阶贴立面停滞 ×0.2、停滞 12 s 截断、其余全部参数。
#   - v2.19.0 (2026-11-21): 第二十八轮（从零，种子 42）= 第二十七轮 + 偏离只扣分、不判失败（方案见
#     logs/supermimic_terrain/obstacle_r27_scratch/report/r28_plan.md）。第二十七轮第 36000 轮验收的失败大多只是横偏：
#     把"爬上去、四轮着地、没翻、横偏 0.15–0.75 m"算成功，×40 楼梯和满宽单台阶 95–100%；第 38000/40000 轮 16/17.5×40
#     严格判 2–8%、软判 100%，到顶横偏中位 0.29–0.31 m。critic 探针（r27@36k）：单台阶、平地 46–48% 的回合因到终点
#     |y| > 0.15 m 被真终止，楼梯爬升段 critic 给横偏的估值反而更高（偏 0.40 m 比居中高 +5.0）。
#     A 到终点一律成功截断 --curriculum_goal_lane_termination 0，出车道只剩开出台面 --curriculum_fail_lane_m 0.40→0.75
#       （与验收的 0.75 一致；0.40 是 v1.1.0 与车道代价同时加的）；
#     B 晋级横偏门单台阶/平地 --curriculum_success_lane_m 0.15→0.40，楼梯 --stairs_success_lane_schedule 0 0
#       （从第 0 轮起 = success_lane_m 0.40；fail_lane 改成 0.75 后原 90000 100000 会让楼梯门变成 0.75）；
#     C 车轮行进方向对准路线 --course -20（preview_task v0.24.0，−20·sin²(ψ + δ̄)，δ̄ = 四轮转向平均；航向 10° 不蟹行
#       ≈ −0.6/步，与 y = 0.15 m 的车道代价相当；已有轨迹楼梯段 r22@20k −0.18、r27@36k −0.69、r26@34k −1.14/步）；
#       同版本修正平地滚动 / 堵转空转"轮心沿轮向速度"的符号（转向角正 = 轮头朝左，原式按朝右写反）；
#     D 日志按地形记回合结局占比、到终点 |y|、行进方向误差（train_terrain v0.23.0）。
#     不变：车道代价 −30·y²、楼梯区 2000→8000 轮 ×0→×1、critic 车道/路线特征、停滞 12 s 截断、其余全部参数。
#   - v2.18.0 (2026-11-20): 第二十七轮（从零，种子 42）= 第二十六轮 + 楼梯区车道代价早开 --stairs_lane_schedule 2000 8000，
#     晋级横偏门不收紧 --stairs_success_lane_schedule 90000 100000（只给前者时晋级门会跟着收紧）。第二十六轮第 34000 轮
#     16/17.5×40 楼梯 47/64、25/64 回合登顶但 |y| 0.20–0.43 m（验收门 0.15），另 14/37 回合前轴顶最后一级、后轴在第二级
#     时停住。16×40 轨迹：车头被左右轮替拧到 ±10–13°，第二十二轮第 20000 轮反向打转向约 16° 蟹行、横偏 ≤0.15 m，
#     第二十六轮转向 ≈0、斜着开，每上一级横偏约 +0.1 m。第十八到二十六轮楼梯区车道代价与额外横向速度倍数整轮 ×0
#     （日志 stairs_lane_scale 一直 0.0），横偏只靠碰巧学会的反打转向。第二十三轮（斜行 0°、代价同样 2000→8000）
#     第 16000 轮横偏 >15 cm 只有 11%；它的晋级门 16000 起收紧导致楼梯降级，这里不收紧。斜行保持 6°。
#   - v2.17.0 (2026-11-19): 第二十六轮（从零，种子 42）= 第二十二轮 + 登顶后回正窗口 --post_climb_relief 1.0
#     --post_climb_window_m 1.0（preview_task v0.23.0）。第二十二轮 10k–32k 的 40 cm 踏面楼梯到顶率 98–100%、停住 0–1%，
#     失败全部是登顶横偏 |y| 0.19–0.30 m（验收门 0.15）。第 22000 轮 16×40 轨迹：楼梯上左右轮先后上台阶把车头扭到
#     ±9–12°（第二十三轮斜行 0° 时同样 6–12°，斜行目标不是来源），横偏主要在最后两级和平台上累计——平台上航向还有
#     6–10°、1–2 m 才回到 5°。轨迹代回奖励：平台上平地滚动四项（wheel_sync / wheel_slip / toe / flat_steer）每步
#     −7～−11，航向 + 车道只有 −1.5～−3，回正比留着横偏贵。后轴刚翻上去的 1 m 内免除这四项；平地直行身后没有立面，
#     不受影响。第二十三轮的斜行 0° + 楼梯车道代价早开把第一级停留从 1.5 s 拉到 3–4 s、停住 10–93%，不采用。
#   - v2.16.0 (2026-11-19): 回退到第二十二轮配置（用户要求：先回到最好的一版再往上加）。同一验收（accept.py 11-11 起未改，
#     环境/地形文件 sha 相同）下各轮最好存档的楼梯 12/28、12/40、16/40、17.5/40：第二十二轮第 20000 轮 100/98/84/86%，
#     第二十三到二十五轮最好只有 0/100/80/47、20/95/59/42、78/84/25/6。撤回第二十三到二十五轮的全部参数改动：楼梯斜行 6°、
#     楼梯车道代价 ×0 且晋级横偏门一直 0.40 m（--stairs_lane_schedule 90000 100000）、--stall -5 --stall_linear -3
#     --stall_linear_cap 1（=原行为）、--progress 1、--prelift_front 25、--terminated -20 --tilt_temperature 0.05（默认值，
#     显式写出）、去掉 --leg_action_bound。之后各轮新增的代码开关默认关闭，训练日志与第二十二轮逐位核对。
#   - v2.15.0 (2026-11-19): 第二十五轮（从零，种子 42）= 第二十四轮 + 翻车代价。第二十四轮第 14000 轮楼梯 12/16/17.5×40 =
#     95/59/42%，16000 轮起 ≥12 cm 楼梯几乎全部翻车（第一级 3.7 s 内后翻；前轮上第二级时前摆臂反向下压把车头顶起），
#     想象存活率 0.999→0.929、真实终止率 ×4。12 cm 楼梯轨迹按第二十四轮奖励重算（γ=0.99，未乘 dt，从 x=0.45 起）：
#     翻车 −24 > 爬上去 −241 > 原地停住 −2049（翻车只扣一次 −20，等于驻车一步）；第二十三轮奖励下抖动驻车最优、翻车第二。
#     改动：--terminated −20 → −10000（翻车发生在决策后约 2.8 s，折扣约 ×0.25，须 ≲ −9000 才比停住差）、
#     --tilt_temperature 0.05 → 0.02（成功爬楼最大倾角 28°；温度 0.05 时 −5000 的翻车代价会给成功爬楼额外记 −220，
#     0.02 时为 0）。重算：12 cm 楼梯 爬 −240 > 停住 −2049 > 翻车 −2498；16 cm 楼梯 爬 −656 > 停住 −2655 > 抖动 −2763；
#     17.5 cm 单台阶 爬 −595 > 抖动 −1463 > 停住 −2460。其余与第二十四轮相同。
#   - v2.14.0 (2026-11-18): 第二十四轮（从零，种子 42）。第二十三轮第 16000 轮楼梯 12/16/17.5×40 = 100/80/47%（历轮最好，
#     横偏 >15 cm 只有 11%），18000 轮起单台阶整面与楼梯全部驻车：前轴已上一级、后轴离立面 0.3–0.45 m，前蹭约 0.10 m/s
#     3 s 再快速倒回，净位移 0。把两条轨迹代回奖励（γ=0.99 的折扣回报，4 回合均值）：
#       单台阶 17.5：从前轮到立面前起 翻越 −2917 / 停住 −2271；后轴决策点 翻越 −610 / 抖动驻车 −298
#       楼梯 16×40：后轴决策点 翻越 −1545 / 抖动驻车 −657 / 静止 −1306
#     即按原目标"不爬"本来就更优。改动（同一批轨迹重算后翻越在两个决策点都优于停住/抖动/静止）：
#       --stall 0 --stall_linear -20 --stall_linear_cap 5（停滞只留线性项并放开到倒车：对速度凹，来回抖不再便宜）
#       --progress 1→12（世界系前进，线性、有正负，抖动净得 0）、--prelift_front 25→10（前轮没预抬的 0.6 s 原来 −53～−104/步）
#       重算：单台阶 前 −1187 / −3472，后 −82 / −1424 / −1740；楼梯 前 −1606 / −3563，后 −2156 / −2662 / −2506。
#     --leg_action_bound（摆臂动作上界 2π → ≈2.72：驻车时后摆臂动作 −4.4～−4.7 在关节限位之外，梯度为 0）。
#     回退 --stairs_success_lane_schedule（16000 轮起收紧楼梯晋级横偏门，楼梯降级率 0.26–0.48、课程 13.8→11.2 cm）；
#     保留楼梯不斜行与车道代价 2000→8000 轮打开。
#   - v2.13.0 (2026-11-17): 第二十三轮（从零，种子 42）= 第二十二轮 + 楼梯不再斜行、楼梯车道分两段收紧。第二十二轮第 32000 轮
#     16×40 楼梯 8 回合轨迹：前轴一侧先上第一级时车头被拧到 +10～13°，之后整段楼梯保持 +8～9°（斜行目标 6°），蟹行只抵消
#     一部分，登顶时横偏 17–27 cm > 验收门 15 cm；40 cm 踏面楼梯到顶率 98–100%，失败几乎都是横偏。轮替已由奖励保证，
#     斜行只剩副作用：--stairs_slant_deg 6→0。车道：--stairs_lane_schedule 2000 8000（楼梯区车道代价与越障横向速度额外
#     倍数 2000→8000 轮 ×0→×1），--stairs_success_lane_schedule 16000 26000（train_terrain v0.19.0：楼梯课程晋级横偏门
#     16000→26000 轮 0.40→0.15 m，楼梯高度先上到 ≈13 cm 再收紧；第十八轮 22000–32000 轮两者一起收紧时楼梯顶层比例 0.33→0.006）。
#   - v2.12.0 (2026-11-16): 第二十二轮（从零）。去掉 --lead_side_obs（不加观测维度），改用奖励做楼梯左右轮替
#     （preview_task v0.20.0）：--stairs_climb_trail_weight 0.25（同一轴两轮都在翻越时领先轮全权重、落后轮 ×0.25；原来
#     --stairs_climb_pair_sum 两轮误差相加，凸 Huber 让两轮平分抬升量最划算，等于要求左右同时上）、--stairs_alt_lift -6
#     （两轮都贴立面且净空差 < 半个立面高时计代价）、--stairs_contact_climb_m 0.08（楼梯立面前 lead ≤ 8 cm 也算翻越中，
#     不计三轮支撑裕度：第二十轮第 30000 轮死锁姿态单抬后轮代价 −18～−36/步，顶死只要 −15.7/步）。
#     探针（同一姿态）：顶死 −31.4/步；单抬一个后臂 +0.1/+0.3/+0.5 rad → −29.1/−22.8/−18.6，一路变好；两后臂同抬每档都比
#     单抬差（+0.4：−21.5 对 −19.9）。窗口 5 cm 时单抬 +0.3 轮心退到 lead 0.052，掉出窗口跳回 −63，故取 8 cm。第二十一轮验收第 2000–14000 轮被 export_ckpt 导出错误污染（已修），
#     训练日志里楼梯课程 3000–14000 轮停在 ≈9.5 cm。种子回到 42，与第十八/二十轮同一种子对比。
#   - v2.11.0 (2026-11-16): 第二十一轮（从零）。--lead_side_obs：actor 多一维每回合随机的先抬侧标记（±1，镜像取反），导出时
#     固定 +1、部署仍 232 维（train_terrain v0.17.0）。依据：第二十轮第 30000 轮 16×40 楼梯 4/4 回合后轴顶第一级卡死，
#     左右动作几乎相同（|a_rl − a_rr| 0.2）；第 24000 轮靠左右不对称一次上一个后轮（|a_rl − a_rr| 1.0–1.5）全部登顶；
#     镜像一致性（--mirror_loss_coef 1.0）在近似对称状态上把动作推向对称。第二十轮不收紧车道照样退步，车道不是原因。
#     种子 42 → 21：训练是确定性的，第二十轮前 22000 轮与第十八轮逐位相同；换种子才是独立样本。其余与 v2.10.0 相同。
#   - v2.10.0 (2026-11-14): 第二十轮（从零，用户要求不热启动：热启动不代表真实水平）。楼梯课程下限 8 cm 写进脚本
#     --curriculum_stairs_min_level 2（第十五到十八轮都在启动命令里另加，第十九轮漏掉，楼梯课程从 4.8 cm 起、停在 8.1–8.4 cm）。
#     楼梯车道不再收紧 --stairs_lane_schedule 90000 100000（等于整轮放松：成功车道 0.40 m、楼梯区车道代价 ×0）：第十八轮
#     平均横偏一直 ≈0.08 m，22000→32000 轮成功车道收到 0.15 m 后楼梯课程顶级比例 0.33→0.006、16/17.5×40 在存档间
#     100↔5% 来回、12×28 归零；actor 看不到横偏，收紧车道只会把偏 0.15 m 以上的回合判失败。
#   - v2.9.0 (2026-11-13): 第十八轮（从零）。第十七轮 it12000 楼梯 12 cm×40 cm 98% 爬上顶，但斜 9–11°
#     没蟹行、横偏 0.21–0.32 m，验收 0%；训练里到终点 |y|>0.15 同样判出车道终止，楼梯课程 it10000 起停在 8.2–8.6 cm，
#     16 cm 以上没练过。先爬上去、后期再加车道：--stairs_lane_schedule 22000 32000（22000 轮前楼梯成功车道 0.40 m、
#     楼梯区车道项与越障横向速度额外倍数 ×0，22000→32000 线性收紧回 0.15 m / ×1；单台阶与平地不变）；
#     楼梯斜行 --stairs_slant_deg 12→6（斜 10° 不蟹行时每米横偏约 0.18 m，5 级楼梯会碰到出车道 0.40 m）。
#   - v2.8.0 (2026-11-13): 第十七轮（从零）。第十六轮 it18000 回放左侧 12 cm 箱：左后轮着地顶立面 13 s（≈50 N、空转
#     2.5 rad/s、车速 ≈0）。贴立面停滞 ×0.2 只看水平位置，后摆臂一抬轮心后移出窗口、折扣消失：顶着不动每步 −10.5，
#     原地抬轮 −12.9。--stall_climb_min_clearance 0.01 --stall_climb_lead_m 0.05（折扣只给立面前 5 cm 内真离地 >1 cm
#     的轮，对净空可导）、--riser_spin -10（贴立面着地轮空转代价）；楼梯课程迟滞 3/3→2/3（第十六轮楼梯课程 it20000
#     才到 10.8 cm，第十五轮 1/1 时 it11000 已 13 cm）。
#   - v2.7.0 (2026-11-12): 第十六轮（从零）。第十五轮 it26000 楼梯死锁：左前轮搭在第二级台阶角、右前轮顶立面、四轮停转直到
#     超时（原地不动每步 −5.9，抬右前臂 −8.6～−9.2）。--stairs_climb_pair_sum 1（楼梯同轴两轮都在翻越时误差相加、只用实际净空）、
#     --stall_linear -3（停滞门槛以下的速度差线性代价，停住时仍有起步梯度）；楼梯课程迟滞 1/1→3/3（第十五轮楼梯课程
#     it14000 升到 13.5 cm 后被降回 9–10 cm，12 cm 楼梯 98%→0–3%→83% 来回）。
#   - v2.6.0 (2026-11-11): 第十五轮（方案 A，减少越障侧偏）：整宽单台阶斜行 --single_slant_deg 12→6、越障区横向速度代价 ×3
#     --obstacle_lateral_vel_scale 3。第十四轮全宽单台阶失败几乎都是翻过后横偏 0.15–0.53 m（"冲出"），偏向随先抬的前轮，
#     后轴爬块阶段侧滑 v_y≈0.1 m/s；actor 看不到横偏，下块后不会修回。
#   - v2.5.0 (2026-11-11): 第十五轮改为从零训练。新增 --trailing_contact_climb 1：同轴另一个轮已上更高一级时本轮
#     不要求接触前预抬，碰到立面后再翻（第十四轮 it40000 右前轮停在第二级立面前 4 cm 直到超时）。楼梯课程下限
#     8 cm 见启动命令（--curriculum_stairs_min_level 2）。
#   - v2.4.0 (2026-11-11): 第十五轮楼梯补练（从第十四轮 it40000 热启动，见 docs/supermimic.md）。第十四轮 12–17.5 cm 楼梯
#     全部"停住超时"：楼梯课程在迟滞 2/2 下从 it6000 起一直停在 8.0–8.6 cm；贴立面停滞 ×0.2、轻碰 −400 / 15 cm 让
#     "顶着第一级不动"几乎不扣分、又压掉了上楼梯的冲劲。楼梯区改为：停滞不放松 --stairs_stall_climb_scale 1.0、
#     轻碰 ×0.5 --stairs_riser_contact_scale 0.5（=−200）、接近距离 --stairs_riser_approach_m 0.05（回到第十三轮）；
#     楼梯课程迟滞 1/1（--curriculum_stairs_promote_streak 1 --curriculum_stairs_demote_streak 1）。单台阶不变。
#     补练另加 --curriculum_freeze_non_stairs 1 --curriculum_stairs_min_level 4（见启动命令）。
#   - v2.3.0 (2026-11-10): 第十四轮（从第十三轮 it20000 热启动，见 docs/supermimic.md）：--stall_climb_scale 0.2、
#     --drop_relief 0.8、轻碰 --riser_contact_speed −400 / --riser_approach_m 0.15、--prelift_front 25、课程迟滞 2/2、
#     --stairs_tread_rule 0.12 0.15 0.36、--actor_ema_decay 0.999。热启动另加 --warm_start_reset --critic_warmup_iterations
#     与较小的 --actor_lr（见启动命令）。
#   - v2.2.0 (2026-11-09): 第十三轮。第十二轮（与第十轮相同）楼梯课程升到 9–11 cm 时停滞比例 ×10、世界模型末步误差 ×2、
#     actor 损失 ×5，平地/单台阶一起退化：楼梯降级比例 --curriculum_stairs_demote_ratio 0.5→0.8（卡在楼梯中段的回合降级），
#     连续停滞 12 s 截断回合 --curriculum_stall_truncate_s 12；平地静止车头低 6–8°（前臂收到 −28°）是 actor 输出饱和
#     （前臂动作 −6.27，界 ±6.28，关节早顶在限位上，平地中立代价的梯度≈0）：--action_saturation_coef 0.02
#     --action_saturation_margin 0.10（腿关节动作超出软限位外 0.1 rad 的 tanh 前代价）；左右偏好在存档间翻转（镜像一致性只占
#     actor 损失 <1%）：--mirror_loss_coef 0.1→1.0；平地想象起点保底 --imag_flat_fraction 0.35；先移质心只对单台阶生效（代码内，
#     楼梯前轴回到第十一轮口径）。
#   - v2.1.0 (2026-11-08): 第十二轮。平地 --toe −2→−20、--wheel_slip −5→−20，新增 --flat_steer −10、--wheel_sync −50
#     （第十一轮 0.30 m/s 横偏 P95 0.52 m：前轮内八约 20°、后轮外八约 13°、后轮快转 35%、转向 ±14° 以 0.57 Hz 摆动）；
#     该抬的轮从立面前 0.25 m 起要求质心裕度 ≥0.8 cm（--prelift_margin_m 0.008 --prelift_shift_zone 0.25），接触前的支撑净空
#     按裕度门控（代码内；第十一轮窗口内摆臂抬 4 cm、轮离地 0.2 mm——质心压在对角线上，收臂只让车身倾倒）。
#   - v2.0.0 (2026-11-08): 第十一轮。单台阶前轴领先轮额外净空跟踪 --prelift_front 15、整宽单台阶斜行 --single_slant_deg 12
#     （第十轮 μ=1.0 单台阶预抬事件 0%，撞上去也能爬过）；平地 --toe -2、--wheel_slip -5、零速 --stand_yaw_rate -5
#     （平地探针：后轮 ±43° 八字、前轮两倍转速打滑、零速前后轴对顶 → 横漂、零速自转、俯仰振动）；--action_rate −0.005→−0.02
#     （第 22000 轮策略本身约 6.7 Hz 抖动）。新增 FROM_RUN/FROM_CKPT 热启动。
#   - v1.9.0 (2026-11-07): 轮胎摩擦按用户决定固定 μ=1.0（terrain_env_cfg.WHEEL_FRICTION，训练与验收同时生效）；摩擦恒定，
#     撤掉 --contact_context（ξ 回到 25 维）。奖励/课程回到第七代 = 第六代 + 楼梯斜 12°、停滞 −5、偏航角速度 −2。
#   - v1.8.0 (2026-11-07): 撤掉楼梯探索回合（第八代：探索回合 ≥12 cm 通过率 6000 轮始终为 0，只产生失败/翻车数据；
#     同期单台阶课程 8.0 cm vs 第六代 13.8 cm、平地等级 3.3 vs 6.7，验收单台阶与平地航向全面退步）；保留 --contact_context。
#   - v1.7.1 (2026-11-07): 想象单步编译改回默认关闭：第八代续训第一次 actor 更新即出现非有限梯度、自动退回 eager，
#     合成基准的提速在真实台阶状态上用不上。
#   - v1.7.0 (2026-11-07): （已撤回，见 v1.7.1）默认编译想象单步（--compile_imagination；合成基准 learner.update 1784→767 ms，数值相对差 ~1e-7；
#     出现非有限 actor 梯度时 learner 自动永久退回 eager）；RESUME=1 从同名 run 的最新存档续训（日志追加）。
#   - v1.6.0 (2026-11-07): 世界模型/critic 的 ξ 加四轮摩擦（--contact_context，摩擦 A/B：8 cm 楼梯 μ=1.0 成功 98%、
#     μ=0.5 只有 3%）；楼梯探索回合 30%（--curriculum_stairs_explore_frac 0.3，前七轮楼梯课程一直在 6–8 cm）。
#   - v1.5.0 (2026-11-07): 楼梯区航向目标 |ψ|=12°（后轴两轮先后碰立面）、停滞 −2→−5、偏航角速度 −2；
#     下台阶落下的轮不计两轮离地/支撑裕度（代码内）。
#   - v1.4.0 (2026-11-07): 航向 Huber −5、v_y −5→−15、平地中立 −1→−5（平地横偏/俯仰）；前轴在楼梯每级也预抬（代码内）。
#   - v1.3.0 (2026-11-06): 支撑裕度门槛 1.5→0.5 cm 且计入预抬窗口内该抬的轮；翻越中的轮不计两轮离地/裕度；净空跟踪 3→5。
#   - v1.2.0 (2026-11-06): 预抬按物理边界重做：删除立面接近代价，加三轮支撑裕度/两轮离地/轻碰/准备行程/横滚超限；
#     预抬只在单台阶、每根轴一个轮先抬另一个滚上去（窗口与早抬 0.12 m），接触后翻越跟踪对所有立面；减速带 min(cmd, 0.12)。
#   - v1.1.0 (2026-11-05): 车道惩罚 −30·y²、v_y Huber；成功/晋级 |y|≤0.15 m、出车道 0.40 m；楼梯无碰撞晋级门、
#     回合 40 s；预抬目标（命令速度项、跟踪 3.0、窗口/早抬 0.14 m）；θ_ref 扣 6°；命令 0.10–0.40 m/s + 10% 静止。
#   - v1.0.0 (2026-11-05): 初始创建；越障目标、地形课程与优化参数全部显式写出。
set -e
RUN=${1:-obstacle_$(date +%Y%m%d_%H%M)}
ENVS=${2:-2048}
ITERS=${3:-40000}
shift $(( $# < 3 ? $# : 3 ))
cd "$(dirname "$0")/../.."
RESUME_ARGS=()
if [[ "${RESUME:-0}" == 1 ]]; then
  if ! ls "logs/supermimic_terrain/$RUN"/ckpt_*.pt > /dev/null 2>&1; then
    echo "续训找不到存档：logs/supermimic_terrain/$RUN/ckpt_*.pt" >&2
    exit 1
  fi
  if [[ -f "logs/supermimic_terrain/$RUN.pid" ]] && kill -0 "$(cat "logs/supermimic_terrain/$RUN.pid")" 2>/dev/null; then
    echo "run 仍在训练（pid $(cat "logs/supermimic_terrain/$RUN.pid")），先停掉再续训" >&2
    exit 1
  fi
  RESUME_ARGS=(--resume --load_run "$RUN")
elif [[ -e "logs/supermimic_terrain/$RUN.log" || -e "logs/supermimic_terrain/$RUN" ]]; then
  echo "run已存在，拒绝覆盖：$RUN" >&2
  exit 1
elif [[ -n "${FROM_RUN:-}" ]]; then
  if [[ ! -f "logs/supermimic_terrain/$FROM_RUN/${FROM_CKPT:?热启动需要 FROM_CKPT}" ]]; then
    echo "热启动找不到存档：logs/supermimic_terrain/$FROM_RUN/$FROM_CKPT" >&2
    exit 1
  fi
  RESUME_ARGS=(--resume --load_run "$FROM_RUN" --checkpoint "$FROM_CKPT")
fi
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
unset PYTHONPATH
source /media/bigdisk/IsaacLab/_isaac_sim/setup_conda_env.sh >/dev/null
mkdir -p logs/supermimic_terrain

TRAIN_ARGS=(
  --task RobotLab-Isaac-SuperMimic-Obstacle-Parallelogram-Rover-v0 --init_terrain_level 0 --seed 42
  --steps_per_iter 2 --save_interval 1000 --log_interval 20 --check_obs_interval 500 --map_res 0.02
  # 1 速度跟踪 / 前进 / 停滞（立面前 0.30 m 内允许减速到 min(cmd, 0.12 m/s)，为轻碰留出余量；停滞须重于越障途中的代价）
  --track_lin_vel 5.0 --track_lin_vel_std 0.5 --slow_band_m 0.30 --slow_band_speed 0.12
  --progress 1.0 --stall -5.0 --stall_speed_ratio 0.25
  #   第十四轮：贴立面翻越时停滞 ×0.2（轻碰后靠摆臂慢慢翻，不再被停滞逼着带速撞）；下行边沿减免落下必然产生的竖直速度/
  #   角速度/俯仰/摆臂速度/两轮离地/支撑裕度代价 80%（整宽单台阶后轴停在下块处直到停滞超时）
  --stall_climb_scale 0.2 --drop_relief 0.8
  #   第二十六轮：登顶后回正窗口（后轴刚翻上去的 1 m 内平地滚动四项免除，让 4WS 把爬楼扭出的航向转回来）
  --post_climb_relief 1.0 --post_climb_window_m 1.0
  #   第十五轮：楼梯区不放松停滞（每级都贴立面，放松后顶着第一级不动几乎不扣分）
  --stairs_stall_climb_scale 1.0
  #   第十六轮：低于停滞门槛的速度差另计线性代价（停滞 sigmoid 在车速 0 处饱和、没有梯度，停住后不会重新起步）
  --stall_linear -3.0 --stall_linear_cap 1.0
  #   第二十九轮：卡住加码（只进真实回报 → critic）：本回合最远前进 x 1 s 没再前进 1 cm 后 2 s 内线性加到 −20/步，
  #   按位置计时（倒车、原地、贴立面前后蹭都不清零）；第二十八轮后期满宽高台阶贴着立面不动 13–15 s
  --stall_hold -20.0 --stall_hold_grace_s 1.0 --stall_hold_ramp_s 2.0 --stall_hold_gain_m 0.01
  #   第十七轮：贴立面停滞折扣只给真离地（>1 cm）、立面前 5 cm 内的轮；贴立面着地轮空转另计代价（堵转）
  --stall_climb_min_clearance 0.01 --stall_climb_lead_m 0.05 --riser_spin -10.0 --riser_spin_lead_m 0.005
  #   第十八轮：楼梯车道约束先放松、后期收紧（22000 轮前楼梯成功车道 = 出车道 0.40 m、楼梯区车道代价 ×0，32000 轮收紧到 0.15 m）
  #   （--stairs_lane_scale 是不给 schedule 时的固定系数；给了 schedule 时每轮覆盖）
  #   第二十轮：不收紧（schedule 起点设在训练轮数之外）
  #   第二十七轮：楼梯区车道代价与越障横向速度额外倍数 2000→8000 轮 ×0→×1；晋级横偏门单独给 schedule 且不收紧（一直 0.40 m）
  #   第二十八轮：出车道改 0.75 m 后，楼梯晋级横偏门 = fail_lane + (success_lane − fail_lane)·s，s 从第 0 轮起取 1（0.40 m）
  --stairs_lane_scale 1.0 --stairs_lane_schedule 2000 8000 --stairs_success_lane_schedule 0 0
  # 2 翻车
  #   第三十轮：翻车（及任何真终止）代价 −20 → −10000、软存活温度 0.05 → 0.02（第二十五轮的值）。卡住加码让贴立面每步
  #   多付到 −20 以后，只记一次 −20 的翻车比停着便宜，第二十九轮学成在第一级立面前主动后翻
  --terminated -10000.0 --tilt_limit 0.8 --tilt_temperature 0.02
  # 3 姿态（θ_ref = sign(θ_geo)·max(|θ_geo| − 6°, 0)，θ_geo 为前后轮下地形的几何俯仰；横滚超过 4.5° 陡增）
  --roll -50.0 --roll_excess -2000.0 --roll_excess_deg 4.5
  --pitch -50.0 --pitch_tol_deg 3.0 --pitch_ref_window 0.15 --pitch_ref_deadband_deg 6.0
  # 4 平顺
  --ang_vel -2.0 --arm_vel -0.5 --action_rate -0.02 --lin_vel_z -0.5
  # 5 航向 / 直行 / 车道（lane 用训练特权横向偏移，只进奖励与 critic；actor 仍为 232 维）；
  #   楼梯区（任一轮前方 0.60 m 内有非单台阶立面）航向目标 |ψ|=stairs_slant_deg（第十八轮 12°→6°），4WS 蟹行沿路线前进，后轴两轮先后碰立面
  --heading -30.0 --heading_huber -5.0 --heading_huber_delta 0.02 --yaw_rate -2.0
  --stairs_slant_deg 6.0 --stairs_slant_zone_m 0.60 --lateral_vel -15.0 --lateral_vel_delta 0.02
  #   第十五轮：越障区（非平地）横向速度代价 ×3（actor 看不到横偏，侧滑多少就偏多少）
  --obstacle_lateral_vel_scale 3.0
  --lane -30.0 --steering -0.1
  #   第二十八轮：车轮行进方向对准路线 −20·sin²(ψ + δ̄)（δ̄ = 四轮转向平均，只约束同相/蟹行，不碰反相/转车头）
  --course -20.0
  #   整宽单台阶（同轴两轮都在单台阶区）也斜 12°：前轴两轮先后到立面，先到的轮去预抬（窄台阶不受影响）
  #   第十五轮 12→6：斜行横向分量减半（同轴两前轮到立面仍错开约 6 cm，另一轮可碰到后再翻）
  --single_slant_deg 6.0
  # 6 平地：中立 + 不对顶（同轴反向打角 toe、转向角平方、四轮同速、轮缘与轮心速度差 slip）+ 零速不自转
  --flat_neutral -5.0 --rest_clearance 0.005 --flat_lookahead 0.60
  --toe -20.0 --flat_steer -10.0 --wheel_sync -50.0 --wheel_sync_delta 0.05
  --wheel_slip -20.0 --wheel_slip_delta 0.05 --stand_yaw_rate -5.0 --stand_yaw_rate_delta 0.02
  # 7 越障：单台阶每根轴一个轮先抬（窗口对齐验收 3–12 cm）、另一个轮轻碰后滚上去 + 三轮支撑裕度；
  #   楼梯前轴每级一个轮先抬，后轴轻碰（接触前 ≤0.15 m/s）+ 接触后翻越跟踪（每轴一次一个轮）；立面前 0.30 m 预留摆臂上抬行程；
  #   支撑裕度计离地轮 + 预抬窗口内该抬的轮（先移质心再抬），翻越中的轮不算悬空
  --riser_min_height 0.03 --riser_contact_gap 0.005
  #   第十四轮：轻碰区 5→15 cm、权重 −200→−400（原来 0.4 m/s 撞上只罚约 6 步，所有档位接触速度 0.32–0.54 m/s）
  --riser_contact_speed -400.0 --riser_safe_speed 0.15 --riser_approach_m 0.15
  #   第十五轮：楼梯区轻碰回到第十三轮口径（×0.5 = −200、5 cm），12–16 cm 楼梯第十三轮靠 0.34–0.41 m/s 的冲劲翻上
  --stairs_riser_contact_scale 0.5 --stairs_riser_approach_m 0.05
  --prelift_track 5.0 --prelift_window 0.12 --early_lift -2.0 --early_lift_lead 0.12
  #   单台阶前轴领先轮（lead 较小者）额外净空跟踪：μ=1.0 下直接撞也能爬过，原跟踪取两轮误差较小者等于不罚先到的轮
  #   第十四轮 15→25：整宽单台阶两前轮在接触瞬间同时翻上（严格预抬 0%），验收要求一个前轮在立面前 3–12 cm 抬 ≥2 cm；
  #   静态可行性：只抬一个前轮 2 cm 横滚 0° 即可（车头抬约 7.5°），第二个前轮在第一个已上台阶时横滚 ≤8° 不可行，只能接触翻越
  --prelift_front 25.0
  #   第十五轮：同轴另一个轮已站上更高一级时，本轮不再要求接触前预抬（静态不可行、又被支撑裕度顶住），碰到立面后翻越
  --trailing_contact_climb 1.0
  #   第十六轮：楼梯同轴两轮都在翻越时两轮误差相加、只用实际净空（原来取较小者：一个轮搭上台阶角，另一个轮的误差被吃掉，
  #   抬它反而让先上的轮的支撑净空变差）
  --stairs_climb_pair_sum 1.0
  #   第二十二轮：楼梯左右轮替——同一轴两轮都在翻越时领先轮全权重、落后轮 ×0.25；两轮都贴立面（lead ≤ 8 cm）且净空差
  #   不足半个立面高时计代价（每轴最多 −6/步）。都对左右互换不变，与镜像一致性兼容，不加观测维度
  #   楼梯立面前 lead ≤ 8 cm 的轮也算翻越中：贴着立面抬后轮不再按悬空轮计三轮支撑裕度代价（车头上仰时 −18～−36/步，
  #   比顶死不动 −15.7/步还差），并跟踪翻越净空
  --stairs_climb_trail_weight 0.25 --stairs_alt_lift -6.0 --stairs_alt_lift_frac 0.5 --stairs_contact_climb_m 0.08
  #   三轮支撑：离地轮裕度 ≥0.5 cm；单台阶该抬的轮从立面前 0.25 m 起裕度 ≥0.8 cm（先移质心，约横滚 3° + 车头抬 3°），
  #   接触前支撑净空只在质心进了三角形时才算数；楼梯前轴只在预抬窗口内、门槛 0.5 cm（代码内）
  --support_margin -10.0 --support_margin_m 0.005 --prelift_margin_m 0.008 --prelift_shift_zone 0.25
  --multi_lift -5.0
  --ready_stroke -1.0 --ready_zone_m 0.30
  --min_step 0.03
  # 8 关节软限位（env 软限位系数 0.95）
  --joint_pos_limits -5.0
  # 课程（按地形类型）与终止：楼梯不设碰撞晋级门（比例 1.0）
  #   第二十八轮：到终点一律成功截断（不看横偏）；出车道只剩 |y| > 0.75 m（开出台面）；晋级要求 |y| ≤ 0.40 m
  --curriculum_promote_m 3.6 --curriculum_success_lane_m 0.40 --curriculum_fail_lane_m 0.75
  --curriculum_goal_lane_termination 0
  --curriculum_demote_ratio 0.5 --curriculum_posture_bad_frac 0.05
  --curriculum_collision_frac 0.10 --curriculum_stairs_collision_frac 1.0
  --curriculum_roll_limit_deg 10 --curriculum_pitch_dev_limit_deg 12
  # 楼梯没走到命令距离 80% 即降级（原 50%：卡在第 3–5 级的回合不升不降、反复卡满 40 s）；命令前进时连续停滞 12 s 截断回合
  --curriculum_stairs_demote_ratio 0.8 --curriculum_stall_truncate_s 12
  # 楼梯探索回合关闭（v1.8.0）：随机抽到的高立面在 μ 0.4–1.0 下几乎爬不上，30% 探索只喂失败数据，拖慢全部课程
  --curriculum_stairs_explore_frac 0.0
  # 第二十轮：楼梯课程下限 8 cm（第 2 级；原来只在启动命令里给）
  --curriculum_stairs_min_level 2
  # 第十四轮：课程迟滞（连续 2 次通过才升、连续 2 次失败才降），减少课程冲击；--curriculum_freeze 1 用于打磨阶段
  --curriculum_promote_streak 2 --curriculum_demote_streak 2 --curriculum_freeze 0
  #   第十五轮：楼梯不加迟滞（2/2 下楼梯升级率 0.02–0.12，课程停在 8–8.6 cm）；第十六轮 3/3：1/1 下楼梯课程升到 13.5 cm
  #   后被降回 9–10 cm，12 cm 楼梯在存档间 98%→0%→83%
  #   第十七轮 2/3：3/3 下楼梯升级率 0.03（"既不过也不挂"的回合也清零连胜），it20000 才到 10.8 cm
  --curriculum_stairs_promote_streak 2 --curriculum_stairs_demote_streak 3
  # 楼梯踏面下限随立面提高：立面 ≤12 cm 不变（0.26 m），≥15 cm 时 ≥0.36 m（16–17.5 cm × 踏面 ≤32 cm 前轮静态不可抬、
  # 前后轴同时顶立面，训练里必然失败）
  --stairs_tread_rule 0.12 0.15 0.36
  #   第二十九轮：另留一列短踏面楼梯（立面 14–17.5 cm × 踏面 0.26–0.30 m，楼梯 9 列里的 1 列），常驻探索：随机等级、
  #   不改课程、不进楼梯课程统计（验收 16/17.5×28 在训练里从未出现；r22@20k 在 16×28 上 5%，几何上不是完全不可行）
  #   第三十轮：去掉（这一列从第 0 轮起回合全部失败，第二十九轮在这里学会后翻；短踏面先单独核几何可行性）
  --stairs_short_frac 0.0
  # 命令与回合（只改本任务）：前进 0.10–0.40 m/s、10% 静止命令、回合 40 s
  --command_lin_vel_x 0.10 0.40 --command_standing_fraction 0.10 --episode_length_s 40
  # critic / 世界模型特权输入（actor 观测仍为 232 维）
  --critic_lane --critic_route_features
  # 轮胎摩擦固定 μ=1.0（静 1.2 / 动 1.0，见 terrain_env_cfg.WHEEL_FRICTION），摩擦恒定，ξ 不再追加摩擦（--contact_context 关）
  # 优化（scratch_40k_r8k + 上游式）
  --buffer_len 600 --horizon 32 --imag_batch 4096 --wm_window 8 --wm_batch 2048 --wm_updates 10 --wm_lr 3e-4
  --warmup_iterations 200 --actor_lr 1e-4 --actor_lr_final 1e-5 --actor_lr_decay_iterations "$(( ITERS > 200 ? ITERS - 200 : 1 ))"
  --critic_lr 3e-4
  --action_noise 0.1 --warmup_action_noise 0.3 --actor_update_every 1 --actor_value_grad
  --actor_value_grad_scale 1.0 --gamma 0.99 --lam 0.95 --priority_fraction 0.25
  --critic_real_returns --real_return_window 64 --real_return_batch 2048
  --compile_wm --no-compile_imagination --mirror_aug --mirror_loss_coef 1.0 --diagnostics_interval 20
  # 腿关节动作超出软限位外 0.1 rad 的 tanh 前原始输出计平方代价（防止输出顶在界上、梯度消失）；平地想象起点保底 35%
  --action_saturation_coef 0.02 --action_saturation_margin 0.10 --imag_flat_fraction 0.35
  # 第十四轮：actor 权重 EMA（时间常数约 1000 轮），存档与导出用 EMA（第十三轮楼梯 12 cm 存档间 92%→56%）
  --actor_ema_decay 0.999
)

# nohup只忽略SIGHUP；独立会话还需避开启动终端对原进程组的清理。
nohup setsid env PYTHONUNBUFFERED=1 python scripts/supermimic/train_terrain.py --headless \
  --num_envs "$ENVS" --max_iterations "$ITERS" --run_name "$RUN" "${TRAIN_ARGS[@]}" "${RESUME_ARGS[@]}" "$@" \
  < /dev/null >> "logs/supermimic_terrain/$RUN.log" 2>&1 &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "logs/supermimic_terrain/$RUN.pid"
echo "started $RUN pid=$TRAIN_PID log=logs/supermimic_terrain/$RUN.log"
