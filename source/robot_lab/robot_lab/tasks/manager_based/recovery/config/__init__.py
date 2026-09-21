# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""恢复任务的配置子包。

导出 `agents` 供 `recovery/__init__.py` 构造 `rsl_rl_cfg_entry_point`。
"""

from . import agents

__all__ = ["agents"]
