# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""直立任务的配置子包。导出 `agents` 供 `upright/__init__.py` 构造 entry point。"""

from . import agents

__all__ = ["agents"]
