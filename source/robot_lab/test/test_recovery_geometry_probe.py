"""倒置碰撞支点扫描的真实资产回归；不把运动学采样包装成恢复验收。

@author pony
@date 2026-09-30
@version v0.1.0
@last_modified 2026-09-30
@changelog
    - v0.1.0 (2026-09-30): 验证几何间隙、缺碰撞链和网格输入边界。
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
spec = importlib.util.spec_from_file_location(
    "recovery_geometry_probe", ROOT / "scripts/tools/probe_recovery_geometry.py",
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_current_inverted_geometry_has_no_sampled_moving_ground_contact():
    result = probe.scan_inverted_clearance(samples_per_joint=21)
    assert result["root_height_m"] == pytest.approx(.073)
    assert result["samples_per_branch"] == 441
    assert len(result["collision_records"]) == 8
    for record in result["collision_records"]:
        assert record["min_gap_m"] > .03
        assert not record["sampled_ground_reachable"]
        assert record["min_gap_m"] <= record["max_gap_m"]
    assert "fl_arm" in result["missing_collision_visual_links"]
    assert "fl_lower_arm" in result["missing_collision_visual_links"]
    assert not result["dynamic_reachability_proven"]
    assert not result["hardware_feasibility_verified"]


@pytest.mark.parametrize("samples", [0, 2, 402, True, 5.5])
def test_geometry_probe_rejects_invalid_grid(samples):
    with pytest.raises(ValueError, match="samples_per_joint"):
        probe.scan_inverted_clearance(samples_per_joint=samples)
