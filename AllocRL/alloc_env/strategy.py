"""
격자(Grid) + 지번(Lot) 기반 배치 전략.

C# BaseGridStrategy의 Python 재구현.
- 9앵커 지번 배치
- 1지번 1블록 규칙 (1/3 점유 기준)
- AABB 충돌 검사
- PlainWorkSpace 격자 스윕
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Tuple

from .block import Block, PrePlacedBlock, EPSILON
from .workspace import LotRegion, Workspace

# 1지번 1블록 점유 판정 기준
OCCUPATION_THRESHOLD = 1.0 / 3.0


class BaseGridStrategy:
    """격자 + 지번 기반 통합 배치 전략."""

    def __init__(self, step: float = 5.0):
        self.step = step if step > 0 else 5.0

    def determine_position(
        self, ws: Workspace, block: Block, env_date: date,
    ) -> Optional[Tuple[float, float]]:
        """배치 좌표를 결정합니다."""
        if ws.has_lots:
            return self._determine_in_lot_workspace(ws, block, env_date)
        return self._determine_in_plain_workspace(ws, block, env_date)

    # ── 지번 기반 배치 ────────────────────────────────────────────

    def _determine_in_lot_workspace(
        self, ws: Workspace, block: Block, env_date: date,
    ) -> Optional[Tuple[float, float]]:
        # 1. 탐색 대상 지번 + 가중치 수집
        lot_weights = self._build_lot_weight_list(ws)
        if not lot_weights:
            return None

        # 2. 가중치 내림차순 정렬
        lot_weights.sort(key=lambda lw: lw[1], reverse=True)

        # 3. 충돌 검사용 기배치 블록
        pre_placements = ws.get_overlapping_pre_placements(
            block.in_date, block.out_date)

        # 4. 1지번 1블록: 점유 지번 세트
        occupied = self._build_occupied_lot_set(
            ws, block.in_date, block.out_date)

        ws_left   = ws.origin_x
        ws_bottom = ws.origin_y
        ws_right  = ws.origin_x + ws.length
        ws_top    = ws.origin_y + ws.breadth

        half_l = block.length  / 2.0
        half_b = block.breadth / 2.0

        # 5. 각 지번에서 9앵커 배치 시도
        for lot, _ in lot_weights:
            if lot.lot_id in occupied:
                continue

            anchors = _generate_lot_anchors(lot, half_l, half_b)
            for cx, cy in anchors:
                # 작업장 범위 이탈 검사
                if cx - half_l < ws_left   - EPSILON:
                    continue
                if cy - half_b < ws_bottom - EPSILON:
                    continue
                if cx + half_l > ws_right  + EPSILON:
                    continue
                if cy + half_b > ws_top    + EPSILON:
                    continue

                # 충돌 검사
                if not _has_collision(ws, block, cx, cy, pre_placements):
                    return (cx, cy)

        return None

    # ── PlainWorkSpace 격자 스윕 ──────────────────────────────────

    def _determine_in_plain_workspace(
        self, ws: Workspace, block: Block, env_date: date,
    ) -> Optional[Tuple[float, float]]:
        ws_left   = ws.origin_x
        ws_bottom = ws.origin_y
        ws_right  = ws.origin_x + ws.length
        ws_top    = ws.origin_y + ws.breadth

        half_l = block.length  / 2.0
        half_b = block.breadth / 2.0

        pre_placements = ws.get_overlapping_pre_placements(
            block.in_date, block.out_date)

        y = ws_bottom + half_b
        while y + half_b <= ws_top:
            x = ws_left + half_l
            while x + half_l <= ws_right:
                if not _has_collision(ws, block, x, y, pre_placements):
                    return (x, y)
                x += self.step
            y += self.step

        return None

    # ── 지번 가중치 생성 ──────────────────────────────────────────

    def _build_lot_weight_list(
        self, ws: Workspace,
    ) -> List[Tuple[LotRegion, float]]:
        custom_weights = self.get_lot_weights(ws)
        result = []
        for lot in ws.lots:
            weight = 1.0
            if custom_weights and lot.lot_id in custom_weights:
                weight = custom_weights[lot.lot_id]
            if weight <= 0:
                continue
            result.append((lot, weight))
        return result

    def get_lot_weights(self, ws: Workspace) -> Optional[Dict[str, float]]:
        """서브클래스에서 override하여 커스터마이즈."""
        return None

    # ── 1지번 1블록 점유 판정 ─────────────────────────────────────

    @staticmethod
    def _build_occupied_lot_set(
        ws: Workspace, block_in: date, block_out: date,
    ) -> set:
        occupied: set[str] = set()
        lots = ws.lots

        # 현재 배치된 블록
        for blk in ws.blocks:
            b_left   = blk.ref_x - blk.length  / 2.0
            b_right  = blk.ref_x + blk.length  / 2.0
            b_bottom = blk.ref_y - blk.breadth / 2.0
            b_top    = blk.ref_y + blk.breadth / 2.0
            for lot in lots:
                if lot.lot_id in occupied:
                    continue
                if _is_occupied_by_rect(lot, b_left, b_right, b_bottom, b_top):
                    occupied.add(lot.lot_id)

        # 기배치 블록 (기간 겹침)
        pps = ws.get_overlapping_pre_placements(block_in, block_out)
        for pp in pps:
            b_left   = pp.pos_x - pp.length  / 2.0
            b_right  = pp.pos_x + pp.length  / 2.0
            b_bottom = pp.pos_y - pp.breadth / 2.0
            b_top    = pp.pos_y + pp.breadth / 2.0
            for lot in lots:
                if lot.lot_id in occupied:
                    continue
                if _is_occupied_by_rect(lot, b_left, b_right, b_bottom, b_top):
                    occupied.add(lot.lot_id)

        return occupied

    def occupied_lot_ids(
        self,
        workspace: Workspace,
        block_in: date,
        block_out: date,
    ) -> set[str]:
        return set(
            self._build_occupied_lot_set(workspace, block_in, block_out)
        )


# ── 유틸리티 함수 (모듈 레벨) ──────────────────────────────────────


def _generate_lot_anchors(
    lot: LotRegion, half_l: float, half_b: float,
) -> List[Tuple[float, float]]:
    """지번 내 9개 앵커 포인트(블록 중심 좌표)를 생성합니다."""
    x_left   = lot.origin_x + half_l
    x_center = lot.origin_x + lot.length / 2.0
    x_right  = lot.origin_x + lot.length - half_l

    y_bottom = lot.origin_y + half_b
    y_center = lot.origin_y + lot.breadth / 2.0
    y_top    = lot.origin_y + lot.breadth - half_b

    # 블록 > 지번인 경우 clamp
    if x_right < x_left:
        x_right = x_left
    if y_top < y_bottom:
        y_top = y_bottom
    x_center = max(x_left, min(x_center, x_right))
    y_center = max(y_bottom, min(y_center, y_top))

    # 9개 후보 (BL→BC→BR→ML→MC→MR→TL→TC→TR)
    raw = [
        (x_left,   y_bottom),
        (x_center, y_bottom),
        (x_right,  y_bottom),
        (x_left,   y_center),
        (x_center, y_center),
        (x_right,  y_center),
        (x_left,   y_top),
        (x_center, y_top),
        (x_right,  y_top),
    ]

    # 중복 제거
    deduped: List[Tuple[float, float]] = []
    for pt in raw:
        is_dup = False
        for existing in deduped:
            if abs(pt[0] - existing[0]) < EPSILON and abs(pt[1] - existing[1]) < EPSILON:
                is_dup = True
                break
        if not is_dup:
            deduped.append(pt)

    return deduped


def _has_collision(
    ws: Workspace, block: Block, cx: float, cy: float,
    pre_placements: List[PrePlacedBlock],
) -> bool:
    """후보 좌표에서 블록이 기존 블록/기배치와 충돌하는지 검사."""
    dx = cx - block.ref_x
    dy = cy - block.ref_y
    block.move(dx, dy)

    overlaps = False

    # 기존 배치 블록과의 충돌
    for existing in ws.blocks:
        if existing is block:
            continue
        if block.intersects(existing):
            overlaps = True
            break

    # 기배치 블록과의 충돌
    if not overlaps:
        for pp in pre_placements:
            if pp.intersects_block_in_period(block, block.in_date, block.out_date):
                overlaps = True
                break

    block.move(-dx, -dy)
    return overlaps


def _is_occupied_by_rect(
    lot: LotRegion,
    rect_left: float, rect_right: float,
    rect_bottom: float, rect_top: float,
) -> bool:
    """AABB가 지번을 X 또는 Y 방향으로 1/3 이상 침범하면 True."""
    lot_left   = lot.origin_x
    lot_right  = lot.origin_x + lot.length
    lot_bottom = lot.origin_y
    lot_top    = lot.origin_y + lot.breadth

    overlap_x = max(0.0, min(rect_right, lot_right)  - max(rect_left, lot_left))
    overlap_y = max(0.0, min(rect_top,   lot_top)    - max(rect_bottom, lot_bottom))

    if overlap_x <= 1e-4 or overlap_y <= 1e-4:
        return False

    return (overlap_x / lot.length >= OCCUPATION_THRESHOLD
            or overlap_y / lot.breadth >= OCCUPATION_THRESHOLD)
