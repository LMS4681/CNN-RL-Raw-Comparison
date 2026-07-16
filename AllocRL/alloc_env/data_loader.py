"""
CSV 데이터 로더.

C# DataLoader의 Python 재구현.
- 작업장 마스터 CSV
- 지번 CSV
- 블록 데이터 CSV (기배치 + 미배치)
"""

from __future__ import annotations

import copy
import csv
from datetime import datetime, date
from pathlib import Path
from typing import Dict, IO, List, Mapping, Optional, Sequence, Tuple

from . import calendar as cal


def _open_csv(path: str) -> IO:
    """CSV 파일을 적절한 인코딩으로 엽니다 (utf-8-sig → cp949 fallback)."""
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            f = open(path, encoding=enc)
            f.readline()  # 첫 줄 읽어서 인코딩 검증
            f.seek(0)
            return f
        except (UnicodeDecodeError, UnicodeError):
            f.close()
    # 최후 수단: errors=replace
    return open(path, encoding="utf-8", errors="replace")
from .block import Block, PrePlacedBlock
from .workspace import LotRegion, Workspace
from .strategy import BaseGridStrategy

DEFAULT_WORKSPACE_LIMIT = None
DEFAULT_TARGET_NAME_PREFIX = None

# ── 블록 CSV 열 인덱스 (0-based) ─────────────────────────────────

COL_SHIP_NO      = 0   # A: 호선
COL_BLOCK_NAME   = 1   # B: 블록
COL_WORKSPACE    = 6   # G: 배치 작업장 코드
COL_COORD        = 8   # I: 블록 좌표값 (x,y)
COL_LENGTH       = 9   # J: 길이
COL_BREADTH      = 10  # K: 폭
COL_HEIGHT       = 11  # L: 높이
COL_WEIGHT       = 12  # M: 중량
COL_PLACED_IN    = 15  # P: 기배치 배치일
COL_PLACED_OUT   = 16  # Q: 기배치 출고일
COL_SCHEDULE_IN  = 17  # R: 미배치 예정 입고일
COL_SCHEDULE_OUT = 18  # S: 미배치 예정 출고일


# ── 공개 API ─────────────────────────────────────────────────────

def load_workspaces(
    workspace_csv: str,
    lot_csv: str,
    strategy: Optional[BaseGridStrategy] = None,
    workspace_limit: Optional[int] = DEFAULT_WORKSPACE_LIMIT,
    target_name_prefix: Optional[str] = DEFAULT_TARGET_NAME_PREFIX,
    supplemental_workspaces: Optional[
        Mapping[str, Tuple[str, float, float]]
    ] = None,
) -> List[Workspace]:
    """작업장 CSV + 지번 CSV → Workspace 목록 생성."""
    if strategy is None:
        strategy = BaseGridStrategy()
    if workspace_limit is not None and workspace_limit < 1:
        raise ValueError("workspace_limit must be at least 1")

    ws_master = _parse_workspace_csv(workspace_csv)
    lots_per_ws = _parse_lot_csv(lot_csv)
    for code, spec in (supplemental_workspaces or {}).items():
        normalized_code = code.strip().upper()
        if normalized_code and normalized_code not in ws_master:
            ws_master[normalized_code] = spec
    workspace_items = list(ws_master.items())
    if workspace_limit is not None:
        workspace_items = workspace_items[:workspace_limit]

    workspaces: List[Workspace] = []
    for idx, (code, (name, width, height)) in enumerate(workspace_items, start=1):
        display_name = f"{target_name_prefix}{idx}" if target_name_prefix else name
        ws = Workspace(
            code=code,
            origin_x=0.0,
            origin_y=0.0,
            breadth=height,
            length=width,
            name=display_name,
            strategy=strategy,
        )

        if code in lots_per_ws:
            for lot_id, csv_x, csv_y, lot_w, lot_h in lots_per_ws[code]:
                converted_y = height - (csv_y + lot_h)
                ws.add_lot(LotRegion(
                    lot_id=lot_id,
                    origin_x=csv_x,
                    origin_y=converted_y,
                    breadth=lot_h,
                    length=lot_w,
                ))

        workspaces.append(ws)

    total_lots = sum(len(ws.lots) for ws in workspaces)
    print(f"[DataLoader] 작업장 {len(workspaces)}개 로드 (지번 총 {total_lots}개)")
    return workspaces


def clone_empty_workspaces(workspaces: Sequence[Workspace]) -> List[Workspace]:
    """Clone workspace geometry while removing all placement state."""
    result = copy.deepcopy(list(workspaces))
    for workspace in result:
        workspace.blocks.clear()
        workspace.pre_placements.clear()
    return result


def apply_allowable_block_patterns(
    workspaces: List[Workspace],
    patterns_by_workspace: Optional[Mapping[str, Sequence[str]]] = None,
) -> List[Workspace]:
    """
    작업장별 허용 블록명 패턴을 적용합니다.

    패턴 기준 데이터가 없는 현재 학습 CSV에서는 기본적으로 제약을 두지 않습니다.
    호출자가 매핑을 넘기면 작업장 코드(case-insensitive)를 기준으로 패턴을 주입합니다.
    """
    if not patterns_by_workspace:
        return workspaces

    normalized = {
        code.upper(): list(patterns)
        for code, patterns in patterns_by_workspace.items()
    }
    for ws in workspaces:
        ws.set_allowable_block_patterns(normalized.get(ws.code.upper()))
    return workspaces


def select_workspaces(
    workspaces: List[Workspace],
    active_codes: Optional[List[str]],
) -> List[Workspace]:
    if not active_codes:
        return list(workspaces)

    wanted = {
        code.strip().upper()
        for code in active_codes
        if code.strip()
    }
    known = {workspace.code.upper() for workspace in workspaces}
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(
            "Unknown active workspace code(s): " + ", ".join(unknown)
        )

    selected = [
        workspace
        for workspace in workspaces
        if workspace.code.upper() in wanted
    ]
    if not selected:
        raise ValueError("At least one active workspace is required.")
    return selected


def select_workspaces_in_order(
    workspaces: List[Workspace],
    ordered_codes: Optional[Sequence[str]],
) -> List[Workspace]:
    """Select workspaces in a recorded model action-index order."""
    if not ordered_codes:
        return list(workspaces)

    normalized = [
        code.strip().upper()
        for code in ordered_codes
        if code.strip()
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Recorded workspace codes must be unique.")

    by_code = {workspace.code.upper(): workspace for workspace in workspaces}
    unknown = [code for code in normalized if code not in by_code]
    if unknown:
        raise ValueError(
            "Unknown recorded workspace code(s): " + ", ".join(unknown)
        )
    return [by_code[code] for code in normalized]


def load_blocks(
    block_csv: str,
    workspaces: List[Workspace],
) -> List[Block]:
    """블록 CSV → 미배치 블록 리스트 (기배치는 작업장에 직접 등록)."""
    ws_map = {ws.code.upper(): ws for ws in workspaces}
    unplaced: List[Block] = []
    pre_placed_count = 0
    skipped = 0

    with _open_csv(block_csv) as f:
        reader = csv.reader(f)
        next(reader)  # 헤더 skip
        for row in reader:
            if len(row) < 19:
                continue

            ship_no    = row[COL_SHIP_NO].strip()
            block_name = row[COL_BLOCK_NAME].strip()
            ws_code    = row[COL_WORKSPACE].strip()
            length     = _parse_float(row[COL_LENGTH])
            breadth    = _parse_float(row[COL_BREADTH])
            height     = _parse_float(row[COL_HEIGHT])
            weight     = _parse_float(row[COL_WEIGHT])

            if length <= 0 or breadth <= 0:
                skipped += 1
                continue

            # ── 기배치 블록 ───────────────────────────────────────
            placed_in  = _try_parse_date(row[COL_PLACED_IN])
            placed_out = _try_parse_date(row[COL_PLACED_OUT])

            if ws_code and placed_in and placed_out:
                ws = ws_map.get(ws_code.upper())
                if ws:
                    csv_x, csv_y = _parse_coord(row[COL_COORD])
                    center_x = csv_x + length / 2.0
                    center_y = ws.breadth - (csv_y + breadth / 2.0)

                    pp = PrePlacedBlock(
                        label=f"{ship_no}-{block_name}",
                        pos_x=center_x,
                        pos_y=center_y,
                        length=length,
                        breadth=breadth,
                        start_date=placed_in,
                        end_date=placed_out,
                    )
                    ws.add_pre_placement(pp)
                    pre_placed_count += 1
                continue

            # ── 미배치 블록 ───────────────────────────────────────
            sched_in  = _try_parse_date(row[COL_SCHEDULE_IN])
            sched_out = _try_parse_date(row[COL_SCHEDULE_OUT])

            if sched_in and sched_out:
                sched_in = cal.adjust_to_working_day(sched_in, forward=True)
                sched_out = cal.adjust_to_working_day(sched_out, forward=True)

                block = Block(
                    name=block_name,
                    ship_no=ship_no,
                    block_type="BUILD",
                    length=length,
                    breadth=breadth,
                    height=height,
                    weight=weight,
                    in_date=sched_in,
                    out_date=sched_out,
                )
                unplaced.append(block)
                continue

            skipped += 1

    print(f"[DataLoader] 기배치 {pre_placed_count}개, "
          f"미배치 {len(unplaced)}개, 스킵 {skipped}개")
    return unplaced


def load_target_blocks(
    block_csv: str,
    excluded_start_months: Sequence[int] = (7, 11),
) -> List[Block]:
    """Load every eligible CSV row as a new allocation target.

    Every row uses its construction start/end dates. Historical workspace
    codes, coordinates, actual placement dates, and placement state are
    deliberately ignored.
    """
    excluded = {int(month) for month in excluded_start_months}
    if any(month < 1 or month > 12 for month in excluded):
        raise ValueError("excluded_start_months must contain months 1 through 12")

    targets: List[Block] = []
    skipped = 0
    excluded_count = 0
    with _open_csv(block_csv) as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            if len(row) < 19:
                skipped += 1
                continue

            length = _parse_float(row[COL_LENGTH])
            breadth = _parse_float(row[COL_BREADTH])
            if length <= 0 or breadth <= 0:
                skipped += 1
                continue

            scheduled_pair = (
                _try_parse_date(row[COL_SCHEDULE_IN]),
                _try_parse_date(row[COL_SCHEDULE_OUT]),
            )
            if not all(scheduled_pair):
                skipped += 1
                continue
            start_date, end_date = scheduled_pair

            assert start_date is not None and end_date is not None
            if start_date.month in excluded:
                excluded_count += 1
                continue
            if end_date < start_date:
                skipped += 1
                continue

            start_date = cal.adjust_to_working_day(start_date, forward=True)
            end_date = cal.adjust_to_working_day(end_date, forward=True)
            targets.append(Block(
                name=row[COL_BLOCK_NAME].strip(),
                ship_no=row[COL_SHIP_NO].strip(),
                block_type="BUILD",
                length=length,
                breadth=breadth,
                height=_parse_float(row[COL_HEIGHT]),
                weight=_parse_float(row[COL_WEIGHT]),
                in_date=start_date,
                out_date=end_date,
            ))

    targets.sort(key=lambda block: (block.in_date, block.ship_no, block.name))
    print(
        f"[DataLoader] allocation targets={len(targets)}, "
        f"excluded_month={excluded_count}, skipped={skipped}"
    )
    return targets


# ── 내부 파서 ─────────────────────────────────────────────────────

def _parse_workspace_csv(
    path: str,
) -> Dict[str, Tuple[str, float, float]]:
    """작업장 마스터 CSV → {code: (name, width, height)}"""
    result: Dict[str, Tuple[str, float, float]] = {}
    with _open_csv(path) as f:
        reader = csv.reader(f)
        next(reader)  # 헤더 skip
        for row in reader:
            if len(row) < 6:
                continue
            code   = row[2].strip()
            name   = row[3].strip()
            width  = _parse_float(row[4])
            height = _parse_float(row[5])
            if code and width > 0 and height > 0:
                result[code] = (name, width, height)
    return result


def _parse_lot_csv(
    path: str,
) -> Dict[str, List[Tuple[str, float, float, float, float]]]:
    """지번 CSV → {ws_code: [(lot_id, x, y, width, height), ...]}"""
    result: Dict[str, List[Tuple[str, float, float, float, float]]] = {}
    with _open_csv(path) as f:
        reader = csv.reader(f)
        next(reader)  # 헤더 skip
        for row in reader:
            if len(row) < 6:
                continue
            ws_code = row[0].strip()
            lot_id  = row[1].strip()
            x       = _parse_float(row[2])
            y       = _parse_float(row[3])
            width   = _parse_float(row[4])
            height  = _parse_float(row[5])
            if not ws_code:
                continue
            result.setdefault(ws_code, []).append((lot_id, x, y, width, height))
    return result


# ── 유틸리티 ─────────────────────────────────────────────────────

def _parse_float(value: str) -> float:
    try:
        return float(value.strip())
    except (ValueError, AttributeError):
        return 0.0


def _try_parse_date(value: str) -> Optional[date]:
    value = value.strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _parse_coord(value: str) -> Tuple[float, float]:
    value = value.strip()
    if not value:
        return 0.0, 0.0
    parts = value.split(",")
    if len(parts) >= 2:
        return _parse_float(parts[0]), _parse_float(parts[1])
    return 0.0, 0.0
