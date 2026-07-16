"""
가상 블록 데이터 생성기.

실제 CSV 데이터의 분포를 기반으로 랜덤 블록을 생성하여
매 에피소드마다 다양한 데이터로 학습, 과적합(overfitting) 방지.
"""

from __future__ import annotations

import calendar as month_calendar
import csv
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import calendar as cal
from .block import Block, PrePlacedBlock
from .workspace import LotRegion, Workspace


# ── 분포 통계 ────────────────────────────────────────────────────

@dataclass
class DistParam:
    """단일 속성의 분포 파라미터 (truncated normal)."""
    mean: float
    std: float
    low: float      # 클리핑 하한 (P5 기반)
    high: float     # 클리핑 상한 (P95 기반)

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        """Truncated normal 분포에서 n개 샘플링."""
        vals = rng.normal(self.mean, self.std, size=n)
        return np.clip(vals, self.low, self.high)


@dataclass
class BlockDistribution:
    """블록 속성 분포 파라미터 집합."""
    length: DistParam
    breadth: DistParam
    height: DistParam
    weight: DistParam
    duration_days: DistParam   # 달력일 기준 공기

    @staticmethod
    def from_defaults() -> BlockDistribution:
        """분석된 하드코딩 분포 (918블록 기준)."""
        return BlockDistribution(
            length=DistParam(mean=19.75, std=7.19, low=5.0, high=35.0),
            breadth=DistParam(mean=20.58, std=8.12, low=5.0, high=35.0),
            height=DistParam(mean=6.47, std=4.82, low=1.0, high=20.0),
            weight=DistParam(mean=180.16, std=110.34, low=10.0, high=500.0),
            duration_days=DistParam(mean=19.90, std=10.71, low=3.0, high=50.0),
        )

    @staticmethod
    def from_csv(csv_path: str) -> BlockDistribution:
        """CSV 파일에서 분포 통계 학습."""
        lengths, breadths, heights, weights, durations = [], [], [], [], []

        for enc in ("utf-8-sig", "cp949", "euc-kr"):
            try:
                with open(csv_path, encoding=enc) as f:
                    reader = csv.reader(f)
                    next(reader)  # header
                    for row in reader:
                        if len(row) < 19:
                            continue
                        l = _safe_float(row[9])
                        b = _safe_float(row[10])
                        h = _safe_float(row[11])
                        w = _safe_float(row[12])
                        if l <= 0 or b <= 0:
                            continue

                        lengths.append(l)
                        breadths.append(b)
                        if h > 0:
                            heights.append(h)
                        if w > 0:
                            weights.append(w)

                        # 공기 계산
                        d_in = _safe_date(row[15]) or _safe_date(row[17])
                        d_out = _safe_date(row[16]) or _safe_date(row[18])
                        if d_in and d_out:
                            dur = (d_out - d_in).days
                            if dur > 0:
                                durations.append(dur)
                break
            except UnicodeDecodeError:
                continue

        return BlockDistribution(
            length=_fit_dist(lengths, floor=1.0),
            breadth=_fit_dist(breadths, floor=1.0),
            height=_fit_dist(heights, floor=0.5),
            weight=_fit_dist(weights, floor=1.0),
            duration_days=_fit_dist(durations, floor=2.0),
        )

    @staticmethod
    def from_blocks(blocks: Sequence[Block]) -> BlockDistribution:
        """Fit normalization bounds from complete source block records."""
        if not blocks:
            raise ValueError("At least one source block is required")
        return BlockDistribution(
            length=_fit_observed_dist([block.length for block in blocks], 1.0),
            breadth=_fit_observed_dist([block.breadth for block in blocks], 1.0),
            height=_fit_observed_dist([block.height for block in blocks], 0.5),
            weight=_fit_observed_dist([block.weight for block in blocks], 1.0),
            duration_days=_fit_observed_dist(
                [block.original_duration for block in blocks], 1.0
            ),
        )


# ── 블록 생성기 ──────────────────────────────────────────────────

class SyntheticBlockGenerator:
    """
    가상 블록 데이터 생성기.

    매 에피소드마다 새로운 블록 세트를 생성하여
    학습 데이터 다양성 확보.
    """

    def __init__(
        self,
        dist: Optional[BlockDistribution] = None,
        seed: Optional[int] = None,
        source_blocks: Optional[Sequence[Block]] = None,
        monthly_jitter: int = 20,
        empirical_profile_probability: float = 0.2,
        target_month_counts: Optional[Mapping[Tuple[int, int], int]] = None,
    ):
        self._dist = dist or BlockDistribution.from_defaults()
        self._rng = np.random.default_rng(seed)
        self._block_counter = 0
        if monthly_jitter < 0:
            raise ValueError("monthly_jitter must be non-negative")
        if not 0.0 <= empirical_profile_probability <= 1.0:
            raise ValueError(
                "empirical_profile_probability must be between 0 and 1"
            )
        self._source_blocks = tuple(
            block.clone() for block in (source_blocks or ())
        )
        self._monthly_jitter = int(monthly_jitter)
        self._empirical_profile_probability = float(
            empirical_profile_probability
        )
        self._source_month_counts = Counter(
            (block.in_date.year, block.in_date.month)
            for block in self._source_blocks
        )
        target_counts = {
            (int(year), int(month)): int(count)
            for (year, month), count in (
                target_month_counts
                if target_month_counts is not None
                else self._source_month_counts
            ).items()
        }
        for count in target_counts.values():
            if count < 0:
                raise ValueError("target month counts must be non-negative")
        self._target_month_counts = Counter(
            dict(sorted(
                (key, count)
                for key, count in target_counts.items()
                if count > 0
            ))
        )
        if target_month_counts is not None and not self._target_month_counts:
            raise ValueError(
                "target month counts must include at least one positive count"
            )
        for key in self._target_month_counts:
            if key not in self._source_month_counts:
                raise ValueError(f"Target month {key} has no source templates")

    @property
    def dist(self) -> BlockDistribution:
        """블록 속성 분포(정규화 상수 등 외부 참조용)."""
        return self._dist

    @property
    def source_blocks(self) -> Tuple[Block, ...]:
        return self._source_blocks

    @property
    def target_month_counts(self) -> dict[tuple[int, int], int]:
        return dict(self._target_month_counts)

    @property
    def monthly_jitter(self) -> int:
        return self._monthly_jitter

    @property
    def empirical_profile_probability(self) -> float:
        return self._empirical_profile_probability

    @classmethod
    def from_csv(cls, csv_path: str, seed: Optional[int] = None):
        """CSV 파일의 분포를 학습하여 생성기 초기화."""
        dist = BlockDistribution.from_csv(csv_path)
        return cls(dist=dist, seed=seed)

    @classmethod
    def from_blocks(
        cls,
        blocks: Sequence[Block],
        seed: Optional[int] = None,
        monthly_jitter: int = 20,
        empirical_profile_probability: float = 0.2,
        target_month_counts: Optional[Mapping[Tuple[int, int], int]] = None,
    ) -> SyntheticBlockGenerator:
        """Create a row-bootstrap generator from allocation targets."""
        source = list(blocks)
        return cls(
            dist=BlockDistribution.from_blocks(source),
            seed=seed,
            source_blocks=source,
            monthly_jitter=monthly_jitter,
            empirical_profile_probability=empirical_profile_probability,
            target_month_counts=target_month_counts,
        )

    @classmethod
    def from_defaults(cls, seed: Optional[int] = None):
        """기본 분포로 생성기 초기화."""
        return cls(dist=BlockDistribution.from_defaults(), seed=seed)

    def generate(
        self,
        n_blocks: int,
        base_date: date,
        spread_days: int | Tuple[int, int] = 90,
    ) -> List[Block]:
        """
        n_blocks개의 가상 미배치 블록 생성.

        Args:
            n_blocks: 생성할 블록 수
            base_date: 기준 날짜 (입고일 시작점)
            spread_days: 입고일 분산 범위. 정수이면 고정 폭, 튜플이면
                        (min, max) 범위에서 에피소드별 샘플링.
                        기본값 90은 기존 30~90일 랜덤 동작을 유지.
        """
        if self._source_blocks:
            return self._generate_monthly_bootstrap(n_blocks)

        dist = self._dist

        if isinstance(spread_days, tuple):
            spread_min, spread_max = spread_days
            spread_min = max(0, int(spread_min))
            spread_max = max(spread_min, int(spread_max))
            actual_spread = int(self._rng.integers(spread_min, spread_max + 1))
        else:
            # Historical default: no explicit override means 30-90 day random.
            actual_spread = int(self._rng.integers(30, 91))
            if spread_days != 90:
                actual_spread = max(0, int(spread_days))

        # 속성 일괄 샘플링 (벡터 연산)
        lens = dist.length.sample(self._rng, n_blocks)
        bres = dist.breadth.sample(self._rng, n_blocks)
        hgts = dist.height.sample(self._rng, n_blocks)
        wgts = dist.weight.sample(self._rng, n_blocks)
        durs = dist.duration_days.sample(self._rng, n_blocks).astype(int)
        durs = np.maximum(durs, 2)  # 최소 2일

        # 입고일: base_date ~ base_date + actual_spread 범위 내 랜덤
        day_offsets = self._rng.integers(0, actual_spread + 1, size=n_blocks)
        # 정렬: 입고일 순으로 정렬 (시뮬레이터 효율)
        day_offsets.sort()

        blocks: List[Block] = []
        for i in range(n_blocks):
            self._block_counter += 1

            in_date = base_date + timedelta(days=int(day_offsets[i]))
            in_date = cal.adjust_to_working_day(in_date, forward=True)

            # 공기(working days) 기반 출고일 계산
            working_dur = max(2, int(durs[i] * 0.7))  # 달력일 → 근무일 근사
            out_date = cal.calculate_end_date(in_date, working_dur)

            block = Block(
                name=f"SYN-{self._block_counter:05d}",
                ship_no=f"SH{self._rng.integers(1000, 9999):04d}",
                block_type="BUILD",
                length=round(float(lens[i]), 2),
                breadth=round(float(bres[i]), 2),
                height=round(float(hgts[i]), 2),
                weight=round(float(wgts[i]), 2),
                in_date=in_date,
                out_date=out_date,
            )
            blocks.append(block)

        return blocks

    def _generate_monthly_bootstrap(self, n_blocks: int) -> List[Block]:
        if n_blocks < 1:
            return []

        month_keys = sorted(self._target_month_counts)
        if self._rng.random() < self._empirical_profile_probability:
            month_counts = self._empirical_month_counts(n_blocks, month_keys)
        else:
            month_counts = self._balanced_month_counts(n_blocks, month_keys)

        source_by_month = {
            key: [
                block for block in self._source_blocks
                if (block.in_date.year, block.in_date.month) == key
            ]
            for key in month_keys
        }
        blocks: List[Block] = []
        for key, count in zip(month_keys, month_counts):
            year, month = key
            templates = source_by_month[key]
            working_dates = _working_dates_in_month(year, month)
            template_indices = self._rng.integers(
                0, len(templates), size=int(count)
            )
            date_indices = self._rng.integers(
                0, len(working_dates), size=int(count)
            )
            for template_index, date_index in zip(
                template_indices, date_indices
            ):
                self._block_counter += 1
                template = templates[int(template_index)]
                in_date = working_dates[int(date_index)]
                duration = max(int(template.original_duration), 1)
                out_date = cal.calculate_end_date(in_date, duration)
                blocks.append(Block(
                    name=f"SYN-{self._block_counter:05d}",
                    ship_no=template.ship_no,
                    block_type=template.block_type,
                    length=template.length,
                    breadth=template.breadth,
                    height=template.height,
                    weight=template.weight,
                    in_date=in_date,
                    out_date=out_date,
                ))

        blocks.sort(key=lambda block: (block.in_date, block.name))
        return blocks

    def _balanced_month_counts(
        self, n_blocks: int, month_keys: Sequence[Tuple[int, int]]
    ) -> np.ndarray:
        n_months = len(month_keys)
        base = np.full(n_months, n_blocks // n_months, dtype=np.int64)
        remainder = n_blocks % n_months
        if remainder:
            indices = self._rng.choice(n_months, size=remainder, replace=False)
            base[indices] += 1

        jitter = self._monthly_jitter
        if jitter == 0:
            return base
        lower = -np.minimum(base, jitter)
        offsets = np.array([
            self._rng.integers(int(low), jitter + 1)
            for low in lower
        ], dtype=np.int64)
        correction = -int(offsets.sum())
        while correction != 0:
            if correction > 0:
                candidates = np.flatnonzero(offsets < jitter)
                index = int(self._rng.choice(candidates))
                step = min(correction, jitter - int(offsets[index]))
                offsets[index] += step
                correction -= step
            else:
                candidates = np.flatnonzero(offsets > lower)
                index = int(self._rng.choice(candidates))
                step = min(-correction, int(offsets[index] - lower[index]))
                offsets[index] -= step
                correction += step
        return base + offsets

    def _empirical_month_counts(
        self, n_blocks: int, month_keys: Sequence[Tuple[int, int]]
    ) -> np.ndarray:
        target_counts = np.array(
            [self._target_month_counts[key] for key in month_keys],
            dtype=np.float64,
        )
        raw = target_counts * (float(n_blocks) / float(target_counts.sum()))
        counts = np.floor(raw).astype(np.int64)
        remainder = n_blocks - int(counts.sum())
        if remainder:
            fractions = raw - counts
            order = np.argsort(-fractions, kind="stable")
            counts[order[:remainder]] += 1
        return counts

    def generate_preplaced(
        self,
        n_blocks: int,
        workspaces: List[Workspace],
        base_date: date,
    ) -> List[Tuple[str, PrePlacedBlock]]:
        """
        가상 기배치 블록 생성 (작업장에 미리 배치된 블록).

        Returns:
            [(workspace_code, PrePlacedBlock), ...] 리스트
        """
        dist = self._dist
        ws_codes = [ws.code for ws in workspaces]
        ws_dims = {ws.code: (ws.length, ws.breadth) for ws in workspaces}

        results: List[Tuple[str, PrePlacedBlock]] = []

        for _ in range(n_blocks):
            self._block_counter += 1

            # 랜덤 작업장 선택
            ws_code = ws_codes[self._rng.integers(0, len(ws_codes))]
            ws_len, ws_bre = ws_dims[ws_code]

            # 블록 크기 (작업장보다 작게)
            bl = min(float(dist.length.sample(self._rng, 1)[0]), ws_len * 0.4)
            bb = min(float(dist.breadth.sample(self._rng, 1)[0]), ws_bre * 0.4)
            bl = max(bl, 2.0)
            bb = max(bb, 2.0)

            # 위치: 작업장 내 유효 범위에서 랜덤
            margin_x = bl / 2.0
            margin_y = bb / 2.0
            px = self._rng.uniform(margin_x, max(margin_x + 0.1, ws_len - margin_x))
            py = self._rng.uniform(margin_y, max(margin_y + 0.1, ws_bre - margin_y))

            # 배치 시작일: base_date 전 30일 ~ base_date
            start_offset = self._rng.integers(-30, 1)
            start_d = base_date + timedelta(days=int(start_offset))
            start_d = cal.adjust_to_working_day(start_d, forward=True)

            # 공기
            dur = max(2, int(dist.duration_days.sample(self._rng, 1)[0] * 0.7))
            end_d = cal.calculate_end_date(start_d, dur)

            pp = PrePlacedBlock(
                label=f"PRE-{self._block_counter:05d}",
                pos_x=round(float(px), 2),
                pos_y=round(float(py), 2),
                length=round(bl, 2),
                breadth=round(bb, 2),
                start_date=start_d,
                end_date=end_d,
            )
            results.append((ws_code, pp))

        return results

    def generate_workspaces(
        self,
        base_workspaces: List[Workspace],
        scale_range: Tuple[float, float] = (0.7, 1.3),
    ) -> List[Workspace]:
        """
        기준 작업장의 크기를 랜덤 스케일링하여 변형된 작업장 세트 생성.

        작업장 수는 유지하되, 각 작업장의 크기·지번 구성을 변형하여
        다양한 레이아웃에 대한 범용적 학습을 가능하게 한다.

        Args:
            base_workspaces: 원본 작업장 목록
            scale_range: 스케일링 범위 (min, max), 예: (0.7, 1.3) → ±30%

        Returns:
            변형된 작업장 목록 (원본과 동일 수)
        """
        import copy

        lo, hi = scale_range
        result: List[Workspace] = []

        for ws in base_workspaces:
            # 각 축 독립 스케일링
            scale_l = float(self._rng.uniform(lo, hi))
            scale_b = float(self._rng.uniform(lo, hi))

            new_length = max(10.0, round(ws.length * scale_l, 1))
            new_breadth = max(10.0, round(ws.breadth * scale_b, 1))

            new_ws = Workspace(
                code=ws.code,
                origin_x=ws.origin_x,
                origin_y=ws.origin_y,
                breadth=new_breadth,
                length=new_length,
                max_weight=ws.max_weight,
                max_breadth=ws.max_breadth,
                max_height=ws.max_height,
                name=ws.name,
                allowable_block_patterns=(
                    list(ws.allowable_block_patterns)
                    if ws.allowable_block_patterns else None
                ),
                strategy=ws.strategy,
            )

            # 지번 재배치: 스케일에 비례하여 조정
            for lot in ws.lots:
                new_ws.add_lot(LotRegion(
                    lot_id=lot.lot_id,
                    origin_x=round(lot.origin_x * scale_l, 1),
                    origin_y=round(lot.origin_y * scale_b, 1),
                    breadth=round(lot.breadth * scale_b, 1),
                    length=round(lot.length * scale_l, 1),
                ))

            result.append(new_ws)

        return result


# ── 유틸리티 ─────────────────────────────────────────────────────

def _safe_float(value: str) -> float:
    try:
        return float(value.strip())
    except (ValueError, AttributeError):
        return 0.0


def _safe_date(value: str) -> Optional[date]:
    from datetime import datetime
    value = value.strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _fit_dist(values: list, floor: float = 0.0) -> DistParam:
    """값 리스트에서 DistParam 피팅."""
    arr = np.array(values, dtype=np.float64)
    p5, p95 = np.percentile(arr, [5, 95])
    return DistParam(
        mean=float(arr.mean()),
        std=float(arr.std()),
        low=max(floor, float(p5)),
        high=float(p95),
    )


def _fit_observed_dist(values: Sequence[float], floor: float) -> DistParam:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot fit an empty observed distribution")
    return DistParam(
        mean=float(arr.mean()),
        std=float(arr.std()),
        low=max(floor, float(arr.min())),
        high=max(floor, float(arr.max())),
    )


def _working_dates_in_month(year: int, month: int) -> List[date]:
    last_day = month_calendar.monthrange(year, month)[1]
    result = [
        date(year, month, day)
        for day in range(1, last_day + 1)
        if cal.is_working_day(date(year, month, day))
    ]
    if not result:
        raise ValueError(f"Month has no working dates: {year:04d}-{month:02d}")
    return result
