from typing import Dict, List
from pydantic import BaseModel


class DailyTrend(BaseModel):
    date: str
    total_scans: int
    violations: int


class MissingFieldStat(BaseModel):
    field_name: str
    count: int
    percentage: float


class DashboardStatsResponse(BaseModel):
    total_scans: int
    total_violations: int
    overall_violation_rate_pct: float
    field_violation_rate_pct: float = 0.0  # excludes NOT_REQUIRED from denominator
    category_violation_rates: Dict[str, float]
    status_breakdown: Dict[str, int]
    compliance_breakdown: Dict[str, int]
    last_30_days_trend: List[DailyTrend]
    most_commonly_missing_fields: List[MissingFieldStat]
