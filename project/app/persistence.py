"""
설정 저장/로드 모듈

애플리케이션 설정, 프로젝트 파일, 결과 내보내기를 담당합니다.

향후 확장 포인트:
- 클라우드 동기화
- 설정 마이그레이션
- 암호화 저장
"""

import json
import os
from pathlib import Path
from datetime import date, datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import pandas as pd

from app.utils import logger
from app.backtest_engine import BacktestConfig, HolidayAdjustment, RebalanceFrequency
from app.analytics import BacktestResult, PerformanceMetrics


# ============================================================================
# 기본 경로 설정
# ============================================================================

# 사용자 설정 디렉토리
CONFIG_DIR = Path.home() / ".dca_backtest"
CONFIG_FILE = CONFIG_DIR / "config.json"
RECENT_PROJECTS_FILE = CONFIG_DIR / "recent_projects.json"

# 프로젝트 디렉토리
DEFAULT_PROJECT_DIR = Path.home() / "Documents" / "DCA_Backtest_Projects"


# ============================================================================
# 애플리케이션 설정
# ============================================================================

@dataclass
class AppSettings:
    """애플리케이션 전역 설정"""
    theme: str = "dark"  # "dark" or "light"
    language: str = "ko"  # "ko" or "en"
    cache_dir: str = ""
    default_project_dir: str = ""
    auto_save: bool = True
    auto_save_interval: int = 300  # 초
    recent_files: List[str] = None
    max_recent_files: int = 10

    # 마지막 사용 설정
    last_tickers: List[str] = None
    last_start_date: str = ""
    last_end_date: str = ""
    last_strategy: str = "Pure DCA"

    def __post_init__(self):
        if self.recent_files is None:
            self.recent_files = []
        if self.last_tickers is None:
            self.last_tickers = ["QQQ"]


class SettingsManager:
    """
    애플리케이션 설정 관리자

    설정을 JSON 파일로 저장/로드합니다.
    """

    def __init__(self, config_path: Path = CONFIG_FILE):
        self.config_path = config_path
        self.settings: AppSettings = AppSettings()
        self._ensure_config_dir()

    def _ensure_config_dir(self) -> None:
        """설정 디렉토리 생성"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> AppSettings:
        """설정 로드"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.settings = AppSettings(**data)
                logger.debug("설정 로드 완료")
            except Exception as e:
                logger.warning(f"설정 로드 실패: {e}")
                self.settings = AppSettings()
        return self.settings

    def save(self) -> bool:
        """설정 저장"""
        try:
            self._ensure_config_dir()
            data = asdict(self.settings)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug("설정 저장 완료")
            return True
        except Exception as e:
            logger.error(f"설정 저장 실패: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """설정값 조회"""
        return getattr(self.settings, key, default)

    def set(self, key: str, value: Any) -> None:
        """설정값 변경"""
        if hasattr(self.settings, key):
            setattr(self.settings, key, value)

    def add_recent_file(self, file_path: str) -> None:
        """최근 파일 추가"""
        if file_path in self.settings.recent_files:
            self.settings.recent_files.remove(file_path)
        self.settings.recent_files.insert(0, file_path)
        self.settings.recent_files = self.settings.recent_files[:self.settings.max_recent_files]

    def clear_recent_files(self) -> None:
        """최근 파일 목록 초기화"""
        self.settings.recent_files = []


# 전역 설정 매니저 인스턴스
settings_manager = SettingsManager()


# ============================================================================
# 프로젝트 파일
# ============================================================================

@dataclass
class Project:
    """프로젝트 데이터"""
    name: str = "Untitled"
    created_at: str = ""
    modified_at: str = ""

    # 백테스트 설정
    config: Dict = None

    # 결과 (선택적)
    has_results: bool = False
    results_file: str = ""  # CSV 파일 경로

    def __post_init__(self):
        if self.config is None:
            self.config = {}
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.modified_at = datetime.now().isoformat()


class ProjectManager:
    """
    프로젝트 파일 관리자

    백테스트 설정과 결과를 프로젝트 파일로 저장/로드합니다.
    """

    def __init__(self, project_dir: Path = DEFAULT_PROJECT_DIR):
        self.project_dir = Path(project_dir)
        self.current_project: Optional[Project] = None
        self.current_path: Optional[Path] = None

    def _ensure_project_dir(self) -> None:
        """프로젝트 디렉토리 생성"""
        self.project_dir.mkdir(parents=True, exist_ok=True)

    def new_project(self, name: str = "Untitled") -> Project:
        """새 프로젝트 생성"""
        self.current_project = Project(name=name)
        self.current_path = None
        return self.current_project

    def save_project(self, path: Optional[Path] = None) -> bool:
        """
        프로젝트 저장

        Args:
            path: 저장 경로 (None이면 현재 경로 사용)

        Returns:
            성공 여부
        """
        if self.current_project is None:
            logger.error("저장할 프로젝트가 없습니다.")
            return False

        if path:
            self.current_path = Path(path)
        elif self.current_path is None:
            logger.error("저장 경로가 지정되지 않았습니다.")
            return False

        try:
            self.current_project.modified_at = datetime.now().isoformat()
            data = asdict(self.current_project)

            with open(self.current_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"프로젝트 저장: {self.current_path}")
            settings_manager.add_recent_file(str(self.current_path))
            return True

        except Exception as e:
            logger.error(f"프로젝트 저장 실패: {e}")
            return False

    def load_project(self, path: Path) -> Optional[Project]:
        """
        프로젝트 로드

        Args:
            path: 프로젝트 파일 경로

        Returns:
            Project 객체 또는 None
        """
        path = Path(path)

        if not path.exists():
            logger.error(f"파일을 찾을 수 없습니다: {path}")
            return None

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.current_project = Project(**data)
            self.current_path = path

            logger.info(f"프로젝트 로드: {path}")
            settings_manager.add_recent_file(str(path))
            return self.current_project

        except Exception as e:
            logger.error(f"프로젝트 로드 실패: {e}")
            return None

    def update_config(self, config: BacktestConfig) -> None:
        """프로젝트 설정 업데이트"""
        if self.current_project is None:
            self.new_project()

        self.current_project.config = config_to_dict(config)

    def get_config(self) -> Optional[BacktestConfig]:
        """프로젝트에서 설정 추출"""
        if self.current_project is None or not self.current_project.config:
            return None

        return dict_to_config(self.current_project.config)


# 전역 프로젝트 매니저 인스턴스
project_manager = ProjectManager()


# ============================================================================
# 설정 직렬화
# ============================================================================

def config_to_dict(config: BacktestConfig) -> Dict[str, Any]:
    """BacktestConfig를 딕셔너리로 변환"""
    return {
        'start_date': config.start_date.isoformat() if config.start_date else None,
        'end_date': config.end_date.isoformat() if config.end_date else None,
        'tickers': config.tickers,
        'weights': config.weights,
        'payday': config.payday,
        'monthly_investment': config.monthly_investment,
        'annual_increase_rate': config.annual_increase_rate,
        'commission_rate': config.commission_rate,
        'slippage_rate': config.slippage_rate,
        'holiday_adjustment': config.holiday_adjustment.value,
        'rebalance_frequency': config.rebalance_frequency.value,
        'reserve_cash': config.reserve_cash,
        'cash_interest_rate': config.cash_interest_rate,
        'benchmark': config.benchmark,
        'strategy_name': config.strategy_name,
        'strategy_params': config.strategy_params,
    }


def dict_to_config(data: Dict[str, Any]) -> BacktestConfig:
    """딕셔너리를 BacktestConfig로 변환"""
    start_date = None
    end_date = None

    if data.get('start_date'):
        start_date = date.fromisoformat(data['start_date'])
    if data.get('end_date'):
        end_date = date.fromisoformat(data['end_date'])

    holiday_adj = HolidayAdjustment.PREVIOUS
    if data.get('holiday_adjustment'):
        holiday_adj = HolidayAdjustment(data['holiday_adjustment'])

    rebalance = RebalanceFrequency.NONE
    if data.get('rebalance_frequency'):
        rebalance = RebalanceFrequency(data['rebalance_frequency'])

    return BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        tickers=data.get('tickers', []),
        weights=data.get('weights', []),
        payday=data.get('payday', 25),
        monthly_investment=data.get('monthly_investment', 4000.0),
        annual_increase_rate=data.get('annual_increase_rate', 0.0),
        commission_rate=data.get('commission_rate', 0.001),
        slippage_rate=data.get('slippage_rate', 0.001),
        holiday_adjustment=holiday_adj,
        rebalance_frequency=rebalance,
        reserve_cash=data.get('reserve_cash', True),
        cash_interest_rate=data.get('cash_interest_rate', 0.0),
        benchmark=data.get('benchmark', 'SPY'),
        strategy_name=data.get('strategy_name', 'Pure DCA'),
        strategy_params=data.get('strategy_params', {}),
    )


# ============================================================================
# 결과 내보내기
# ============================================================================

def export_results_to_csv(result: BacktestResult, path: Path) -> bool:
    """
    백테스트 결과를 CSV로 내보내기

    Args:
        result: 백테스트 결과
        path: 저장 경로

    Returns:
        성공 여부
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 자산 곡선
        df = pd.DataFrame({
            'date': result.equity_curve.index,
            'equity': result.equity_curve.values,
            'drawdown': result.drawdown_curve.values if result.drawdown_curve is not None else None
        })

        df.to_csv(path, index=False)
        logger.info(f"결과 내보내기: {path}")
        return True

    except Exception as e:
        logger.error(f"결과 내보내기 실패: {e}")
        return False


def export_metrics_to_json(metrics: PerformanceMetrics, path: Path) -> bool:
    """
    성과 지표를 JSON으로 내보내기

    Args:
        metrics: 성과 지표
        path: 저장 경로

    Returns:
        성공 여부
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = metrics.to_dict()

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"지표 내보내기: {path}")
        return True

    except Exception as e:
        logger.error(f"지표 내보내기 실패: {e}")
        return False


def export_full_report(result: BacktestResult, directory: Path) -> bool:
    """
    전체 보고서 내보내기

    디렉토리에 다음 파일들을 생성:
    - equity_curve.csv
    - monthly_returns.csv
    - metrics.json
    - summary.txt

    Args:
        result: 백테스트 결과
        directory: 저장 디렉토리

    Returns:
        성공 여부
    """
    try:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # 자산 곡선
        export_results_to_csv(result, directory / "equity_curve.csv")

        # 월별 수익률
        if result.monthly_returns is not None:
            result.monthly_returns.to_csv(directory / "monthly_returns.csv")

        # 성과 지표
        if result.metrics:
            export_metrics_to_json(result.metrics, directory / "metrics.json")

        # 요약 텍스트
        with open(directory / "summary.txt", 'w', encoding='utf-8') as f:
            f.write("DCA Backtest Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Period: {result.start_date} ~ {result.end_date}\n")
            f.write(f"Tickers: {', '.join(result.tickers)}\n")
            f.write(f"Strategy: {result.strategy_name}\n\n")

            if result.metrics:
                m = result.metrics
                f.write("Performance Metrics\n")
                f.write("-" * 30 + "\n")
                f.write(f"CAGR: {m.cagr*100:.2f}%\n")
                f.write(f"Total Return: {m.total_return*100:.2f}%\n")
                f.write(f"MDD: {m.mdd*100:.2f}%\n")
                f.write(f"Sharpe Ratio: {m.sharpe_ratio:.2f}\n")
                f.write(f"Sortino Ratio: {m.sortino_ratio:.2f}\n")
                f.write(f"Calmar Ratio: {m.calmar_ratio:.2f}\n\n")
                f.write(f"Total Invested: ${m.total_invested:,.2f}\n")
                f.write(f"Final Value: ${m.final_value:,.2f}\n")
                f.write(f"Profit: ${m.profit:,.2f}\n")
                f.write(f"Win Rate: {m.win_rate*100:.1f}%\n")

        logger.info(f"전체 보고서 내보내기: {directory}")
        return True

    except Exception as e:
        logger.error(f"보고서 내보내기 실패: {e}")
        return False


# ============================================================================
# 프리셋 관리
# ============================================================================

# 내장 프리셋
BUILT_IN_PRESETS = {
    "QQQ Pure DCA": {
        "tickers": ["QQQ"],
        "weights": [1.0],
        "monthly_investment": 4000.0,
        "payday": 25,
        "strategy_name": "Pure DCA",
        "strategy_params": {},
    },
    "VOO+QQQ Drawdown Tier": {
        "tickers": ["VOO", "QQQ"],
        "weights": [0.5, 0.5],
        "monthly_investment": 4000.0,
        "payday": 25,
        "strategy_name": "Drawdown Tier DCA",
        "strategy_params": {
            "tier1_threshold": -10.0,
            "tier1_multiplier": 1.5,
            "tier2_threshold": -20.0,
            "tier2_multiplier": 2.0,
            "tier3_threshold": -30.0,
            "tier3_multiplier": 2.5,
        },
    },
    "SPY Trend Filter": {
        "tickers": ["SPY"],
        "weights": [1.0],
        "monthly_investment": 4000.0,
        "payday": 25,
        "strategy_name": "Trend Filter",
        "strategy_params": {
            "ma_period": 200,
            "below_ma_action": "reduce",
            "deploy_on_recovery": True,
        },
    },
}


def get_preset_names() -> List[str]:
    """프리셋 이름 목록"""
    return list(BUILT_IN_PRESETS.keys())


def load_preset(name: str) -> Optional[Dict[str, Any]]:
    """프리셋 로드"""
    return BUILT_IN_PRESETS.get(name)


def apply_preset(name: str) -> Optional[BacktestConfig]:
    """프리셋을 BacktestConfig로 변환"""
    preset = load_preset(name)
    if preset is None:
        return None

    # 기본 날짜 설정 (최근 5년)
    end = date.today()
    start = date(end.year - 5, end.month, end.day)

    return BacktestConfig(
        start_date=start,
        end_date=end,
        tickers=preset['tickers'],
        weights=preset['weights'],
        payday=preset.get('payday', 25),
        monthly_investment=preset.get('monthly_investment', 4000.0),
        strategy_name=preset.get('strategy_name', 'Pure DCA'),
        strategy_params=preset.get('strategy_params', {}),
    )


# ============================================================================
# 향후 확장 포인트
# ============================================================================

# TODO: 클라우드 동기화
# class CloudSync:
#     """클라우드 백업/동기화"""
#     pass

# TODO: 설정 마이그레이션
# def migrate_settings(old_version: str, new_version: str):
#     """구버전 설정을 신버전으로 마이그레이션"""
#     pass

# TODO: 암호화 저장
# def encrypt_project(project: Project, password: str) -> bytes:
#     """프로젝트 암호화"""
#     pass
