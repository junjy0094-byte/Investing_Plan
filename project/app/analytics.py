"""
성과 분석 모듈

백테스트 결과에 대한 성과 지표를 계산합니다.

주요 지표:
- CAGR (연평균 수익률)
- MDD (최대 낙폭)
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- 총 납입액, 최종 자산, 수익금

향후 확장 포인트:
- 추가 위험 지표 (VaR, CVaR)
- 팩터 분석
- 상관관계 분석
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import date


# ============================================================================
# 성과 지표 데이터 클래스
# ============================================================================

@dataclass
class PerformanceMetrics:
    """
    백테스트 성과 지표를 담는 데이터 클래스
    """
    # 기본 수익률 지표
    total_return: float = 0.0        # 총 수익률 (예: 1.5 = 150%)
    cagr: float = 0.0                # 연평균 수익률
    monthly_return_avg: float = 0.0  # 월평균 수익률
    monthly_return_std: float = 0.0  # 월 수익률 표준편차

    # 위험 지표
    mdd: float = 0.0                 # 최대 낙폭 (음수)
    volatility: float = 0.0          # 연율화 변동성
    downside_volatility: float = 0.0 # 하방 변동성

    # 위험조정 수익률
    sharpe_ratio: float = 0.0        # 샤프 비율
    sortino_ratio: float = 0.0       # 소르티노 비율
    calmar_ratio: float = 0.0        # 칼마 비율

    # 투자 관련
    total_invested: float = 0.0      # 총 납입액
    final_value: float = 0.0         # 최종 자산
    profit: float = 0.0              # 수익금 (최종 - 납입)
    profit_rate: float = 0.0         # 수익률 (수익금/납입액)

    # 승률 관련
    winning_months: int = 0          # 수익 월 수
    losing_months: int = 0           # 손실 월 수
    win_rate: float = 0.0            # 월 기준 승률

    # 기간
    total_days: int = 0
    total_months: int = 0
    total_years: float = 0.0

    # 벤치마크 대비
    alpha: float = 0.0               # 알파 (벤치마크 대비 초과수익)
    beta: float = 0.0                # 베타 (시장 민감도)
    tracking_error: float = 0.0      # 추적 오차
    information_ratio: float = 0.0   # 정보 비율

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            'CAGR': self.cagr,
            'Total Return': self.total_return,
            'MDD': self.mdd,
            'Volatility': self.volatility,
            'Sharpe Ratio': self.sharpe_ratio,
            'Sortino Ratio': self.sortino_ratio,
            'Calmar Ratio': self.calmar_ratio,
            'Total Invested': self.total_invested,
            'Final Value': self.final_value,
            'Profit': self.profit,
            'Profit Rate': self.profit_rate,
            'Win Rate': self.win_rate,
            'Alpha': self.alpha,
        }


@dataclass
class BacktestResult:
    """
    백테스트 결과를 담는 데이터 클래스
    """
    # 시계열 데이터
    equity_curve: pd.Series = None           # 자산 곡선
    drawdown_curve: pd.Series = None         # 드로우다운 곡선
    investment_history: pd.DataFrame = None  # 투자 내역

    # 성과 지표
    metrics: PerformanceMetrics = None

    # 월별/연도별 수익률
    monthly_returns: pd.Series = None
    yearly_returns: pd.Series = None

    # 벤치마크 데이터
    benchmark_curve: pd.Series = None
    relative_performance: pd.Series = None

    # 메타 정보
    start_date: date = None
    end_date: date = None
    tickers: List[str] = field(default_factory=list)
    strategy_name: str = ""


# ============================================================================
# 성과 계산 함수
# ============================================================================

def calculate_cagr(initial_value: float, final_value: float, years: float) -> float:
    """
    CAGR (연평균 복합 성장률) 계산

    Args:
        initial_value: 초기 자산
        final_value: 최종 자산
        years: 투자 기간 (년)

    Returns:
        CAGR (예: 0.12 = 12%)
    """
    if initial_value <= 0 or final_value <= 0 or years <= 0:
        return 0.0

    return (final_value / initial_value) ** (1 / years) - 1


def calculate_mdd(equity_curve: pd.Series) -> float:
    """
    MDD (최대 낙폭) 계산

    Args:
        equity_curve: 자산 곡선

    Returns:
        MDD (음수, 예: -0.25 = -25%)
    """
    if equity_curve.empty:
        return 0.0

    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak

    return drawdown.min()


def calculate_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """
    드로우다운 시리즈 계산

    Args:
        equity_curve: 자산 곡선

    Returns:
        드로우다운 시리즈
    """
    peak = equity_curve.cummax()
    return (equity_curve - peak) / peak


def calculate_volatility(returns: pd.Series, annualize: bool = True,
                         trading_days: int = 252) -> float:
    """
    변동성 계산

    Args:
        returns: 수익률 시리즈
        annualize: 연율화 여부
        trading_days: 연간 거래일

    Returns:
        변동성 (예: 0.20 = 20%)
    """
    if returns.empty or len(returns) < 2:
        return 0.0

    vol = returns.std()

    if annualize:
        vol *= np.sqrt(trading_days)

    return vol


def calculate_downside_volatility(returns: pd.Series, threshold: float = 0,
                                   annualize: bool = True) -> float:
    """
    하방 변동성 (Downside Deviation) 계산

    Args:
        returns: 수익률 시리즈
        threshold: 임계 수익률 (기본: 0)
        annualize: 연율화 여부

    Returns:
        하방 변동성
    """
    if returns.empty:
        return 0.0

    downside = returns[returns < threshold]

    if len(downside) < 2:
        return 0.0

    downside_dev = np.sqrt(((downside - threshold) ** 2).mean())

    if annualize:
        downside_dev *= np.sqrt(252)

    return downside_dev


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                           trading_days: int = 252) -> float:
    """
    Sharpe Ratio 계산

    Args:
        returns: 일별 수익률 시리즈
        risk_free_rate: 무위험 수익률 (연율)
        trading_days: 연간 거래일

    Returns:
        Sharpe Ratio
    """
    if returns.empty or len(returns) < 2:
        return 0.0

    excess_return = returns.mean() * trading_days - risk_free_rate
    vol = returns.std() * np.sqrt(trading_days)

    if vol == 0:
        return 0.0

    return excess_return / vol


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                            trading_days: int = 252) -> float:
    """
    Sortino Ratio 계산

    Args:
        returns: 일별 수익률 시리즈
        risk_free_rate: 무위험 수익률 (연율)
        trading_days: 연간 거래일

    Returns:
        Sortino Ratio
    """
    if returns.empty or len(returns) < 2:
        return 0.0

    excess_return = returns.mean() * trading_days - risk_free_rate
    downside_vol = calculate_downside_volatility(returns, 0, True)

    if downside_vol == 0:
        return 0.0

    return excess_return / downside_vol


def calculate_calmar_ratio(cagr: float, mdd: float) -> float:
    """
    Calmar Ratio 계산

    Args:
        cagr: 연평균 수익률
        mdd: 최대 낙폭 (음수)

    Returns:
        Calmar Ratio
    """
    if mdd == 0:
        return 0.0

    return cagr / abs(mdd)


def calculate_monthly_returns(equity_curve: pd.Series) -> pd.Series:
    """
    월별 수익률 계산

    Args:
        equity_curve: 자산 곡선

    Returns:
        월별 수익률 시리즈
    """
    if equity_curve.empty:
        return pd.Series()

    # 월말 기준 리샘플링
    monthly = equity_curve.resample('M').last()
    returns = monthly.pct_change().dropna()

    return returns


def calculate_yearly_returns(equity_curve: pd.Series) -> pd.Series:
    """
    연도별 수익률 계산

    Args:
        equity_curve: 자산 곡선

    Returns:
        연도별 수익률 시리즈
    """
    if equity_curve.empty:
        return pd.Series()

    # 연말 기준 리샘플링
    yearly = equity_curve.resample('Y').last()
    returns = yearly.pct_change().dropna()

    return returns


def calculate_win_rate(monthly_returns: pd.Series) -> Tuple[int, int, float]:
    """
    승률 계산

    Args:
        monthly_returns: 월별 수익률

    Returns:
        (수익 월 수, 손실 월 수, 승률)
    """
    if monthly_returns.empty:
        return 0, 0, 0.0

    winning = (monthly_returns > 0).sum()
    losing = (monthly_returns < 0).sum()
    total = winning + losing

    if total == 0:
        return 0, 0, 0.0

    return int(winning), int(losing), winning / total


# ============================================================================
# 벤치마크 대비 분석
# ============================================================================

def calculate_alpha_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> Tuple[float, float]:
    """
    알파, 베타 계산 (CAPM 기반)

    Args:
        portfolio_returns: 포트폴리오 수익률
        benchmark_returns: 벤치마크 수익률

    Returns:
        (alpha, beta)
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        return 0.0, 0.0

    # 인덱스 맞추기
    aligned = pd.DataFrame({
        'portfolio': portfolio_returns,
        'benchmark': benchmark_returns
    }).dropna()

    if len(aligned) < 2:
        return 0.0, 0.0

    cov = aligned['portfolio'].cov(aligned['benchmark'])
    var = aligned['benchmark'].var()

    if var == 0:
        return 0.0, 0.0

    beta = cov / var
    alpha = aligned['portfolio'].mean() - beta * aligned['benchmark'].mean()

    # 연율화
    alpha = alpha * 252

    return alpha, beta


def calculate_tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    추적 오차 계산

    Args:
        portfolio_returns: 포트폴리오 수익률
        benchmark_returns: 벤치마크 수익률

    Returns:
        추적 오차 (연율화)
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        return 0.0

    # 인덱스 맞추기
    aligned = pd.DataFrame({
        'portfolio': portfolio_returns,
        'benchmark': benchmark_returns
    }).dropna()

    if len(aligned) < 2:
        return 0.0

    diff = aligned['portfolio'] - aligned['benchmark']
    return diff.std() * np.sqrt(252)


def calculate_information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    정보 비율 계산

    Args:
        portfolio_returns: 포트폴리오 수익률
        benchmark_returns: 벤치마크 수익률

    Returns:
        정보 비율
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        return 0.0

    alpha, _ = calculate_alpha_beta(portfolio_returns, benchmark_returns)
    tracking_err = calculate_tracking_error(portfolio_returns, benchmark_returns)

    if tracking_err == 0:
        return 0.0

    return alpha / tracking_err


# ============================================================================
# 종합 분석
# ============================================================================

def analyze_backtest(
    equity_curve: pd.Series,
    investment_history: pd.DataFrame,
    benchmark_curve: Optional[pd.Series] = None
) -> PerformanceMetrics:
    """
    백테스트 결과 종합 분석

    Args:
        equity_curve: 자산 곡선
        investment_history: 투자 내역 (date, amount 컬럼)
        benchmark_curve: 벤치마크 자산 곡선

    Returns:
        PerformanceMetrics 객체
    """
    metrics = PerformanceMetrics()

    if equity_curve.empty:
        return metrics

    # 기간 계산
    start_date = equity_curve.index[0]
    end_date = equity_curve.index[-1]
    metrics.total_days = (end_date - start_date).days
    metrics.total_years = metrics.total_days / 365.25

    # 일별 수익률
    daily_returns = equity_curve.pct_change().dropna()

    # 총 납입액
    if investment_history is not None and 'amount' in investment_history.columns:
        metrics.total_invested = investment_history['amount'].sum()
    else:
        metrics.total_invested = equity_curve.iloc[0]

    # 최종 자산
    metrics.final_value = equity_curve.iloc[-1]

    # 수익금
    metrics.profit = metrics.final_value - metrics.total_invested
    if metrics.total_invested > 0:
        metrics.profit_rate = metrics.profit / metrics.total_invested

    # 총 수익률
    if equity_curve.iloc[0] > 0:
        metrics.total_return = (metrics.final_value / equity_curve.iloc[0]) - 1

    # CAGR
    metrics.cagr = calculate_cagr(
        equity_curve.iloc[0],
        metrics.final_value,
        metrics.total_years
    )

    # MDD
    metrics.mdd = calculate_mdd(equity_curve)

    # 변동성
    metrics.volatility = calculate_volatility(daily_returns)
    metrics.downside_volatility = calculate_downside_volatility(daily_returns)

    # 위험조정 수익률
    metrics.sharpe_ratio = calculate_sharpe_ratio(daily_returns)
    metrics.sortino_ratio = calculate_sortino_ratio(daily_returns)
    metrics.calmar_ratio = calculate_calmar_ratio(metrics.cagr, metrics.mdd)

    # 월별 수익률 분석
    monthly_returns = calculate_monthly_returns(equity_curve)
    metrics.total_months = len(monthly_returns)
    if not monthly_returns.empty:
        metrics.monthly_return_avg = monthly_returns.mean()
        metrics.monthly_return_std = monthly_returns.std()
        win, lose, rate = calculate_win_rate(monthly_returns)
        metrics.winning_months = win
        metrics.losing_months = lose
        metrics.win_rate = rate

    # 벤치마크 대비 분석
    if benchmark_curve is not None and not benchmark_curve.empty:
        benchmark_returns = benchmark_curve.pct_change().dropna()
        metrics.alpha, metrics.beta = calculate_alpha_beta(daily_returns, benchmark_returns)
        metrics.tracking_error = calculate_tracking_error(daily_returns, benchmark_returns)
        metrics.information_ratio = calculate_information_ratio(daily_returns, benchmark_returns)

    return metrics


def create_backtest_result(
    equity_curve: pd.Series,
    investment_history: pd.DataFrame,
    benchmark_curve: Optional[pd.Series] = None,
    tickers: List[str] = None,
    strategy_name: str = ""
) -> BacktestResult:
    """
    백테스트 결과 객체 생성

    Args:
        equity_curve: 자산 곡선
        investment_history: 투자 내역
        benchmark_curve: 벤치마크 곡선
        tickers: 투자 티커 목록
        strategy_name: 전략 이름

    Returns:
        BacktestResult 객체
    """
    result = BacktestResult()

    result.equity_curve = equity_curve
    result.drawdown_curve = calculate_drawdown_series(equity_curve)
    result.investment_history = investment_history

    result.metrics = analyze_backtest(equity_curve, investment_history, benchmark_curve)

    result.monthly_returns = calculate_monthly_returns(equity_curve)
    result.yearly_returns = calculate_yearly_returns(equity_curve)

    result.benchmark_curve = benchmark_curve
    if benchmark_curve is not None and not benchmark_curve.empty:
        # 상대 성과: 포트폴리오/벤치마크 (1부터 시작하도록 정규화)
        aligned_port = equity_curve / equity_curve.iloc[0]
        aligned_bench = benchmark_curve.reindex(equity_curve.index, method='ffill')
        if not aligned_bench.empty:
            aligned_bench = aligned_bench / aligned_bench.iloc[0]
            result.relative_performance = aligned_port - aligned_bench

    if equity_curve.index.size > 0:
        result.start_date = equity_curve.index[0].date() if hasattr(equity_curve.index[0], 'date') else equity_curve.index[0]
        result.end_date = equity_curve.index[-1].date() if hasattr(equity_curve.index[-1], 'date') else equity_curve.index[-1]

    result.tickers = tickers or []
    result.strategy_name = strategy_name

    return result


# ============================================================================
# 히트맵 데이터 생성
# ============================================================================

def create_returns_heatmap_data(monthly_returns: pd.Series) -> pd.DataFrame:
    """
    월별 수익률 히트맵 데이터 생성

    Args:
        monthly_returns: 월별 수익률 시리즈

    Returns:
        행: 연도, 열: 월 인 DataFrame
    """
    if monthly_returns.empty:
        return pd.DataFrame()

    # 연도와 월 추출
    df = pd.DataFrame({'return': monthly_returns})
    df['year'] = df.index.year
    df['month'] = df.index.month

    # 피벗
    heatmap = df.pivot(index='year', columns='month', values='return')
    heatmap.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(heatmap.columns)]

    return heatmap


# ============================================================================
# 향후 확장 포인트
# ============================================================================

# TODO: VaR (Value at Risk)
# def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
#     """Value at Risk 계산"""
#     pass

# TODO: CVaR (Conditional VaR)
# def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
#     """Conditional VaR 계산"""
#     pass

# TODO: 팩터 분석
# def factor_analysis(returns: pd.Series, factors: pd.DataFrame) -> Dict:
#     """팩터 분석"""
#     pass

# TODO: 롤링 지표
# def calculate_rolling_metrics(equity_curve: pd.Series, window: int = 252):
#     """롤링 윈도우 기반 지표 계산"""
#     pass
