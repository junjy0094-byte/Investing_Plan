"""
기술적 지표 모듈

이동평균, 변동성, 드로우다운 등의 기술적 지표를 계산합니다.

향후 확장 포인트:
- RSI, MACD, Bollinger Bands 등 추가 지표
- 지표 플러그인 시스템
- 지표 조합 (composite indicators)
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple
from dataclasses import dataclass


# ============================================================================
# 이동평균 지표
# ============================================================================

def sma(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    단순 이동평균 (Simple Moving Average)

    Args:
        prices: 가격 시리즈
        period: 기간 (기본: 20일)

    Returns:
        SMA 시리즈
    """
    return prices.rolling(window=period, min_periods=1).mean()


def ema(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    지수 이동평균 (Exponential Moving Average)

    Args:
        prices: 가격 시리즈
        period: 기간 (기본: 20일)

    Returns:
        EMA 시리즈
    """
    return prices.ewm(span=period, adjust=False, min_periods=1).mean()


def wma(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    가중 이동평균 (Weighted Moving Average)

    Args:
        prices: 가격 시리즈
        period: 기간

    Returns:
        WMA 시리즈
    """
    weights = np.arange(1, period + 1)

    def weighted_avg(x):
        if len(x) < period:
            w = np.arange(1, len(x) + 1)
            return np.dot(x, w) / w.sum()
        return np.dot(x, weights) / weights.sum()

    return prices.rolling(window=period, min_periods=1).apply(weighted_avg, raw=True)


# ============================================================================
# 변동성 지표
# ============================================================================

def volatility(prices: pd.Series, period: int = 20,
               annualize: bool = True, trading_days: int = 252) -> pd.Series:
    """
    변동성 (표준편차 기반)

    Args:
        prices: 가격 시리즈
        period: 기간 (기본: 20일)
        annualize: 연율화 여부 (기본: True)
        trading_days: 연간 거래일 수 (기본: 252)

    Returns:
        변동성 시리즈
    """
    returns = prices.pct_change()
    vol = returns.rolling(window=period, min_periods=2).std()

    if annualize:
        vol = vol * np.sqrt(trading_days)

    return vol


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """
    평균 진정 범위 (Average True Range)

    Args:
        high: 고가 시리즈
        low: 저가 시리즈
        close: 종가 시리즈
        period: 기간 (기본: 14일)

    Returns:
        ATR 시리즈
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return true_range.rolling(window=period, min_periods=1).mean()


def realized_volatility(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    실현 변동성 (로그 수익률 기반)

    Args:
        prices: 가격 시리즈
        period: 기간

    Returns:
        실현 변동성 시리즈
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.rolling(window=period, min_periods=2).std() * np.sqrt(252)


# ============================================================================
# 드로우다운 관련
# ============================================================================

def rolling_max(prices: pd.Series) -> pd.Series:
    """
    누적 최고점 (Rolling Maximum)

    Args:
        prices: 가격 시리즈

    Returns:
        누적 최고점 시리즈
    """
    return prices.cummax()


def drawdown(prices: pd.Series) -> pd.Series:
    """
    드로우다운 비율

    Args:
        prices: 가격 시리즈

    Returns:
        드로우다운 시리즈 (음수 값, 예: -0.1 = -10%)
    """
    peak = rolling_max(prices)
    dd = (prices - peak) / peak
    return dd


def drawdown_duration(prices: pd.Series) -> pd.Series:
    """
    드로우다운 지속 기간 (거래일 수)

    Args:
        prices: 가격 시리즈

    Returns:
        드로우다운 지속 기간 시리즈
    """
    peak = rolling_max(prices)
    is_drawdown = prices < peak

    duration = pd.Series(0, index=prices.index)
    count = 0

    for i in range(len(prices)):
        if is_drawdown.iloc[i]:
            count += 1
        else:
            count = 0
        duration.iloc[i] = count

    return duration


@dataclass
class DrawdownInfo:
    """드로우다운 정보를 담는 데이터 클래스"""
    current_drawdown: float      # 현재 드로우다운 비율
    max_drawdown: float          # 최대 드로우다운
    peak_value: float            # 고점 가격
    current_value: float         # 현재 가격
    drawdown_duration: int       # 현재 드로우다운 지속 기간
    recovery_needed: float       # 원금 회복에 필요한 수익률


def get_drawdown_info(prices: pd.Series) -> DrawdownInfo:
    """
    현재 드로우다운 상세 정보

    Args:
        prices: 가격 시리즈

    Returns:
        DrawdownInfo 객체
    """
    dd = drawdown(prices)
    peak = rolling_max(prices)
    duration = drawdown_duration(prices)

    current_dd = dd.iloc[-1]
    mdd = dd.min()
    peak_val = peak.iloc[-1]
    current_val = prices.iloc[-1]
    current_duration = int(duration.iloc[-1])

    # 원금 회복에 필요한 수익률 계산
    if current_dd < 0:
        recovery = (peak_val - current_val) / current_val
    else:
        recovery = 0.0

    return DrawdownInfo(
        current_drawdown=current_dd,
        max_drawdown=mdd,
        peak_value=peak_val,
        current_value=current_val,
        drawdown_duration=current_duration,
        recovery_needed=recovery
    )


# ============================================================================
# 추세 지표
# ============================================================================

def trend_direction(prices: pd.Series, ma_period: int = 200) -> pd.Series:
    """
    추세 방향 (이동평균 대비)

    Args:
        prices: 가격 시리즈
        ma_period: 이동평균 기간 (기본: 200일)

    Returns:
        추세 방향 시리즈 (1: 상승추세, -1: 하락추세, 0: 횡보)
    """
    ma = sma(prices, ma_period)
    diff = (prices - ma) / ma

    direction = pd.Series(0, index=prices.index)
    direction[diff > 0.02] = 1   # 2% 이상 위 = 상승추세
    direction[diff < -0.02] = -1  # 2% 이상 아래 = 하락추세

    return direction


def price_momentum(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    가격 모멘텀 (N일 수익률)

    Args:
        prices: 가격 시리즈
        period: 기간

    Returns:
        모멘텀 시리즈
    """
    return prices.pct_change(periods=period)


def rate_of_change(prices: pd.Series, period: int = 10) -> pd.Series:
    """
    변화율 (Rate of Change)

    Args:
        prices: 가격 시리즈
        period: 기간

    Returns:
        ROC 시리즈
    """
    return ((prices - prices.shift(period)) / prices.shift(period)) * 100


# ============================================================================
# 지표 캐시 매니저
# ============================================================================

class IndicatorCache:
    """
    지표 계산 결과 캐시

    동일한 데이터에 대해 지표를 반복 계산하는 것을 방지합니다.

    향후 확장 포인트:
    - LRU 캐시 구현
    - 디스크 캐시 지원
    - 캐시 무효화 전략
    """

    def __init__(self):
        self._cache = {}

    def get_key(self, indicator_name: str, ticker: str, period: int) -> str:
        """캐시 키 생성"""
        return f"{indicator_name}_{ticker}_{period}"

    def get(self, key: str) -> Optional[pd.Series]:
        """캐시에서 지표 조회"""
        return self._cache.get(key)

    def set(self, key: str, value: pd.Series) -> None:
        """캐시에 지표 저장"""
        self._cache[key] = value

    def clear(self) -> None:
        """캐시 초기화"""
        self._cache.clear()

    def compute_if_absent(self, key: str, compute_fn) -> pd.Series:
        """
        캐시에 없으면 계산 후 저장

        Args:
            key: 캐시 키
            compute_fn: 계산 함수 (lambda)

        Returns:
            지표 시리즈
        """
        if key not in self._cache:
            self._cache[key] = compute_fn()
        return self._cache[key]


# 전역 캐시 인스턴스
indicator_cache = IndicatorCache()


# ============================================================================
# 지표 계산 헬퍼 함수
# ============================================================================

def compute_all_indicators(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    주어진 가격 데이터에 대해 모든 주요 지표 계산

    Args:
        prices: OHLCV 데이터프레임
        ticker: 티커 심볼

    Returns:
        지표가 추가된 데이터프레임
    """
    close = prices['Close'] if 'Close' in prices.columns else prices['Adj Close']

    result = prices.copy()

    # 이동평균
    result['SMA_20'] = sma(close, 20)
    result['SMA_50'] = sma(close, 50)
    result['SMA_200'] = sma(close, 200)
    result['EMA_20'] = ema(close, 20)

    # 변동성
    result['Volatility_20'] = volatility(close, 20)

    # 드로우다운
    result['Peak'] = rolling_max(close)
    result['Drawdown'] = drawdown(close)

    # 추세
    result['Trend'] = trend_direction(close, 200)
    result['Momentum_20'] = price_momentum(close, 20)

    return result


# ============================================================================
# 향후 확장 포인트
# ============================================================================

# TODO: RSI (Relative Strength Index)
# def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
#     """상대강도지수"""
#     pass

# TODO: MACD
# def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
#     """MACD 지표"""
#     pass

# TODO: Bollinger Bands
# def bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0):
#     """볼린저 밴드"""
#     pass

# TODO: Stochastic Oscillator
# def stochastic(high, low, close, k_period=14, d_period=3):
#     """스토캐스틱 오실레이터"""
#     pass

# TODO: 지표 팩토리 (동적 지표 생성)
# class IndicatorFactory:
#     """지표 팩토리 - 문자열로 지표 생성"""
#     @staticmethod
#     def create(name: str, params: dict):
#         pass
