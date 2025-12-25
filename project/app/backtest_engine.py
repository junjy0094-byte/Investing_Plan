"""
백테스트 엔진 모듈

DCA 투자 백테스트의 핵심 로직을 구현합니다.

주요 기능:
- 월급날 매수 시뮬레이션
- 전략 기반 투자 결정
- 포지션 및 현금 관리
- 리밸런싱
- 수수료/슬리피지 처리

향후 확장 포인트:
- 병렬 백테스트
- 워크 어헤드 (walk-forward) 분석
- 몬테카를로 시뮬레이션
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Callable, Tuple
from enum import Enum

from app.strategies import BaseStrategy, StrategyContext, StrategyDecision, StrategyRegistry
from app.data_provider import DataProvider, data_provider, us_calendar
from app.indicators import sma, volatility, drawdown, compute_all_indicators
from app.analytics import BacktestResult, create_backtest_result, PerformanceMetrics
from app.utils import logger, safe_divide


# ============================================================================
# 설정 및 열거형
# ============================================================================

class HolidayAdjustment(Enum):
    """휴장일 조정 방식"""
    PREVIOUS = "previous"  # 직전 거래일
    NEXT = "next"          # 직후 거래일


class RebalanceFrequency(Enum):
    """리밸런싱 주기"""
    NONE = "none"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


# ============================================================================
# 백테스트 설정
# ============================================================================

@dataclass
class BacktestConfig:
    """
    백테스트 설정을 담는 데이터 클래스
    """
    # 기간
    start_date: date = None
    end_date: date = None

    # 종목 및 비중
    tickers: List[str] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)  # 0~1 사이 값

    # 투자 규칙
    payday: int = 25  # 월급날 (1~28)
    monthly_investment: float = 4000.0  # 월 투자금 (USD)
    annual_increase_rate: float = 0.0  # 연간 투자금 증가율 (0~1)

    # 비용
    commission_rate: float = 0.001  # 수수료율 (0.1%)
    slippage_rate: float = 0.001    # 슬리피지율 (0.1%)

    # 휴장일 처리
    holiday_adjustment: HolidayAdjustment = HolidayAdjustment.PREVIOUS

    # 리밸런싱
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.NONE

    # 현금 관리
    reserve_cash: bool = True  # 매수 보류 시 현금 적립
    cash_interest_rate: float = 0.0  # 현금 이자율 (연율)

    # 벤치마크
    benchmark: str = "SPY"

    # 전략
    strategy_name: str = "Pure DCA"
    strategy_params: Dict = field(default_factory=dict)

    def validate(self) -> Tuple[bool, List[str]]:
        """설정 유효성 검증"""
        errors = []

        if not self.tickers:
            errors.append("최소 1개 이상의 티커를 입력하세요.")

        if len(self.weights) != len(self.tickers):
            errors.append("비중 개수가 티커 개수와 일치하지 않습니다.")
        elif abs(sum(self.weights) - 1.0) > 0.01:
            errors.append(f"비중 합이 100%가 아닙니다: {sum(self.weights)*100:.1f}%")

        if self.start_date >= self.end_date:
            errors.append("시작일이 종료일 이후입니다.")

        if self.payday < 1 or self.payday > 28:
            errors.append("월급날은 1~28 사이여야 합니다.")

        if self.monthly_investment <= 0:
            errors.append("월 투자금은 0보다 커야 합니다.")

        return len(errors) == 0, errors


# ============================================================================
# 포지션 및 거래 기록
# ============================================================================

@dataclass
class Position:
    """포지션 정보"""
    shares: float = 0.0
    avg_cost: float = 0.0
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return self.shares * (self.current_price - self.avg_cost)


@dataclass
class Trade:
    """거래 기록"""
    date: date
    ticker: str
    action: str  # "BUY" or "SELL"
    shares: float
    price: float
    amount: float
    commission: float
    reason: str = ""


# ============================================================================
# 백테스트 상태
# ============================================================================

@dataclass
class BacktestState:
    """백테스트 진행 상태"""
    current_date: date = None
    cash: float = 0.0
    reserved_cash: float = 0.0  # 보류된 현금
    positions: Dict[str, Position] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)

    # 포트폴리오 가치 추적
    portfolio_peak: float = 0.0
    portfolio_values: List[float] = field(default_factory=list)
    portfolio_dates: List[date] = field(default_factory=list)

    # 투자 내역
    investment_dates: List[date] = field(default_factory=list)
    investment_amounts: List[float] = field(default_factory=list)

    # 현재 연도 투자금 (연간 증가 적용)
    current_year_investment: float = 0.0
    last_investment_year: int = 0

    @property
    def total_value(self) -> float:
        """총 포트폴리오 가치"""
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + self.reserved_cash + positions_value


# ============================================================================
# 백테스트 엔진
# ============================================================================

class BacktestEngine:
    """
    DCA 백테스트 엔진

    월급날 적립식 투자를 시뮬레이션합니다.

    향후 확장 포인트:
    - 병렬 백테스트 (multiprocessing)
    - 실시간 모니터링
    - 중단/재개 기능
    """

    def __init__(self, config: BacktestConfig,
                 data_provider: DataProvider = data_provider):
        """
        엔진 초기화

        Args:
            config: 백테스트 설정
            data_provider: 데이터 공급자
        """
        self.config = config
        self.data_provider = data_provider
        self.strategy: Optional[BaseStrategy] = None
        self.state: Optional[BacktestState] = None
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.benchmark_data: Optional[pd.DataFrame] = None

        # 콜백
        self.progress_callback: Optional[Callable[[int, int, str], None]] = None

    def _initialize_strategy(self) -> None:
        """전략 초기화"""
        self.strategy = StrategyRegistry.create(
            self.config.strategy_name,
            **self.config.strategy_params
        )

        if self.strategy is None:
            logger.warning(f"전략 '{self.config.strategy_name}'을 찾을 수 없습니다. Pure DCA 사용.")
            self.strategy = StrategyRegistry.create("Pure DCA")

    def _load_data(self) -> bool:
        """
        가격 데이터 로드

        Returns:
            성공 여부
        """
        logger.info("가격 데이터 로딩 중...")

        # 지표 계산을 위해 여유 기간 추가
        buffer_start = self.config.start_date - timedelta(days=365)

        # 티커 데이터 로드
        all_tickers = self.config.tickers.copy()
        if self.config.benchmark and self.config.benchmark not in all_tickers:
            all_tickers.append(self.config.benchmark)

        for i, ticker in enumerate(all_tickers):
            if self.progress_callback:
                self.progress_callback(
                    i + 1, len(all_tickers),
                    f"{ticker} 데이터 로딩..."
                )

            data = self.data_provider.get_price_data(
                ticker, buffer_start, self.config.end_date
            )

            if data is None or data.empty:
                logger.error(f"{ticker} 데이터를 가져올 수 없습니다.")
                return False

            # 지표 추가
            data = compute_all_indicators(data, ticker)

            if ticker == self.config.benchmark:
                self.benchmark_data = data
            else:
                self.price_data[ticker] = data

        logger.info(f"{len(self.price_data)}개 티커 데이터 로드 완료")
        return True

    def _initialize_state(self) -> None:
        """백테스트 상태 초기화"""
        self.state = BacktestState()
        self.state.current_date = self.config.start_date
        self.state.cash = 0.0
        self.state.reserved_cash = 0.0
        self.state.current_year_investment = self.config.monthly_investment
        self.state.last_investment_year = self.config.start_date.year

        # 포지션 초기화
        for ticker in self.config.tickers:
            self.state.positions[ticker] = Position()

    def _get_current_prices(self) -> Dict[str, float]:
        """현재 가격 조회"""
        prices = {}
        current_ts = pd.Timestamp(self.state.current_date)

        for ticker, data in self.price_data.items():
            # 현재 날짜 이하의 가장 최근 가격
            mask = data.index <= current_ts
            if mask.any():
                close_col = 'Close' if 'Close' in data.columns else 'Adj Close'
                prices[ticker] = data.loc[mask, close_col].iloc[-1]

        return prices

    def _update_positions_prices(self, prices: Dict[str, float]) -> None:
        """포지션의 현재 가격 업데이트"""
        for ticker, price in prices.items():
            if ticker in self.state.positions:
                self.state.positions[ticker].current_price = price

    def _get_price_history(self) -> Dict[str, pd.Series]:
        """현재 날짜까지의 가격 히스토리"""
        history = {}
        current_ts = pd.Timestamp(self.state.current_date)

        for ticker, data in self.price_data.items():
            mask = data.index <= current_ts
            if mask.any():
                close_col = 'Close' if 'Close' in data.columns else 'Adj Close'
                history[ticker] = data.loc[mask, close_col]

        return history

    def _get_indicators(self) -> Dict[str, Dict[str, float]]:
        """현재 날짜의 지표값"""
        indicators = {}
        current_ts = pd.Timestamp(self.state.current_date)

        for ticker, data in self.price_data.items():
            mask = data.index <= current_ts
            if mask.any():
                row = data.loc[mask].iloc[-1]
                indicators[ticker] = {
                    'SMA_20': row.get('SMA_20', np.nan),
                    'SMA_50': row.get('SMA_50', np.nan),
                    'SMA_200': row.get('SMA_200', np.nan),
                    'Volatility_20': row.get('Volatility_20', np.nan),
                    'Drawdown': row.get('Drawdown', np.nan),
                }

        return indicators

    def _is_payday(self, d: date) -> bool:
        """월급날 여부 확인"""
        target_day = self.config.payday

        # 해당 월의 목표 날짜
        try:
            target_date = date(d.year, d.month, target_day)
        except ValueError:
            # 28일 이상인 경우 (예: 2월)
            import calendar
            last_day = calendar.monthrange(d.year, d.month)[1]
            target_date = date(d.year, d.month, min(target_day, last_day))

        # 휴장일 조정
        adjusted = self.data_provider.adjust_to_trading_day(
            target_date,
            self.config.holiday_adjustment.value
        )

        return d == adjusted

    def _get_investment_amount(self) -> float:
        """현재 투자금 계산 (연간 증가 적용)"""
        current_year = self.state.current_date.year

        if current_year > self.state.last_investment_year:
            # 연간 증가 적용
            years_diff = current_year - self.state.last_investment_year
            increase_factor = (1 + self.config.annual_increase_rate) ** years_diff
            self.state.current_year_investment *= increase_factor
            self.state.last_investment_year = current_year

        return self.state.current_year_investment

    def _calculate_portfolio_drawdown(self) -> float:
        """포트폴리오 드로우다운 계산"""
        current_value = self.state.total_value

        if current_value > self.state.portfolio_peak:
            self.state.portfolio_peak = current_value

        if self.state.portfolio_peak > 0:
            return (current_value - self.state.portfolio_peak) / self.state.portfolio_peak
        return 0.0

    def _build_strategy_context(self, is_payday: bool) -> StrategyContext:
        """전략 컨텍스트 생성"""
        prices = self._get_current_prices()
        self._update_positions_prices(prices)

        return StrategyContext(
            current_date=self.state.current_date,
            current_prices=prices,
            price_history=self._get_price_history(),
            positions={t: p.shares for t, p in self.state.positions.items()},
            cash=self.state.cash,
            total_value=self.state.total_value,
            portfolio_peak=self.state.portfolio_peak,
            portfolio_drawdown=self._calculate_portfolio_drawdown(),
            indicators=self._get_indicators(),
            base_investment=self._get_investment_amount(),
            is_payday=is_payday,
            year=self.state.current_date.year,
            month=self.state.current_date.month,
            reserved_cash=self.state.reserved_cash
        )

    def _execute_buy(self, ticker: str, amount: float, reason: str = "") -> Optional[Trade]:
        """
        매수 실행

        Args:
            ticker: 티커
            amount: 투자금
            reason: 매수 사유

        Returns:
            Trade 객체 또는 None
        """
        if amount <= 0:
            return None

        current_price = self.state.positions[ticker].current_price
        if current_price <= 0:
            return None

        # 수수료 및 슬리피지 계산
        total_cost_rate = self.config.commission_rate + self.config.slippage_rate
        effective_price = current_price * (1 + self.config.slippage_rate)
        commission = amount * self.config.commission_rate

        # 실제 매수 금액 (수수료 제외)
        buy_amount = amount - commission
        shares = buy_amount / effective_price

        if shares <= 0:
            return None

        # 포지션 업데이트
        position = self.state.positions[ticker]
        old_value = position.shares * position.avg_cost
        new_value = old_value + buy_amount
        position.shares += shares

        if position.shares > 0:
            position.avg_cost = new_value / position.shares

        # 거래 기록
        trade = Trade(
            date=self.state.current_date,
            ticker=ticker,
            action="BUY",
            shares=shares,
            price=effective_price,
            amount=buy_amount,
            commission=commission,
            reason=reason
        )
        self.state.trades.append(trade)

        return trade

    def _execute_investment(self, total_amount: float, decision: StrategyDecision) -> None:
        """
        투자 실행

        Args:
            total_amount: 총 투자금
            decision: 전략 결정
        """
        if total_amount <= 0:
            return

        # 비중에 따라 분배
        for i, ticker in enumerate(self.config.tickers):
            weight = self.config.weights[i]
            ticker_amount = total_amount * weight

            if ticker_amount > 0:
                self._execute_buy(ticker, ticker_amount, decision.reason)

        # 투자 기록
        self.state.investment_dates.append(self.state.current_date)
        self.state.investment_amounts.append(total_amount)

    def _apply_cash_interest(self) -> None:
        """현금 이자 적용 (일별)"""
        if self.config.cash_interest_rate > 0 and self.state.reserved_cash > 0:
            daily_rate = self.config.cash_interest_rate / 365
            interest = self.state.reserved_cash * daily_rate
            self.state.reserved_cash += interest

    def _check_rebalancing(self) -> bool:
        """리밸런싱 필요 여부 확인"""
        if self.config.rebalance_frequency == RebalanceFrequency.NONE:
            return False

        d = self.state.current_date

        if self.config.rebalance_frequency == RebalanceFrequency.MONTHLY:
            # 매월 첫 거래일
            return d.day <= 7 and d == self.data_provider.adjust_to_trading_day(
                date(d.year, d.month, 1), "next"
            )

        elif self.config.rebalance_frequency == RebalanceFrequency.QUARTERLY:
            # 분기 첫 거래일 (1, 4, 7, 10월)
            if d.month in [1, 4, 7, 10]:
                return d == self.data_provider.adjust_to_trading_day(
                    date(d.year, d.month, 1), "next"
                )

        elif self.config.rebalance_frequency == RebalanceFrequency.YEARLY:
            # 연초 첫 거래일
            if d.month == 1:
                return d == self.data_provider.adjust_to_trading_day(
                    date(d.year, 1, 1), "next"
                )

        return False

    def _execute_rebalancing(self) -> None:
        """리밸런싱 실행"""
        total_value = sum(p.market_value for p in self.state.positions.values())

        if total_value <= 0:
            return

        logger.info(f"[{self.state.current_date}] 리밸런싱 실행")

        # 목표 금액 계산
        target_values = {}
        for i, ticker in enumerate(self.config.tickers):
            target_values[ticker] = total_value * self.config.weights[i]

        # 현재와 목표의 차이 계산 및 조정
        # 간단한 구현: 전체 매도 후 재매수 (실제로는 최소 거래 방식 권장)
        # 향후 확장: 최소 거래 리밸런싱 알고리즘

        for ticker in self.config.tickers:
            position = self.state.positions[ticker]
            current_value = position.market_value
            target_value = target_values[ticker]
            diff = target_value - current_value

            if abs(diff) > total_value * 0.01:  # 1% 이상 차이 시만
                if diff > 0:
                    # 추가 매수
                    self._execute_buy(ticker, diff, "리밸런싱")
                # 매도는 현재 구현 생략 (DCA 특성상 주로 매수만)

    def _process_day(self) -> None:
        """하루 처리"""
        d = self.state.current_date

        # 거래일인지 확인
        if not self.data_provider.is_trading_day(d):
            return

        # 가격 업데이트
        prices = self._get_current_prices()
        if not prices:
            return

        self._update_positions_prices(prices)

        # 현금 이자 적용
        self._apply_cash_interest()

        # 월급날 체크
        is_payday = self._is_payday(d)

        if is_payday:
            # 투자금 준비
            monthly_amount = self._get_investment_amount()
            self.state.cash += monthly_amount

            # 전략 컨텍스트 생성
            context = self._build_strategy_context(True)

            # 전략 결정
            decision = self.strategy.decide(context)

            logger.debug(f"[{d}] 월급날 - {decision.reason}")

            if decision.should_invest:
                # 투자 실행
                invest_amount = self.state.cash * decision.multiplier

                # 보류금 투입
                if decision.deploy_reserved and self.state.reserved_cash > 0:
                    invest_amount += self.state.reserved_cash
                    self.state.reserved_cash = 0
                    logger.debug(f"[{d}] 보류금 투입")

                # 현금 한도 내에서만
                invest_amount = min(invest_amount, self.state.cash + self.state.reserved_cash)

                self._execute_investment(invest_amount, decision)
                self.state.cash = max(0, self.state.cash - invest_amount)

            elif decision.reserve_cash:
                # 현금 보류
                self.state.reserved_cash += self.state.cash
                self.state.cash = 0
                logger.debug(f"[{d}] 현금 보류: ${self.state.reserved_cash:.2f}")

        # 리밸런싱 체크
        if self._check_rebalancing():
            self._execute_rebalancing()

        # 포트폴리오 가치 기록
        self.state.portfolio_values.append(self.state.total_value)
        self.state.portfolio_dates.append(d)

    def run(self, progress_callback: Optional[Callable[[int, int, str], None]] = None
            ) -> Optional[BacktestResult]:
        """
        백테스트 실행

        Args:
            progress_callback: 진행 상황 콜백 (current, total, message)

        Returns:
            BacktestResult 객체
        """
        self.progress_callback = progress_callback

        # 설정 검증
        is_valid, errors = self.config.validate()
        if not is_valid:
            for error in errors:
                logger.error(error)
            return None

        # 초기화
        self._initialize_strategy()
        self._initialize_state()

        # 데이터 로드
        if not self._load_data():
            return None

        # 거래일 목록
        trading_days = self.data_provider.get_trading_days(
            self.config.start_date,
            self.config.end_date
        )

        if not trading_days:
            logger.error("거래일이 없습니다.")
            return None

        logger.info(f"백테스트 시작: {self.config.start_date} ~ {self.config.end_date}")
        logger.info(f"전략: {self.config.strategy_name}")
        logger.info(f"종목: {', '.join(self.config.tickers)}")

        total_days = len(trading_days)

        # 일별 시뮬레이션
        for i, d in enumerate(trading_days):
            self.state.current_date = d
            self._process_day()

            if progress_callback and i % 20 == 0:
                progress_callback(
                    i + 1, total_days,
                    f"시뮬레이션 중... {d.strftime('%Y-%m-%d')}"
                )

        logger.info("백테스트 완료")

        # 결과 생성
        return self._create_result()

    def _create_result(self) -> BacktestResult:
        """백테스트 결과 생성"""
        # 자산 곡선
        equity_curve = pd.Series(
            self.state.portfolio_values,
            index=pd.DatetimeIndex(self.state.portfolio_dates)
        )

        # 투자 내역
        investment_history = pd.DataFrame({
            'date': self.state.investment_dates,
            'amount': self.state.investment_amounts
        })

        # 벤치마크 곡선 (투자금 비례)
        benchmark_curve = None
        if self.benchmark_data is not None and not investment_history.empty:
            # 간단한 벤치마크: SPY에 동일 금액 투자했다고 가정
            benchmark_curve = self._calculate_benchmark_curve(investment_history)

        return create_backtest_result(
            equity_curve=equity_curve,
            investment_history=investment_history,
            benchmark_curve=benchmark_curve,
            tickers=self.config.tickers,
            strategy_name=self.config.strategy_name
        )

    def _calculate_benchmark_curve(self, investment_history: pd.DataFrame) -> pd.Series:
        """벤치마크 자산 곡선 계산"""
        if self.benchmark_data is None:
            return None

        close_col = 'Close' if 'Close' in self.benchmark_data.columns else 'Adj Close'
        benchmark_prices = self.benchmark_data[close_col]

        # 투자 시점마다 벤치마크 주식 매수 가정
        shares = 0.0
        benchmark_values = []
        benchmark_dates = []

        investment_dict = dict(zip(
            [pd.Timestamp(d) for d in investment_history['date']],
            investment_history['amount']
        ))

        for d in self.state.portfolio_dates:
            ts = pd.Timestamp(d)

            # 투자일이면 주식 매수
            if ts in investment_dict:
                amount = investment_dict[ts]
                price_mask = benchmark_prices.index <= ts
                if price_mask.any():
                    price = benchmark_prices[price_mask].iloc[-1]
                    shares += amount / price

            # 현재 가치 계산
            price_mask = benchmark_prices.index <= ts
            if price_mask.any():
                current_price = benchmark_prices[price_mask].iloc[-1]
                benchmark_values.append(shares * current_price)
            else:
                benchmark_values.append(0)

            benchmark_dates.append(d)

        return pd.Series(
            benchmark_values,
            index=pd.DatetimeIndex(benchmark_dates)
        )


# ============================================================================
# 편의 함수
# ============================================================================

def run_backtest(
    tickers: List[str],
    weights: List[float],
    start_date: date,
    end_date: date,
    monthly_investment: float = 4000.0,
    payday: int = 25,
    strategy_name: str = "Pure DCA",
    strategy_params: Dict = None,
    progress_callback: Callable = None
) -> Optional[BacktestResult]:
    """
    백테스트 실행 편의 함수

    Args:
        tickers: 티커 리스트
        weights: 비중 리스트 (합이 1.0)
        start_date: 시작일
        end_date: 종료일
        monthly_investment: 월 투자금
        payday: 월급날
        strategy_name: 전략 이름
        strategy_params: 전략 파라미터
        progress_callback: 진행 콜백

    Returns:
        BacktestResult 객체
    """
    config = BacktestConfig(
        tickers=tickers,
        weights=weights,
        start_date=start_date,
        end_date=end_date,
        monthly_investment=monthly_investment,
        payday=payday,
        strategy_name=strategy_name,
        strategy_params=strategy_params or {}
    )

    engine = BacktestEngine(config)
    return engine.run(progress_callback)


def compare_strategies(
    tickers: List[str],
    weights: List[float],
    start_date: date,
    end_date: date,
    strategies: List[Tuple[str, Dict]],  # [(name, params), ...]
    monthly_investment: float = 4000.0,
    progress_callback: Callable = None
) -> Dict[str, BacktestResult]:
    """
    여러 전략 비교 백테스트

    Args:
        tickers: 티커 리스트
        weights: 비중 리스트
        start_date: 시작일
        end_date: 종료일
        strategies: (전략이름, 파라미터) 튜플 리스트
        monthly_investment: 월 투자금
        progress_callback: 진행 콜백

    Returns:
        {전략이름: BacktestResult} 딕셔너리
    """
    results = {}

    for i, (strategy_name, strategy_params) in enumerate(strategies):
        if progress_callback:
            progress_callback(i + 1, len(strategies), f"전략 '{strategy_name}' 실행 중...")

        result = run_backtest(
            tickers=tickers,
            weights=weights,
            start_date=start_date,
            end_date=end_date,
            monthly_investment=monthly_investment,
            strategy_name=strategy_name,
            strategy_params=strategy_params
        )

        if result:
            results[strategy_name] = result

    return results


# ============================================================================
# 향후 확장 포인트
# ============================================================================

# TODO: 병렬 백테스트
# def run_parallel_backtest(configs: List[BacktestConfig], n_jobs: int = -1):
#     """여러 설정을 병렬로 백테스트"""
#     pass

# TODO: 워크 어헤드 분석
# def walk_forward_analysis(config: BacktestConfig, train_period: int, test_period: int):
#     """워크 어헤드 분석"""
#     pass

# TODO: 몬테카를로 시뮬레이션
# def monte_carlo_simulation(config: BacktestConfig, n_simulations: int = 1000):
#     """몬테카를로 시뮬레이션"""
#     pass

# TODO: 파라미터 최적화
# def optimize_parameters(config: BacktestConfig, param_ranges: Dict):
#     """전략 파라미터 최적화"""
#     pass
