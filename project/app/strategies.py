"""
전략 모듈

DCA 투자 전략을 정의하는 클래스들을 포함합니다.

기본 제공 전략:
1. Pure DCA: 무조건 매월 정액 매수
2. Drawdown Tier DCA: 하락률 구간별 투자금 배수
3. Trend Filter: 이동평균선 기반 매수 조절
4. Volatility Control: 변동성 기반 투자금 조절

향후 확장 포인트:
- 커스텀 전략 플러그인 시스템
- 전략 조합 (Composite Strategy)
- 머신러닝 기반 전략
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Type
from datetime import date
import pandas as pd
import numpy as np

from app.indicators import sma, volatility, drawdown, get_drawdown_info


# ============================================================================
# 전략 컨텍스트
# ============================================================================

@dataclass
class StrategyContext:
    """
    전략 결정에 필요한 컨텍스트 정보

    백테스트 엔진이 매 거래일마다 이 컨텍스트를 생성하여 전략에 전달합니다.
    """
    # 현재 상태
    current_date: date
    current_prices: Dict[str, float]  # {ticker: price}

    # 가격 히스토리 (지표 계산용)
    price_history: Dict[str, pd.Series]  # {ticker: Series}

    # 포지션 정보
    positions: Dict[str, float]  # {ticker: shares}
    cash: float
    total_value: float

    # 드로우다운 정보
    portfolio_peak: float
    portfolio_drawdown: float  # 음수 값 (예: -0.15 = -15%)

    # 지표값
    indicators: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # 예: {"QQQ": {"SMA_200": 350.5, "Volatility_20": 0.25}}

    # 투자 규칙
    base_investment: float = 0.0  # 기본 월 투자금
    is_payday: bool = False  # 월급날 여부

    # 기타
    year: int = 0
    month: int = 0
    reserved_cash: float = 0.0  # 보류된 현금


@dataclass
class StrategyDecision:
    """
    전략의 결정 결과

    전략이 매수 여부와 투자금 배수를 결정합니다.
    """
    should_invest: bool = True
    multiplier: float = 1.0  # 투자금 배수 (예: 1.5 = 150% 투자)
    reason: str = ""  # 결정 이유 (로깅용)
    reserve_cash: bool = False  # 현금 보류 여부
    deploy_reserved: bool = False  # 보류된 현금 투입 여부


# ============================================================================
# 기본 전략 클래스
# ============================================================================

class BaseStrategy(ABC):
    """
    전략 기본 클래스

    모든 전략은 이 클래스를 상속받아 구현합니다.

    향후 확장 포인트:
    - validate() 메서드로 파라미터 검증
    - backtest_result 분석 메서드
    """

    # 전략 메타정보 (서브클래스에서 오버라이드)
    name: str = "Base Strategy"
    description: str = "기본 전략 클래스"

    # 파라미터 정의 (GUI에서 동적 생성)
    # 형식: {"param_name": {"type": "float", "default": 1.0, "min": 0, "max": 10, "label": "표시명"}}
    parameters: Dict[str, Dict[str, Any]] = {}

    def __init__(self, **params):
        """
        전략 초기화

        Args:
            **params: 전략 파라미터
        """
        self.params = {}

        # 기본값으로 초기화
        for name, config in self.parameters.items():
            self.params[name] = config.get('default', 0)

        # 전달된 파라미터로 업데이트
        self.params.update(params)

    @abstractmethod
    def decide(self, context: StrategyContext) -> StrategyDecision:
        """
        투자 결정 수행

        Args:
            context: 전략 컨텍스트

        Returns:
            StrategyDecision 객체
        """
        pass

    def get_param(self, name: str, default: Any = None) -> Any:
        """파라미터 값 조회"""
        return self.params.get(name, default)

    def set_param(self, name: str, value: Any) -> None:
        """파라미터 값 설정"""
        self.params[name] = value

    def to_dict(self) -> Dict[str, Any]:
        """전략 설정을 딕셔너리로 변환"""
        return {
            'name': self.name,
            'params': self.params.copy()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseStrategy':
        """딕셔너리에서 전략 복원"""
        return cls(**data.get('params', {}))


# ============================================================================
# 1. Pure DCA 전략
# ============================================================================

class PureDCAStrategy(BaseStrategy):
    """
    순수 DCA 전략

    매월 월급날에 무조건 정해진 금액을 투자합니다.
    가장 단순하고 기본적인 적립식 투자 전략입니다.
    """

    name = "Pure DCA"
    description = "매월 정액 매수 (시장 상황 무관)"
    parameters = {}  # 추가 파라미터 없음

    def decide(self, context: StrategyContext) -> StrategyDecision:
        """
        항상 투자 결정
        """
        return StrategyDecision(
            should_invest=True,
            multiplier=1.0,
            reason="Pure DCA: 정액 매수"
        )


# ============================================================================
# 2. Drawdown Tier DCA 전략
# ============================================================================

class DrawdownTierStrategy(BaseStrategy):
    """
    드로우다운 티어 DCA 전략

    직전 고점 대비 하락률에 따라 투자금을 증가시킵니다.
    하락장에서 더 많이 매수하여 평균 매입가를 낮춥니다.

    티어 예시:
    - 0% ~ -10%: 1.0x (기본)
    - -10% ~ -20%: 1.5x
    - -20% ~ -30%: 2.0x
    - -30% 이하: 2.5x
    """

    name = "Drawdown Tier DCA"
    description = "하락률 구간별 추가 매수"
    parameters = {
        "tier1_threshold": {
            "type": "float", "default": -10.0, "min": -50, "max": 0,
            "label": "1단계 하락률 (%)", "step": 1
        },
        "tier1_multiplier": {
            "type": "float", "default": 1.5, "min": 1.0, "max": 5.0,
            "label": "1단계 투자 배수", "step": 0.1
        },
        "tier2_threshold": {
            "type": "float", "default": -20.0, "min": -50, "max": 0,
            "label": "2단계 하락률 (%)", "step": 1
        },
        "tier2_multiplier": {
            "type": "float", "default": 2.0, "min": 1.0, "max": 5.0,
            "label": "2단계 투자 배수", "step": 0.1
        },
        "tier3_threshold": {
            "type": "float", "default": -30.0, "min": -50, "max": 0,
            "label": "3단계 하락률 (%)", "step": 1
        },
        "tier3_multiplier": {
            "type": "float", "default": 2.5, "min": 1.0, "max": 5.0,
            "label": "3단계 투자 배수", "step": 0.1
        },
    }

    def decide(self, context: StrategyContext) -> StrategyDecision:
        """
        드로우다운에 따라 투자 배수 결정
        """
        dd = context.portfolio_drawdown * 100  # 퍼센트로 변환

        tier1 = self.get_param('tier1_threshold', -10)
        tier2 = self.get_param('tier2_threshold', -20)
        tier3 = self.get_param('tier3_threshold', -30)

        mult1 = self.get_param('tier1_multiplier', 1.5)
        mult2 = self.get_param('tier2_multiplier', 2.0)
        mult3 = self.get_param('tier3_multiplier', 2.5)

        if dd <= tier3:
            multiplier = mult3
            reason = f"Tier 3: {dd:.1f}% 하락 → {mult3}x 매수"
        elif dd <= tier2:
            multiplier = mult2
            reason = f"Tier 2: {dd:.1f}% 하락 → {mult2}x 매수"
        elif dd <= tier1:
            multiplier = mult1
            reason = f"Tier 1: {dd:.1f}% 하락 → {mult1}x 매수"
        else:
            multiplier = 1.0
            reason = f"기본: {dd:.1f}% (임계값 미만) → 1x 매수"

        return StrategyDecision(
            should_invest=True,
            multiplier=multiplier,
            reason=reason
        )


# ============================================================================
# 3. Trend Filter 전략
# ============================================================================

class TrendFilterStrategy(BaseStrategy):
    """
    추세 필터 전략

    가격이 이동평균선 위에 있을 때만 투자하거나,
    이동평균선 아래일 때 현금을 보류했다가 반등 시 투입합니다.

    옵션:
    - 이동평균 기간 (기본: 200일)
    - 하락 시 행동: 보류/감액/정상 투자
    - 반등 시 보류 현금 투입 여부
    """

    name = "Trend Filter"
    description = "이동평균선 기반 매수 조절"
    parameters = {
        "ma_period": {
            "type": "int", "default": 200, "min": 20, "max": 300,
            "label": "이동평균 기간", "step": 10
        },
        "below_ma_action": {
            "type": "choice", "default": "reduce",
            "choices": ["invest", "reduce", "reserve"],
            "labels": ["정상 투자", "50% 감액", "전액 보류"],
            "label": "MA 하회 시 행동"
        },
        "deploy_on_recovery": {
            "type": "bool", "default": True,
            "label": "반등 시 보류금 투입"
        },
    }

    def __init__(self, **params):
        super().__init__(**params)
        self._was_below_ma = False

    def decide(self, context: StrategyContext) -> StrategyDecision:
        """
        이동평균 대비 가격 위치에 따라 결정
        """
        ma_period = int(self.get_param('ma_period', 200))
        below_action = self.get_param('below_ma_action', 'reduce')
        deploy_on_recovery = self.get_param('deploy_on_recovery', True)

        # 첫 번째 티커의 가격으로 판단 (포트폴리오 전체 기준 개선 가능)
        tickers = list(context.price_history.keys())
        if not tickers:
            return StrategyDecision(should_invest=True, reason="데이터 없음")

        primary_ticker = tickers[0]
        prices = context.price_history[primary_ticker]

        if len(prices) < ma_period:
            return StrategyDecision(
                should_invest=True,
                reason=f"데이터 부족 ({len(prices)}일 < {ma_period}일)"
            )

        # 이동평균 계산
        ma_value = sma(prices, ma_period).iloc[-1]
        current_price = context.current_prices[primary_ticker]

        is_below_ma = current_price < ma_value
        just_recovered = self._was_below_ma and not is_below_ma

        self._was_below_ma = is_below_ma

        if is_below_ma:
            if below_action == "invest":
                return StrategyDecision(
                    should_invest=True,
                    multiplier=1.0,
                    reason=f"MA{ma_period} 하회 중이나 정상 투자"
                )
            elif below_action == "reduce":
                return StrategyDecision(
                    should_invest=True,
                    multiplier=0.5,
                    reason=f"MA{ma_period} 하회 → 50% 감액"
                )
            else:  # reserve
                return StrategyDecision(
                    should_invest=False,
                    reserve_cash=True,
                    reason=f"MA{ma_period} 하회 → 현금 보류"
                )
        else:
            # MA 위에 있음
            if just_recovered and deploy_on_recovery and context.reserved_cash > 0:
                return StrategyDecision(
                    should_invest=True,
                    multiplier=1.0,
                    deploy_reserved=True,
                    reason=f"MA{ma_period} 회복 → 보류금 투입"
                )
            return StrategyDecision(
                should_invest=True,
                multiplier=1.0,
                reason=f"MA{ma_period} 상회 → 정상 투자"
            )


# ============================================================================
# 4. Volatility Control 전략
# ============================================================================

class VolatilityControlStrategy(BaseStrategy):
    """
    변동성 조절 전략

    시장 변동성이 높을 때 투자금을 줄이고,
    변동성이 낮을 때 정상 또는 증가된 금액을 투자합니다.

    높은 변동성 = 시장 불안 → 보수적 투자
    낮은 변동성 = 시장 안정 → 적극적 투자
    """

    name = "Volatility Control"
    description = "변동성 기반 투자금 조절"
    parameters = {
        "vol_period": {
            "type": "int", "default": 20, "min": 5, "max": 60,
            "label": "변동성 계산 기간", "step": 5
        },
        "high_vol_threshold": {
            "type": "float", "default": 30.0, "min": 10, "max": 80,
            "label": "고변동성 임계값 (%)", "step": 5
        },
        "low_vol_threshold": {
            "type": "float", "default": 15.0, "min": 5, "max": 40,
            "label": "저변동성 임계값 (%)", "step": 5
        },
        "high_vol_multiplier": {
            "type": "float", "default": 0.5, "min": 0, "max": 1.0,
            "label": "고변동성 투자 배수", "step": 0.1
        },
        "low_vol_multiplier": {
            "type": "float", "default": 1.2, "min": 1.0, "max": 2.0,
            "label": "저변동성 투자 배수", "step": 0.1
        },
    }

    def decide(self, context: StrategyContext) -> StrategyDecision:
        """
        현재 변동성 수준에 따라 투자금 조절
        """
        vol_period = int(self.get_param('vol_period', 20))
        high_threshold = self.get_param('high_vol_threshold', 30) / 100
        low_threshold = self.get_param('low_vol_threshold', 15) / 100
        high_mult = self.get_param('high_vol_multiplier', 0.5)
        low_mult = self.get_param('low_vol_multiplier', 1.2)

        # 첫 번째 티커 기준
        tickers = list(context.price_history.keys())
        if not tickers:
            return StrategyDecision(should_invest=True, reason="데이터 없음")

        primary_ticker = tickers[0]
        prices = context.price_history[primary_ticker]

        if len(prices) < vol_period + 1:
            return StrategyDecision(
                should_invest=True,
                reason=f"데이터 부족 ({len(prices)}일)"
            )

        # 변동성 계산 (연율화)
        current_vol = volatility(prices, vol_period).iloc[-1]

        if pd.isna(current_vol):
            return StrategyDecision(should_invest=True, reason="변동성 계산 불가")

        if current_vol >= high_threshold:
            return StrategyDecision(
                should_invest=True,
                multiplier=high_mult,
                reason=f"고변동성 {current_vol*100:.1f}% → {high_mult}x 감액"
            )
        elif current_vol <= low_threshold:
            return StrategyDecision(
                should_invest=True,
                multiplier=low_mult,
                reason=f"저변동성 {current_vol*100:.1f}% → {low_mult}x 증액"
            )
        else:
            return StrategyDecision(
                should_invest=True,
                multiplier=1.0,
                reason=f"보통 변동성 {current_vol*100:.1f}% → 1x 정상"
            )


# ============================================================================
# 전략 레지스트리
# ============================================================================

class StrategyRegistry:
    """
    전략 등록 및 관리

    새 전략을 등록하고 이름으로 조회할 수 있습니다.

    향후 확장 포인트:
    - 플러그인 디렉토리에서 자동 로드
    - 전략 유효성 검증
    - 전략 버전 관리
    """

    _strategies: Dict[str, Type[BaseStrategy]] = {}

    @classmethod
    def register(cls, strategy_class: Type[BaseStrategy]) -> None:
        """전략 클래스 등록"""
        cls._strategies[strategy_class.name] = strategy_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseStrategy]]:
        """이름으로 전략 클래스 조회"""
        return cls._strategies.get(name)

    @classmethod
    def get_all(cls) -> Dict[str, Type[BaseStrategy]]:
        """등록된 모든 전략 반환"""
        return cls._strategies.copy()

    @classmethod
    def get_names(cls) -> List[str]:
        """등록된 전략 이름 목록"""
        return list(cls._strategies.keys())

    @classmethod
    def create(cls, name: str, **params) -> Optional[BaseStrategy]:
        """전략 인스턴스 생성"""
        strategy_class = cls.get(name)
        if strategy_class:
            return strategy_class(**params)
        return None


# 기본 전략 등록
StrategyRegistry.register(PureDCAStrategy)
StrategyRegistry.register(DrawdownTierStrategy)
StrategyRegistry.register(TrendFilterStrategy)
StrategyRegistry.register(VolatilityControlStrategy)


# ============================================================================
# 향후 확장 포인트
# ============================================================================

# TODO: 복합 전략 (여러 전략 조합)
# class CompositeStrategy(BaseStrategy):
#     """여러 전략을 조합하여 사용"""
#     def __init__(self, strategies: List[BaseStrategy], mode: str = "and"):
#         self.strategies = strategies
#         self.mode = mode  # "and", "or", "vote"
#     def decide(self, context):
#         decisions = [s.decide(context) for s in self.strategies]
#         # 모드에 따라 결합
#         pass

# TODO: 커스텀 전략 (사용자 정의 수식)
# class CustomStrategy(BaseStrategy):
#     """사용자가 GUI에서 정의한 조건 기반 전략"""
#     name = "Custom Strategy"
#     def __init__(self, condition_expr: str, **params):
#         self.condition = compile(condition_expr, '<string>', 'eval')
#     def decide(self, context):
#         result = eval(self.condition, context.__dict__)
#         pass

# TODO: 머신러닝 전략
# class MLStrategy(BaseStrategy):
#     """머신러닝 모델 기반 전략"""
#     pass
