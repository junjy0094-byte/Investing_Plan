"""
유틸리티 모듈

공용 유틸리티 함수, 검증 로직, 로깅 설정을 제공합니다.

향후 확장 포인트:
- 커스텀 예외 클래스 추가
- 국제화(i18n) 지원
- 성능 프로파일링 유틸리티
"""

import logging
import sys
from datetime import datetime, date
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass
import re

# ============================================================================
# 로깅 설정
# ============================================================================

class LogHandler(logging.Handler):
    """
    GUI 로그 창에 로그를 전달하기 위한 커스텀 핸들러

    향후 확장 포인트: 로그 레벨별 색상 지정, 로그 필터링
    """

    def __init__(self, callback=None):
        super().__init__()
        self.callback = callback
        self.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        ))

    def emit(self, record):
        if self.callback:
            msg = self.format(record)
            self.callback(msg, record.levelno)


def setup_logger(name: str = "dca_backtest", level: int = logging.INFO) -> logging.Logger:
    """
    애플리케이션 로거 설정

    Args:
        name: 로거 이름
        level: 로그 레벨

    Returns:
        설정된 로거 인스턴스
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 기존 핸들러 제거 (중복 방지)
    logger.handlers.clear()

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(console_handler)

    return logger


# 기본 로거 생성
logger = setup_logger()


# ============================================================================
# 검증 유틸리티
# ============================================================================

@dataclass
class ValidationResult:
    """검증 결과를 담는 데이터 클래스"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]

    def __bool__(self):
        return self.is_valid


def validate_ticker(ticker: str) -> Tuple[bool, str]:
    """
    티커 심볼 유효성 검증

    Args:
        ticker: 검증할 티커 심볼

    Returns:
        (유효 여부, 에러 메시지)
    """
    if not ticker:
        return False, "티커가 비어있습니다."

    ticker = ticker.strip().upper()

    # 기본 형식 검증 (영문자, 숫자, 일부 특수문자만 허용)
    if not re.match(r'^[A-Z0-9\.\-\^]+$', ticker):
        return False, f"'{ticker}'는 유효하지 않은 티커 형식입니다."

    # 길이 검증
    if len(ticker) > 10:
        return False, f"'{ticker}'가 너무 깁니다 (최대 10자)."

    return True, ""


def validate_tickers(tickers: List[str]) -> ValidationResult:
    """
    티커 리스트 전체 검증

    Args:
        tickers: 검증할 티커 리스트

    Returns:
        ValidationResult 객체
    """
    errors = []
    warnings = []

    if not tickers:
        errors.append("최소 1개의 티커를 입력해주세요.")
        return ValidationResult(False, errors, warnings)

    seen = set()
    for ticker in tickers:
        is_valid, error = validate_ticker(ticker)
        if not is_valid:
            errors.append(error)
        elif ticker.upper() in seen:
            warnings.append(f"'{ticker}'가 중복됩니다.")
        else:
            seen.add(ticker.upper())

    return ValidationResult(len(errors) == 0, errors, warnings)


def validate_weights(weights: List[float], tickers: List[str]) -> ValidationResult:
    """
    자산 비중 검증

    Args:
        weights: 비중 리스트 (0~100 사이 값)
        tickers: 티커 리스트

    Returns:
        ValidationResult 객체
    """
    errors = []
    warnings = []

    if len(weights) != len(tickers):
        errors.append("비중 개수가 티커 개수와 일치하지 않습니다.")
        return ValidationResult(False, errors, warnings)

    # 개별 비중 검증
    for i, w in enumerate(weights):
        if w < 0:
            errors.append(f"'{tickers[i]}'의 비중이 음수입니다.")
        elif w > 100:
            errors.append(f"'{tickers[i]}'의 비중이 100%를 초과합니다.")

    # 합계 검증
    total = sum(weights)
    if abs(total - 100) > 0.01:  # 소수점 오차 허용
        if total < 100:
            errors.append(f"비중 합계가 100%가 아닙니다. (현재: {total:.1f}%)")
        else:
            errors.append(f"비중 합계가 100%를 초과합니다. (현재: {total:.1f}%)")

    return ValidationResult(len(errors) == 0, errors, warnings)


def validate_date_range(start_date: date, end_date: date) -> ValidationResult:
    """
    날짜 범위 검증

    Args:
        start_date: 시작일
        end_date: 종료일

    Returns:
        ValidationResult 객체
    """
    errors = []
    warnings = []

    today = date.today()

    if start_date >= end_date:
        errors.append("시작일이 종료일보다 이후입니다.")

    if end_date > today:
        warnings.append("종료일이 오늘 이후입니다. 오늘까지의 데이터만 사용됩니다.")

    if start_date > today:
        errors.append("시작일이 미래입니다.")

    # 최소 기간 검증 (1개월)
    diff_days = (end_date - start_date).days
    if diff_days < 30:
        warnings.append("백테스트 기간이 1개월 미만입니다.")

    # 너무 오래된 데이터 경고
    if start_date.year < 1990:
        warnings.append("1990년 이전 데이터는 부정확할 수 있습니다.")

    return ValidationResult(len(errors) == 0, errors, warnings)


def validate_investment_params(
    monthly_investment: float,
    annual_increase: float,
    commission: float,
    slippage: float
) -> ValidationResult:
    """
    투자 파라미터 검증

    Args:
        monthly_investment: 월 투자금
        annual_increase: 연간 투자금 증가율 (%)
        commission: 수수료 (%)
        slippage: 슬리피지 (%)

    Returns:
        ValidationResult 객체
    """
    errors = []
    warnings = []

    if monthly_investment <= 0:
        errors.append("월 투자금은 0보다 커야 합니다.")
    elif monthly_investment < 100:
        warnings.append("월 투자금이 $100 미만입니다.")

    if annual_increase < 0:
        errors.append("연간 증가율은 0% 이상이어야 합니다.")
    elif annual_increase > 50:
        warnings.append("연간 증가율이 50%를 초과합니다.")

    if commission < 0:
        errors.append("수수료는 0% 이상이어야 합니다.")
    elif commission > 5:
        warnings.append("수수료가 5%를 초과합니다. 확인해주세요.")

    if slippage < 0:
        errors.append("슬리피지는 0% 이상이어야 합니다.")
    elif slippage > 5:
        warnings.append("슬리피지가 5%를 초과합니다. 확인해주세요.")

    return ValidationResult(len(errors) == 0, errors, warnings)


def validate_all_inputs(
    tickers: List[str],
    weights: List[float],
    start_date: date,
    end_date: date,
    monthly_investment: float,
    annual_increase: float = 0,
    commission: float = 0,
    slippage: float = 0
) -> ValidationResult:
    """
    모든 입력값 통합 검증

    Args:
        tickers: 티커 리스트
        weights: 비중 리스트
        start_date: 시작일
        end_date: 종료일
        monthly_investment: 월 투자금
        annual_increase: 연간 증가율
        commission: 수수료
        slippage: 슬리피지

    Returns:
        ValidationResult 객체
    """
    all_errors = []
    all_warnings = []

    # 각 검증 수행
    results = [
        validate_tickers(tickers),
        validate_weights(weights, tickers),
        validate_date_range(start_date, end_date),
        validate_investment_params(monthly_investment, annual_increase, commission, slippage)
    ]

    for result in results:
        all_errors.extend(result.errors)
        all_warnings.extend(result.warnings)

    return ValidationResult(len(all_errors) == 0, all_errors, all_warnings)


# ============================================================================
# 날짜 유틸리티
# ============================================================================

def parse_date(date_str: str) -> Optional[date]:
    """
    문자열을 날짜로 파싱

    지원 형식: YYYY-MM-DD, YYYY/MM/DD, DD-MM-YYYY

    Args:
        date_str: 날짜 문자열

    Returns:
        date 객체 또는 None
    """
    formats = [
        '%Y-%m-%d',
        '%Y/%m/%d',
        '%d-%m-%Y',
        '%d/%m/%Y',
        '%m/%d/%Y',
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue

    return None


def format_date(d: date, fmt: str = '%Y-%m-%d') -> str:
    """날짜를 문자열로 포맷"""
    return d.strftime(fmt)


def get_month_end(d: date) -> date:
    """해당 월의 마지막 날짜 반환"""
    import calendar
    last_day = calendar.monthrange(d.year, d.month)[1]
    return date(d.year, d.month, last_day)


def get_year_start(d: date) -> date:
    """해당 연도의 첫날 반환"""
    return date(d.year, 1, 1)


# ============================================================================
# 숫자 포맷팅
# ============================================================================

def format_currency(value: float, currency: str = "USD", decimals: int = 2) -> str:
    """
    금액을 통화 형식으로 포맷

    Args:
        value: 금액
        currency: 통화 코드
        decimals: 소수점 자릿수

    Returns:
        포맷된 문자열 (예: "$1,234.56")
    """
    symbols = {"USD": "$", "EUR": "€", "KRW": "₩", "JPY": "¥"}
    symbol = symbols.get(currency, currency + " ")

    if abs(value) >= 1_000_000:
        return f"{symbol}{value/1_000_000:,.{decimals}f}M"
    elif abs(value) >= 1_000:
        return f"{symbol}{value:,.{decimals}f}"
    else:
        return f"{symbol}{value:.{decimals}f}"


def format_percent(value: float, decimals: int = 2, include_sign: bool = False) -> str:
    """
    숫자를 퍼센트 형식으로 포맷

    Args:
        value: 값 (예: 0.15 = 15%)
        decimals: 소수점 자릿수
        include_sign: 양수에 + 기호 포함 여부

    Returns:
        포맷된 문자열 (예: "15.00%" 또는 "+15.00%")
    """
    pct = value * 100
    if include_sign and pct > 0:
        return f"+{pct:.{decimals}f}%"
    return f"{pct:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """숫자를 천단위 구분자로 포맷"""
    return f"{value:,.{decimals}f}"


# ============================================================================
# 기타 유틸리티
# ============================================================================

def clamp(value: float, min_val: float, max_val: float) -> float:
    """값을 최소/최대 범위 내로 제한"""
    return max(min_val, min(max_val, value))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """안전한 나눗셈 (0으로 나누기 방지)"""
    if denominator == 0:
        return default
    return numerator / denominator


def interpolate(value: float, in_min: float, in_max: float,
                out_min: float, out_max: float) -> float:
    """선형 보간"""
    if in_max == in_min:
        return out_min
    ratio = (value - in_min) / (in_max - in_min)
    return out_min + ratio * (out_max - out_min)


# ============================================================================
# 색상 유틸리티 (차트용)
# ============================================================================

def get_color_for_value(value: float, positive_color: str = "#26A69A",
                        negative_color: str = "#EF5350",
                        neutral_color: str = "#9E9E9E") -> str:
    """
    값에 따른 색상 반환 (양수/음수/중립)

    Args:
        value: 값
        positive_color: 양수일 때 색상
        negative_color: 음수일 때 색상
        neutral_color: 0일 때 색상

    Returns:
        HEX 색상 코드
    """
    if value > 0:
        return positive_color
    elif value < 0:
        return negative_color
    return neutral_color


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    """HEX 색상을 RGBA 튜플로 변환"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    return (r, g, b, alpha)


# ============================================================================
# 향후 확장 포인트
# ============================================================================

# TODO: 다국어 지원 (i18n)
# def translate(key: str, lang: str = "ko") -> str:
#     """다국어 문자열 반환"""
#     pass

# TODO: 성능 프로파일링
# def profile(func):
#     """함수 실행 시간 측정 데코레이터"""
#     pass

# TODO: 메모이제이션
# def memoize(func):
#     """결과 캐싱 데코레이터"""
#     pass
