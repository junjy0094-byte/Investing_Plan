"""
데이터 공급자 모듈

yfinance를 통한 주가 데이터 다운로드 및 로컬 캐싱을 담당합니다.

향후 확장 포인트:
- 다중 데이터 소스 지원 (Alpha Vantage, IEX Cloud 등)
- 실시간 데이터 스트리밍
- 분/시간 단위 데이터 지원
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Tuple, Callable
from pathlib import Path
import yfinance as yf
from dataclasses import dataclass
import hashlib
import json

from app.utils import logger


# ============================================================================
# 설정
# ============================================================================

# 기본 캐시 디렉토리
DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "cache"

# 캐시 유효 기간 (초)
CACHE_TTL_SECONDS = 24 * 60 * 60  # 24시간

# US 주식 시장 휴장일 (간단 버전 - 실제로는 더 정교한 캘린더 필요)
# 향후 확장: exchange_calendars 라이브러리 사용
US_HOLIDAYS_2024_2025 = [
    # 2024
    date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19),
    date(2024, 3, 29), date(2024, 5, 27), date(2024, 6, 19),
    date(2024, 7, 4), date(2024, 9, 2), date(2024, 11, 28),
    date(2024, 12, 25),
    # 2025
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17),
    date(2025, 4, 18), date(2025, 5, 26), date(2025, 6, 19),
    date(2025, 7, 4), date(2025, 9, 1), date(2025, 11, 27),
    date(2025, 12, 25),
]


# ============================================================================
# 휴장일 처리
# ============================================================================

@dataclass
class TradingCalendar:
    """
    거래일 캘린더

    향후 확장 포인트:
    - exchange_calendars 라이브러리 통합
    - 다중 거래소 지원 (NYSE, NASDAQ, KRX 등)
    """

    holidays: List[date]

    def is_trading_day(self, d: date) -> bool:
        """주어진 날짜가 거래일인지 확인"""
        # 주말 체크
        if d.weekday() >= 5:
            return False
        # 휴장일 체크
        if d in self.holidays:
            return False
        return True

    def get_previous_trading_day(self, d: date) -> date:
        """직전 거래일 반환"""
        current = d - timedelta(days=1)
        while not self.is_trading_day(current):
            current -= timedelta(days=1)
            # 무한 루프 방지
            if (d - current).days > 30:
                break
        return current

    def get_next_trading_day(self, d: date) -> date:
        """직후 거래일 반환"""
        current = d + timedelta(days=1)
        while not self.is_trading_day(current):
            current += timedelta(days=1)
            if (current - d).days > 30:
                break
        return current

    def adjust_to_trading_day(self, d: date, direction: str = "previous") -> date:
        """
        주어진 날짜를 거래일로 조정

        Args:
            d: 조정할 날짜
            direction: "previous" (직전) 또는 "next" (직후)

        Returns:
            조정된 거래일
        """
        if self.is_trading_day(d):
            return d

        if direction == "previous":
            return self.get_previous_trading_day(d)
        else:
            return self.get_next_trading_day(d)

    def get_trading_days_between(self, start: date, end: date) -> List[date]:
        """기간 내 모든 거래일 반환"""
        days = []
        current = start
        while current <= end:
            if self.is_trading_day(current):
                days.append(current)
            current += timedelta(days=1)
        return days


# 기본 거래 캘린더 인스턴스
us_calendar = TradingCalendar(holidays=US_HOLIDAYS_2024_2025)


# ============================================================================
# 캐시 매니저
# ============================================================================

class CacheManager:
    """
    데이터 캐시 매니저

    Parquet 형식으로 주가 데이터를 로컬에 캐싱합니다.

    향후 확장 포인트:
    - 압축 옵션
    - 캐시 정리 (오래된 파일 삭제)
    - 캐시 통계
    """

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """메타데이터 파일 로드"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"캐시 메타데이터 로드 실패: {e}")
        return {}

    def _save_metadata(self) -> None:
        """메타데이터 파일 저장"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"캐시 메타데이터 저장 실패: {e}")

    def _get_cache_path(self, ticker: str) -> Path:
        """티커별 캐시 파일 경로"""
        safe_ticker = ticker.replace("^", "_").replace("/", "_")
        return self.cache_dir / f"{safe_ticker}.parquet"

    def _get_cache_key(self, ticker: str) -> str:
        """캐시 키 생성"""
        return ticker.upper().strip()

    def is_cache_valid(self, ticker: str, start_date: date, end_date: date) -> bool:
        """
        캐시가 유효한지 확인

        Args:
            ticker: 티커 심볼
            start_date: 필요한 시작일
            end_date: 필요한 종료일

        Returns:
            캐시 유효 여부
        """
        key = self._get_cache_key(ticker)
        cache_path = self._get_cache_path(ticker)

        if not cache_path.exists():
            return False

        if key not in self._metadata:
            return False

        meta = self._metadata[key]

        # 캐시 시간 확인
        cached_time = datetime.fromisoformat(meta.get('cached_at', '1970-01-01'))
        age = (datetime.now() - cached_time).total_seconds()

        if age > CACHE_TTL_SECONDS:
            # 오늘 장이 끝났으면 캐시 유효
            now = datetime.now()
            if now.hour >= 21 and end_date < date.today():  # 미국 장 마감 후
                pass
            else:
                return False

        # 날짜 범위 확인
        cached_start = date.fromisoformat(meta.get('start_date', '2100-01-01'))
        cached_end = date.fromisoformat(meta.get('end_date', '1970-01-01'))

        if start_date < cached_start or end_date > cached_end:
            return False

        return True

    def get_cached_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """캐시된 데이터 조회"""
        cache_path = self._get_cache_path(ticker)

        if not cache_path.exists():
            return None

        try:
            df = pd.read_parquet(cache_path)
            logger.debug(f"캐시에서 {ticker} 데이터 로드 ({len(df)}행)")
            return df
        except Exception as e:
            logger.warning(f"캐시 읽기 실패 ({ticker}): {e}")
            return None

    def save_to_cache(self, ticker: str, data: pd.DataFrame,
                      start_date: date, end_date: date) -> None:
        """데이터를 캐시에 저장"""
        cache_path = self._get_cache_path(ticker)
        key = self._get_cache_key(ticker)

        try:
            data.to_parquet(cache_path, index=True)

            self._metadata[key] = {
                'cached_at': datetime.now().isoformat(),
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'rows': len(data)
            }
            self._save_metadata()

            logger.debug(f"{ticker} 데이터 캐시 저장 ({len(data)}행)")
        except Exception as e:
            logger.warning(f"캐시 저장 실패 ({ticker}): {e}")

    def clear_cache(self, ticker: Optional[str] = None) -> None:
        """캐시 삭제"""
        if ticker:
            cache_path = self._get_cache_path(ticker)
            if cache_path.exists():
                cache_path.unlink()
            key = self._get_cache_key(ticker)
            if key in self._metadata:
                del self._metadata[key]
                self._save_metadata()
        else:
            # 전체 캐시 삭제
            for f in self.cache_dir.glob("*.parquet"):
                f.unlink()
            self._metadata = {}
            self._save_metadata()
            logger.info("전체 캐시 삭제됨")


# 기본 캐시 매니저 인스턴스
cache_manager = CacheManager()


# ============================================================================
# 데이터 공급자
# ============================================================================

class DataProvider:
    """
    주가 데이터 공급자

    yfinance를 통해 데이터를 다운로드하고 로컬에 캐싱합니다.

    향후 확장 포인트:
    - 다중 데이터 소스 지원
    - 데이터 품질 검증
    - 분할/배당 조정 옵션
    """

    def __init__(self, cache_manager: CacheManager = cache_manager,
                 calendar: TradingCalendar = us_calendar):
        self.cache = cache_manager
        self.calendar = calendar
        self._data_cache: Dict[str, pd.DataFrame] = {}  # 인메모리 캐시

    def get_price_data(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        use_cache: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Optional[pd.DataFrame]:
        """
        주가 데이터 조회

        Args:
            ticker: 티커 심볼
            start_date: 시작일
            end_date: 종료일
            use_cache: 캐시 사용 여부
            progress_callback: 진행 상태 콜백

        Returns:
            OHLCV 데이터프레임 (Date 인덱스)
        """
        ticker = ticker.upper().strip()

        if progress_callback:
            progress_callback(f"{ticker} 데이터 조회 중...")

        # 인메모리 캐시 확인
        cache_key = f"{ticker}_{start_date}_{end_date}"
        if cache_key in self._data_cache:
            logger.debug(f"인메모리 캐시에서 {ticker} 반환")
            return self._data_cache[cache_key]

        # 로컬 캐시 확인
        if use_cache and self.cache.is_cache_valid(ticker, start_date, end_date):
            data = self.cache.get_cached_data(ticker)
            if data is not None:
                # 날짜 필터링
                data = self._filter_date_range(data, start_date, end_date)
                self._data_cache[cache_key] = data
                return data

        # yfinance에서 다운로드
        try:
            if progress_callback:
                progress_callback(f"{ticker} 다운로드 중...")

            logger.info(f"{ticker} 데이터 다운로드: {start_date} ~ {end_date}")

            # 여유 기간 추가 (지표 계산용)
            buffer_start = start_date - timedelta(days=365)

            yf_ticker = yf.Ticker(ticker)
            data = yf_ticker.history(
                start=buffer_start.isoformat(),
                end=(end_date + timedelta(days=1)).isoformat(),
                auto_adjust=True  # 분할/배당 조정
            )

            if data.empty:
                logger.warning(f"{ticker}: 데이터 없음")
                return None

            # 인덱스를 date로 변환
            data.index = pd.to_datetime(data.index).date
            data.index = pd.DatetimeIndex(data.index)

            # 캐시 저장
            if use_cache:
                self.cache.save_to_cache(ticker, data, buffer_start, end_date)

            # 필터링
            data = self._filter_date_range(data, start_date, end_date)
            self._data_cache[cache_key] = data

            if progress_callback:
                progress_callback(f"{ticker} 완료 ({len(data)}일)")

            return data

        except Exception as e:
            logger.error(f"{ticker} 데이터 다운로드 실패: {e}")
            return None

    def _filter_date_range(self, data: pd.DataFrame,
                           start_date: date, end_date: date) -> pd.DataFrame:
        """날짜 범위로 데이터 필터링"""
        mask = (data.index >= pd.Timestamp(start_date)) & \
               (data.index <= pd.Timestamp(end_date))
        return data[mask].copy()

    def get_multiple_tickers(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date,
        use_cache: bool = True,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        여러 티커의 데이터 조회

        Args:
            tickers: 티커 리스트
            start_date: 시작일
            end_date: 종료일
            use_cache: 캐시 사용 여부
            progress_callback: 진행 상태 콜백 (msg, current, total)

        Returns:
            {ticker: DataFrame} 딕셔너리
        """
        result = {}
        total = len(tickers)

        for i, ticker in enumerate(tickers):
            if progress_callback:
                progress_callback(f"{ticker} 처리 중...", i + 1, total)

            data = self.get_price_data(ticker, start_date, end_date, use_cache)
            if data is not None:
                result[ticker] = data

        return result

    def validate_ticker(self, ticker: str) -> Tuple[bool, str]:
        """
        티커 유효성 검증 (실제 데이터 존재 확인)

        Args:
            ticker: 티커 심볼

        Returns:
            (유효 여부, 메시지)
        """
        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info

            if not info or 'regularMarketPrice' not in info:
                return False, f"'{ticker}'를 찾을 수 없습니다."

            return True, f"'{ticker}' 확인됨: {info.get('shortName', ticker)}"

        except Exception as e:
            return False, f"'{ticker}' 검증 실패: {str(e)}"

    def get_trading_days(self, start_date: date, end_date: date) -> List[date]:
        """기간 내 거래일 목록 반환"""
        return self.calendar.get_trading_days_between(start_date, end_date)

    def adjust_to_trading_day(self, d: date, direction: str = "previous") -> date:
        """날짜를 거래일로 조정"""
        return self.calendar.adjust_to_trading_day(d, direction)

    def is_trading_day(self, d: date) -> bool:
        """거래일 여부 확인"""
        return self.calendar.is_trading_day(d)

    def clear_memory_cache(self) -> None:
        """인메모리 캐시 초기화"""
        self._data_cache.clear()


# 기본 데이터 공급자 인스턴스
data_provider = DataProvider()


# ============================================================================
# 편의 함수
# ============================================================================

def download_data(
    tickers: List[str],
    start_date: date,
    end_date: date,
    progress_callback: Optional[Callable] = None
) -> Dict[str, pd.DataFrame]:
    """
    데이터 다운로드 편의 함수

    Args:
        tickers: 티커 리스트
        start_date: 시작일
        end_date: 종료일
        progress_callback: 진행 콜백

    Returns:
        {ticker: DataFrame} 딕셔너리
    """
    return data_provider.get_multiple_tickers(
        tickers, start_date, end_date,
        progress_callback=progress_callback
    )


def get_benchmark_data(
    benchmark: str = "SPY",
    start_date: date = None,
    end_date: date = None
) -> Optional[pd.DataFrame]:
    """
    벤치마크 데이터 조회

    Args:
        benchmark: 벤치마크 티커 (기본: SPY)
        start_date: 시작일
        end_date: 종료일

    Returns:
        벤치마크 데이터프레임
    """
    if start_date is None:
        start_date = date(2010, 1, 1)
    if end_date is None:
        end_date = date.today()

    return data_provider.get_price_data(benchmark, start_date, end_date)


# ============================================================================
# 향후 확장 포인트
# ============================================================================

# TODO: 다중 데이터 소스 지원
# class AlphaVantageProvider(DataProvider):
#     """Alpha Vantage 데이터 공급자"""
#     pass

# TODO: 실시간 데이터
# class RealtimeDataProvider:
#     """실시간 데이터 스트리밍"""
#     pass

# TODO: 데이터 품질 검증
# def validate_data_quality(data: pd.DataFrame) -> List[str]:
#     """데이터 품질 문제 검사"""
#     pass
