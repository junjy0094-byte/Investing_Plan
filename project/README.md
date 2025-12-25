# DCA Backtest GUI

**월급날 적립식 투자 백테스트 시뮬레이터**

매월 정해진 날짜에 무지성으로 미국 주식(ETF/개별종목)을 매수하는 적립식 투자 전략을 백테스트하고,
다양한 하락장 대응 전략을 실험/비교할 수 있는 GUI 애플리케이션입니다.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Features

### 기본 기능
- **멀티 티커 지원**: QQQ, VOO, AAPL 등 여러 종목 동시 투자
- **자산 비중 설정**: 동일 비중 또는 커스텀 비중
- **월급날 매수**: 매월 특정 일자에 자동 매수 (휴장일 처리 옵션)
- **연간 투자금 증가**: 매년 일정 비율로 투자금 증가 설정

### 하락장 대응 전략
1. **Pure DCA**: 무조건 매월 정액 매수
2. **Drawdown Tier DCA**: 고점 대비 하락률에 따라 추가 매수
3. **Trend Filter**: 200일 이동평균선 기반 매수 조절
4. **Volatility Control**: 변동성에 따른 투자금 조절

### 분석 지표
- CAGR (연평균 수익률)
- MDD (최대 낙폭)
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- 총 납입액, 최종 자산, 수익금
- 월별/연도별 수익률 히트맵

### UI 특징
- 모던한 다크/라이트 테마
- 실시간 차트 (Equity Curve, Drawdown, 상대성과)
- 전략 비교 탭
- 설정 저장/불러오기

---

## Installation

### 1. 요구사항
- Python 3.10 이상
- pip (Python 패키지 관리자)

### 2. 설치 방법

```bash
# 저장소 클론 또는 다운로드
cd project

# 가상환경 생성 (권장)
python -m venv venv

# 가상환경 활성화
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 3. 실행

```bash
python run.py
```

---

## Usage

### Quick Start

1. **티커 입력**: 좌측 패널에서 투자할 종목 입력 (예: QQQ, VOO)
2. **비중 설정**: 동일 비중 또는 직접 비중 입력
3. **기간 설정**: 백테스트 시작일/종료일 선택
4. **월급날 설정**: 매월 매수할 날짜 선택 (기본: 25일)
5. **투자금 설정**: 월 투자금 입력 (예: 4000 USD)
6. **전략 선택**: 드롭다운에서 전략 선택 및 파라미터 조정
7. **Run 버튼 클릭**: 백테스트 실행

### 프리셋 예시

애플리케이션에는 3가지 프리셋이 내장되어 있습니다:

- **QQQ Pure DCA**: QQQ 단일 종목, 월 $4,000, 순수 DCA
- **VOO+QQQ Drawdown Tier**: 50/50 비중, 하락 시 추가 매수
- **SPY Trend Filter**: SPY 단일, 200DMA 필터 적용

---

## Project Structure

```
project/
├── README.md              # 이 파일
├── requirements.txt       # 의존성 목록
├── run.py                 # 애플리케이션 진입점
├── cache/                 # 데이터 캐시 디렉토리
└── app/
    ├── __init__.py
    ├── gui_main.py        # PyQt 메인 윈도우
    ├── widgets.py         # 커스텀 위젯
    ├── styles.qss         # QSS 테마 스타일
    ├── backtest_engine.py # 백테스트 엔진
    ├── strategies.py      # 전략 클래스
    ├── indicators.py      # 기술적 지표
    ├── data_provider.py   # 데이터 공급자 (yfinance + 캐시)
    ├── analytics.py       # 성과 분석
    ├── persistence.py     # 설정 저장/로드
    └── utils.py           # 유틸리티 함수
```

---

## Extending

### 새 전략 추가

`app/strategies.py`에서 `BaseStrategy`를 상속하여 새 전략을 구현할 수 있습니다:

```python
from app.strategies import BaseStrategy, StrategyContext, StrategyDecision

class MyCustomStrategy(BaseStrategy):
    name = "My Custom Strategy"
    description = "나만의 전략 설명"

    # 파라미터 정의
    parameters = {
        "my_param": {"type": "float", "default": 1.0, "min": 0, "max": 10, "label": "My Param"}
    }

    def decide(self, context: StrategyContext) -> StrategyDecision:
        # 전략 로직 구현
        multiplier = self.params.get("my_param", 1.0)
        return StrategyDecision(
            should_invest=True,
            multiplier=multiplier,
            reason="My strategy decision"
        )
```

### 새 지표 추가

`app/indicators.py`에 새 지표 함수를 추가할 수 있습니다:

```python
def my_indicator(prices: pd.Series, period: int = 14) -> pd.Series:
    """내 커스텀 지표"""
    # 지표 계산 로직
    return result
```

---

## Configuration

설정 파일은 `~/.dca_backtest/config.json`에 저장됩니다.

주요 설정:
- 테마 (다크/라이트)
- 마지막 사용 설정
- 캐시 디렉토리 경로

---

## Data Caching

- 주가 데이터는 `cache/` 디렉토리에 Parquet 형식으로 캐시됩니다
- 캐시 유효 기간: 1일 (장중에는 실시간 데이터 요청)
- 캐시를 강제로 새로고침하려면 캐시 디렉토리를 삭제하세요

---

## Troubleshooting

### 일반적인 문제

1. **yfinance 데이터 다운로드 실패**
   - 인터넷 연결 확인
   - 티커 심볼이 올바른지 확인 (Yahoo Finance 기준)
   - 잠시 후 재시도 (API 제한일 수 있음)

2. **GUI가 표시되지 않음**
   - PyQt6가 올바르게 설치되었는지 확인
   - Python 버전 확인 (3.10+)

3. **차트가 표시되지 않음**
   - matplotlib 설치 확인
   - 창 크기 조절 시도

---

## License

MIT License

---

## Contributing

버그 리포트, 기능 제안, PR을 환영합니다!

---

## Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance 데이터
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - GUI 프레임워크
- [pandas](https://pandas.pydata.org/) - 데이터 분석
- [matplotlib](https://matplotlib.org/) - 차트 시각화
