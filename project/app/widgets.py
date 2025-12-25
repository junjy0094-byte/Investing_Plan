"""
커스텀 위젯 모듈

PyQt6 기반의 커스텀 위젯들을 정의합니다.

주요 위젯:
- InputPanel: 좌측 입력 패널
- ChartPanel: 차트 표시 패널
- ResultsPanel: 결과 및 지표 표시
- LogWidget: 로그 출력

향후 확장 포인트:
- 커스텀 차트 위젯
- 드래그앤드롭 티커 입력
- 키보드 단축키
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QDateEdit, QCheckBox, QRadioButton, QButtonGroup, QSlider,
    QGroupBox, QFrame, QTabWidget, QTableWidget, QTableWidgetItem,
    QTextEdit, QProgressBar, QSplitter, QScrollArea, QSizePolicy,
    QHeaderView, QAbstractItemView, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, QDate, pyqtSignal, QThread, QSize
from PyQt6.QtGui import QFont, QColor, QPalette, QIntValidator, QDoubleValidator

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from datetime import date, datetime
from typing import Dict, List, Optional, Any, Callable

from app.strategies import StrategyRegistry, BaseStrategy
from app.backtest_engine import BacktestConfig, HolidayAdjustment, RebalanceFrequency
from app.analytics import BacktestResult, PerformanceMetrics, create_returns_heatmap_data
from app.persistence import get_preset_names, apply_preset
from app.utils import format_currency, format_percent, format_number, logger


# ============================================================================
# 티커 입력 위젯
# ============================================================================

class TickerInputWidget(QWidget):
    """
    티커 및 비중 입력 위젯

    여러 티커와 각 비중을 입력받습니다.
    """

    tickersChanged = pyqtSignal(list, list)  # (tickers, weights)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 티커 입력
        ticker_layout = QHBoxLayout()
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("티커 입력 (예: QQQ, VOO)")
        self.ticker_input.returnPressed.connect(self._add_ticker)

        self.add_btn = QPushButton("추가")
        self.add_btn.clicked.connect(self._add_ticker)

        ticker_layout.addWidget(self.ticker_input)
        ticker_layout.addWidget(self.add_btn)
        layout.addLayout(ticker_layout)

        # 동일비중 체크박스
        self.equal_weight_cb = QCheckBox("동일 비중")
        self.equal_weight_cb.setChecked(True)
        self.equal_weight_cb.stateChanged.connect(self._on_equal_weight_changed)
        layout.addWidget(self.equal_weight_cb)

        # 티커 테이블
        self.ticker_table = QTableWidget()
        self.ticker_table.setColumnCount(3)
        self.ticker_table.setHorizontalHeaderLabels(["티커", "비중 (%)", ""])
        self.ticker_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.ticker_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.ticker_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.ticker_table.setColumnWidth(1, 80)
        self.ticker_table.setColumnWidth(2, 60)
        self.ticker_table.setMaximumHeight(150)
        layout.addWidget(self.ticker_table)

        # 비중 합계
        self.weight_sum_label = QLabel("비중 합계: 0%")
        layout.addWidget(self.weight_sum_label)

    def _add_ticker(self):
        """티커 추가"""
        text = self.ticker_input.text().strip().upper()
        if not text:
            return

        # 쉼표로 분리된 여러 티커 처리
        tickers = [t.strip() for t in text.split(',') if t.strip()]

        for ticker in tickers:
            if not self._ticker_exists(ticker):
                self._add_ticker_row(ticker)

        self.ticker_input.clear()
        self._update_weights()
        self._emit_change()

    def _ticker_exists(self, ticker: str) -> bool:
        """티커 존재 여부 확인"""
        for row in range(self.ticker_table.rowCount()):
            item = self.ticker_table.item(row, 0)
            if item and item.text() == ticker:
                return True
        return False

    def _add_ticker_row(self, ticker: str, weight: float = 0):
        """티커 행 추가"""
        row = self.ticker_table.rowCount()
        self.ticker_table.insertRow(row)

        # 티커
        ticker_item = QTableWidgetItem(ticker)
        ticker_item.setFlags(ticker_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.ticker_table.setItem(row, 0, ticker_item)

        # 비중
        weight_spin = QDoubleSpinBox()
        weight_spin.setRange(0, 100)
        weight_spin.setValue(weight)
        weight_spin.setSuffix("%")
        weight_spin.setEnabled(not self.equal_weight_cb.isChecked())
        weight_spin.valueChanged.connect(self._on_weight_changed)
        self.ticker_table.setCellWidget(row, 1, weight_spin)

        # 삭제 버튼
        delete_btn = QPushButton("X")
        delete_btn.setMaximumWidth(40)
        delete_btn.clicked.connect(lambda: self._remove_ticker_row(row))
        self.ticker_table.setCellWidget(row, 2, delete_btn)

    def _remove_ticker_row(self, row: int):
        """티커 행 삭제"""
        self.ticker_table.removeRow(row)
        self._update_weights()
        self._emit_change()

    def _on_equal_weight_changed(self, state):
        """동일비중 체크 변경"""
        is_equal = state == Qt.CheckState.Checked.value
        for row in range(self.ticker_table.rowCount()):
            spin = self.ticker_table.cellWidget(row, 1)
            if spin:
                spin.setEnabled(not is_equal)
        self._update_weights()
        self._emit_change()

    def _on_weight_changed(self):
        """비중 변경"""
        self._update_weight_sum()
        self._emit_change()

    def _update_weights(self):
        """비중 업데이트 (동일비중 시)"""
        if self.equal_weight_cb.isChecked():
            count = self.ticker_table.rowCount()
            if count > 0:
                equal_weight = 100.0 / count
                for row in range(count):
                    spin = self.ticker_table.cellWidget(row, 1)
                    if spin:
                        spin.blockSignals(True)
                        spin.setValue(equal_weight)
                        spin.blockSignals(False)
        self._update_weight_sum()

    def _update_weight_sum(self):
        """비중 합계 업데이트"""
        total = self._get_weight_sum()
        color = "green" if abs(total - 100) < 0.01 else "red"
        self.weight_sum_label.setText(f"비중 합계: <span style='color:{color}'>{total:.1f}%</span>")

    def _get_weight_sum(self) -> float:
        """비중 합계 계산"""
        total = 0.0
        for row in range(self.ticker_table.rowCount()):
            spin = self.ticker_table.cellWidget(row, 1)
            if spin:
                total += spin.value()
        return total

    def _emit_change(self):
        """변경 시그널 발생"""
        tickers, weights = self.get_values()
        self.tickersChanged.emit(tickers, weights)

    def get_values(self) -> tuple:
        """티커와 비중 반환"""
        tickers = []
        weights = []

        for row in range(self.ticker_table.rowCount()):
            ticker_item = self.ticker_table.item(row, 0)
            weight_spin = self.ticker_table.cellWidget(row, 1)

            if ticker_item and weight_spin:
                tickers.append(ticker_item.text())
                weights.append(weight_spin.value() / 100.0)  # 0~1로 변환

        return tickers, weights

    def set_values(self, tickers: List[str], weights: List[float]):
        """값 설정"""
        self.ticker_table.setRowCount(0)

        for i, ticker in enumerate(tickers):
            weight = weights[i] * 100 if i < len(weights) else 0
            self._add_ticker_row(ticker, weight)

        self._update_weights()


# ============================================================================
# 전략 파라미터 위젯
# ============================================================================

class StrategyParamsWidget(QWidget):
    """
    전략 파라미터 동적 생성 위젯

    선택된 전략에 따라 파라미터 입력 UI를 동적으로 생성합니다.
    """

    paramsChanged = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_strategy_name = ""
        self.param_widgets: Dict[str, QWidget] = {}
        self._init_ui()

    def _init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.params_frame = QFrame()
        self.params_layout = QFormLayout(self.params_frame)
        self.params_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.params_frame)

    def set_strategy(self, strategy_name: str):
        """전략 설정 및 파라미터 UI 생성"""
        if strategy_name == self.current_strategy_name:
            return

        self.current_strategy_name = strategy_name
        self._clear_params()

        strategy_class = StrategyRegistry.get(strategy_name)
        if strategy_class is None:
            return

        for name, config in strategy_class.parameters.items():
            widget = self._create_param_widget(name, config)
            if widget:
                label = config.get('label', name)
                self.params_layout.addRow(f"{label}:", widget)
                self.param_widgets[name] = widget

    def _clear_params(self):
        """파라미터 위젯 초기화"""
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.param_widgets.clear()

    def _create_param_widget(self, name: str, config: dict) -> Optional[QWidget]:
        """파라미터 타입에 따른 위젯 생성"""
        param_type = config.get('type', 'float')
        default = config.get('default', 0)

        if param_type == 'float':
            widget = QDoubleSpinBox()
            widget.setRange(config.get('min', -1000), config.get('max', 1000))
            widget.setValue(default)
            widget.setSingleStep(config.get('step', 0.1))
            widget.valueChanged.connect(self._on_param_changed)
            return widget

        elif param_type == 'int':
            widget = QSpinBox()
            widget.setRange(config.get('min', 0), config.get('max', 1000))
            widget.setValue(int(default))
            widget.setSingleStep(config.get('step', 1))
            widget.valueChanged.connect(self._on_param_changed)
            return widget

        elif param_type == 'bool':
            widget = QCheckBox()
            widget.setChecked(bool(default))
            widget.stateChanged.connect(self._on_param_changed)
            return widget

        elif param_type == 'choice':
            widget = QComboBox()
            choices = config.get('choices', [])
            labels = config.get('labels', choices)
            for i, choice in enumerate(choices):
                label = labels[i] if i < len(labels) else choice
                widget.addItem(label, choice)
            if default in choices:
                widget.setCurrentIndex(choices.index(default))
            widget.currentIndexChanged.connect(self._on_param_changed)
            return widget

        return None

    def _on_param_changed(self):
        """파라미터 변경 시 시그널 발생"""
        self.paramsChanged.emit(self.get_params())

    def get_params(self) -> dict:
        """현재 파라미터 값 반환"""
        params = {}

        strategy_class = StrategyRegistry.get(self.current_strategy_name)
        if strategy_class is None:
            return params

        for name, config in strategy_class.parameters.items():
            widget = self.param_widgets.get(name)
            if widget is None:
                continue

            param_type = config.get('type', 'float')

            if param_type in ('float', 'int'):
                params[name] = widget.value()
            elif param_type == 'bool':
                params[name] = widget.isChecked()
            elif param_type == 'choice':
                params[name] = widget.currentData()

        return params

    def set_params(self, params: dict):
        """파라미터 값 설정"""
        for name, value in params.items():
            widget = self.param_widgets.get(name)
            if widget is None:
                continue

            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                index = widget.findData(value)
                if index >= 0:
                    widget.setCurrentIndex(index)


# ============================================================================
# 입력 패널
# ============================================================================

class InputPanel(QScrollArea):
    """
    좌측 입력 패널

    백테스트에 필요한 모든 입력을 받습니다.
    """

    runClicked = pyqtSignal()
    configChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumWidth(320)
        self.setMaximumWidth(400)

        # 메인 컨테이너
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(16)

        # 프리셋
        preset_group = QGroupBox("Quick Start")
        preset_layout = QVBoxLayout(preset_group)
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("-- 프리셋 선택 --", None)
        for name in get_preset_names():
            self.preset_combo.addItem(name, name)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self.preset_combo)
        layout.addWidget(preset_group)

        # 종목/자산
        ticker_group = QGroupBox("종목 / 자산")
        ticker_layout = QVBoxLayout(ticker_group)
        self.ticker_widget = TickerInputWidget()
        self.ticker_widget.tickersChanged.connect(lambda t, w: self.configChanged.emit())
        ticker_layout.addWidget(self.ticker_widget)

        # 벤치마크
        bench_layout = QHBoxLayout()
        bench_layout.addWidget(QLabel("벤치마크:"))
        self.benchmark_combo = QComboBox()
        self.benchmark_combo.addItems(["SPY", "QQQ", "VOO", "IWM", "VTI"])
        self.benchmark_combo.setEditable(True)
        bench_layout.addWidget(self.benchmark_combo)
        ticker_layout.addLayout(bench_layout)

        layout.addWidget(ticker_group)

        # 기간 설정
        period_group = QGroupBox("기간 설정")
        period_layout = QFormLayout(period_group)

        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate.currentDate().addYears(-5))
        period_layout.addRow("시작일:", self.start_date)

        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate())
        period_layout.addRow("종료일:", self.end_date)

        layout.addWidget(period_group)

        # 투자 규칙
        invest_group = QGroupBox("투자 규칙")
        invest_layout = QFormLayout(invest_group)

        self.payday_spin = QSpinBox()
        self.payday_spin.setRange(1, 28)
        self.payday_spin.setValue(25)
        invest_layout.addRow("월급날:", self.payday_spin)

        self.monthly_invest = QDoubleSpinBox()
        self.monthly_invest.setRange(100, 1000000)
        self.monthly_invest.setValue(4000)
        self.monthly_invest.setPrefix("$ ")
        self.monthly_invest.setSingleStep(100)
        invest_layout.addRow("월 투자금:", self.monthly_invest)

        self.annual_increase = QDoubleSpinBox()
        self.annual_increase.setRange(0, 50)
        self.annual_increase.setValue(0)
        self.annual_increase.setSuffix(" %")
        invest_layout.addRow("연간 증가율:", self.annual_increase)

        layout.addWidget(invest_group)

        # 비용
        cost_group = QGroupBox("비용")
        cost_layout = QFormLayout(cost_group)

        self.commission = QDoubleSpinBox()
        self.commission.setRange(0, 5)
        self.commission.setValue(0.1)
        self.commission.setSuffix(" %")
        self.commission.setDecimals(2)
        cost_layout.addRow("수수료:", self.commission)

        self.slippage = QDoubleSpinBox()
        self.slippage.setRange(0, 5)
        self.slippage.setValue(0.1)
        self.slippage.setSuffix(" %")
        self.slippage.setDecimals(2)
        cost_layout.addRow("슬리피지:", self.slippage)

        layout.addWidget(cost_group)

        # 전략
        strategy_group = QGroupBox("전략")
        strategy_layout = QVBoxLayout(strategy_group)

        self.strategy_combo = QComboBox()
        for name in StrategyRegistry.get_names():
            strategy_class = StrategyRegistry.get(name)
            self.strategy_combo.addItem(f"{name}", name)
        self.strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        strategy_layout.addWidget(self.strategy_combo)

        self.strategy_desc = QLabel()
        self.strategy_desc.setWordWrap(True)
        self.strategy_desc.setStyleSheet("color: gray; font-size: 11px;")
        strategy_layout.addWidget(self.strategy_desc)

        self.strategy_params = StrategyParamsWidget()
        self.strategy_params.paramsChanged.connect(lambda p: self.configChanged.emit())
        strategy_layout.addWidget(self.strategy_params)

        layout.addWidget(strategy_group)

        # 고급 설정
        advanced_group = QGroupBox("고급 설정")
        advanced_group.setCheckable(True)
        advanced_group.setChecked(False)
        advanced_layout = QFormLayout(advanced_group)

        self.holiday_combo = QComboBox()
        self.holiday_combo.addItem("직전 거래일", HolidayAdjustment.PREVIOUS)
        self.holiday_combo.addItem("직후 거래일", HolidayAdjustment.NEXT)
        advanced_layout.addRow("휴장일 처리:", self.holiday_combo)

        self.rebalance_combo = QComboBox()
        self.rebalance_combo.addItem("없음", RebalanceFrequency.NONE)
        self.rebalance_combo.addItem("월간", RebalanceFrequency.MONTHLY)
        self.rebalance_combo.addItem("분기", RebalanceFrequency.QUARTERLY)
        self.rebalance_combo.addItem("연간", RebalanceFrequency.YEARLY)
        advanced_layout.addRow("리밸런싱:", self.rebalance_combo)

        self.cash_interest = QDoubleSpinBox()
        self.cash_interest.setRange(0, 10)
        self.cash_interest.setValue(0)
        self.cash_interest.setSuffix(" %")
        advanced_layout.addRow("현금 이자율:", self.cash_interest)

        layout.addWidget(advanced_group)

        # 실행 버튼
        self.run_btn = QPushButton("Run Backtest")
        self.run_btn.setProperty("primary", True)
        self.run_btn.setMinimumHeight(48)
        self.run_btn.setFont(QFont("", 14, QFont.Weight.Bold))
        self.run_btn.clicked.connect(self.runClicked.emit)
        layout.addWidget(self.run_btn)

        # 진행바
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addStretch()

        self.setWidget(container)

        # 초기 전략 설정
        self._on_strategy_changed()

    def _on_preset_changed(self, index):
        """프리셋 선택 변경"""
        preset_name = self.preset_combo.currentData()
        if preset_name:
            config = apply_preset(preset_name)
            if config:
                self.set_config(config)

    def _on_strategy_changed(self):
        """전략 선택 변경"""
        strategy_name = self.strategy_combo.currentData()
        self.strategy_params.set_strategy(strategy_name)

        strategy_class = StrategyRegistry.get(strategy_name)
        if strategy_class:
            self.strategy_desc.setText(strategy_class.description)

        self.configChanged.emit()

    def get_config(self) -> BacktestConfig:
        """현재 설정 반환"""
        tickers, weights = self.ticker_widget.get_values()

        return BacktestConfig(
            start_date=self.start_date.date().toPyDate(),
            end_date=self.end_date.date().toPyDate(),
            tickers=tickers,
            weights=weights,
            payday=self.payday_spin.value(),
            monthly_investment=self.monthly_invest.value(),
            annual_increase_rate=self.annual_increase.value() / 100,
            commission_rate=self.commission.value() / 100,
            slippage_rate=self.slippage.value() / 100,
            holiday_adjustment=self.holiday_combo.currentData(),
            rebalance_frequency=self.rebalance_combo.currentData(),
            cash_interest_rate=self.cash_interest.value() / 100,
            benchmark=self.benchmark_combo.currentText(),
            strategy_name=self.strategy_combo.currentData(),
            strategy_params=self.strategy_params.get_params(),
        )

    def set_config(self, config: BacktestConfig):
        """설정 적용"""
        if config.tickers and config.weights:
            self.ticker_widget.set_values(config.tickers, config.weights)

        if config.start_date:
            self.start_date.setDate(QDate(config.start_date.year,
                                          config.start_date.month,
                                          config.start_date.day))
        if config.end_date:
            self.end_date.setDate(QDate(config.end_date.year,
                                        config.end_date.month,
                                        config.end_date.day))

        self.payday_spin.setValue(config.payday)
        self.monthly_invest.setValue(config.monthly_investment)
        self.annual_increase.setValue(config.annual_increase_rate * 100)
        self.commission.setValue(config.commission_rate * 100)
        self.slippage.setValue(config.slippage_rate * 100)

        # 전략
        index = self.strategy_combo.findData(config.strategy_name)
        if index >= 0:
            self.strategy_combo.setCurrentIndex(index)
            self.strategy_params.set_params(config.strategy_params)

        # 벤치마크
        self.benchmark_combo.setCurrentText(config.benchmark)

    def set_running(self, running: bool):
        """실행 중 상태 설정"""
        self.run_btn.setEnabled(not running)
        self.progress_bar.setVisible(running)
        if running:
            self.progress_bar.setValue(0)

    def set_progress(self, current: int, total: int, message: str = ""):
        """진행 상태 업데이트"""
        if total > 0:
            self.progress_bar.setValue(int(current / total * 100))


# ============================================================================
# 차트 캔버스
# ============================================================================

class ChartCanvas(FigureCanvas):
    """
    matplotlib 차트 캔버스

    향후 확장 포인트:
    - 인터랙티브 줌/팬
    - 크로스헤어
    - 데이터 포인트 툴팁
    """

    def __init__(self, parent=None, figsize=(8, 6), dpi=100):
        self.fig = Figure(figsize=figsize, dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

        # 여백 설정
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

    def clear(self):
        """차트 초기화"""
        self.fig.clear()
        self.draw()

    def set_dark_theme(self, dark: bool = True):
        """테마 설정"""
        if dark:
            self.fig.patch.set_facecolor('#1e1e1e')
            plt.style.use('dark_background')
        else:
            self.fig.patch.set_facecolor('white')
            plt.style.use('default')


# ============================================================================
# 차트 패널
# ============================================================================

class ChartPanel(QWidget):
    """
    차트 표시 패널

    여러 차트를 탭으로 구성합니다.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self.result: Optional[BacktestResult] = None

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # 자산 곡선 탭
        self.equity_tab = QWidget()
        equity_layout = QVBoxLayout(self.equity_tab)
        self.equity_canvas = ChartCanvas(figsize=(10, 6))
        self.equity_toolbar = NavigationToolbar(self.equity_canvas, self)
        equity_layout.addWidget(self.equity_toolbar)
        equity_layout.addWidget(self.equity_canvas)
        self.tabs.addTab(self.equity_tab, "자산 곡선")

        # 드로우다운 탭
        self.dd_tab = QWidget()
        dd_layout = QVBoxLayout(self.dd_tab)
        self.dd_canvas = ChartCanvas(figsize=(10, 4))
        dd_layout.addWidget(self.dd_canvas)
        self.tabs.addTab(self.dd_tab, "Drawdown")

        # 월별 수익률 탭
        self.monthly_tab = QWidget()
        monthly_layout = QVBoxLayout(self.monthly_tab)
        self.monthly_canvas = ChartCanvas(figsize=(10, 6))
        monthly_layout.addWidget(self.monthly_canvas)
        self.tabs.addTab(self.monthly_tab, "월별 수익률")

        # 전략 비교 탭
        self.compare_tab = QWidget()
        compare_layout = QVBoxLayout(self.compare_tab)
        self.compare_canvas = ChartCanvas(figsize=(10, 6))
        compare_layout.addWidget(self.compare_canvas)
        self.tabs.addTab(self.compare_tab, "전략 비교")

    def set_result(self, result: BacktestResult):
        """백테스트 결과 설정 및 차트 갱신"""
        self.result = result
        try:
            self._plot_equity_curve()
        except Exception as e:
            logger.error(f"자산 곡선 차트 오류: {e}")
        try:
            self._plot_drawdown()
        except Exception as e:
            logger.error(f"드로우다운 차트 오류: {e}")
        try:
            self._plot_monthly_returns()
        except Exception as e:
            logger.error(f"월별 수익률 차트 오류: {e}")

    def _plot_equity_curve(self):
        """자산 곡선 플롯"""
        if self.result is None or self.result.equity_curve is None:
            return

        self.equity_canvas.fig.clear()
        ax = self.equity_canvas.fig.add_subplot(111)

        # 포트폴리오 곡선
        dates = self.result.equity_curve.index
        values = self.result.equity_curve.values

        ax.plot(dates, values, label='Portfolio', linewidth=2, color='#2196F3')

        # 벤치마크
        if self.result.benchmark_curve is not None and len(self.result.benchmark_curve) > 0:
            bench_dates = self.result.benchmark_curve.index
            bench_values = self.result.benchmark_curve.values
            ax.plot(bench_dates, bench_values, label='Benchmark', linewidth=1.5,
                   color='#FF9800', linestyle='--', alpha=0.7)

        # 투자 타이밍 마커
        if self.result.investment_history is not None:
            inv_dates = pd.to_datetime(self.result.investment_history['date'])
            for d in inv_dates:
                ax.axvline(x=d, color='green', alpha=0.2, linewidth=0.5)

        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Equity Curve')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # 날짜 포맷
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        self.equity_canvas.fig.autofmt_xdate()

        self.equity_canvas.draw()

    def _plot_drawdown(self):
        """드로우다운 플롯"""
        if self.result is None or self.result.drawdown_curve is None:
            return

        self.dd_canvas.fig.clear()
        ax = self.dd_canvas.fig.add_subplot(111)

        dates = self.result.drawdown_curve.index
        values = self.result.drawdown_curve.values * 100  # 퍼센트

        ax.fill_between(dates, values, 0, alpha=0.5, color='#EF5350')
        ax.plot(dates, values, color='#EF5350', linewidth=1)

        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Drawdown')
        ax.grid(True, alpha=0.3)

        # MDD 표시
        if self.result.metrics:
            mdd = self.result.metrics.mdd * 100
            ax.axhline(y=mdd, color='red', linestyle='--', alpha=0.7,
                      label=f'MDD: {mdd:.1f}%')
            ax.legend()

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        self.dd_canvas.fig.autofmt_xdate()

        self.dd_canvas.draw()

    def _plot_monthly_returns(self):
        """월별 수익률 히트맵"""
        if self.result is None or self.result.monthly_returns is None:
            return

        self.monthly_canvas.fig.clear()

        heatmap_data = create_returns_heatmap_data(self.result.monthly_returns)
        if heatmap_data.empty:
            return

        ax = self.monthly_canvas.fig.add_subplot(111)

        # 히트맵
        data = heatmap_data.values * 100
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto',
                       vmin=-10, vmax=10)

        # 축 설정
        ax.set_xticks(np.arange(len(heatmap_data.columns)))
        ax.set_yticks(np.arange(len(heatmap_data.index)))
        ax.set_xticklabels(heatmap_data.columns)
        ax.set_yticklabels(heatmap_data.index)

        # 값 표시
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                val = data[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > 5 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                           color=color, fontsize=8)

        ax.set_title('Monthly Returns (%)')
        self.monthly_canvas.fig.colorbar(im, ax=ax)

        self.monthly_canvas.draw()

    def plot_comparison(self, results: Dict[str, BacktestResult]):
        """전략 비교 플롯"""
        self.compare_canvas.fig.clear()
        ax = self.compare_canvas.fig.add_subplot(111)

        colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']

        for i, (name, result) in enumerate(results.items()):
            if result.equity_curve is not None:
                # 정규화 (첫 값을 1로)
                normalized = result.equity_curve / result.equity_curve.iloc[0]
                color = colors[i % len(colors)]
                ax.plot(normalized.index, normalized.values, label=name,
                       linewidth=2, color=color)

        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Value')
        ax.set_title('Strategy Comparison')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        self.compare_canvas.fig.autofmt_xdate()

        self.compare_canvas.draw()

    def clear_charts(self):
        """모든 차트 초기화"""
        self.equity_canvas.clear()
        self.dd_canvas.clear()
        self.monthly_canvas.clear()
        self.compare_canvas.clear()


# ============================================================================
# 결과 패널
# ============================================================================

class ResultsPanel(QWidget):
    """
    결과 및 성과 지표 표시 패널
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 헤더
        header = QLabel("Performance Metrics")
        header.setObjectName("headerLabel")
        layout.addWidget(header)

        # 주요 지표 그리드
        metrics_frame = QFrame()
        metrics_frame.setObjectName("card")
        metrics_layout = QGridLayout(metrics_frame)
        metrics_layout.setSpacing(16)

        # 지표 라벨들
        self.metric_labels = {}
        metrics = [
            ("cagr", "CAGR", 0, 0),
            ("total_return", "Total Return", 0, 1),
            ("mdd", "Max Drawdown", 0, 2),
            ("sharpe", "Sharpe Ratio", 1, 0),
            ("sortino", "Sortino Ratio", 1, 1),
            ("calmar", "Calmar Ratio", 1, 2),
            ("invested", "Total Invested", 2, 0),
            ("final", "Final Value", 2, 1),
            ("profit", "Profit", 2, 2),
        ]

        for key, label, row, col in metrics:
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(4)

            name_label = QLabel(label)
            name_label.setObjectName("subHeaderLabel")
            container_layout.addWidget(name_label)

            value_label = QLabel("--")
            value_label.setObjectName("metricValue")
            container_layout.addWidget(value_label)

            self.metric_labels[key] = value_label
            metrics_layout.addWidget(container, row, col)

        layout.addWidget(metrics_frame)

        # 상세 테이블
        self.detail_table = QTableWidget()
        self.detail_table.setColumnCount(2)
        self.detail_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.detail_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.detail_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.detail_table)

    def set_result(self, result: BacktestResult):
        """결과 표시"""
        if result.metrics is None:
            return

        m = result.metrics

        # 주요 지표 업데이트
        self._set_metric("cagr", format_percent(m.cagr), m.cagr >= 0)
        self._set_metric("total_return", format_percent(m.total_return), m.total_return >= 0)
        self._set_metric("mdd", format_percent(m.mdd), False)  # MDD는 항상 음수
        self._set_metric("sharpe", f"{m.sharpe_ratio:.2f}", m.sharpe_ratio >= 1)
        self._set_metric("sortino", f"{m.sortino_ratio:.2f}", m.sortino_ratio >= 1)
        self._set_metric("calmar", f"{m.calmar_ratio:.2f}", m.calmar_ratio >= 1)
        self._set_metric("invested", format_currency(m.total_invested))
        self._set_metric("final", format_currency(m.final_value))
        self._set_metric("profit", format_currency(m.profit), m.profit >= 0)

        # 상세 테이블
        details = [
            ("Win Rate (Monthly)", format_percent(m.win_rate)),
            ("Winning Months", str(m.winning_months)),
            ("Losing Months", str(m.losing_months)),
            ("Volatility (Annual)", format_percent(m.volatility)),
            ("Profit Rate", format_percent(m.profit_rate)),
            ("Total Days", str(m.total_days)),
            ("Total Years", f"{m.total_years:.1f}"),
            ("Alpha", format_percent(m.alpha)),
            ("Beta", f"{m.beta:.2f}"),
        ]

        self.detail_table.setRowCount(len(details))
        for i, (name, value) in enumerate(details):
            self.detail_table.setItem(i, 0, QTableWidgetItem(name))
            self.detail_table.setItem(i, 1, QTableWidgetItem(value))

    def _set_metric(self, key: str, value: str, positive: bool = None):
        """지표 값 설정"""
        label = self.metric_labels.get(key)
        if label:
            label.setText(value)
            if positive is not None:
                label.setObjectName("positiveValue" if positive else "negativeValue")
                label.style().unpolish(label)
                label.style().polish(label)

    def clear(self):
        """결과 초기화"""
        for label in self.metric_labels.values():
            label.setText("--")
        self.detail_table.setRowCount(0)


# ============================================================================
# 로그 위젯
# ============================================================================

class LogWidget(QTextEdit):
    """
    로그 출력 위젯
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("logWidget")
        self.setReadOnly(True)
        self.setMaximumHeight(150)

    def append_log(self, message: str, level: int = 20):
        """로그 추가"""
        color = "white"
        if level >= 40:  # ERROR
            color = "#EF5350"
        elif level >= 30:  # WARNING
            color = "#FFC107"
        elif level >= 20:  # INFO
            color = "#4CAF50"
        else:  # DEBUG
            color = "#9E9E9E"

        self.append(f'<span style="color:{color}">{message}</span>')

    def clear_log(self):
        """로그 초기화"""
        self.clear()


# ============================================================================
# 향후 확장 포인트
# ============================================================================

# TODO: 드래그앤드롭 티커 입력
# class DragDropTickerWidget(QWidget):
#     pass

# TODO: 인터랙티브 차트 (plotly)
# class InteractiveChartWidget(QWidget):
#     pass

# TODO: 비교 테이블 위젯
# class ComparisonTableWidget(QWidget):
#     pass
