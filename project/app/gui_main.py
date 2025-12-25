"""
메인 윈도우 모듈

PyQt6 기반의 메인 애플리케이션 윈도우를 정의합니다.

향후 확장 포인트:
- 멀티 윈도우 지원
- 플러그인 메뉴
- 키보드 단축키 커스터마이징
"""

import sys
from pathlib import Path
from typing import Optional, Dict

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QMenuBar, QMenu, QToolBar, QStatusBar,
    QMessageBox, QFileDialog, QLabel, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QAction, QIcon, QFont, QKeySequence

from app.widgets import InputPanel, ChartPanel, ResultsPanel, LogWidget
from app.backtest_engine import BacktestEngine, BacktestConfig
from app.analytics import BacktestResult
from app.persistence import (
    settings_manager, project_manager,
    export_full_report, config_to_dict, dict_to_config
)
from app.utils import logger, LogHandler, validate_all_inputs


# ============================================================================
# 테마 관리
# ============================================================================

# 다크 테마 색상
DARK_THEME = {
    'bg_primary': '#1e1e1e',
    'bg_secondary': '#252526',
    'bg_tertiary': '#2d2d2d',
    'text_primary': '#ffffff',
    'text_secondary': '#b0b0b0',
    'text_on_accent': '#ffffff',
    'border_color': '#3c3c3c',
    'accent_color': '#0078d4',
    'accent_hover': '#1084d8',
    'accent_pressed': '#006cc1',
    'hover_color': '#3a3a3a',
    'pressed_color': '#4a4a4a',
    'input_bg': '#1e1e1e',
    'disabled_bg': '#2d2d2d',
    'disabled_text': '#6e6e6e',
    'success_color': '#4caf50',
    'error_color': '#f44336',
    'warning_color': '#ff9800',
    'scrollbar_color': '#4a4a4a',
    'scrollbar_hover': '#5a5a5a',
}

# 라이트 테마 색상
LIGHT_THEME = {
    'bg_primary': '#ffffff',
    'bg_secondary': '#f5f5f5',
    'bg_tertiary': '#e0e0e0',
    'text_primary': '#212121',
    'text_secondary': '#757575',
    'text_on_accent': '#ffffff',
    'border_color': '#e0e0e0',
    'accent_color': '#1976d2',
    'accent_hover': '#1565c0',
    'accent_pressed': '#0d47a1',
    'hover_color': '#eeeeee',
    'pressed_color': '#e0e0e0',
    'input_bg': '#ffffff',
    'disabled_bg': '#f5f5f5',
    'disabled_text': '#bdbdbd',
    'success_color': '#4caf50',
    'error_color': '#f44336',
    'warning_color': '#ff9800',
    'scrollbar_color': '#bdbdbd',
    'scrollbar_hover': '#9e9e9e',
}


def load_stylesheet(theme: str = 'dark') -> str:
    """QSS 스타일시트 로드 및 테마 적용"""
    qss_path = Path(__file__).parent / "styles.qss"

    try:
        with open(qss_path, 'r', encoding='utf-8') as f:
            stylesheet = f.read()
    except FileNotFoundError:
        logger.warning("styles.qss 파일을 찾을 수 없습니다.")
        return ""

    # 테마 색상 적용
    colors = DARK_THEME if theme == 'dark' else LIGHT_THEME

    for key, value in colors.items():
        stylesheet = stylesheet.replace(f'{{{key}}}', value)

    return stylesheet


# ============================================================================
# 백테스트 워커 스레드
# ============================================================================

class BacktestWorker(QThread):
    """
    백테스트 실행을 위한 워커 스레드

    UI 블로킹 없이 백테스트를 실행합니다.
    """

    finished = pyqtSignal(object)  # BacktestResult 또는 None
    progress = pyqtSignal(int, int, str)  # current, total, message
    error = pyqtSignal(str)

    def __init__(self, config: BacktestConfig):
        super().__init__()
        self.config = config
        self._is_cancelled = False

    def run(self):
        """백테스트 실행"""
        try:
            engine = BacktestEngine(self.config)
            result = engine.run(self._progress_callback)

            if not self._is_cancelled:
                self.finished.emit(result)

        except Exception as e:
            logger.error(f"백테스트 오류: {e}")
            self.error.emit(str(e))

    def _progress_callback(self, current: int, total: int, message: str):
        """진행 상황 콜백"""
        if not self._is_cancelled:
            self.progress.emit(current, total, message)

    def cancel(self):
        """백테스트 취소"""
        self._is_cancelled = True


# ============================================================================
# 메인 윈도우
# ============================================================================

class MainWindow(QMainWindow):
    """
    DCA Backtest GUI 메인 윈도우

    향후 확장 포인트:
    - 도킹 가능한 패널
    - 멀티 탭 프로젝트
    - 실시간 데이터 연동
    """

    def __init__(self):
        super().__init__()

        self.current_theme = 'dark'
        self.worker: Optional[BacktestWorker] = None
        self.current_result: Optional[BacktestResult] = None
        self.comparison_results: Dict[str, BacktestResult] = {}

        self._init_ui()
        self._init_menu()
        self._init_toolbar()
        self._init_statusbar()
        self._setup_logging()
        self._load_settings()

        self.setWindowTitle("DCA Backtest - 월급날 적립식 투자 시뮬레이터")
        self.resize(1400, 900)

    def _init_ui(self):
        """UI 초기화"""
        # 메인 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 메인 스플리터 (좌: 입력, 우: 차트+결과)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)

        # 좌측: 입력 패널
        self.input_panel = InputPanel()
        self.input_panel.runClicked.connect(self._on_run_clicked)
        main_splitter.addWidget(self.input_panel)

        # 우측 컨테이너
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)

        # 우측 스플리터 (상: 차트, 하: 결과+로그)
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_layout.addWidget(right_splitter)

        # 차트 패널
        self.chart_panel = ChartPanel()
        right_splitter.addWidget(self.chart_panel)

        # 하단 컨테이너 (결과 + 로그)
        bottom_container = QWidget()
        bottom_layout = QHBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        # 결과 패널
        self.results_panel = ResultsPanel()
        bottom_layout.addWidget(self.results_panel, stretch=2)

        # 로그 위젯
        log_container = QWidget()
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(0, 0, 0, 0)

        log_header = QLabel("Log")
        log_header.setObjectName("subHeaderLabel")
        log_layout.addWidget(log_header)

        self.log_widget = LogWidget()
        log_layout.addWidget(self.log_widget)

        bottom_layout.addWidget(log_container, stretch=1)

        right_splitter.addWidget(bottom_container)

        # 스플리터 비율
        right_splitter.setSizes([600, 200])

        main_splitter.addWidget(right_container)
        main_splitter.setSizes([320, 1080])

    def _init_menu(self):
        """메뉴바 초기화"""
        menubar = self.menuBar()

        # 파일 메뉴
        file_menu = menubar.addMenu("파일(&F)")

        new_action = QAction("새 프로젝트(&N)", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._new_project)
        file_menu.addAction(new_action)

        open_action = QAction("열기(&O)...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_project)
        file_menu.addAction(open_action)

        save_action = QAction("저장(&S)", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._save_project)
        file_menu.addAction(save_action)

        save_as_action = QAction("다른 이름으로 저장(&A)...", self)
        save_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_as_action.triggered.connect(self._save_project_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        export_action = QAction("결과 내보내기(&E)...", self)
        export_action.triggered.connect(self._export_results)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction("종료(&X)", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 보기 메뉴
        view_menu = menubar.addMenu("보기(&V)")

        self.theme_action = QAction("라이트 테마", self)
        self.theme_action.setCheckable(True)
        self.theme_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(self.theme_action)

        clear_log_action = QAction("로그 지우기", self)
        clear_log_action.triggered.connect(self.log_widget.clear_log)
        view_menu.addAction(clear_log_action)

        # 백테스트 메뉴
        backtest_menu = menubar.addMenu("백테스트(&B)")

        run_action = QAction("실행(&R)", self)
        run_action.setShortcut(QKeySequence("F5"))
        run_action.triggered.connect(self._on_run_clicked)
        backtest_menu.addAction(run_action)

        compare_action = QAction("전략 비교(&C)...", self)
        compare_action.triggered.connect(self._compare_strategies)
        backtest_menu.addAction(compare_action)

        backtest_menu.addSeparator()

        clear_cache_action = QAction("캐시 삭제", self)
        clear_cache_action.triggered.connect(self._clear_cache)
        backtest_menu.addAction(clear_cache_action)

        # 도움말 메뉴
        help_menu = menubar.addMenu("도움말(&H)")

        about_action = QAction("정보(&A)", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _init_toolbar(self):
        """툴바 초기화"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # 실행 버튼
        run_action = QAction("Run", self)
        run_action.setToolTip("백테스트 실행 (F5)")
        run_action.triggered.connect(self._on_run_clicked)
        toolbar.addAction(run_action)

        toolbar.addSeparator()

        # 테마 토글
        theme_action = QAction("Theme", self)
        theme_action.setToolTip("테마 전환")
        theme_action.triggered.connect(self._toggle_theme)
        toolbar.addAction(theme_action)

    def _init_statusbar(self):
        """상태바 초기화"""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        self.status_label = QLabel("준비")
        self.statusbar.addWidget(self.status_label)

    def _setup_logging(self):
        """로깅 설정"""
        def log_callback(message: str, level: int):
            self.log_widget.append_log(message, level)

        handler = LogHandler(log_callback)
        logger.addHandler(handler)

    def _load_settings(self):
        """설정 로드"""
        settings = settings_manager.load()

        # 테마 적용
        self.current_theme = settings.theme
        self._apply_theme()

        # 마지막 사용 설정
        if settings.last_tickers:
            # 마지막 설정 복원은 선택적
            pass

    def _save_settings(self):
        """설정 저장"""
        settings_manager.settings.theme = self.current_theme
        settings_manager.save()

    def _apply_theme(self):
        """테마 적용"""
        stylesheet = load_stylesheet(self.current_theme)
        QApplication.instance().setStyleSheet(stylesheet)

        # 차트 테마
        self.chart_panel.equity_canvas.set_dark_theme(self.current_theme == 'dark')
        self.chart_panel.dd_canvas.set_dark_theme(self.current_theme == 'dark')
        self.chart_panel.monthly_canvas.set_dark_theme(self.current_theme == 'dark')
        self.chart_panel.compare_canvas.set_dark_theme(self.current_theme == 'dark')

        self.theme_action.setChecked(self.current_theme == 'light')

    def _toggle_theme(self):
        """테마 토글"""
        self.current_theme = 'light' if self.current_theme == 'dark' else 'dark'
        self._apply_theme()

    # ========================================================================
    # 파일 작업
    # ========================================================================

    def _new_project(self):
        """새 프로젝트"""
        project_manager.new_project()
        self.setWindowTitle("DCA Backtest - 새 프로젝트")
        self.log_widget.clear_log()
        self.chart_panel.clear_charts()
        self.results_panel.clear()
        logger.info("새 프로젝트 생성")

    def _open_project(self):
        """프로젝트 열기"""
        path, _ = QFileDialog.getOpenFileName(
            self, "프로젝트 열기", "",
            "DCA Project (*.dcaproj);;All Files (*)"
        )

        if path:
            project = project_manager.load_project(Path(path))
            if project:
                config = project_manager.get_config()
                if config:
                    self.input_panel.set_config(config)
                self.setWindowTitle(f"DCA Backtest - {project.name}")
                logger.info(f"프로젝트 로드: {path}")

    def _save_project(self):
        """프로젝트 저장"""
        if project_manager.current_path is None:
            self._save_project_as()
        else:
            config = self.input_panel.get_config()
            project_manager.update_config(config)
            if project_manager.save_project():
                logger.info("프로젝트 저장 완료")

    def _save_project_as(self):
        """다른 이름으로 저장"""
        path, _ = QFileDialog.getSaveFileName(
            self, "프로젝트 저장", "",
            "DCA Project (*.dcaproj);;All Files (*)"
        )

        if path:
            if not path.endswith('.dcaproj'):
                path += '.dcaproj'

            config = self.input_panel.get_config()
            project_manager.update_config(config)
            project_manager.current_project.name = Path(path).stem

            if project_manager.save_project(Path(path)):
                self.setWindowTitle(f"DCA Backtest - {Path(path).stem}")
                logger.info(f"프로젝트 저장: {path}")

    def _export_results(self):
        """결과 내보내기"""
        if self.current_result is None:
            QMessageBox.warning(self, "경고", "내보낼 결과가 없습니다.")
            return

        path = QFileDialog.getExistingDirectory(self, "결과 저장 폴더 선택")

        if path:
            if export_full_report(self.current_result, Path(path)):
                QMessageBox.information(self, "완료", f"결과가 {path}에 저장되었습니다.")
            else:
                QMessageBox.warning(self, "오류", "결과 저장에 실패했습니다.")

    # ========================================================================
    # 백테스트 실행
    # ========================================================================

    def _on_run_clicked(self):
        """백테스트 실행 버튼 클릭"""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            return

        config = self.input_panel.get_config()

        # 유효성 검증
        is_valid, errors = config.validate()
        if not is_valid:
            QMessageBox.warning(self, "입력 오류", "\n".join(errors))
            return

        # 실행
        self.input_panel.set_running(True)
        self.status_label.setText("백테스트 실행 중...")

        self.worker = BacktestWorker(config)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

        logger.info("백테스트 시작")

    def _on_progress(self, current: int, total: int, message: str):
        """진행 상황 업데이트"""
        self.input_panel.set_progress(current, total, message)
        self.status_label.setText(message)

    def _on_finished(self, result: Optional[BacktestResult]):
        """백테스트 완료"""
        self.input_panel.set_running(False)
        self.worker = None

        if result is None:
            self.status_label.setText("백테스트 실패")
            logger.error("백테스트 실패")
            return

        self.current_result = result

        try:
            self.chart_panel.set_result(result)
        except Exception as e:
            logger.error(f"차트 표시 오류: {e}")

        try:
            self.results_panel.set_result(result)
        except Exception as e:
            logger.error(f"결과 표시 오류: {e}")

        self.status_label.setText("백테스트 완료")
        logger.info("백테스트 완료")

        # 성과 요약 로그
        if result.metrics:
            m = result.metrics
            logger.info(f"CAGR: {m.cagr*100:.2f}% | MDD: {m.mdd*100:.2f}% | "
                       f"Sharpe: {m.sharpe_ratio:.2f}")

    def _on_error(self, error_msg: str):
        """백테스트 오류"""
        self.input_panel.set_running(False)
        self.worker = None
        self.status_label.setText("오류 발생")
        QMessageBox.critical(self, "오류", f"백테스트 중 오류가 발생했습니다:\n{error_msg}")

    def _compare_strategies(self):
        """전략 비교"""
        # 현재 설정으로 여러 전략 비교
        config = self.input_panel.get_config()

        is_valid, errors = config.validate()
        if not is_valid:
            QMessageBox.warning(self, "입력 오류", "\n".join(errors))
            return

        from app.strategies import StrategyRegistry
        from app.backtest_engine import compare_strategies

        strategies = [
            (name, {}) for name in StrategyRegistry.get_names()
        ]

        self.status_label.setText("전략 비교 실행 중...")
        self.input_panel.set_running(True)

        # 간단한 구현 (실제로는 별도 스레드에서 실행 권장)
        try:
            results = compare_strategies(
                config.tickers,
                config.weights,
                config.start_date,
                config.end_date,
                strategies,
                config.monthly_investment
            )

            self.comparison_results = results
            self.chart_panel.plot_comparison(results)
            self.chart_panel.tabs.setCurrentIndex(3)  # 비교 탭

            logger.info(f"{len(results)}개 전략 비교 완료")

        except Exception as e:
            logger.error(f"전략 비교 오류: {e}")
            QMessageBox.critical(self, "오류", str(e))

        finally:
            self.input_panel.set_running(False)
            self.status_label.setText("준비")

    def _clear_cache(self):
        """캐시 삭제"""
        from app.data_provider import cache_manager

        reply = QMessageBox.question(
            self, "확인",
            "모든 캐시 데이터를 삭제하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            cache_manager.clear_cache()
            logger.info("캐시 삭제 완료")

    def _show_about(self):
        """정보 대화상자"""
        QMessageBox.about(
            self, "DCA Backtest",
            "<h3>DCA Backtest</h3>"
            "<p>월급날 적립식 투자 백테스트 시뮬레이터</p>"
            "<p>Version 1.0.0</p>"
            "<hr>"
            "<p>매월 정해진 날짜에 미국 주식(ETF/개별종목)을 "
            "적립식으로 투자하는 전략을 백테스트합니다.</p>"
            "<p><b>제공 전략:</b></p>"
            "<ul>"
            "<li>Pure DCA</li>"
            "<li>Drawdown Tier DCA</li>"
            "<li>Trend Filter</li>"
            "<li>Volatility Control</li>"
            "</ul>"
        )

    def closeEvent(self, event):
        """종료 이벤트"""
        self._save_settings()

        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()

        event.accept()


# ============================================================================
# 애플리케이션 실행
# ============================================================================

def run_app():
    """애플리케이션 실행"""
    import os

    # Windows에서 DirectWrite 폰트 오류 방지
    if sys.platform == "win32":
        # FreeType 폰트 엔진 사용으로 DirectWrite 우회
        os.environ.setdefault("QT_QPA_PLATFORM", "windows:fontengine=freetype")

    app = QApplication(sys.argv)

    # 문제가 되는 폰트를 시스템 폰트로 대체
    QFont.insertSubstitution("MS Sans Serif", "Segoe UI")
    QFont.insertSubstitution("MS Shell Dlg", "Segoe UI")
    QFont.insertSubstitution("MS Shell Dlg 2", "Segoe UI")

    # 애플리케이션 기본 폰트 설정
    font = QFont("Segoe UI", 10)
    if not font.exactMatch():
        # Segoe UI가 없으면 대체 폰트 사용
        for fallback in ["Malgun Gothic", "Arial", "Helvetica"]:
            font = QFont(fallback, 10)
            if font.exactMatch():
                break
    app.setFont(font)

    # 메인 윈도우 생성 및 표시
    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(run_app())
