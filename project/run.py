#!/usr/bin/env python3
"""
DCA Backtest GUI - Entry Point

월급날 적립식 투자 백테스트 시뮬레이터

Usage:
    python run.py

Requirements:
    - Python 3.10+
    - PyQt6
    - pandas, numpy
    - yfinance
    - matplotlib
"""

import sys
import os

# Windows에서 DirectWrite 폰트 오류 방지 (QApplication 생성 전에 설정 필요)
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_PLATFORM", "windows:fontengine=freetype")

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def check_dependencies():
    """의존성 확인"""
    missing = []

    try:
        import PyQt6
    except ImportError:
        missing.append("PyQt6")

    try:
        import pandas
    except ImportError:
        missing.append("pandas")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import yfinance
    except ImportError:
        missing.append("yfinance")

    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")

    if missing:
        print("=" * 50)
        print("필수 패키지가 설치되지 않았습니다:")
        print()
        for pkg in missing:
            print(f"  - {pkg}")
        print()
        print("다음 명령어로 설치하세요:")
        print()
        print("  pip install -r requirements.txt")
        print()
        print("또는:")
        print()
        print(f"  pip install {' '.join(missing)}")
        print("=" * 50)
        return False

    return True


def setup_exception_handler():
    """전역 예외 핸들러 설정"""
    import traceback

    def exception_hook(exc_type, exc_value, exc_traceback):
        """처리되지 않은 예외를 잡아서 출력"""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        print("=" * 50)
        print("오류가 발생했습니다:")
        print("=" * 50)
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print("=" * 50)

    sys.excepthook = exception_hook


def main():
    """메인 함수"""
    # 전역 예외 핸들러 설정
    setup_exception_handler()

    # 의존성 확인
    if not check_dependencies():
        sys.exit(1)

    # 애플리케이션 실행
    from app.gui_main import run_app

    print("=" * 50)
    print("DCA Backtest GUI 시작...")
    print("=" * 50)

    try:
        return run_app()
    except Exception as e:
        print(f"애플리케이션 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
