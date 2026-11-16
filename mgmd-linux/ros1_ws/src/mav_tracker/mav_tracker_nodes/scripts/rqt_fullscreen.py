#!/usr/bin/env python3
"""Use the rqt ImageView plugin with the full WSLg display area."""

import sys

from rqt_gui.main import Main


class FullscreenImageView(Main):
    def create_application(self, argv):
        from python_qt_binding.QtCore import QTimer
        from python_qt_binding.QtWidgets import QMainWindow, QShortcut
        from python_qt_binding.QtGui import QKeySequence

        app = super().create_application(argv)
        timer = QTimer(app)

        def show_viewer():
            for window in app.topLevelWidgets():
                if isinstance(window, QMainWindow) and window.isVisible():
                    timer.stop()
                    window.showFullScreen()
                    QShortcut(QKeySequence("Escape"), window, activated=window.showNormal)
                    QShortcut(QKeySequence("F11"), window, activated=window.showFullScreen)
                    return

        timer.timeout.connect(show_viewer)
        timer.start(100)
        return app


def add_arguments(parser):
    parser.add_argument("topic", nargs="?")


if __name__ == "__main__":
    sys.exit(FullscreenImageView().main(sys.argv, standalone="rqt_image_view/ImageView",
                                        plugin_argument_provider=add_arguments))
