from PyQt5 import QtWidgets, QtCore, QtGui

class AnimatedButton(QtWidgets.QPushButton):
    def __init__(self, *args, glow_color=QtGui.QColor("#88C0D0"), max_blur=30, **kwargs):
        super().__init__(*args, **kwargs)
        # effetto glow esterno
        self._effect = QtWidgets.QGraphicsDropShadowEffect(self)
        self._effect.setOffset(0, 0)
        self._effect.setColor(glow_color)
        self._effect.setBlurRadius(0)
        self.setGraphicsEffect(self._effect)

        # animazione bloom all’entrata
        self._enter_anim = QtCore.QPropertyAnimation(self._effect, b"blurRadius", self)
        self._enter_anim.setDuration(300)
        self._enter_anim.setStartValue(0)
        self._enter_anim.setEndValue(max_blur)

        # animazione bloom all’uscita
        self._leave_anim = QtCore.QPropertyAnimation(self._effect, b"blurRadius", self)
        self._leave_anim.setDuration(300)
        self._leave_anim.setStartValue(max_blur)
        self._leave_anim.setEndValue(0)

        # animazione colore glow (opzionale)
        self._color_anim = QtCore.QPropertyAnimation(self._effect, b"color", self)
        self._color_anim.setDuration(300)
        self._color_anim.setStartValue(glow_color.lighter(150))
        self._color_anim.setEndValue(glow_color)

    def enterEvent(self, event):
        self._enter_anim.start()
        self._color_anim.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._leave_anim.start()
        super().leaveEvent(event)
