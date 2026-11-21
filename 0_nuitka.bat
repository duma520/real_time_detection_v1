cls
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

nuitka --standalone ^
    --enable-plugin=pyside6 ^
    --windows-console-mode=disable ^
    --windows-icon-from-ico=icon.ico ^
    --include-package-data=pypinyin ^
    --include-package=cv2 ^
    --include-package=numpy ^
    --include-package=psutil ^
    --include-package=win32gui ^
    --include-package=win32process ^
    --include-package=mss ^
    --include-package=PySide6 ^
    --include-qt-plugins=all ^
    --follow-imports ^
    --jobs=4 ^
    --output-dir=dist ^
    real_time_detection.py

