# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for AI 学术写作助手
Fixed: 强制排除 PyQt/PySide 依赖，解决 "Failed to load platform plugin windows" 错误
"""

import os
import sys
from PyInstaller.utils.hooks import collect_submodules

# 获取 spec 文件所在目录
spec_dir = os.path.dirname(os.path.abspath(SPEC))

# ----------------------------------------------------------------------
# 1. 显式排除列表：强制 PyInstaller 忽略 GUI 库
# ----------------------------------------------------------------------
excluded_modules = [
    # 标准 GUI 库
    'tkinter',
    '_tkinter',
    
    # 数据科学库 (如果你的后端没用到这些)
    'matplotlib', 
    'numpy', 
    'pandas', 
    'scipy', 
    'PIL',
    
    # === 关键修改：排除所有 Qt 相关库 ===
    'PyQt4', 'PyQt5', 'PyQt6',
    'PySide', 'PySide2', 'PySide6',
    'Qt', 'QtCore', 'QtGui', 'QtWidgets',
    'Qt5', 'Qt6',
]

# ----------------------------------------------------------------------
# 2. 隐式导入收集
# ----------------------------------------------------------------------
hidden_imports = [
    # === 修复 jaraco 报错的关键部分 ===
    'platformdirs',
    'platformdirs.windows', # 针对 Windows 平台的特定模块
    'jaraco',
    'jaraco.text',
    'jaraco.classes',
    'jaraco.context',
    'jaraco.functools',
    'more_itertools',
    'packaging',
    'packaging.version',
    'packaging.specifiers',
    'packaging.requirements',
    
    # 原有配置
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'uvicorn.lifespan.off',
    'httptools',
    'websockets',
    'sqlalchemy.dialects.sqlite',
    'pydantic',
    'pydantic_settings',
    'passlib.handlers.bcrypt',
    'jose',
    'openai',
    'httpx',
    'aiofiles',
    'sse_starlette',
    'redis',
    'dotenv',
]


# 收集常用依赖的子模块
hidden_imports += collect_submodules('uvicorn')
hidden_imports += collect_submodules('sqlalchemy')
hidden_imports += collect_submodules('pydantic')
hidden_imports += collect_submodules('pydantic_settings')
hidden_imports += collect_submodules('fastapi')
hidden_imports += collect_submodules('starlette')

# ----------------------------------------------------------------------
# 3. 打包分析
# ----------------------------------------------------------------------
a = Analysis(
    ['main.py'],
    pathex=[spec_dir, os.path.join(spec_dir, 'backend')],
    binaries=[],
    datas=[
        # 包含前端静态文件
        ('static', 'static'),
        # 包含后端 app 目录
        ('backend/app', 'app'),
    ],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_modules,  # 应用排除列表
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AI学术写作助手',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # === 建议修改：关闭 UPX，防止 DLL 损坏 ===
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)