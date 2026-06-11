# test_import.py
import os
import sys

# 设置环境变量
os.environ['MACOSX_DEPLOYMENT_TARGET'] = '11.0'

print(f"Python: {sys.version}")
print(f"Platform: {sys.platform}")
print(f"Machine: {sys.implementation._multiarch}")

try:
    from pynqs.libs.C_extension_MAX_SORB_64 import *
    print("import suceed!")
except:
    print("import fail")
