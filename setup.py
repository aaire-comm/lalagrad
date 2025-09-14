import os


supported = ['posix']

if os.name not in supported:
    raise EnvironmentError(f"system {os.name} not supported yet")



args = ["-shared", "-fopenmp", "-fPIC", "./lala/c/lib.c", "-o", "./lala/c/lib.so"]

import subprocess
print("compiling lib...")

try:
    s = subprocess.call(["gcc"] + args)
    print("setup done")
except FileNotFoundError:
    print('gcc not found on you system')
except:
    def download_lib(): pass
    print("Error compiling lib")
    print("Downloading lib binary")

    download_lib()




