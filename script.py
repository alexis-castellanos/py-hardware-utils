import sys
import platform
import psutil
import pkg_resources
import os
import torch
import subprocess

def check_python_version():
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}\n")
    virtual_env = os.getenv("VIRTUAL_ENV")
    print(f"Virtual Environment: {virtual_env if virtual_env else 'Not using a virtual environment'}\n")

def check_hardware_resources():
    print(f"System: {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"Processor: {platform.processor()}")
    print(f"CPU Cores: {psutil.cpu_count(logical=False)}")
    print(f"Logical CPUs: {psutil.cpu_count(logical=True)}")
    print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB\n")
    
    # Detect NVIDIA GPU (Linux/Windows)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} (NVIDIA)")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\n")
    
    elif platform.system() == "Darwin":  # macOS AMD GPU Detection
        try:
            output = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], text=True)
            if "AMD" in output or "Radeon" in output:
                print("Detected AMD GPU(s) on macOS:")
                for line in output.split("\n"):
                    if "Chipset Model" in line:
                        print(f"- {line.strip().split(': ')[1]}")
                
                # Check if MPS is available for computation
                if torch.backends.mps.is_available():
                    print("MPS (Metal Performance Shaders) is available for computation.\n")
                else:
                    print("MPS is NOT available for computation. Upgrade PyTorch or check macOS version.\n")
            else:
                print("No AMD GPU detected on macOS.\n")
        except Exception as e:
            print(f"Error detecting AMD GPU on macOS: {e}\n")
    
    elif platform.system() == "Linux":  # AMD GPU detection on Linux
        try:
            output = subprocess.check_output(["rocm-smi"], stderr=subprocess.DEVNULL).decode()
            if "GPU" in output:
                print("Detected AMD GPU(s) via ROCm:")
                print(output)
            else:
                raise Exception("No AMD GPU detected via ROCm.")
        except FileNotFoundError:
            if os.path.exists("/sys/class/drm/"):
                amd_gpus = [f for f in os.listdir("/sys/class/drm/") if "card" in f and "renderD" not in f]
                if amd_gpus:
                    print("Detected AMD GPU(s) via system scan:")
                    for gpu in amd_gpus:
                        print(f"- {gpu}")
                else:
                    print("GPU: Not Available\n")
            else:
                print("GPU: Not Available\n")
    else:
        print("GPU: Not Available\n")

def check_installed_modules():
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    print("Installed Modules:")
    for pkg, version in installed_packages.items():
        print(f"{pkg}=={version}")

if __name__ == "__main__":
    check_python_version()
    check_hardware_resources()
    check_installed_modules()
