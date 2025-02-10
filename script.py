import sys
import platform
import psutil
import pkg_resources
import os
import torch
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def check_python_version():
    logging.info("Checking Python version...")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}\n")
    virtual_env = os.getenv("VIRTUAL_ENV")
    print(f"Virtual Environment: {virtual_env if virtual_env else 'Not using a virtual environment'}\n")

def get_nvidia_gpu_info():
    try:
        device_index = 0
        gpu_name = torch.cuda.get_device_name(device_index)
        gpu_memory = torch.cuda.get_device_properties(device_index).total_memory / (1024**3)
        cuda_version = torch.version.cuda
        print(f"GPU: {gpu_name} (NVIDIA)")
        print(f"CUDA Version: {cuda_version}")
        print(f"GPU Memory: {gpu_memory:.2f} GB\n")
    except Exception as e:
        logging.error(f"Error fetching NVIDIA GPU info: {e}")

def get_macos_amd_gpu_info():
    try:
        output = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"],
            text=True,
            encoding='utf-8'
        )
        if "AMD" in output or "Radeon" in output:
            print("Detected AMD GPU(s) on macOS:")
            for line in output.splitlines():
                if "Chipset Model:" in line:
                    model = line.split(":", 1)[1].strip()
                    print(f"- {model}")
            if torch.backends.mps.is_available():
                print("MPS (Metal Performance Shaders) is available for computation.\n")
            else:
                print("MPS is NOT available for computation. Upgrade PyTorch or check macOS version.\n")
        else:
            print("No AMD GPU detected on macOS.\n")
    except subprocess.CalledProcessError as e:
        logging.error(f"system_profiler command failed: {e}")
    except Exception as e:
        logging.error(f"Error detecting AMD GPU on macOS: {e}")

def get_linux_amd_gpu_info():
    # Try using ROCm first
    try:
        output = subprocess.check_output(
            ["rocm-smi"],
            stderr=subprocess.DEVNULL,
            text=True,
            encoding='utf-8'
        )
        if "GPU" in output:
            print("Detected AMD GPU(s) via ROCm:")
            print(output)
        else:
            raise Exception("No AMD GPU detected via ROCm.")
    except FileNotFoundError:
        # Fallback: scan /sys/class/drm for AMD GPU devices
        if os.path.exists("/sys/class/drm/"):
            amd_gpus = [entry for entry in os.listdir("/sys/class/drm/")
                        if "card" in entry and "renderD" not in entry]
            if amd_gpus:
                print("Detected AMD GPU(s) via system scan:")
                for gpu in amd_gpus:
                    print(f"- {gpu}")
                print()  # Extra newline for readability
            else:
                print("GPU: Not Available\n")
        else:
            print("GPU: Not Available\n")
    except subprocess.CalledProcessError as e:
        logging.error(f"rocm-smi command failed: {e}")
    except Exception as e:
        logging.error(f"Error detecting AMD GPU on Linux: {e}")

def get_windows_amd_gpu_info():
    try:
        output = subprocess.check_output(
            ["wmic", "path", "win32_VideoController", "get", "name"],
            text=True,
            encoding='utf-8'
        )
        lines = [line.strip() for line in output.splitlines() if line.strip() and "Name" not in line]
        amd_gpus = [gpu for gpu in lines if "AMD" in gpu or "Radeon" in gpu]
        if amd_gpus:
            print("Detected AMD GPU(s) on Windows:")
            for gpu in amd_gpus:
                print(f"- {gpu}")
            print()
        else:
            print("No AMD GPU detected on Windows.\n")
    except Exception as e:
        logging.error(f"Error detecting AMD GPU on Windows: {e}")

def check_hardware_resources():
    logging.info("Checking hardware resources...")
    system_info = f"{platform.system()} {platform.release()} ({platform.architecture()[0]})"
    print(f"System: {system_info}")
    print(f"Processor: {platform.processor()}")
    
    cpu_physical = psutil.cpu_count(logical=False)
    cpu_logical = psutil.cpu_count(logical=True)
    print(f"CPU Cores (Physical): {cpu_physical if cpu_physical is not None else 'N/A'}")
    print(f"Logical CPUs: {cpu_logical if cpu_logical is not None else 'N/A'}")
    
    total_ram = psutil.virtual_memory().total / (1024**3)
    print(f"Total RAM: {total_ram:.2f} GB\n")
    
    if torch.cuda.is_available():
        get_nvidia_gpu_info()
    else:
        system_platform = platform.system()
        if system_platform == "Darwin":
            get_macos_amd_gpu_info()
        elif system_platform == "Linux":
            get_linux_amd_gpu_info()
        elif system_platform == "Windows":
            get_windows_amd_gpu_info()
        else:
            print("GPU: Not Available\n")

def check_installed_modules():
    logging.info("Checking installed Python modules...")
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    print("Installed Modules:")
    for pkg, version in sorted(installed_packages.items()):
        print(f"{pkg}=={version}")

def main():
    check_python_version()
    check_hardware_resources()
    check_installed_modules()

if __name__ == "__main__":
    main()