import torch
import tensorflow as tf
import subprocess

def check_gpu(verbose: bool = True) -> int:
    """
    Vérifie la présence et la compatibilité GPU via torch, tensorflow et nvidia-smi.
    Retourne : 
    - 0 si un GPU est détecté (device ID pour torch)
    - -1 si aucun GPU utilisable
    """
    gpu_detected = False

    if verbose:
        print("🔍 Vérification GPU (torch, tensorflow, nvidia-smi)...")

    # PyTorch
    if torch.cuda.is_available():
        gpu_detected = True
        if verbose:
            print(f"✅ torch.cuda : {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
    else:
        if verbose:
            print("❌ torch.cuda : Aucun GPU détecté")

    # TensorFlow
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        gpu_detected = True
        if verbose:
            print(f"✅ TensorFlow : {gpus[0].name} (CUDA {tf.sysconfig.get_build_info().get('cuda_version')}, cuDNN {tf.sysconfig.get_build_info().get('cudnn_version')})")
    else:
        if verbose:
            print("❌ TensorFlow : Aucun GPU détecté")

    # nvidia-smi
    try:
        output = subprocess.check_output("nvidia-smi", shell=True).decode()
        if verbose:
            print("✅ nvidia-smi : disponible")
            print(output.split('\n')[2])  # Affiche ligne infos GPU
        gpu_detected = True
    except Exception:
        if verbose:
            print("❌ nvidia-smi : indisponible ou non installé")

    return 0 if gpu_detected else -1
