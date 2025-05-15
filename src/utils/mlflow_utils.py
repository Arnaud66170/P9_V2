# src/utils/mlflow_utils.py

import mlflow
import time
import psutil
from functools import wraps

def log_mlflow_run(run_name: str = "mlflow_tracked_function"):
    """
    Décorateur MLOps complet pour logger automatiquement dans MLflow :
    - durée d'exécution
    - paramètres (depuis kwargs simples)
    - usage CPU et RAM en fin de run
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with mlflow.start_run(run_name=run_name):
                # Démarrage du chronomètre
                start_time = time.time()

                # Exécution de la fonction décorée
                result = func(*args, **kwargs)

                # Fin du chronomètre
                duration = round(time.time() - start_time, 2)
                mlflow.log_metric("duration_seconds", duration)

                # Log automatique des kwargs simples
                for k, v in kwargs.items():
                    if isinstance(v, (str, int, float, bool)):
                        mlflow.log_param(k, v)

                # Log de la charge CPU et RAM (à la fin du run)
                cpu = psutil.cpu_percent()
                ram = psutil.virtual_memory().percent
                mlflow.log_metric("cpu_percent", cpu)
                mlflow.log_metric("ram_percent", ram)

                return result
        return wrapper
    return decorator
