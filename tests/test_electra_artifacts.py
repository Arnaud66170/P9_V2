import os
import pytest

@pytest.mark.parametrize("model_dir", ["models/electra_model"])
def test_electra_artifacts_present(model_dir):
    required_files = [
        "pytorch_model.bin",
        "config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "vocab.txt"  # parfois généré selon le tokenizer
    ]

    missing = [f for f in required_files if not os.path.isfile(os.path.join(model_dir, f))]

    if missing:
        missing_list = "\n".join(f"❌ {f}" for f in missing)
        pytest.fail(f"Des fichiers critiques manquent dans {model_dir} :\n{missing_list}")

# lancement du test dans le terminal :
# pytest tests/test_electra_artifacts.py