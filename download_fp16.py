import urllib.request
from pathlib import Path

# Fetching the standard FP16 model for maximum hardware compatibility (CPU/GPU/FPGA)
base_url = (
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/"
    "2022.1/models_bin/3/human-pose-estimation-0001/FP16/"
)

save_dir = Path("models/human-pose-estimation-0001/FP16")
save_dir.mkdir(parents=True, exist_ok=True)

for name in ["human-pose-estimation-0001.xml", "human-pose-estimation-0001.bin"]:
    print("Downloading", name)
    urllib.request.urlretrieve(base_url + name, save_dir / name)

print("Done. FP16 Files saved to", save_dir)