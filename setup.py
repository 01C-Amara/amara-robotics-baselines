from setuptools import setup, find_packages

setup(
    name="amara_robotics_baselines",
    version="0.1.0",
    packages=["amara_robotics_baselines"]
    + [f"amara_robotics_baselines.{p}" for p in ["checks", "utils", "scripts"]],
    install_requires=["pandas", "pyarrow", "tqdm", "numpy", "scipy", "trimesh", "huggingface_hub"],
    python_requires=">=3.8",
)
