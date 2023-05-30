from setuptools import setup, find_packages

setup(
    name="chao_par",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.1+cu113",
        "torchvision>=0.11.2+cu113",
        "opencv-python",
        "numpy",
        "tqdm",
        "Pillow",
        # 其他依赖项...
    ],
)

