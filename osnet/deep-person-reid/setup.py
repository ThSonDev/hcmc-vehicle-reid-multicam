import numpy as np
import os.path as osp
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize

def readme():
    if osp.exists('README.rst'):
        with open('README.rst') as f:
            content = f.read()
        return content
    return "A library for deep learning person re-ID in PyTorch"

def find_version():
    version_file = 'torchreid/__init__.py'
    with open(version_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                # Trích xuất '1.4.0' từ dòng __version__ = '1.4.0'
                return line.split('=')[1].strip().strip("'").strip('"')
    return "1.4.0"

def numpy_include():
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()
    return numpy_include

ext_modules = [
    Extension(
        'torchreid.metrics.rank_cylib.rank_cy',
        ['torchreid/metrics/rank_cylib/rank_cy.pyx'],
        include_dirs=[numpy_include()],
    )
]

def get_requirements(filename='requirements.txt'):
    # CHỈNH SỬA TẠI ĐÂY: Loại bỏ các thư viện cốt lõi để tránh downgrade
    excluded_packages = ['torch', 'torchvision', 'numpy', 'scipy', 'six']
    
    here = osp.dirname(osp.realpath(__file__))
    requires = []
    
    if osp.exists(osp.join(here, filename)):
        with open(osp.join(here, filename), 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Tách tên package khỏi version (ví dụ: torch>=1.2.0 -> torch)
                pkg_name = line.split('>=')[0].split('==')[0].split('<')[0].strip().lower()
                if pkg_name not in excluded_packages:
                    requires.append(line)
    
    # Bổ sung các thư viện hỗ trợ cần thiết mà thường không gây xung đột
    # Nếu bạn thiếu gì hãy thêm vào đây
    return requires

setup(
    name='torchreid',
    version=find_version(),
    description='A library for deep learning person re-ID in PyTorch',
    author='Kaiyang Zhou',
    license='MIT',
    long_description=readme(),
    url='https://github.com/KaiyangZhou/deep-person-reid',
    packages=find_packages(),
    # install_requires sẽ chỉ cài những thứ râu ria, không động vào Torch/Numpy của bạn
    install_requires=get_requirements(),
    keywords=['Person Re-Identification', 'Deep Learning', 'Computer Vision'],
    ext_modules=cythonize(ext_modules),
    include_package_data=True, # Đảm bảo copy các file non-python
)
