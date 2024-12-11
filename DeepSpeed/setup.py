"""
DeepSpeed library setup

Create a new wheel via the following command: python setup.py bdist_wheel
The wheel will be located at: dist/*.whl
"""

import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# Command class for custom build extensions
cmdclass = {'build_ext': BuildExtension.with_options(use_ninja=False)}

# Detect PyTorch version
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

# Check for CUDA availability
if not torch.cuda.is_available():
    print(
        "[WARNING] CUDA not detected. Ensure proper CUDA installation if GPU acceleration is required."
    )
    if os.environ.get("TORCH_CUDA_ARCH_LIST") is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"

# Handle version-dependent macros for compatibility
version_ge_1_1 = ['-DVERSION_GE_1_1'] if (TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR > 0)) else []
version_ge_1_3 = ['-DVERSION_GE_1_3'] if (TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR > 2)) else []
version_ge_1_5 = ['-DVERSION_GE_1_5'] if (TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR > 4)) else []
version_dependent_macros = version_ge_1_1 + version_ge_1_3 + version_ge_1_5

# Define extensions for custom CUDA kernels
ext_modules = [
    CUDAExtension(
        name='deepspeed_lamb_cuda',
        sources=['csrc/lamb/fused_lamb_cuda.cpp',
                 'csrc/lamb/fused_lamb_cuda_kernel.cu'],
        include_dirs=['csrc/includes'],
        extra_compile_args={
            'cxx': ['-O3'] + version_dependent_macros,
            'nvcc': ['-O3', '--use_fast_math'] + version_dependent_macros
        }
    ),
    CUDAExtension(
        name='deepspeed_transformer_cuda',
        sources=[
            'csrc/transformer/ds_transformer_cuda.cpp',
            'csrc/transformer/cublas_wrappers.cu',
            'csrc/transformer/transform_kernels.cu',
            'csrc/transformer/gelu_kernels.cu',
            'csrc/transformer/dropout_kernels.cu',
            'csrc/transformer/normalize_kernels.cu',
            'csrc/transformer/softmax_kernels.cu',
            'csrc/transformer/general_kernels.cu'
        ],
        include_dirs=['csrc/includes'],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++14', '-g', '-Wno-reorder'],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '-gencode', 'arch=compute_70,code=sm_70',
                '-gencode', 'arch=compute_80,code=sm_80',
                '-std=c++14',
                '-U__CUDA_NO_HALF_OPERATORS__',
                '-U__CUDA_NO_HALF_CONVERSIONS__',
                '-U__CUDA_NO_HALF2_OPERATORS__'
            ]
        }
    ),
    CUDAExtension(
        name='deepspeed_stochastic_transformer_cuda',
        sources=[
            'csrc/transformer/ds_transformer_cuda.cpp',
            'csrc/transformer/cublas_wrappers.cu',
            'csrc/transformer/transform_kernels.cu',
            'csrc/transformer/gelu_kernels.cu',
            'csrc/transformer/dropout_kernels.cu',
            'csrc/transformer/normalize_kernels.cu',
            'csrc/transformer/softmax_kernels.cu',
            'csrc/transformer/general_kernels.cu'
        ],
        include_dirs=['csrc/includes'],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++14', '-g', '-Wno-reorder'],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '-gencode', 'arch=compute_70,code=sm_70',
                '-gencode', 'arch=compute_80,code=sm_80',
                '-std=c++14',
                '-U__CUDA_NO_HALF_OPERATORS__',
                '-U__CUDA_NO_HALF_CONVERSIONS__',
                '-U__CUDA_NO_HALF2_OPERATORS__',
                '-D__STOCHASTIC_MODE__'
            ]
        }
    ),
]

# Setup function
setup(
    name='deepspeed',
    version='0.16.2',
    description='DeepSpeed library with custom optimizations',
    author='DeepSpeed Team',
    author_email='deepspeed@microsoft.com',
    url='http://aka.ms/deepspeed',
    packages=find_packages(exclude=["docker", "third_party", "csrc"]),
    scripts=[
        'bin/deepspeed',
        'bin/deepspeed.pt',
        'bin/ds',
        'bin/ds_ssh'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
    ],
    ext_modules=ext_modules,
    cmdclass=cmdclass
)
