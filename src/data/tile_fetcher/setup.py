from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fetcher_cpp',
    ext_modules=[
        CUDAExtension(
            name='fetcher_cpp',
            sources=['fetcher.cpp'],
            libraries=['nvjpeg'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)