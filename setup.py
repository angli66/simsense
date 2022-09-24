import os
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir='./'):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the extension")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        extdir = os.path.join(extdir, self.distribution.get_name())

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        config = 'Debug' if self.debug else 'Release'

        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DCMAKE_BUILD_TYPE=' + config
        ]

        build_args = [
            '--config', config,
            '--', '-j8'
        ]

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.', "--target", "pysimsense"] + build_args, cwd=self.build_temp)

def read_requirements():
    with open('requirements.txt', 'r') as f:
        lines = f.readlines()
    install_requires = [line.strip() for line in lines if line]
    return install_requires

setup(
    name="simsense",
    version="1.0.0",
    author="Ang Li",
    author_email="ang6li98@gmail.com",
    description="SimSense: A Real-Time Depth Sensor Simulator",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=["simsense"],
    package_dir={"simsense": "python/package"},
    python_requires=">=3.6",
    install_requires=read_requirements(),
    ext_modules=[CMakeExtension("pysimsense")],
    cmdclass={"build_ext": CMakeBuild},
    license="MIT",
    zip_safe=False
)
