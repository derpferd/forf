#!/usr/bin/env python
import os.path
import sys

import setuptools

#  https://stackoverflow.com/questions/4529555/building-a-ctypes-based-c-library-with-distutils
from distutils.command.build_ext import build_ext as build_ext_orig


class CTypesExtension(setuptools.Extension):
    pass


class build_ext(build_ext_orig):
    def build_extension(self, ext):
        self._ctypes = isinstance(ext, CTypesExtension)
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            return ext_name + ".so"
        return super().get_ext_filename(ext_name)


# here = os.path.abspath(os.path.dirname(__file__))
# include_dirs=[os.path.join(here, "cforf/src")]
# if __name__ == "__main__":
setuptools.setup(
    ext_modules=[
        CTypesExtension(
            "forf", ["cforf/src/libcforf.c", "cforf/src/forf.c", "cforf/src/rand.c"]
        )
    ]
)
