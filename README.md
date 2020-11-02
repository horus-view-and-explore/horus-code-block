# Horus Code Block C API

The [Horus](https://horus.nu/) Code Block C API is a single C header file.  By
implementing the functions in this header file using either C or C++ and
compiling/linking it into a shared library, you can create your own [Horus
System V2](https://horus.nu/horus-inside/) component by loading that shared
library into the Horus Code Block component.  You can find more information in
the header file [Horus_code_block.h](Horus_code_block.h) itself.

The directories [c_example](c_example/) and [cpp_examples](cpp_examples/)
contain example implementations of the Horus Code Block C API.
[cpp_examples/Horus_code_block_cpu_temperature.cpp](cpp_examples/Horus_code_block_cpu_temperature.cpp)
is a good place to start.
