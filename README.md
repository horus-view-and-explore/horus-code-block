# Horus Code Block C API

The [Horus](https://horus.nu/) Code Block C API is a single C header file.  By
implementing the functions in this header file using either C or C++ and
compiling/linking it into a shared library, you can create your own [Horus
System V2](https://horus.nu/horus-inside/) component by loading that shared
library into the Horus Code Block component.  You can find more information in
the header file [Horus_code_block.h](Horus_code_block.h) itself.

## Examples

The directory [examples](examples/) contains a number of subdirectories with
CMake-based example implementations of the Horus Code Block C API.

[PTZ Orientation RPY Eight
Sectors](examples/ptz_orientation_rpy_eight_sectors/Horus_code_block_ptz_orientation_rpy_eight_sectors.cpp)
is a good place to start.  It is a simple example using C++ that receives PTZ
roll-pitch-yaw orientations and sends the sector number of that orientation as
sensor data.

[CPU Temperature](examples/cpu_temperature/Horus_code_block_cpu_temperature.cpp)
is another simple example using C++.  On Linux, it produces CPU temperature
readings as sensor data.

[Debug Sink](examples/debug_sink/Horus_code_block_debug_sink.c) is an elaborate
example in pure C that prints information on the shared library calls.  You can
use it to let a Code Block component print all Horus Code Block supported data
structures produced for incoming messages.

[Framework](examples/framework/) is an elaborate example that presents a small
C++ framework to implement the Code Block C API as well as two exmples that use
this framework.  [Debug
Source](examples/framework/Horus_code_block_debug_source.cpp) produces various
types of output messages.  [LSB
Eraser](examples/framework/Horus_code_block_lsb_eraser.cpp) demonstrates buffer
manipulation by erasing the four least significant bits of raw video buffer
bytes of input messages.

## Building for an embedded device

In order to build a Code Block shared library for your embedded device, the
corresponding SDK (yocto) needs to be installed.  These SDK's can be requested
or downloaded from [http://embed.horus.nu](http://embed.horus.nu).

### Installing the SDK

In this example we downloaded and executed
`fslc-x11-glibc-x86_64-core-image-x11-cortexa9t2hf-neon-imx6qdl-variscite-som-toolchain-3.1.sh`:

```console
FSLC X11 SDK installer version 3.1
==================================
Enter target directory for SDK (default: /opt/fslc-x11/3.1): /home/auke/yoctosdk/fslc-x11/3.1
You are about to install the SDK to "/home/auke/yoctosdk/fslc-x11/3.1". Proceed [Y/n]? y
Extracting SDK....................done
Setting it up...done
Setting up IceCream distributed compiling...
creating /home/auke/yoctosdk/fslc-x11/3.1/sysroots/x86_64-fslcsdk-linux/usr/share/arm-fslc-linux-gnueabi-icecream/fslc-x11-glibc-x86_64-arm-fslc-linux-gnueabi-3.1.tar.gz
SDK has been successfully set up and is ready to be used.
Each time you wish to use the SDK in a new shell session, you need to source the environment setup script e.g.
 $ . /home/auke/yoctosdk/fslc-x11/3.1/environment-setup-cortexa9t2hf-neon-fslc-linux-gnueabi
```

### Build Environment

After sourcing the environment, use CMake to build the Code Block examples.
