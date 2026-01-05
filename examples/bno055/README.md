# BNO055 Code Block

This code block reads the orientation quaternion from a Bosch BNO055 IMU sensor through I<sup>2</sup>C. 

## Usage

To use the code block, place it in the configured `code-block-shared-library-path` and (re)start Horus System V2. The 
code block should show up in the Discovered tab of the Graph Builder. 

The code block defaults to I<sup>2</sup>C device `/dev/i2c-1`. This can be configured by adding a Key-Value Pair
to the code block with the key `i2c-device`, and setting the value to the desired device path. 

Attach a desired trigger signal to the Clock input pipe, and the code block outputs the most recent quaternion on the
Data 0 output pipe.