# CUDA Video Flip Tutorial

## Introduction

This example demonstrates a CUDA-based video processing application that
performs frame inversion using a two-buffer approach. Instead of
modifying the incoming video data in place, the system utilizes a
separate destination buffer for the transformed output. The process
works as follows: The original video content arrives in an input CUDA
buffer, and the flipping operation is performed by copying and
transforming the data into a second, separate output buffer. This
approach preserves the integrity of the source material while creating
the inverted version in the destination buffer.

This example demonstrates the implementation of efficient buffer
synchronization in CUDA-based video processing, with a focus on the
notification system that enables multiple buffers to communicate their
completion status.

The `notify_done_flags` mechanism is particularly noteworthy as it
allows a single buffer to signal the completion of buffers. This is
achieved through a bit-flag system in the `Notify_done_flags`
enumeration:

```cpp
enum Notify_done_flags
{
    Horus_cuda_code_block_data_buffer_notify_done_0 = 1 << 0,
    Horus_cuda_code_block_data_buffer_notify_done_1 = 1 << 1,
};
```


  
  

## Core Components

  

### Header Dependencies

  

The implementation relies on three essential headers:

  

```cpp
#include  "../../../Horus_code_block.h"
#include  "../../../Horus_cuda_code_block.h"
#include  "../Helper_functions.hpp"
```

These headers provide the Horus data structures and operational helpers.

### Processing Functions () 

The code defines two primary CUDA processing functions for different memory layouts:

  ```cpp
void  video_flip_run_surface(dim3  blockDim, dim3  gridDim, ...);
void  video_flip_run_linear(dim3  blockDim, dim3  gridDim, ...);
```

All video processing operations are executed through dedicated CUDA
kernels, which are defined and implemented in the separate Kernel.cu
file. By leveraging CUDA kernel operations, the system ensures that all
video transformations are performed directly on the GPU, maximizing
parallel processing capabilities and computational efficiency.
  

### Context Management

  

The `User_context` structure manages the processing state:

  

```cpp

struct  User_context {
 dim3 blockDim;
 dim3 gridDim;
 
 bool debug_schedule;

 Horus_cuda_code_block_data_buffer *src;
 Horus_cuda_code_block_data_buffer *dst;

 bool src_inited;
 bool dst_inited;

 bool src_claimed;
 bool dst_claimed;

 bool configured;
 bool use_surface;
 bool use_linear;

// Additional methods...
    
};
```
## Implementation Details

### Thread Organization

The code implements efficient CUDA thread management:

```cpp
void  create_grid_dimensions(size_t  width, size_t  height) {
blockDim = dim3(32, 32);
gridDim = dim3(
  (static_cast<unsigned  int>(width) + blockDim.x - 1) / blockDim.x,
  (static_cast<unsigned  int>(height) + blockDim.y - 1) / blockDim.y);
}

```

This creates a 2D grid optimized for image processing, with each block
containing 32x32 threads.

### Memory Management

The implementation supports two GPU memory types:

1. Surface Memory: Optimized for 2D access patterns

2. Linear Memory: Traditional sequential memory layout

  

The scheduling function demonstrates this dual support:

  

```cpp
void  schedule(Horus_cuda_code_block_data_buffer  *cuda_buffer) {
 if (use_linear)
        {
            video_flip_run_linear(...);
        }
        else if (use_surface)
        {
            video_flip_run_surface(...);
        }
}
```

### API Functions

The code implements four essential Horus API functions:

 
```cpp

Horus_code_block_result  horus_code_block_get_version(unsigned  int  *const  version);

Horus_code_block_result  horus_code_block_open(...);

Horus_code_block_result  horus_code_block_close(...);

Horus_code_block_result  horus_code_block_write(...);

```

  

## Event Processing

  

The write function handles three primary events:

  

1. Initialization Event: Configures processing dimensions and initializes resources

2. Slot Change Event: Manages active processing slots and triggers buffer processing

3. Stop Event: Handles cleanup and resource deallocation

  
  

## Usage Guide

  

To implement this code effectively:

  

```cpp
// 1. Initialize the code block

Horus_code_block_user_context *context;

horus_code_block_open(&block_context, &context);

  
// 2. Process video frames

horus_code_block_write(&block_context, &data);

 // 3. Cleanup resources

horus_code_block_close(&block_context);
```

  

## Horus_cuda_code_block_data_buffer

  

The `Horus_cuda_code_block_data_buffer` structure, which serves as a
comprehensive data container for managing CUDA-based buffer
(video) operations. This structure is essential for coordinating memory
management and media processing between CPU and GPU.

  

The structure contains several enumeration types that define various
operational states and flags:

  

The `Event_type` enumeration tracks the buffer's lifecycle states:

-  `initialized`: Marks when the buffer is first set up

-  `slot_changed`: Indicates a change in the active processing slot

-  `stopped`: Signals when processing has terminated

  

The `Memory_flags` enumeration manages memory states and configurations
across device and host:

- Tracks initialization states for both device and host memory

- Defines memory organization types (array, linear)

- Specifies GPU-specific memory features (texture, surface)

  

The `Media_flags` enumeration monitors which media properties have been configured, including width, height, codec, and pixel format settings.

  

The `Notify_done_flags` enumeration provides notification mechanisms for buffer processing completion.

  

The structure's memory section manages both device and host memory resources:

-  `mem_dev`: Points to GPU memory (supports various CUDA memory types)

-  `mem_host`: Points to CPU memory

-  `mem_bytes`: Tracks memory allocation size

-  `mem_flags`: Stores memory configuration settings

  

The media section handles video-specific properties:

- Dimensions (width and height)

- Codec information

- Pixel format specifications

- Configuration flags

  

The structure also includes buffer management components:

-  `index`: References external buffer identification

-  `slot_idx`: Our processing slot

-  `name`: Stores buffer identification string

  

For synchronization and processing:

-  `event`: Tracks the current buffer state

-  `active_slot_idx`: Identifies the currently active processing slot

-  `notify_done_flags`: Manages buffer completion signaling

-  `cuda_stream`: Handles CUDA processing queue management

  

This structure effectively encapsulates all necessary information for
managing buffer(video) processing in a Horus Code Block environment,
providing a robust foundation for video processing applications.
