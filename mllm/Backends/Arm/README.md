# Arm Backend

## Armv7

We did not optimized mllm's arm backend for armv7 arch.

## Armv8

Armv8 has 32 x 128-bit vector reg. Naming from v0-v31.

2 x 64-bit -> vn.2d

4 x 32-bit -> vn.4s

8 x 16-bit -> vn.8h

16 x 8-bit -> vn.16b
