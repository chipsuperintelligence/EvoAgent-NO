# csrc: C and CUDA kernels

Empty on purpose. EvoAgent-NO starts in Python with `torch.fft`. A kernel lands here
only when profiling shows Python is the bottleneck, and each one ships with a
pure-PyTorch reference and a test that the two agree.

First candidates:

- a fused nonlinear term + ETDRK4 step for the batched spectral solver (`physics/`)
- a forward-mode (JVP) attention kernel for the self-play score (`score/`)
