# Nerf lab

Pet project in which I am going to create a reconstruction & rendering engine using NERF technology. Inspired by [NERFs](https://www.matthewtancik.com/nerf) and Instant-NGP ([paper](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf)/[code](https://github.com/NVlabs/instant-ngp)) which is already awesome but I believe there still a room for possibilities. To be more specific I'm aiming to:

 * Extend perfomance of `tiny-cuda-nn` inference which is the heart of real-time nerf inference (with use of QAT or even XNOR networks for example).
 * Achieve real-time (at least 20 fps) full-hd (1920x1080) on my laptop GPU (hope it's possible, but if not, I'll try anyway).
 * Add possibility to make unbounded scenes (with use of block nerf or/and nerf 360)
 * Create simple and useful API to add nerf rendering in other projects like 3d scanners, video editing application or even games.
 * Improve my experience in CUDA, OpenGL, and DeepLearning.

<img src="https://static.vecteezy.com/system/resources/previews/001/218/694/original/under-construction-warning-sign-vector.jpg" alt="Currently under construction" width="200"/>
