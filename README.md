# Depth Estimation on Android Using Rust

Real-time, on-device monocular depth on Android powered by a Rust NDK library and PyTorch Mobile (TorchScript). 

## Demo
[![Watch demo](thumb.jpg)](https://cdn.discordapp.com/attachments/1194007460629987479/1408479044328095765/screen-20250822-212118.mp4?ex=68a9e3d4&is=68a89254&hm=db26175747d2be5c6951ca97ec43b159740ad2c162e8315aa00a1334177ef52b&)

## Overview
- Android app (Kotlin, CameraX) streams camera frames
- Rust `cdylib` (`librustdepth.so`) handles fast YUV→RGBA and buffer reuse
- TorchScript (PyTorch Mobile) performs depth inference; Kornia transforms can be embedded inside the scripted model
- Clean UI: mode (Rust vs Depth), palette (Gray vs Color), preview toggle, opacity slider, FPS + stats

Use cases: robotics, AR, SLAM, 3D scene understanding.

## Project Layout
```
depthRust/
  android-app/             # Android Studio project (app id: com.sujal.depth)
  rustdepth/               # Rust crate (JNI + pre/post processing)
  models/                  # optional local models
```

## Prerequisites
- Android Studio + SDK + NDK + CMake
- Rust toolchain with Android target
```bash
rustup target add aarch64-linux-android
cargo install cargo-ndk
```
- Device: ARM64 Android phone (recommended). Enable USB debugging.

## Build the Rust library
```bash
cd rustdepth
cargo ndk -t arm64-v8a -o ../android-app/app/src/main/jniLibs build --release
```
This produces `android-app/app/src/main/jniLibs/arm64-v8a/librustdepth.so`.

## Run the Android app
```bash
cd android-app
./gradlew :app:installDebug
$HOME/Android/Sdk/platform-tools/adb shell am start -n com.sujal.depth/.MainActivity
```

## Model setup (TorchScript)
- Export a TorchScript Lite model with Kornia-based preprocessing baked in, or use a scripted MiDaS small variant.
- Place it in assets:
```bash
cp ~/Downloads/depth_kornia.torchscript.ptl android-app/app/src/main/assets/
```
- App expects `depth_kornia.torchscript.ptl` by default. Reinstall after copying.

## How it works
- Kotlin `ImageAnalysis` provides Y, U, V planes. We copy them into compact arrays.
- JNI → Rust converts YUV420→RGBA using a reusable scratch buffer.
- If Depth mode is enabled, Kotlin downsizes to model input, runs TorchScript via PyTorch Mobile, min–max normalizes, upsamples to screen, and colorizes.
- UI draws the processed bitmap atop the preview (or preview hidden in Depth mode).

## Performance tips
- Lower analysis resolution (now 640×480) for better FPS
- Buffer reuse (implemented): Y/U/V/RGBA arrays and a single `Bitmap`
- Hide preview in Depth mode to avoid composition cost
- Consider scripting the entire pipeline (resize/normalize/inference/postprocess) in TorchScript

Further optimizations (optional):
- Move resize + CHW conversion into Rust JNI to cut one copy
- NEON-accelerated YUV→RGB in Rust for 2–3× speedup
- All-Rust inference with `ort` crate (if you ever want to compare)

## Troubleshooting
- “Model missing”: ensure `android-app/app/src/main/assets/depth_kornia.torchscript.ptl` exists and is multi‑MB; reinstall the app
- ADB device not found: enable USB debugging, accept RSA prompt, run `adb devices`
- ABI issues: ensure device is ARM64; app is limited to `arm64-v8a`

## Credits & Resources
- Android’s official Rust/NDK docs
- PyTorch Mobile / TorchScript
- Kornia (geometric vision ops)
- Depth models: MiDaS, Depth-Anything

## License
MIT (adjust as needed).
