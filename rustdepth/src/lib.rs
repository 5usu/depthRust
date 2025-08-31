#![allow(non_snake_case)]
use jni::objects::{JByteArray, JClass, JString};
use jni::sys::{jint, jbyteArray, jstring};
use jni::JNIEnv;
use std::sync::{OnceLock, Mutex};

static SCRATCH: OnceLock<Mutex<Vec<u8>>> = OnceLock::new();

#[inline]
fn ensure_scratch(cap: usize) -> &'static Mutex<Vec<u8>> {
    let m = SCRATCH.get_or_init(|| Mutex::new(Vec::new()));
    let mut v = m.lock().unwrap();
    if v.len() < cap { v.resize(cap, 0); }
    drop(v);
    m
}

// --- YUV420 (planar) -> interleaved RGB (naive, good enough for MVP) ---
fn yuv420_to_rgb(y: &[u8], u: &[u8], v: &[u8], w: usize, h: usize, stride_y: usize, stride_u: usize, stride_v: usize, out_rgb: &mut [u8]) {
    for j in 0..h {
        for i in 0..w {
            let yv = y[j*stride_y + i] as i32;
            let uidx = (j/2)*stride_u + (i/2);
            let vidx = (j/2)*stride_v + (i/2);
            let u8v = u[uidx] as i32;
            let v8v = v[vidx] as i32;
            // BT.601 approx
            let c = yv - 16;
            let d = u8v - 128;
            let e = v8v - 128;
            // Use fixed-point coeffs closer to ITU-R BT.601 full range
            let mut r = (256*c + 359*e + 128) >> 8;
            let mut g = (256*c -  88*d - 183*e + 128) >> 8;
            let mut b = (256*c + 454*d + 128) >> 8;
            r = r.clamp(0,255); g = g.clamp(0,255); b = b.clamp(0,255);
            let o = (j*w + i)*3;
            out_rgb[o] = r as u8; out_rgb[o+1] = g as u8; out_rgb[o+2] = b as u8;
        }
    }
}

// Simple bilinear resize (RGB u8 -> RGB f32 chw)
fn resize_to_tensor(rgb: &[u8], w: usize, h: usize, tw: usize, th: usize, out_chw: &mut [f32]) {
    for oy in 0..th {
        let fy = (oy as f32 + 0.5) * (h as f32 / th as f32) - 0.5;
        let y0 = fy.floor().clamp(0.0, (h-1) as f32) as usize;
        let y1 = (y0 + 1).min(h-1);
        let wy = fy - y0 as f32;

        for ox in 0..tw {
            let fx = (ox as f32 + 0.5) * (w as f32 / tw as f32) - 0.5;
            let x0 = fx.floor().clamp(0.0, (w-1) as f32) as usize;
            let x1 = (x0 + 1).min(w-1);
            let wx = fx - x0 as f32;

            let idx = |x:usize,y:usize,c:usize| -> u8 { rgb[(y*w + x)*3 + c] };
            for c in 0..3 {
                let p00 = idx(x0,y0,c) as f32;
                let p10 = idx(x1,y0,c) as f32;
                let p01 = idx(x0,y1,c) as f32;
                let p11 = idx(x1,y1,c) as f32;
                let top = p00 + wx*(p10 - p00);
                let bot = p01 + wx*(p11 - p01);
                let val = top + wy*(bot - top);
                out_chw[c*tw*th + oy*tw + ox] = (val / 255.0 - 0.5) / 0.5; // normalize to [-1,1]
            }
        }
    }
}

// JNI: init buffers once (optional)
#[no_mangle]
pub extern "system" fn Java_com_sujal_depth_Native_initBuffers(_env: JNIEnv, _cls: JClass, maxWidth: jint, maxHeight: jint) {
    let cap = (maxWidth as usize) * (maxHeight as usize) * 4;
    let m = SCRATCH.get_or_init(|| Mutex::new(Vec::new()));
    let mut v = m.lock().unwrap();
    v.resize(cap, 0);
}

// JNI: YUV->RGB (and later inference) returning RGBA for preview
#[no_mangle]
pub extern "system" fn Java_com_sujal_depth_Native_yuvToRgba(
    mut env: JNIEnv, _cls: JClass,
    yArr: JByteArray, uArr: JByteArray, vArr: JByteArray,
    w: jint, h: jint, strideY: jint, strideU: jint, strideV: jint
) -> jbyteArray {
    let w = w as usize; let h = h as usize;
    let y = env.convert_byte_array(yArr).unwrap();
    let u = env.convert_byte_array(uArr).unwrap();
    let v = env.convert_byte_array(vArr).unwrap();

    // Reuse scratch buffers to avoid per-frame allocations
    let total_rgb = w*h*3;
    let total_rgba = w*h*4;
    let total_needed = total_rgb + total_rgba;
    let m = ensure_scratch(total_needed);
    let mut guard = m.lock().unwrap();
    if guard.len() < total_needed { guard.resize(total_needed, 0); }
    let (rgb_slice, out_slice) = guard.split_at_mut(total_rgb);

    yuv420_to_rgb(&y, &u, &v, w, h, strideY as usize, strideU as usize, strideV as usize, rgb_slice);
    // Pack RGB -> RGBA
    for i in 0..(w*h) {
        let rgb_i = i*3; let rgba_i = i*4;
        out_slice[rgba_i] = rgb_slice[rgb_i];
        out_slice[rgba_i+1] = rgb_slice[rgb_i+1];
        out_slice[rgba_i+2] = rgb_slice[rgb_i+2];
        out_slice[rgba_i+3] = 255;
    }
    let jarr = env.byte_array_from_slice(&out_slice[..total_rgba]).unwrap();
    jarr.into_raw()
}

// --- Simple hello() for initial JNI wiring test ---
#[no_mangle]
pub extern "system" fn Java_com_sujal_depth_Native_hello(
    mut env: JNIEnv,
    _class: JClass,
    input: JString,
) -> jstring {
    let name: String = env.get_string(&input).unwrap().into();
    let output = format!("Hello from Rust, {}!", name);
    env.new_string(output).unwrap().into_raw()
}

// Track B only: add JNI initSession() + infer() that use ONNX Runtime in Rust.

