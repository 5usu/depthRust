package com.sujal.depth

object Native {
    init { System.loadLibrary("rustdepth") }

    external fun hello(name: String): String

    external fun initBuffers(maxWidth: Int, maxHeight: Int)

    external fun yuvToRgba(
        y: ByteArray,
        u: ByteArray,
        v: ByteArray,
        width: Int,
        height: Int,
        strideY: Int,
        strideU: Int,
        strideV: Int
    ): ByteArray
}
