package com.sujal.depth

import android.content.Context
import java.nio.FloatBuffer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlin.math.max
import kotlin.math.min

class DepthSession(
    context: Context,
    private val modelAssetName: String = "depth_anything_small.onnx",
    private val inputWidth: Int = 256,
    private val inputHeight: Int = 256
) {
    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var inputName: String? = null
    private var outputName: String? = null

    var lastMin: Float = 0f; private set
    var lastMax: Float = 0f; private set
    var lastMs: Long = 0; private set

    init {
        try {
            val bytes = context.assets.open(modelAssetName).use { it.readBytes() }
            val e = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply {
                try { addNnapi() } catch (_: Throwable) {}
                setIntraOpNumThreads(1)
                setInterOpNumThreads(1)
            }
            val s = e.createSession(bytes, opts)
            session = s
            env = e
            inputName = s.inputNames.firstOrNull()
            outputName = s.outputNames.firstOrNull()
        } catch (_: Throwable) {
            env = null
            session = null
        }
    }

    fun isReady(): Boolean = env != null && session != null && inputName != null && outputName != null

    fun inferDepthRgba(
        rgba: ByteArray,
        width: Int,
        height: Int,
        colorize: Boolean = false
    ): ByteArray {
        val s = session ?: return rgba
        val e = env ?: return rgba
        val inName = inputName ?: return rgba
        val outName = outputName ?: return rgba

        val start = System.nanoTime()
        val chw = FloatArray(3 * inputWidth * inputHeight)
        downscaleRgbaToChw(rgba, width, height, chw, inputWidth, inputHeight)
        val fb = FloatBuffer.wrap(chw)
        val inputTensor = OnnxTensor.createTensor(e, fb, longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong()))
        val output: FloatArray
        s.run(mapOf(inName to inputTensor)).use { results ->
            val value = results[0] as OnnxTensor
            val ob = value.floatBuffer
            output = FloatArray(ob.remaining())
            ob.get(output)
        }
        lastMs = (System.nanoTime() - start) / 1_000_000

        var vmin = Float.POSITIVE_INFINITY
        var vmax = Float.NEGATIVE_INFINITY
        for (v in output) { if (v < vmin) vmin = v; if (v > vmax) vmax = v }
        lastMin = vmin; lastMax = vmax
        val range = if (vmax > vmin) (vmax - vmin) else 1f
        val ow = guessOutWidth(output)
        val oh = output.size / ow

        val outRgba = ByteArray(width * height * 4)
        for (y in 0 until height) {
            val sy = (y * oh) / height
            for (x in 0 until width) {
                val sx = (x * ow) / width
                val d = output[sy * ow + sx]
                val n = ((d - vmin) / range).coerceIn(0f, 1f)
                val idx = (y * width + x) * 4
                if (colorize) {
                    val c = turboColor(n)
                    outRgba[idx] = c[0]
                    outRgba[idx + 1] = c[1]
                    outRgba[idx + 2] = c[2]
                } else {
                    val g = (n * 255f).toInt().coerceIn(0, 255)
                    outRgba[idx] = g.toByte()
                    outRgba[idx + 1] = g.toByte()
                    outRgba[idx + 2] = g.toByte()
                }
                outRgba[idx + 3] = 0xFF.toByte()
            }
        }
        return outRgba
    }

    private fun guessOutWidth(arr: FloatArray): Int {
        val n = arr.size
        val candidates = intArrayOf(64, 80, 96, 128, 160, 192, 224, 256, 320, 384)
        for (c in candidates) if (n % c == 0) return c
        return kotlin.math.sqrt(n.toDouble()).toInt().coerceAtLeast(1)
    }

    private fun downscaleRgbaToChw(
        rgba: ByteArray, srcW: Int, srcH: Int, outChw: FloatArray, dstW: Int, dstH: Int
    ) {
        for (oy in 0 until dstH) {
            val sy = (oy * srcH) / dstH
            for (ox in 0 until dstW) {
                val sx = (ox * srcW) / dstW
                val si = (sy * srcW + sx) * 4
                val r = (rgba[si].toInt() and 0xFF) / 255f
                val g = (rgba[si + 1].toInt() and 0xFF) / 255f
                val b = (rgba[si + 2].toInt() and 0xFF) / 255f
                val di = oy * dstW + ox
                outChw[0 * dstW * dstH + di] = r
                outChw[1 * dstW * dstH + di] = g
                outChw[2 * dstW * dstH + di] = b
            }
        }
    }

    private fun turboColor(n: Float): ByteArray {
        val x = n.coerceIn(0f, 1f)
        val r = (34.61 + x*(1172.33 + x*(-10793.56 + x*(33300.12 + x*(-38394.49 + x*14825.05))))) / 255.0
        val g = (23.31 + x*(557.33 + x*(1225.33 + x*(-3574.96 + x*(2326.65 + x*0.0))))) / 255.0
        val b = (27.2 + x*(3211.1 + x*(-15327.97 + x*(27814.0 + x*(-22569.18 + x*6838.66))))) / 255.0
        return byteArrayOf(
            (r.coerceIn(0.0,1.0)*255.0).toInt().toByte(),
            (g.coerceIn(0.0,1.0)*255.0).toInt().toByte(),
            (b.coerceIn(0.0,1.0)*255.0).toInt().toByte(),
        )
    }
}
