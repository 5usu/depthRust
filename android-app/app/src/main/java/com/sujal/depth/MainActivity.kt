package com.sujal.depth

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButtonToggleGroup
import com.google.android.material.materialswitch.MaterialSwitch
import com.google.android.material.slider.Slider
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var rustImageView: ImageView
    private lateinit var fpsText: TextView
    private lateinit var statsText: TextView

    private lateinit var modeToggle: MaterialButtonToggleGroup
    private lateinit var paletteToggle: MaterialButtonToggleGroup
    private lateinit var previewSwitch: MaterialSwitch
    private lateinit var opacitySlider: Slider

    private lateinit var cameraExecutor: ExecutorService

    private var useDepth = false
    private var useColor = false
    private var depthSession: DepthSession? = null

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else finish()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        rustImageView = findViewById(R.id.rustImage)
        fpsText = findViewById(R.id.fpsText)
        statsText = findViewById(R.id.statsText)
        modeToggle = findViewById(R.id.modeToggle)
        paletteToggle = findViewById(R.id.paletteToggle)
        previewSwitch = findViewById(R.id.previewSwitch)
        opacitySlider = findViewById(R.id.opacitySlider)
        cameraExecutor = Executors.newSingleThreadExecutor()

        Log.d("RustJNI", Native.hello("Sujal"))
        Native.initBuffers(1920, 1080)

        depthSession = DepthSession(this)

        // Mode selection
        modeToggle.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (!isChecked) return@addOnButtonCheckedListener
            useDepth = (checkedId == R.id.btnDepth)
            if (useDepth && depthSession?.isReady() != true) {
                Toast.makeText(this, "Depth model missing. Add assets/depth_kornia.torchscript.ptl", Toast.LENGTH_SHORT).show()
                // revert to Rust
                modeToggle.check(R.id.btnRust)
                useDepth = false
            }
            applyPreviewVisibility()
        }
        // default select Rust
        modeToggle.check(R.id.btnRust)

        // Palette selection
        paletteToggle.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (!isChecked) return@addOnButtonCheckedListener
            useColor = (checkedId == R.id.btnColor)
        }
        paletteToggle.check(R.id.btnGray)

        // Preview visibility switch
        previewSwitch.setOnCheckedChangeListener { _, isChecked ->
            previewView.visibility = if (isChecked) View.VISIBLE else View.GONE
        }
        previewSwitch.isChecked = false

        // Opacity slider
        opacitySlider.addOnChangeListener { _, value, _ ->
            rustImageView.alpha = value / 100f
        }
        rustImageView.alpha = opacitySlider.value / 100f

        if (hasCameraPermission()) startCamera()
        else requestPermissionLauncher.launch(Manifest.permission.CAMERA)
    }

    private fun applyPreviewVisibility() {
        // If depth mode, default to hiding preview to avoid ghosting
        previewView.visibility = if (useDepth) View.GONE else if (previewSwitch.isChecked) View.VISIBLE else View.GONE
        rustImageView.alpha = opacitySlider.value / 100f
    }

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val rotation = previewView.display.rotation

            val preview = Preview.Builder()
                .setTargetRotation(rotation)
                .build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val analyzer = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(480, 360))
                .setTargetRotation(rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, FrameAnalyzer(
                        onBitmap = { bitmap -> rustImageView.post { rustImageView.setImageBitmap(bitmap) } },
                        onFps = { f -> fpsText.post { fpsText.text = String.format("%.1f FPS", f) } },
                        onStats = { min, max, ms -> statsText.post { statsText.text = String.format("min=%.3f max=%.3f inf=%dms", min, max, ms) } },
                        onRotation = { deg -> rustImageView.post { rustImageView.rotation = deg } },
                        useDepth = { useDepth },
                        useColor = { useColor },
                        depthInfer = { rgba, w, h ->
                            val ds = depthSession
                            if (ds != null && ds.isReady()) ds.inferDepthRgba(rgba, w, h, colorize = useColor) else rgba
                        },
                        fetchStats = {
                            val ds = depthSession
                            if (ds != null && ds.isReady()) Triple(ds.lastMin, ds.lastMax, ds.lastMs) else Triple(0f,0f,0)
                        }
                    ))
                }

            try {
                cameraProvider.unbindAll()
                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, analyzer)
            } catch (e: Exception) {
                Log.e("CameraX", "Binding failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}

private class FrameAnalyzer(
    private val onBitmap: (Bitmap) -> Unit,
    private val onFps: (Float) -> Unit,
    private val onStats: (Float, Float, Long) -> Unit,
    private val onRotation: (Float) -> Unit,
    private val useDepth: () -> Boolean,
    private val useColor: () -> Boolean,
    private val depthInfer: (ByteArray, Int, Int) -> ByteArray,
    private val fetchStats: () -> Triple<Float, Float, Long>
) : ImageAnalysis.Analyzer {

    private var lastTs = 0L
    private var frameCount = 0
    private var lastFpsTs = 0L

    private var yBuf: ByteArray? = null
    private var uBuf: ByteArray? = null
    private var vBuf: ByteArray? = null
    private var rgbaBuf: ByteArray? = null
    private var bmp: Bitmap? = null

    override fun analyze(imageProxy: ImageProxy) {
        val img = imageProxy.image ?: run { imageProxy.close(); return }

        // Apply current frame rotation to the overlay view
        val deg = imageProxy.imageInfo.rotationDegrees.toFloat()
        onRotation(deg)

        val now = System.currentTimeMillis()
        if (now - lastTs < 33) { imageProxy.close(); return }
        lastTs = now

        val w = img.width
        val h = img.height

        if (bmp == null || bmp!!.width != w || bmp!!.height != h) {
            bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        }
        val cw = (w + 1) / 2
        val ch = (h + 1) / 2
        if (yBuf == null || yBuf!!.size != w*h) yBuf = ByteArray(w*h)
        if (uBuf == null || uBuf!!.size != cw*ch) uBuf = ByteArray(cw*ch)
        if (vBuf == null || vBuf!!.size != cw*ch) vBuf = ByteArray(cw*ch)

        val yPlane = img.planes[0]
        val uPlane = img.planes[1]
        val vPlane = img.planes[2]

        copyPlaneToArray(yPlane.buffer, yPlane.rowStride, yPlane.pixelStride, w, h, yBuf!!)
        copyChromaToArray(uPlane.buffer, uPlane.rowStride, uPlane.pixelStride, cw, ch, uBuf!!)
        copyChromaToArray(vPlane.buffer, vPlane.rowStride, vPlane.pixelStride, cw, ch, vBuf!!)

        var rgba = Native.yuvToRgba(yBuf!!, uBuf!!, vBuf!!, w, h, w, cw, cw)
        if (rgbaBuf == null || rgbaBuf!!.size != rgba.size) rgbaBuf = ByteArray(rgba.size)
        System.arraycopy(rgba, 0, rgbaBuf!!, 0, rgba.size)

        if (useDepth()) {
            val out = depthInfer(rgbaBuf!!, w, h)
            if (out.size != rgbaBuf!!.size) rgbaBuf = ByteArray(out.size)
            System.arraycopy(out, 0, rgbaBuf!!, 0, out.size)
        }

        bmp!!.copyPixelsFromBuffer(ByteBuffer.wrap(rgbaBuf!!))
        onBitmap(bmp!!)

        val (mn, mx, ms) = fetchStats()
        onStats(mn, mx, ms)

        frameCount++
        if (lastFpsTs == 0L) lastFpsTs = now
        val dt = now - lastFpsTs
        if (dt >= 1000) {
            val fps = frameCount * 1000f / dt
            onFps(fps)
            frameCount = 0
            lastFpsTs = now
        }

        imageProxy.close()
    }

    private fun copyPlaneToArray(
        buffer: ByteBuffer,
        rowStride: Int,
        pixelStride: Int,
        width: Int,
        height: Int,
        out: ByteArray
    ) {
        buffer.rewind()
        val row = ByteArray(rowStride)
        var dst = 0
        for (y in 0 until height) {
            val toRead = minOf(rowStride, buffer.remaining())
            if (toRead <= 0) break
            buffer.get(row, 0, toRead)
            var x = 0
            while (x < width) {
                out[dst++] = row[x * pixelStride]
                x++
            }
        }
    }

    private fun copyChromaToArray(
        buffer: ByteBuffer,
        rowStride: Int,
        pixelStride: Int,
        width: Int,
        height: Int,
        out: ByteArray
    ) {
        buffer.rewind()
        val row = ByteArray(rowStride)
        var dst = 0
        for (y in 0 until height) {
            val toRead = minOf(rowStride, buffer.remaining())
            if (toRead <= 0) break
            buffer.get(row, 0, toRead)
            var x = 0
            while (x < width) {
                out[dst++] = row[x * pixelStride]
                x++
            }
        }
    }
}
