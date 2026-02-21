package com.deeplivecam.ui

import android.graphics.*
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.deeplivecam.R
import com.deeplivecam.databinding.ActivityPartSwapLiveBinding
import com.deeplivecam.processing.PartSwapPipeline
import com.deeplivecam.utils.BitmapUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Live camera with Part Swap overlay.
 *
 * Users can drag 4 control points on the live camera view
 * to position the source overlay in real-time.
 */
class PartSwapLiveActivity : AppCompatActivity() {

    private lateinit var binding: ActivityPartSwapLiveBinding
    private val pipeline = PartSwapPipeline()
    private lateinit var cameraExecutor: ExecutorService
    private var isProcessingFrame = AtomicBoolean(false)
    private var useFrontCamera = true
    private var processingEnabled = true

    private var frameCount = 0
    private var lastFpsTime = System.currentTimeMillis()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityPartSwapLiveBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

        setupUI()
        loadSourceFromIntent()
        startCamera()
    }

    private fun setupUI() {
        binding.backButton.setOnClickListener { finish() }

        binding.switchCameraButton.setOnClickListener {
            useFrontCamera = !useFrontCamera
            startCamera()
        }

        binding.processingSwitch.setOnCheckedChangeListener { _, checked ->
            processingEnabled = checked
            binding.processedOverlay.visibility = if (checked) View.VISIBLE else View.GONE
            binding.touchPointOverlay.visibility = if (checked) View.VISIBLE else View.GONE
        }

        binding.opacitySlider.addOnChangeListener { _, value, _ ->
            pipeline.setOpacity(value / 100f)
        }

        binding.captureButton.setOnClickListener {
            capturePhoto()
        }
    }

    private fun loadSourceFromIntent() {
        // Load source image URI passed from PartSwapActivity
        val sourceUriStr = intent.getStringExtra("source_uri")
        if (sourceUriStr != null) {
            val uri = android.net.Uri.parse(sourceUriStr)
            lifecycleScope.launch {
                val bitmap = withContext(Dispatchers.IO) {
                    BitmapUtils.loadBitmap(this@PartSwapLiveActivity, uri, 512)
                }
                if (bitmap != null) {
                    pipeline.setSourceImage(bitmap)
                    binding.sourcePartPreview.setImageBitmap(bitmap)
                }
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also { it.setSurfaceProvider(binding.cameraPreview.surfaceProvider) }

            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                processFrame(imageProxy)
            }

            val cameraSelector = if (useFrontCamera) {
                CameraSelector.DEFAULT_FRONT_CAMERA
            } else {
                CameraSelector.DEFAULT_BACK_CAMERA
            }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
            } catch (e: Exception) {
                Toast.makeText(this, "Camera error: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processFrame(imageProxy: ImageProxy) {
        if (!processingEnabled || !pipeline.hasSource() ||
            !isProcessingFrame.compareAndSet(false, true)
        ) {
            imageProxy.close()
            return
        }

        val bitmap = imageProxyToBitmap(imageProxy)
        imageProxy.close()

        if (bitmap == null) {
            isProcessingFrame.set(false)
            return
        }

        // Get current control points from the overlay
        val points = binding.touchPointOverlay.getPoints()

        // Scale points from overlay view coords to frame coords
        val scaleX = bitmap.width.toFloat() / binding.touchPointOverlay.width
        val scaleY = bitmap.height.toFloat() / binding.touchPointOverlay.height
        val scaledPoints = Array(4) { i ->
            PointF(points[i].x * scaleX, points[i].y * scaleY)
        }

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val result = pipeline.processFrame(bitmap, scaledPoints)
                withContext(Dispatchers.Main) {
                    binding.processedOverlay.setImageBitmap(result)
                    binding.processedOverlay.visibility = View.VISIBLE
                    updateFps()
                }
            } catch (e: Exception) {
                // Skip failed frames
            } finally {
                isProcessingFrame.set(false)
            }
        }
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            val yBuffer = imageProxy.planes[0].buffer
            val uBuffer = imageProxy.planes[1].buffer
            val vBuffer = imageProxy.planes[2].buffer

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)
            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)

            val yuvImage = YuvImage(
                nv21, ImageFormat.NV21,
                imageProxy.width, imageProxy.height, null
            )
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(
                Rect(0, 0, imageProxy.width, imageProxy.height), 80, out
            )
            val bitmap = BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())

            val rotation = imageProxy.imageInfo.rotationDegrees
            if (rotation != 0 || useFrontCamera) {
                val matrix = Matrix()
                matrix.postRotate(rotation.toFloat())
                if (useFrontCamera) matrix.postScale(-1f, 1f)
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            } else {
                bitmap
            }
        } catch (e: Exception) {
            null
        }
    }

    private fun updateFps() {
        frameCount++
        val now = System.currentTimeMillis()
        if (now - lastFpsTime >= 1000) {
            val fps = frameCount * 1000f / (now - lastFpsTime)
            binding.fpsText.text = "FPS: %.1f".format(fps)
            frameCount = 0
            lastFpsTime = now
        }
    }

    private fun capturePhoto() {
        binding.processedOverlay.isDrawingCacheEnabled = true
        val bitmap = binding.processedOverlay.drawingCache
        if (bitmap != null) {
            val filename = "deeplivecam_partswap_live_${System.currentTimeMillis()}.png"
            lifecycleScope.launch {
                val uri = withContext(Dispatchers.IO) {
                    BitmapUtils.saveBitmapToGallery(this@PartSwapLiveActivity, bitmap, filename)
                }
                if (uri != null) {
                    Toast.makeText(this@PartSwapLiveActivity, R.string.save_success, Toast.LENGTH_SHORT).show()
                }
            }
        }
        binding.processedOverlay.isDrawingCacheEnabled = false
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}
