package com.deeplivecam.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.deeplivecam.R
import com.deeplivecam.databinding.ActivityLiveCameraBinding
import com.deeplivecam.processing.FaceSwapPipeline
import com.deeplivecam.utils.BitmapUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class LiveCameraActivity : AppCompatActivity() {

    private lateinit var binding: ActivityLiveCameraBinding
    private var pipeline: FaceSwapPipeline? = null
    private lateinit var cameraExecutor: ExecutorService
    private var isProcessingFrame = AtomicBoolean(false)
    private var useFrontCamera = true
    private var processingEnabled = true

    // FPS tracking
    private var frameCount = 0
    private var lastFpsTime = System.currentTimeMillis()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityLiveCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

        setupUI()
        initPipeline()
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
        }

        binding.captureButton.setOnClickListener {
            capturePhoto()
        }
    }

    private fun initPipeline() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                pipeline = FaceSwapPipeline(this@LiveCameraActivity)

                // Pipeline needs a source face - for live mode, we assume
                // it was set in MainActivity. In a real app, you'd pass this
                // via a shared ViewModel or singleton.
                withContext(Dispatchers.Main) {
                    startCamera()
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(
                        this@LiveCameraActivity,
                        "Failed to load models: ${e.message}",
                        Toast.LENGTH_LONG
                    ).show()
                    finish()
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
        if (!processingEnabled || !isProcessingFrame.compareAndSet(false, true)) {
            imageProxy.close()
            return
        }

        val bitmap = imageProxyToBitmap(imageProxy)
        imageProxy.close()

        if (bitmap == null) {
            isProcessingFrame.set(false)
            return
        }

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val currentPipeline = pipeline
                if (currentPipeline != null && currentPipeline.hasSourceFace()) {
                    val result = currentPipeline.processFrame(bitmap)
                    withContext(Dispatchers.Main) {
                        binding.processedOverlay.setImageBitmap(result)
                        binding.processedOverlay.visibility = View.VISIBLE
                        updateFps()
                    }
                }
            } catch (e: Exception) {
                // Silently skip failed frames
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
                Rect(0, 0, imageProxy.width, imageProxy.height),
                80, out
            )
            val jpegBytes = out.toByteArray()
            val bitmap = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)

            // Handle rotation
            val rotation = imageProxy.imageInfo.rotationDegrees
            if (rotation != 0) {
                val matrix = Matrix()
                matrix.postRotate(rotation.toFloat())
                if (useFrontCamera) {
                    matrix.postScale(-1f, 1f) // Mirror front camera
                }
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
        val elapsed = now - lastFpsTime
        if (elapsed >= 1000) {
            val fps = frameCount * 1000f / elapsed
            binding.fpsText.text = "FPS: %.1f".format(fps)
            frameCount = 0
            lastFpsTime = now
        }
    }

    private fun capturePhoto() {
        // Save the current processed overlay as a photo
        binding.processedOverlay.isDrawingCacheEnabled = true
        val bitmap = binding.processedOverlay.drawingCache
        if (bitmap != null) {
            val filename = "deeplivecam_live_${System.currentTimeMillis()}.png"
            lifecycleScope.launch {
                val uri = withContext(Dispatchers.IO) {
                    BitmapUtils.saveBitmapToGallery(this@LiveCameraActivity, bitmap, filename)
                }
                if (uri != null) {
                    Toast.makeText(this@LiveCameraActivity, R.string.save_success, Toast.LENGTH_SHORT).show()
                }
            }
        }
        binding.processedOverlay.isDrawingCacheEnabled = false
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        pipeline?.close()
    }
}
