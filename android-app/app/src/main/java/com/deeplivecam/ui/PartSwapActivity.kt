package com.deeplivecam.ui

import android.graphics.Bitmap
import android.graphics.PointF
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.deeplivecam.R
import com.deeplivecam.databinding.ActivityPartSwapBinding
import com.deeplivecam.processing.PartSwapPipeline
import com.deeplivecam.utils.BitmapUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Activity for the Part Swap mode.
 *
 * Allows users to:
 * 1. Import any source image (tattoo, logo, texture, body part, etc.)
 * 2. Select a target image
 * 3. Drag 4 control points to define where the source gets warped
 * 4. Adjust opacity and edge feathering
 * 5. Preview and save the result
 */
class PartSwapActivity : AppCompatActivity() {

    private lateinit var binding: ActivityPartSwapBinding
    private val pipeline = PartSwapPipeline()

    private var sourceBitmap: Bitmap? = null
    private var targetBitmap: Bitmap? = null
    private var resultBitmap: Bitmap? = null
    private var isProcessing = false

    private val sourceImagePicker = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { loadSourceImage(it) }
    }

    private val targetImagePicker = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { loadTargetImage(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityPartSwapBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupUI()
    }

    private fun setupUI() {
        // Back button
        binding.backButton.setOnClickListener { finish() }

        // Reset control points
        binding.resetPointsButton.setOnClickListener {
            binding.touchPointOverlay.resetToDefault()
        }

        // Source selection
        binding.sourcePartContainer.setOnClickListener { sourceImagePicker.launch("image/*") }
        binding.selectSourceButton.setOnClickListener { sourceImagePicker.launch("image/*") }

        // Target selection
        binding.selectTargetButton.setOnClickListener { targetImagePicker.launch("image/*") }

        // Opacity slider
        binding.opacitySlider.addOnChangeListener { _, value, _ ->
            val opacity = value / 100f
            pipeline.setOpacity(opacity)
            binding.opacityValue.text = "${value.toInt()}%"
        }

        // Feather slider
        binding.featherSlider.addOnChangeListener { _, value, _ ->
            val feather = value / 100f
            pipeline.setFeather(feather)
            binding.featherValue.text = "${value.toInt()}%"
        }

        // Apply button - process the warp
        binding.applyButton.setOnClickListener {
            applyPartSwap()
        }

        // Save button
        binding.savePartSwapButton.setOnClickListener {
            saveResult()
        }

        // Control points changed - update preview live
        binding.touchPointOverlay.onPointsChanged = { _ ->
            // Live preview during drag (lightweight)
            if (pipeline.hasSource() && targetBitmap != null && !isProcessing) {
                updatePreview()
            }
        }
    }

    private fun loadSourceImage(uri: Uri) {
        lifecycleScope.launch {
            val bitmap = withContext(Dispatchers.IO) {
                BitmapUtils.loadBitmap(this@PartSwapActivity, uri, 1024)
            }
            if (bitmap == null) {
                Toast.makeText(this@PartSwapActivity, "Failed to load image", Toast.LENGTH_SHORT).show()
                return@launch
            }

            sourceBitmap = bitmap
            pipeline.setSourceImage(bitmap)

            binding.sourcePartImage.setImageBitmap(bitmap)
            binding.sourcePartImage.visibility = View.VISIBLE
            binding.sourcePartPlaceholder.visibility = View.GONE
            binding.sourceInfoText.text = "${bitmap.width} x ${bitmap.height}"
        }
    }

    private fun loadTargetImage(uri: Uri) {
        lifecycleScope.launch {
            val bitmap = withContext(Dispatchers.IO) {
                BitmapUtils.loadBitmap(this@PartSwapActivity, uri, 1920)
            }
            if (bitmap == null) {
                Toast.makeText(this@PartSwapActivity, "Failed to load image", Toast.LENGTH_SHORT).show()
                return@launch
            }

            targetBitmap = bitmap
            binding.targetImage.setImageBitmap(bitmap)
            binding.targetImage.visibility = View.VISIBLE
            binding.targetPlaceholder.visibility = View.GONE
            binding.previewImage.visibility = View.GONE

            // Reset control points for new image
            binding.touchPointOverlay.resetToDefault()
            binding.savePartSwapButton.isEnabled = false
            resultBitmap = null
        }
    }

    private fun updatePreview() {
        // Quick preview during drag - run on background thread
        val target = targetBitmap ?: return
        val points = getScaledControlPoints(target) ?: return

        lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                pipeline.setControlPoints(points)
                pipeline.applyToImage(target.copy(Bitmap.Config.ARGB_8888, false))
            }

            if (result.success) {
                binding.previewImage.setImageBitmap(result.resultBitmap)
                binding.previewImage.visibility = View.VISIBLE
            }
        }
    }

    private fun applyPartSwap() {
        if (isProcessing) return

        if (!pipeline.hasSource()) {
            Toast.makeText(this, "Select a source image first", Toast.LENGTH_SHORT).show()
            return
        }

        val target = targetBitmap
        if (target == null) {
            Toast.makeText(this, "Select a target image first", Toast.LENGTH_SHORT).show()
            return
        }

        val points = getScaledControlPoints(target)
        if (points == null) {
            Toast.makeText(this, "Position the control points", Toast.LENGTH_SHORT).show()
            return
        }

        isProcessing = true
        binding.progressBar.visibility = View.VISIBLE
        binding.applyButton.isEnabled = false

        lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                pipeline.setControlPoints(points)
                pipeline.applyToImage(target.copy(Bitmap.Config.ARGB_8888, false))
            }

            isProcessing = false
            binding.progressBar.visibility = View.GONE
            binding.applyButton.isEnabled = true

            if (result.success) {
                resultBitmap = result.resultBitmap
                binding.previewImage.setImageBitmap(result.resultBitmap)
                binding.previewImage.visibility = View.VISIBLE
                binding.savePartSwapButton.isEnabled = true
                Toast.makeText(this@PartSwapActivity, "Part swap applied!", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this@PartSwapActivity, "Swap failed - check points", Toast.LENGTH_SHORT).show()
            }
        }
    }

    /**
     * Scale control points from view coordinates to actual image coordinates.
     */
    private fun getScaledControlPoints(targetBitmap: Bitmap): Array<PointF>? {
        val overlay = binding.touchPointOverlay
        val viewW = overlay.width
        val viewH = overlay.height
        if (viewW <= 0 || viewH <= 0) return null

        // The target image is displayed with fitCenter scaleType,
        // so we need to account for the actual image rect within the view
        val imgW = targetBitmap.width.toFloat()
        val imgH = targetBitmap.height.toFloat()

        val viewRatio = viewW.toFloat() / viewH
        val imgRatio = imgW / imgH

        val displayW: Float
        val displayH: Float
        val offsetX: Float
        val offsetY: Float

        if (imgRatio > viewRatio) {
            // Image is wider - letterbox top/bottom
            displayW = viewW.toFloat()
            displayH = viewW / imgRatio
            offsetX = 0f
            offsetY = (viewH - displayH) / 2f
        } else {
            // Image is taller - pillarbox left/right
            displayH = viewH.toFloat()
            displayW = viewH * imgRatio
            offsetX = (viewW - displayW) / 2f
            offsetY = 0f
        }

        val viewPoints = overlay.getPoints()
        return Array(4) { i ->
            val x = (viewPoints[i].x - offsetX) / displayW * imgW
            val y = (viewPoints[i].y - offsetY) / displayH * imgH
            PointF(
                x.coerceIn(0f, imgW),
                y.coerceIn(0f, imgH)
            )
        }
    }

    private fun saveResult() {
        val bitmap = resultBitmap ?: return
        val filename = "deeplivecam_partswap_${System.currentTimeMillis()}.png"

        lifecycleScope.launch {
            val uri = withContext(Dispatchers.IO) {
                BitmapUtils.saveBitmapToGallery(this@PartSwapActivity, bitmap, filename)
            }
            if (uri != null) {
                Toast.makeText(this@PartSwapActivity, R.string.save_success, Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this@PartSwapActivity, R.string.save_error, Toast.LENGTH_SHORT).show()
            }
        }
    }
}
