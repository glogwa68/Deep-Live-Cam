package com.deeplivecam.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.deeplivecam.R
import com.deeplivecam.databinding.ActivityMainBinding
import com.deeplivecam.ml.ModelDownloader
import com.deeplivecam.ml.ModelManager
import com.deeplivecam.processing.FaceSwapPipeline
import com.deeplivecam.utils.BitmapUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    private var pipeline: FaceSwapPipeline? = null
    private var sourceBitmap: Bitmap? = null
    private var targetBitmap: Bitmap? = null
    private var resultBitmap: Bitmap? = null
    private var isProcessing = false
    private var isDownloading = false

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

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            launchLiveMode()
        } else {
            Toast.makeText(this, R.string.camera_permission_required, Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupClickListeners()
        checkAndDownloadModels()
    }

    private fun setupClickListeners() {
        binding.sourceContainer.setOnClickListener {
            if (!isProcessing) sourceImagePicker.launch("image/*")
        }

        binding.targetContainer.setOnClickListener {
            if (!isProcessing) targetImagePicker.launch("image/*")
        }

        binding.startSwapButton.setOnClickListener {
            startFaceSwap()
        }

        binding.liveModeButton.setOnClickListener {
            if (!ModelManager.areAllModelsAvailable(this)) {
                Toast.makeText(this, "Models are still downloading...", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            requestCameraPermission()
        }

        binding.saveButton.setOnClickListener {
            saveResult()
        }

        // Part Swap mode - no ML models required
        binding.partSwapButton.setOnClickListener {
            val intent = Intent(this, PartSwapActivity::class.java)
            startActivity(intent)
        }

        // Tap download card to retry on failure
        binding.downloadCard.setOnClickListener {
            if (!isDownloading && !ModelManager.areAllModelsAvailable(this)) {
                startModelDownload()
            }
        }
    }

    /**
     * Check models at launch: try assets first, then auto-download if missing.
     */
    private fun checkAndDownloadModels() {
        // Try to copy from assets (if bundled in APK)
        ModelManager.copyModelsFromAssets(this)

        if (ModelManager.areAllModelsAvailable(this)) {
            // Models already present, initialize pipeline
            initializePipeline()
        } else {
            // Auto-download missing models
            startModelDownload()
        }
    }

    /**
     * Start downloading missing models with progress UI.
     */
    private fun startModelDownload() {
        if (isDownloading) return
        isDownloading = true

        // Show download card
        binding.downloadCard.visibility = View.VISIBLE
        binding.downloadTitle.text = getString(R.string.downloading_models)
        binding.downloadModelName.text = getString(R.string.first_launch_info)
        binding.downloadProgressBar.progress = 0
        binding.downloadPercentText.text = "0%"
        binding.downloadSizeText.text = ""

        val downloader = ModelDownloader(this)

        lifecycleScope.launch {
            val result = downloader.downloadMissingModels { progress ->
                // Update UI (already on Main thread from ModelDownloader)
                binding.downloadModelName.text = progress.modelName
                binding.downloadProgressBar.progress = progress.overallPercent
                binding.downloadPercentText.text = "${progress.overallPercent}%"
                binding.downloadSizeText.text =
                    "${progress.megabytesDownloaded} / ${progress.totalMegabytes} MB"
            }

            isDownloading = false

            when (result) {
                is ModelDownloader.DownloadResult.Success -> {
                    binding.downloadTitle.text = getString(R.string.download_complete)
                    binding.downloadModelName.text = ""
                    binding.downloadProgressBar.progress = 100
                    binding.downloadPercentText.text = "100%"

                    // Hide download card after a short delay
                    binding.downloadCard.postDelayed({
                        binding.downloadCard.visibility = View.GONE
                    }, 2000)

                    // Initialize the pipeline now that models are available
                    initializePipeline()
                }
                is ModelDownloader.DownloadResult.Error -> {
                    binding.downloadTitle.text = getString(R.string.download_failed)
                    binding.downloadModelName.text = result.message
                    // Card stays visible so user can tap to retry
                }
            }
        }
    }

    private fun initializePipeline() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                pipeline = FaceSwapPipeline(this@MainActivity)
                withContext(Dispatchers.Main) {
                    updateStatus("Models loaded. Ready.")
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    updateStatus("Error loading models: ${e.message}")
                }
            }
        }
    }

    private fun loadSourceImage(uri: Uri) {
        lifecycleScope.launch {
            val bitmap = withContext(Dispatchers.IO) {
                BitmapUtils.loadBitmap(this@MainActivity, uri, 1024)
            } ?: run {
                Toast.makeText(this@MainActivity, "Failed to load image", Toast.LENGTH_SHORT).show()
                return@launch
            }

            sourceBitmap = bitmap
            binding.sourceImageView.setImageBitmap(bitmap)
            binding.sourceImageView.visibility = View.VISIBLE
            binding.sourcePlaceholder.visibility = View.GONE

            updateStatus("Detecting source face...")
            showProgress(true)

            val success = withContext(Dispatchers.IO) {
                try {
                    pipeline?.setSourceFace(bitmap) ?: false
                } catch (e: Exception) {
                    false
                }
            }

            showProgress(false)
            if (success) {
                updateStatus("Source face detected")
            } else {
                updateStatus("No face found in source image")
                Toast.makeText(this@MainActivity, R.string.no_face_detected, Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun loadTargetImage(uri: Uri) {
        lifecycleScope.launch {
            val bitmap = withContext(Dispatchers.IO) {
                BitmapUtils.loadBitmap(this@MainActivity, uri, 1920)
            } ?: run {
                Toast.makeText(this@MainActivity, "Failed to load image", Toast.LENGTH_SHORT).show()
                return@launch
            }

            targetBitmap = bitmap
            binding.targetImageView.setImageBitmap(bitmap)
            binding.targetImageView.visibility = View.VISIBLE
            binding.targetPlaceholder.visibility = View.GONE

            hideResult()
        }
    }

    private fun startFaceSwap() {
        if (isProcessing) return

        val currentPipeline = pipeline
        if (currentPipeline == null) {
            if (!ModelManager.areAllModelsAvailable(this)) {
                if (!isDownloading) startModelDownload()
                Toast.makeText(this, "Models are downloading...", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Models are still loading...", Toast.LENGTH_SHORT).show()
            }
            return
        }

        if (!currentPipeline.hasSourceFace()) {
            Toast.makeText(this, R.string.no_source_face, Toast.LENGTH_SHORT).show()
            return
        }

        val target = targetBitmap
        if (target == null) {
            Toast.makeText(this, R.string.no_target_image, Toast.LENGTH_SHORT).show()
            return
        }

        isProcessing = true
        showProgress(true)
        updateStatus("Swapping faces...")
        binding.startSwapButton.isEnabled = false

        val swapAll = binding.swapAllFacesSwitch.isChecked

        lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                currentPipeline.swapFace(target, swapAll)
            }

            isProcessing = false
            showProgress(false)
            binding.startSwapButton.isEnabled = true

            when (result) {
                is FaceSwapPipeline.SwapResult.Success -> {
                    resultBitmap = result.resultBitmap
                    showResult(result.resultBitmap)
                    updateStatus("Face swap completed!")
                }
                is FaceSwapPipeline.SwapResult.Error -> {
                    updateStatus("Error: ${result.message}")
                    Toast.makeText(
                        this@MainActivity,
                        result.message,
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }
        }
    }

    private fun showResult(bitmap: Bitmap) {
        binding.resultLabel.visibility = View.VISIBLE
        binding.resultContainer.visibility = View.VISIBLE
        binding.resultImageView.setImageBitmap(bitmap)
        binding.saveButton.visibility = View.VISIBLE
    }

    private fun hideResult() {
        binding.resultLabel.visibility = View.GONE
        binding.resultContainer.visibility = View.GONE
        binding.saveButton.visibility = View.GONE
        resultBitmap = null
    }

    private fun saveResult() {
        val bitmap = resultBitmap ?: return
        val filename = "deeplivecam_${System.currentTimeMillis()}.png"

        lifecycleScope.launch {
            val uri = withContext(Dispatchers.IO) {
                BitmapUtils.saveBitmapToGallery(this@MainActivity, bitmap, filename)
            }

            if (uri != null) {
                Toast.makeText(this@MainActivity, R.string.save_success, Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this@MainActivity, R.string.save_error, Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun requestCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            launchLiveMode()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun launchLiveMode() {
        if (pipeline?.hasSourceFace() != true) {
            Toast.makeText(this, R.string.no_source_face, Toast.LENGTH_SHORT).show()
            return
        }
        val intent = Intent(this, LiveCameraActivity::class.java)
        startActivity(intent)
    }

    private fun showProgress(show: Boolean) {
        binding.progressBar.visibility = if (show) View.VISIBLE else View.GONE
    }

    private fun updateStatus(text: String) {
        binding.statusText.text = text
        binding.statusText.visibility = View.VISIBLE
    }

    override fun onDestroy() {
        super.onDestroy()
        pipeline?.close()
    }
}
