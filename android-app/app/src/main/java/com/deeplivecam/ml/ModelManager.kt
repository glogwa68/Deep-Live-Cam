package com.deeplivecam.ml

import android.content.Context
import java.io.File

/**
 * Manages ML model files for the face swap pipeline.
 *
 * Models are stored in the app's internal files directory.
 * Users must download and place models there before first use.
 *
 * Required models:
 *   - inswapper_128.onnx (~550MB) - Face swap model
 *   - w600k_r50.onnx (~175MB) - Face recognition (ArcFace) model
 */
object ModelManager {

    data class ModelInfo(
        val filename: String,
        val description: String,
        val downloadUrl: String,
        val sizeMB: Int
    )

    val REQUIRED_MODELS = listOf(
        ModelInfo(
            filename = "inswapper_128.onnx",
            description = "Face Swap Model (InSwapper)",
            downloadUrl = "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx",
            sizeMB = 550
        ),
        ModelInfo(
            filename = "w600k_r50.onnx",
            description = "Face Recognition Model (ArcFace)",
            downloadUrl = "https://huggingface.co/hacksider/deep-live-cam/resolve/main/w600k_r50.onnx",
            sizeMB = 175
        )
    )

    /**
     * Get the directory where models are stored.
     */
    fun getModelsDir(context: Context): File {
        val dir = File(context.filesDir, "models")
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    /**
     * Get the full path for a model file.
     */
    fun getModelPath(context: Context, filename: String): String {
        return File(getModelsDir(context), filename).absolutePath
    }

    /**
     * Check if a specific model file exists.
     */
    fun isModelAvailable(context: Context, filename: String): Boolean {
        val file = File(getModelsDir(context), filename)
        return file.exists() && file.length() > 0
    }

    /**
     * Check if all required models are available.
     */
    fun areAllModelsAvailable(context: Context): Boolean {
        return REQUIRED_MODELS.all { isModelAvailable(context, it.filename) }
    }

    /**
     * Get list of missing models.
     */
    fun getMissingModels(context: Context): List<ModelInfo> {
        return REQUIRED_MODELS.filter { !isModelAvailable(context, it.filename) }
    }

    /**
     * Copy a model from assets to internal storage (if bundled with APK).
     */
    fun copyModelFromAssets(context: Context, filename: String): Boolean {
        return try {
            val inputStream = context.assets.open("models/$filename")
            val outputFile = File(getModelsDir(context), filename)
            inputStream.use { input ->
                outputFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            true
        } catch (e: Exception) {
            false
        }
    }

    /**
     * Try to copy all models from assets.
     */
    fun copyModelsFromAssets(context: Context) {
        REQUIRED_MODELS.forEach { model ->
            if (!isModelAvailable(context, model.filename)) {
                copyModelFromAssets(context, model.filename)
            }
        }
    }
}
