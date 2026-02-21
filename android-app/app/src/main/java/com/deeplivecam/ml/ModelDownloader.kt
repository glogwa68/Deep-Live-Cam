package com.deeplivecam.ml

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * Downloads required ML models from HuggingFace at first launch.
 * Shows progress per model and handles resume on interruption.
 */
class ModelDownloader(private val context: Context) {

    data class DownloadProgress(
        val modelName: String,
        val bytesDownloaded: Long,
        val totalBytes: Long,
        val modelIndex: Int,
        val totalModels: Int
    ) {
        val percent: Int
            get() = if (totalBytes > 0) ((bytesDownloaded * 100) / totalBytes).toInt() else 0

        val overallPercent: Int
            get() {
                if (totalModels == 0) return 0
                val perModel = 100f / totalModels
                return (modelIndex * perModel + percent * perModel / 100f).toInt()
            }

        val megabytesDownloaded: String
            get() = "%.1f".format(bytesDownloaded / (1024f * 1024f))

        val totalMegabytes: String
            get() = "%.0f".format(totalBytes / (1024f * 1024f))
    }

    sealed class DownloadResult {
        object Success : DownloadResult()
        data class Error(val message: String, val modelName: String) : DownloadResult()
    }

    /**
     * Download all missing models with progress callback.
     */
    suspend fun downloadMissingModels(
        onProgress: (DownloadProgress) -> Unit
    ): DownloadResult = withContext(Dispatchers.IO) {
        val missing = ModelManager.getMissingModels(context)
        if (missing.isEmpty()) return@withContext DownloadResult.Success

        val modelsDir = ModelManager.getModelsDir(context)

        for ((index, model) in missing.withIndex()) {
            val result = downloadModel(
                model = model,
                outputDir = modelsDir,
                modelIndex = index,
                totalModels = missing.size,
                onProgress = onProgress
            )
            if (result is DownloadResult.Error) {
                return@withContext result
            }
        }

        DownloadResult.Success
    }

    /**
     * Download a single model file with resume support.
     */
    private suspend fun downloadModel(
        model: ModelManager.ModelInfo,
        outputDir: File,
        modelIndex: Int,
        totalModels: Int,
        onProgress: (DownloadProgress) -> Unit
    ): DownloadResult = withContext(Dispatchers.IO) {
        val outputFile = File(outputDir, model.filename)
        val tempFile = File(outputDir, "${model.filename}.tmp")

        try {
            // Check if partially downloaded (resume support)
            var downloadedBytes = if (tempFile.exists()) tempFile.length() else 0L

            val connection = URL(model.downloadUrl).openConnection() as HttpURLConnection
            connection.connectTimeout = 30_000
            connection.readTimeout = 30_000
            connection.setRequestProperty("User-Agent", "DeepLiveCam-Android/1.0")

            // Resume from where we left off
            if (downloadedBytes > 0) {
                connection.setRequestProperty("Range", "bytes=$downloadedBytes-")
            }

            connection.connect()

            val responseCode = connection.responseCode
            val totalBytes: Long

            if (responseCode == HttpURLConnection.HTTP_PARTIAL) {
                // Resume successful
                val contentLength = connection.contentLengthLong
                totalBytes = downloadedBytes + contentLength
            } else if (responseCode == HttpURLConnection.HTTP_OK) {
                // Fresh download (server doesn't support range or start over)
                downloadedBytes = 0L
                totalBytes = connection.contentLengthLong
            } else {
                connection.disconnect()
                return@withContext DownloadResult.Error(
                    "HTTP $responseCode for ${model.filename}",
                    model.filename
                )
            }

            // Report initial progress
            withContext(Dispatchers.Main) {
                onProgress(
                    DownloadProgress(model.filename, downloadedBytes, totalBytes, modelIndex, totalModels)
                )
            }

            val inputStream = connection.inputStream
            val outputStream = FileOutputStream(tempFile, downloadedBytes > 0)

            val buffer = ByteArray(8192)
            var bytesRead: Int
            var lastReportTime = System.currentTimeMillis()

            inputStream.use { input ->
                outputStream.use { output ->
                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        output.write(buffer, 0, bytesRead)
                        downloadedBytes += bytesRead

                        // Report progress every 200ms to avoid UI spam
                        val now = System.currentTimeMillis()
                        if (now - lastReportTime >= 200) {
                            lastReportTime = now
                            val progress = DownloadProgress(
                                model.filename, downloadedBytes, totalBytes,
                                modelIndex, totalModels
                            )
                            withContext(Dispatchers.Main) {
                                onProgress(progress)
                            }
                        }
                    }
                }
            }

            connection.disconnect()

            // Final progress
            withContext(Dispatchers.Main) {
                onProgress(
                    DownloadProgress(model.filename, totalBytes, totalBytes, modelIndex, totalModels)
                )
            }

            // Rename temp to final
            if (outputFile.exists()) outputFile.delete()
            if (!tempFile.renameTo(outputFile)) {
                return@withContext DownloadResult.Error(
                    "Failed to save ${model.filename}",
                    model.filename
                )
            }

            DownloadResult.Success
        } catch (e: Exception) {
            // Keep temp file for resume on next attempt
            DownloadResult.Error(
                "${model.filename}: ${e.message ?: "Download failed"}",
                model.filename
            )
        }
    }
}
