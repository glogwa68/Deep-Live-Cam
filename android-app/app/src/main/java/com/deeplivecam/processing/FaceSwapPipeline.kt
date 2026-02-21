package com.deeplivecam.processing

import android.content.Context
import android.graphics.Bitmap
import com.deeplivecam.ml.FaceDetector
import com.deeplivecam.ml.FaceRecognizer
import com.deeplivecam.ml.FaceSwapper

/**
 * Complete face swap pipeline that orchestrates:
 * 1. Face detection (ML Kit)
 * 2. Face embedding extraction (ArcFace ONNX)
 * 3. Face swapping (InSwapper ONNX)
 * 4. Face paste-back with blending
 */
class FaceSwapPipeline(context: Context) {

    private val faceDetector = FaceDetector()
    private val faceRecognizer = FaceRecognizer(context)
    private val faceSwapper = FaceSwapper(context)

    // Cached source embedding
    private var sourceEmbedding: FloatArray? = null
    private var sourceBitmap: Bitmap? = null

    sealed class SwapResult {
        data class Success(val resultBitmap: Bitmap) : SwapResult()
        data class Error(val message: String) : SwapResult()
    }

    /**
     * Set the source face image. Extracts and caches the face embedding.
     *
     * @param bitmap Source face image
     * @return true if a face was detected and embedding extracted
     */
    suspend fun setSourceFace(bitmap: Bitmap): Boolean {
        val face = faceDetector.detectLargestFace(bitmap) ?: return false
        val landmarks = face.landmarks ?: return false

        sourceEmbedding = faceRecognizer.extractEmbedding(bitmap, landmarks)
        sourceBitmap = bitmap
        return true
    }

    /**
     * Check if a source face has been set.
     */
    fun hasSourceFace(): Boolean = sourceEmbedding != null

    /**
     * Swap faces in a target image.
     * Replaces the first (or all) detected face(s) with the source face.
     *
     * @param targetBitmap Target image
     * @param swapAllFaces If true, swap all detected faces; otherwise only the largest
     * @return SwapResult with the processed image or error message
     */
    suspend fun swapFace(targetBitmap: Bitmap, swapAllFaces: Boolean = false): SwapResult {
        val embedding = sourceEmbedding ?: return SwapResult.Error("No source face set")

        val targetFaces = faceDetector.detectFaces(targetBitmap)
        if (targetFaces.isEmpty()) {
            return SwapResult.Error("No face detected in target image")
        }

        val facesToSwap = if (swapAllFaces) {
            targetFaces
        } else {
            val largest = targetFaces.maxByOrNull {
                it.boundingBox.width() * it.boundingBox.height()
            }
            listOfNotNull(largest)
        }

        var result = targetBitmap.copy(Bitmap.Config.ARGB_8888, true)

        for (face in facesToSwap) {
            val landmarks = face.landmarks ?: continue

            try {
                result = faceSwapper.swapFaceInFrame(embedding, result, landmarks)
            } catch (e: Exception) {
                // Continue with other faces if one fails
                continue
            }
        }

        return SwapResult.Success(result)
    }

    /**
     * Swap face in a single frame (for live/video processing).
     * Optimized for speed with cached source embedding.
     *
     * @param frame Camera/video frame
     * @return Processed frame or original if no face detected
     */
    suspend fun processFrame(frame: Bitmap): Bitmap {
        val embedding = sourceEmbedding ?: return frame

        val face = faceDetector.detectLargestFace(frame) ?: return frame
        val landmarks = face.landmarks ?: return frame

        return try {
            faceSwapper.swapFaceInFrame(embedding, frame, landmarks)
        } catch (e: Exception) {
            frame
        }
    }

    /**
     * Clear the cached source face.
     */
    fun clearSourceFace() {
        sourceEmbedding = null
        sourceBitmap = null
    }

    fun close() {
        faceDetector.close()
        faceRecognizer.close()
        faceSwapper.close()
    }
}
