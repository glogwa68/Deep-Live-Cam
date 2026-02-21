package com.deeplivecam.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import com.deeplivecam.processing.FaceAligner
import java.nio.FloatBuffer

/**
 * ArcFace face recognition model (w600k_r50.onnx).
 * Extracts 512-dimensional face embeddings used by the InSwapper model.
 *
 * Model input: 1x3x112x112 (BGR normalized)
 * Model output: 512-dim embedding vector
 */
class FaceRecognizer(context: Context) {

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    companion object {
        private const val MODEL_FILENAME = "w600k_r50.onnx"
        private const val INPUT_SIZE = 112
        private const val EMBEDDING_DIM = 512
    }

    init {
        val modelPath = ModelManager.getModelPath(context, MODEL_FILENAME)
        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(modelPath, sessionOptions)
    }

    /**
     * Extract face embedding from an aligned face bitmap.
     * The bitmap should already be aligned to 112x112 using ArcFace alignment.
     */
    fun extractEmbedding(alignedFace: Bitmap): FloatArray {
        val inputTensor = preprocessFace(alignedFace)
        val results = session.run(mapOf("input" to inputTensor))
        val outputTensor = results[0] as OnnxTensor
        val embedding = outputTensor.floatBuffer.array().copyOf()

        // L2 normalize the embedding
        val norm = Math.sqrt(embedding.map { (it * it).toDouble() }.sum()).toFloat()
        if (norm > 0) {
            for (i in embedding.indices) {
                embedding[i] /= norm
            }
        }

        inputTensor.close()
        results.close()

        return embedding
    }

    /**
     * Extract embedding from a full image with face landmarks.
     * Performs alignment before embedding extraction.
     */
    fun extractEmbedding(
        bitmap: Bitmap,
        landmarks: FaceDetector.FiveLandmarks
    ): FloatArray {
        val aligned = FaceAligner.alignFace(bitmap, landmarks, INPUT_SIZE)
        return extractEmbedding(aligned)
    }

    /**
     * Preprocess face image for ArcFace model.
     * - Resize to 112x112
     * - Convert to BGR float
     * - Normalize with mean=127.5, std=127.5
     * - Layout: NCHW (1, 3, 112, 112)
     */
    private fun preprocessFace(bitmap: Bitmap): OnnxTensor {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val floatBuffer = FloatBuffer.allocate(1 * 3 * INPUT_SIZE * INPUT_SIZE)

        // NCHW format, BGR order, normalized
        for (c in intArrayOf(2, 1, 0)) { // BGR
            for (y in 0 until INPUT_SIZE) {
                for (x in 0 until INPUT_SIZE) {
                    val pixel = pixels[y * INPUT_SIZE + x]
                    val value = when (c) {
                        0 -> (pixel shr 16 and 0xFF).toFloat() // R -> B channel
                        1 -> (pixel shr 8 and 0xFF).toFloat()  // G
                        2 -> (pixel and 0xFF).toFloat()         // B -> R channel
                        else -> 0f
                    }
                    floatBuffer.put((value - 127.5f) / 127.5f)
                }
            }
        }
        floatBuffer.rewind()

        val shape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        return OnnxTensor.createTensor(env, floatBuffer, shape)
    }

    fun close() {
        session.close()
    }
}
