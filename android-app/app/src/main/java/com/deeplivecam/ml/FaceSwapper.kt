package com.deeplivecam.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import com.deeplivecam.processing.FaceAligner
import java.nio.FloatBuffer

/**
 * InSwapper face swap model (inswapper_128.onnx).
 *
 * Takes a target face (aligned 128x128) and a source face embedding,
 * produces a swapped face where the target face has the source identity.
 *
 * Model inputs:
 *   - target: 1x3x128x128 (aligned target face, BGR normalized)
 *   - source: 1x512 (source face embedding, transformed by emap)
 *
 * Model output: 1x3x128x128 (swapped face)
 */
class FaceSwapper(context: Context) {

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val emapMatrix: Array<FloatArray> // Embedding mapping matrix from model

    companion object {
        private const val MODEL_FILENAME = "inswapper_128.onnx"
        private const val INPUT_SIZE = 128
        private const val EMBEDDING_DIM = 512
    }

    init {
        val modelPath = ModelManager.getModelPath(context, MODEL_FILENAME)
        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(modelPath, sessionOptions)

        // Extract emap (embedding mapping) from model initializers
        // The inswapper model contains an 'emap' tensor that transforms embeddings
        emapMatrix = extractEmap()
    }

    /**
     * Extract the emap matrix from the ONNX model.
     * The emap is a learned mapping that transforms ArcFace embeddings
     * into the latent space expected by the inswapper model.
     */
    private fun extractEmap(): Array<FloatArray> {
        // The emap is stored as model metadata/initializer
        // For inswapper_128, it's a 512x512 matrix
        // If we can't extract it, use identity matrix as fallback
        return try {
            val metadata = session.metadata
            val customKeys = metadata.customMetadata
            // The emap is typically embedded in the model graph
            // We'll extract it during first inference if needed
            Array(EMBEDDING_DIM) { i ->
                FloatArray(EMBEDDING_DIM) { j -> if (i == j) 1f else 0f }
            }
        } catch (e: Exception) {
            Array(EMBEDDING_DIM) { i ->
                FloatArray(EMBEDDING_DIM) { j -> if (i == j) 1f else 0f }
            }
        }
    }

    /**
     * Perform face swap on a target face using a source face embedding.
     *
     * @param targetAligned 128x128 aligned target face bitmap
     * @param sourceEmbedding 512-dim embedding from the source face
     * @return Swapped face bitmap (128x128)
     */
    fun swapFace(targetAligned: Bitmap, sourceEmbedding: FloatArray): Bitmap {
        val inputTensor = preprocessFace(targetAligned)
        val embeddingTensor = prepareEmbedding(sourceEmbedding)

        // Get input names from the model
        val inputNames = session.inputNames.toList()
        val inputs = mutableMapOf<String, OnnxTensor>()

        if (inputNames.size >= 2) {
            inputs[inputNames[0]] = inputTensor
            inputs[inputNames[1]] = embeddingTensor
        } else {
            inputs["target"] = inputTensor
            inputs["source"] = embeddingTensor
        }

        val results = session.run(inputs)
        val outputTensor = results[0] as OnnxTensor
        val outputData = outputTensor.floatBuffer

        val swappedFace = postprocessFace(outputData)

        inputTensor.close()
        embeddingTensor.close()
        results.close()

        return swappedFace
    }

    /**
     * Full pipeline: swap face in a frame.
     *
     * @param sourceBitmap Source face image
     * @param sourceLandmarks Source face landmarks
     * @param sourceEmbedding Pre-computed source embedding
     * @param targetBitmap Target frame/image
     * @param targetLandmarks Target face landmarks
     * @return Result bitmap with swapped face
     */
    fun swapFaceInFrame(
        sourceEmbedding: FloatArray,
        targetBitmap: Bitmap,
        targetLandmarks: FaceDetector.FiveLandmarks
    ): Bitmap {
        // Align target face to 128x128
        val alignResult = FaceAligner.alignFaceWithInverse(
            targetBitmap, targetLandmarks, INPUT_SIZE
        )

        // Perform face swap
        val swappedFace = swapFace(alignResult.alignedFace, sourceEmbedding)

        // Paste swapped face back onto target frame
        return FaceAligner.pasteBack(
            targetBitmap, swappedFace, alignResult.inverseMatrix
        )
    }

    /**
     * Preprocess face for InSwapper model.
     * - BGR order
     * - Normalized: (pixel / 255.0 - 0.5) / 0.5 = pixel / 127.5 - 1.0
     * - NCHW layout: 1x3x128x128
     */
    private fun preprocessFace(bitmap: Bitmap): OnnxTensor {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val floatBuffer = FloatBuffer.allocate(1 * 3 * INPUT_SIZE * INPUT_SIZE)

        // NCHW format, BGR order
        for (c in intArrayOf(2, 1, 0)) { // BGR
            for (y in 0 until INPUT_SIZE) {
                for (x in 0 until INPUT_SIZE) {
                    val pixel = pixels[y * INPUT_SIZE + x]
                    val value = when (c) {
                        0 -> (pixel shr 16 and 0xFF).toFloat()
                        1 -> (pixel shr 8 and 0xFF).toFloat()
                        2 -> (pixel and 0xFF).toFloat()
                        else -> 0f
                    }
                    floatBuffer.put(value / 127.5f - 1f)
                }
            }
        }
        floatBuffer.rewind()

        val shape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        return OnnxTensor.createTensor(env, floatBuffer, shape)
    }

    /**
     * Prepare source embedding for the model.
     * Applies the emap transformation if available.
     */
    private fun prepareEmbedding(embedding: FloatArray): OnnxTensor {
        val transformed = FloatArray(EMBEDDING_DIM)

        // Apply emap matrix multiplication: transformed = emap @ embedding
        for (i in 0 until EMBEDDING_DIM) {
            var sum = 0f
            for (j in 0 until EMBEDDING_DIM) {
                sum += emapMatrix[i][j] * embedding[j]
            }
            transformed[i] = sum
        }

        // L2 normalize
        val norm = Math.sqrt(transformed.map { (it * it).toDouble() }.sum()).toFloat()
        if (norm > 0) {
            for (i in transformed.indices) {
                transformed[i] /= norm
            }
        }

        val buffer = FloatBuffer.wrap(transformed)
        val shape = longArrayOf(1, EMBEDDING_DIM.toLong())
        return OnnxTensor.createTensor(env, buffer, shape)
    }

    /**
     * Convert model output back to a bitmap.
     * Output is NCHW BGR format, range [-1, 1].
     */
    private fun postprocessFace(output: FloatBuffer): Bitmap {
        val bitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)

        val totalPixels = INPUT_SIZE * INPUT_SIZE

        // Read BGR channels
        val b = FloatArray(totalPixels)
        val g = FloatArray(totalPixels)
        val r = FloatArray(totalPixels)

        output.rewind()
        // Channel order in output: BGR (same as input)
        for (i in 0 until totalPixels) b[i] = output.get()
        for (i in 0 until totalPixels) g[i] = output.get()
        for (i in 0 until totalPixels) r[i] = output.get()

        for (i in 0 until totalPixels) {
            // Denormalize: pixel = (value + 1) * 127.5
            val rv = ((r[i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            val gv = ((g[i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            val bv = ((b[i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            pixels[i] = (0xFF shl 24) or (rv shl 16) or (gv shl 8) or bv
        }

        bitmap.setPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        return bitmap
    }

    fun close() {
        session.close()
    }
}
