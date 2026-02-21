package com.deeplivecam.processing

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import com.deeplivecam.ml.FaceDetector

/**
 * Face alignment utilities for ArcFace and InSwapper models.
 *
 * Computes affine transformations to align detected faces to standard
 * templates (112x112 for ArcFace, 128x128 for InSwapper).
 */
object FaceAligner {

    /**
     * Standard ArcFace alignment template (112x112).
     * 5 landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
     */
    private val ARCFACE_TEMPLATE_112 = arrayOf(
        floatArrayOf(38.2946f, 51.6963f),  // left eye
        floatArrayOf(73.5318f, 51.5014f),  // right eye
        floatArrayOf(56.0252f, 71.7366f),  // nose tip
        floatArrayOf(41.5493f, 92.3655f),  // left mouth
        floatArrayOf(70.7299f, 92.2041f)   // right mouth
    )

    /**
     * Scaled template for 128x128 InSwapper input.
     */
    private val INSWAPPER_TEMPLATE_128 = ARCFACE_TEMPLATE_112.map { point ->
        floatArrayOf(point[0] * 128f / 112f, point[1] * 128f / 112f)
    }.toTypedArray()

    data class AlignResult(
        val alignedFace: Bitmap,
        val inverseMatrix: FloatArray  // 2x3 affine inverse matrix for paste-back
    )

    /**
     * Align face to standard template.
     *
     * @param bitmap Source image
     * @param landmarks 5-point facial landmarks
     * @param outputSize Output size (112 for ArcFace, 128 for InSwapper)
     * @return Aligned face bitmap
     */
    fun alignFace(
        bitmap: Bitmap,
        landmarks: FaceDetector.FiveLandmarks,
        outputSize: Int
    ): Bitmap {
        return alignFaceWithInverse(bitmap, landmarks, outputSize).alignedFace
    }

    /**
     * Align face and return the inverse transformation matrix.
     */
    fun alignFaceWithInverse(
        bitmap: Bitmap,
        landmarks: FaceDetector.FiveLandmarks,
        outputSize: Int
    ): AlignResult {
        val template = if (outputSize == 112) ARCFACE_TEMPLATE_112 else INSWAPPER_TEMPLATE_128

        val srcPoints = arrayOf(
            floatArrayOf(landmarks.leftEye.x, landmarks.leftEye.y),
            floatArrayOf(landmarks.rightEye.x, landmarks.rightEye.y),
            floatArrayOf(landmarks.noseTip.x, landmarks.noseTip.y),
            floatArrayOf(landmarks.leftMouth.x, landmarks.leftMouth.y),
            floatArrayOf(landmarks.rightMouth.x, landmarks.rightMouth.y)
        )

        // Compute similarity transform (scale, rotation, translation)
        val matrix = estimateSimilarityTransform(srcPoints, template)
        val inverseMatrix = invertAffineMatrix(matrix)

        // Apply transform
        val aligned = applyAffineTransform(bitmap, matrix, outputSize, outputSize)

        return AlignResult(aligned, inverseMatrix)
    }

    /**
     * Paste a swapped face back onto the original frame.
     *
     * @param frame Original frame
     * @param swappedFace Swapped face (128x128)
     * @param inverseMatrix Inverse affine matrix from alignment
     * @return Frame with face pasted back
     */
    fun pasteBack(
        frame: Bitmap,
        swappedFace: Bitmap,
        inverseMatrix: FloatArray
    ): Bitmap {
        val result = frame.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(result)

        val faceW = swappedFace.width
        val faceH = swappedFace.height

        // Create a mask for smooth blending (elliptical, feathered)
        val mask = createFaceMask(faceW, faceH)

        // For each pixel in the swapped face, compute its position in the frame
        val swappedPixels = IntArray(faceW * faceH)
        swappedFace.getPixels(swappedPixels, 0, faceW, 0, 0, faceW, faceH)

        val maskPixels = IntArray(faceW * faceH)
        mask.getPixels(maskPixels, 0, faceW, 0, 0, faceW, faceH)

        val resultPixels = IntArray(result.width * result.height)
        result.getPixels(resultPixels, 0, result.width, 0, 0, result.width, result.height)

        // inverse matrix components: [a, b, tx, c, d, ty]
        val a = inverseMatrix[0]
        val b = inverseMatrix[1]
        val tx = inverseMatrix[2]
        val c = inverseMatrix[3]
        val d = inverseMatrix[4]
        val ty = inverseMatrix[5]

        for (fy in 0 until faceH) {
            for (fx in 0 until faceW) {
                // Transform face coordinate to frame coordinate
                val frameX = (a * fx + b * fy + tx).toInt()
                val frameY = (c * fx + d * fy + ty).toInt()

                if (frameX < 0 || frameX >= result.width || frameY < 0 || frameY >= result.height) {
                    continue
                }

                val maskAlpha = (maskPixels[fy * faceW + fx] and 0xFF) / 255f
                if (maskAlpha < 0.01f) continue

                val srcPixel = swappedPixels[fy * faceW + fx]
                val dstIdx = frameY * result.width + frameX
                val dstPixel = resultPixels[dstIdx]

                // Alpha blend
                val sr = (srcPixel shr 16 and 0xFF).toFloat()
                val sg = (srcPixel shr 8 and 0xFF).toFloat()
                val sb = (srcPixel and 0xFF).toFloat()

                val dr = (dstPixel shr 16 and 0xFF).toFloat()
                val dg = (dstPixel shr 8 and 0xFF).toFloat()
                val db = (dstPixel and 0xFF).toFloat()

                val nr = (sr * maskAlpha + dr * (1 - maskAlpha)).toInt().coerceIn(0, 255)
                val ng = (sg * maskAlpha + dg * (1 - maskAlpha)).toInt().coerceIn(0, 255)
                val nb = (sb * maskAlpha + db * (1 - maskAlpha)).toInt().coerceIn(0, 255)

                resultPixels[dstIdx] = (0xFF shl 24) or (nr shl 16) or (ng shl 8) or nb
            }
        }

        result.setPixels(resultPixels, 0, result.width, 0, 0, result.width, result.height)
        return result
    }

    /**
     * Create an elliptical face mask with feathered edges.
     */
    private fun createFaceMask(width: Int, height: Int): Bitmap {
        val mask = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height)

        val cx = width / 2f
        val cy = height / 2f
        val rx = width * 0.4f  // ellipse radius X
        val ry = height * 0.45f // ellipse radius Y
        val feather = 0.15f // feather ratio

        for (y in 0 until height) {
            for (x in 0 until width) {
                val dx = (x - cx) / rx
                val dy = (y - cy) / ry
                val dist = Math.sqrt((dx * dx + dy * dy).toDouble()).toFloat()

                val alpha = when {
                    dist <= 1f - feather -> 1f
                    dist >= 1f + feather -> 0f
                    else -> {
                        val t = (1f + feather - dist) / (2f * feather)
                        // Smooth step
                        t * t * (3 - 2 * t)
                    }
                }

                val a = (alpha * 255).toInt().coerceIn(0, 255)
                pixels[y * width + x] = (0xFF shl 24) or (a shl 16) or (a shl 8) or a
            }
        }

        mask.setPixels(pixels, 0, width, 0, 0, width, height)
        return mask
    }

    /**
     * Estimate similarity transform from source to destination points.
     * Uses least squares to find [scale*cos, -scale*sin, tx, scale*sin, scale*cos, ty].
     *
     * Returns 2x3 affine matrix as float array [a, b, tx, c, d, ty].
     */
    private fun estimateSimilarityTransform(
        src: Array<FloatArray>,
        dst: Array<FloatArray>
    ): FloatArray {
        val n = src.size

        // Compute centroids
        var srcCx = 0f; var srcCy = 0f
        var dstCx = 0f; var dstCy = 0f
        for (i in 0 until n) {
            srcCx += src[i][0]; srcCy += src[i][1]
            dstCx += dst[i][0]; dstCy += dst[i][1]
        }
        srcCx /= n; srcCy /= n
        dstCx /= n; dstCy /= n

        // Center the points
        val srcCentered = Array(n) { floatArrayOf(src[it][0] - srcCx, src[it][1] - srcCy) }
        val dstCentered = Array(n) { floatArrayOf(dst[it][0] - dstCx, dst[it][1] - dstCy) }

        // Compute scale and rotation using Procrustes analysis
        var srcNorm = 0f
        var dotAA = 0f; var dotBB = 0f

        for (i in 0 until n) {
            srcNorm += srcCentered[i][0] * srcCentered[i][0] + srcCentered[i][1] * srcCentered[i][1]
            dotAA += srcCentered[i][0] * dstCentered[i][0] + srcCentered[i][1] * dstCentered[i][1]
            dotBB += srcCentered[i][0] * dstCentered[i][1] - srcCentered[i][1] * dstCentered[i][0]
        }

        if (srcNorm < 1e-6f) {
            // Degenerate case: return identity-like transform
            return floatArrayOf(1f, 0f, dstCx - srcCx, 0f, 1f, dstCy - srcCy)
        }

        val a = dotAA / srcNorm
        val b = dotBB / srcNorm

        // Translation
        val tx = dstCx - (a * srcCx + b * srcCy)
        val ty = dstCy - (-b * srcCx + a * srcCy)

        return floatArrayOf(a, b, tx, -b, a, ty)
    }

    /**
     * Invert a 2x3 affine matrix.
     */
    private fun invertAffineMatrix(m: FloatArray): FloatArray {
        val a = m[0]; val b = m[1]; val tx = m[2]
        val c = m[3]; val d = m[4]; val ty = m[5]

        val det = a * d - b * c
        if (Math.abs(det) < 1e-10f) {
            return floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f)
        }

        val invDet = 1f / det
        return floatArrayOf(
            d * invDet,
            -b * invDet,
            (b * ty - d * tx) * invDet,
            -c * invDet,
            a * invDet,
            (c * tx - a * ty) * invDet
        )
    }

    /**
     * Apply affine transform to a bitmap.
     */
    private fun applyAffineTransform(
        bitmap: Bitmap,
        matrix: FloatArray,
        outputW: Int,
        outputH: Int
    ): Bitmap {
        val output = Bitmap.createBitmap(outputW, outputH, Bitmap.Config.ARGB_8888)
        val srcPixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(srcPixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        val outPixels = IntArray(outputW * outputH)

        // Inverse of the forward matrix to map output -> input
        val inv = invertAffineMatrix(matrix)
        val ia = inv[0]; val ib = inv[1]; val itx = inv[2]
        val ic = inv[3]; val id = inv[4]; val ity = inv[5]

        for (dy in 0 until outputH) {
            for (dx in 0 until outputW) {
                // Map output pixel to input pixel
                val sx = ia * dx + ib * dy + itx
                val sy = ic * dx + id * dy + ity

                // Bilinear interpolation
                val x0 = sx.toInt()
                val y0 = sy.toInt()
                val x1 = x0 + 1
                val y1 = y0 + 1

                if (x0 < 0 || y0 < 0 || x1 >= bitmap.width || y1 >= bitmap.height) {
                    outPixels[dy * outputW + dx] = Color.BLACK
                    continue
                }

                val fx = sx - x0
                val fy = sy - y0

                val p00 = srcPixels[y0 * bitmap.width + x0]
                val p10 = srcPixels[y0 * bitmap.width + x1]
                val p01 = srcPixels[y1 * bitmap.width + x0]
                val p11 = srcPixels[y1 * bitmap.width + x1]

                val r = bilinear(
                    (p00 shr 16 and 0xFF).toFloat(), (p10 shr 16 and 0xFF).toFloat(),
                    (p01 shr 16 and 0xFF).toFloat(), (p11 shr 16 and 0xFF).toFloat(),
                    fx, fy
                ).toInt().coerceIn(0, 255)

                val g = bilinear(
                    (p00 shr 8 and 0xFF).toFloat(), (p10 shr 8 and 0xFF).toFloat(),
                    (p01 shr 8 and 0xFF).toFloat(), (p11 shr 8 and 0xFF).toFloat(),
                    fx, fy
                ).toInt().coerceIn(0, 255)

                val bv = bilinear(
                    (p00 and 0xFF).toFloat(), (p10 and 0xFF).toFloat(),
                    (p01 and 0xFF).toFloat(), (p11 and 0xFF).toFloat(),
                    fx, fy
                ).toInt().coerceIn(0, 255)

                outPixels[dy * outputW + dx] = (0xFF shl 24) or (r shl 16) or (g shl 8) or bv
            }
        }

        output.setPixels(outPixels, 0, outputW, 0, 0, outputW, outputH)
        return output
    }

    private fun bilinear(p00: Float, p10: Float, p01: Float, p11: Float, fx: Float, fy: Float): Float {
        return p00 * (1 - fx) * (1 - fy) + p10 * fx * (1 - fy) + p01 * (1 - fx) * fy + p11 * fx * fy
    }
}
