package com.deeplivecam.processing

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PointF
import kotlin.math.roundToInt

/**
 * Perspective warp engine for arbitrary region/part swapping.
 *
 * Takes a source image and 4 control points on the target,
 * computes a perspective transform, and blends the source
 * onto the target with feathered edges.
 *
 * This allows swapping any body part, applying textures,
 * logos, tattoos, etc. onto any surface in real-time.
 */
object RegionWarper {

    /**
     * 4 corners of the source image (normalized 0..1).
     * Default: full image corners.
     */
    val DEFAULT_SOURCE_CORNERS = arrayOf(
        PointF(0f, 0f),     // top-left
        PointF(1f, 0f),     // top-right
        PointF(1f, 1f),     // bottom-right
        PointF(0f, 1f)      // bottom-left
    )

    data class WarpResult(
        val resultBitmap: Bitmap,
        val success: Boolean
    )

    /**
     * Warp source image onto target at the 4 destination points.
     *
     * @param source Source image to warp (the "part" to paste)
     * @param target Target image/frame
     * @param dstPoints 4 destination points on the target (pixel coords)
     * @param opacity Blend opacity (0.0 - 1.0)
     * @param feather Edge feathering amount (0.0 - 0.5)
     * @return Target with source warped and blended onto it
     */
    fun warpAndBlend(
        source: Bitmap,
        target: Bitmap,
        dstPoints: Array<PointF>,
        opacity: Float = 1.0f,
        feather: Float = 0.05f
    ): WarpResult {
        if (dstPoints.size != 4) {
            return WarpResult(target, false)
        }

        // Source corners in pixel coordinates
        val srcPoints = arrayOf(
            PointF(0f, 0f),
            PointF(source.width.toFloat(), 0f),
            PointF(source.width.toFloat(), source.height.toFloat()),
            PointF(0f, source.height.toFloat())
        )

        // Compute perspective transform matrix (dst -> src)
        // We need inverse: for each pixel in dst region, find source pixel
        val matrix = computePerspectiveMatrix(dstPoints, srcPoints) ?: return WarpResult(target, false)

        val result = target.copy(Bitmap.Config.ARGB_8888, true)
        val resultPixels = IntArray(result.width * result.height)
        result.getPixels(resultPixels, 0, result.width, 0, 0, result.width, result.height)

        val srcPixels = IntArray(source.width * source.height)
        source.getPixels(srcPixels, 0, source.width, 0, 0, source.width, source.height)

        // Compute bounding box of destination quad
        var minX = Float.MAX_VALUE; var maxX = Float.MIN_VALUE
        var minY = Float.MAX_VALUE; var maxY = Float.MIN_VALUE
        for (p in dstPoints) {
            if (p.x < minX) minX = p.x
            if (p.x > maxX) maxX = p.x
            if (p.y < minY) minY = p.y
            if (p.y > maxY) maxY = p.y
        }

        val startX = maxOf(0, minX.toInt() - 1)
        val endX = minOf(result.width - 1, maxX.toInt() + 1)
        val startY = maxOf(0, minY.toInt() - 1)
        val endY = minOf(result.height - 1, maxY.toInt() + 1)

        for (dy in startY..endY) {
            for (dx in startX..endX) {
                // Check if point is inside the destination quad
                if (!isPointInQuad(dx.toFloat(), dy.toFloat(), dstPoints)) continue

                // Apply perspective transform to get source coordinates
                val srcCoord = applyPerspective(matrix, dx.toFloat(), dy.toFloat()) ?: continue

                val sx = srcCoord.x
                val sy = srcCoord.y

                // Bounds check on source
                if (sx < 0 || sy < 0 || sx >= source.width - 1 || sy >= source.height - 1) continue

                // Bilinear interpolation on source
                val srcColor = bilinearSample(srcPixels, source.width, source.height, sx, sy)

                // Compute feathered alpha based on distance to edge of quad
                val edgeDist = distToQuadEdge(dx.toFloat(), dy.toFloat(), dstPoints)
                val quadWidth = maxX - minX
                val quadHeight = maxY - minY
                val featherPixels = feather * minOf(quadWidth, quadHeight)
                val featherAlpha = if (featherPixels > 0) {
                    (edgeDist / featherPixels).coerceIn(0f, 1f)
                } else {
                    1f
                }

                // Also use source alpha if present
                val srcAlpha = (srcColor shr 24 and 0xFF) / 255f
                val finalAlpha = opacity * featherAlpha * srcAlpha

                if (finalAlpha < 0.01f) continue

                // Blend
                val idx = dy * result.width + dx
                val dstColor = resultPixels[idx]

                val sr = (srcColor shr 16 and 0xFF).toFloat()
                val sg = (srcColor shr 8 and 0xFF).toFloat()
                val sb = (srcColor and 0xFF).toFloat()

                val dr = (dstColor shr 16 and 0xFF).toFloat()
                val dg = (dstColor shr 8 and 0xFF).toFloat()
                val db = (dstColor and 0xFF).toFloat()

                val nr = (sr * finalAlpha + dr * (1 - finalAlpha)).roundToInt().coerceIn(0, 255)
                val ng = (sg * finalAlpha + dg * (1 - finalAlpha)).roundToInt().coerceIn(0, 255)
                val nb = (sb * finalAlpha + db * (1 - finalAlpha)).roundToInt().coerceIn(0, 255)

                resultPixels[idx] = (0xFF shl 24) or (nr shl 16) or (ng shl 8) or nb
            }
        }

        result.setPixels(resultPixels, 0, result.width, 0, 0, result.width, result.height)
        return WarpResult(result, true)
    }

    /**
     * Compute 3x3 perspective transform matrix from 4 point correspondences.
     * Maps points from dst space to src space.
     *
     * Uses direct linear transform (DLT) algorithm.
     * Returns null if the transform is degenerate.
     */
    private fun computePerspectiveMatrix(
        src: Array<PointF>,  // source quad (in dst image)
        dst: Array<PointF>   // destination quad (in src image)
    ): FloatArray? {
        // Build 8x8 linear system for perspective transform
        // For each point pair (sx,sy) -> (dx,dy):
        //   dx = (a*sx + b*sy + c) / (g*sx + h*sy + 1)
        //   dy = (d*sx + e*sy + f) / (g*sx + h*sy + 1)

        val a = Array(8) { FloatArray(8) }
        val b = FloatArray(8)

        for (i in 0 until 4) {
            val sx = src[i].x; val sy = src[i].y
            val dx = dst[i].x; val dy = dst[i].y

            a[i * 2] = floatArrayOf(sx, sy, 1f, 0f, 0f, 0f, -dx * sx, -dx * sy)
            b[i * 2] = dx

            a[i * 2 + 1] = floatArrayOf(0f, 0f, 0f, sx, sy, 1f, -dy * sx, -dy * sy)
            b[i * 2 + 1] = dy
        }

        // Solve using Gaussian elimination
        val result = solveLinearSystem(a, b) ?: return null

        // Build 3x3 matrix: [a b c; d e f; g h 1]
        return floatArrayOf(
            result[0], result[1], result[2],
            result[3], result[4], result[5],
            result[6], result[7], 1f
        )
    }

    /**
     * Apply perspective transform to a point.
     */
    private fun applyPerspective(matrix: FloatArray, x: Float, y: Float): PointF? {
        val w = matrix[6] * x + matrix[7] * y + matrix[8]
        if (Math.abs(w) < 1e-10f) return null

        val nx = (matrix[0] * x + matrix[1] * y + matrix[2]) / w
        val ny = (matrix[3] * x + matrix[4] * y + matrix[5]) / w
        return PointF(nx, ny)
    }

    /**
     * Gaussian elimination to solve Ax = b.
     */
    private fun solveLinearSystem(a: Array<FloatArray>, b: FloatArray): FloatArray? {
        val n = b.size
        val m = Array(n) { i -> FloatArray(n + 1) { j -> if (j < n) a[i][j] else b[i] } }

        // Forward elimination
        for (col in 0 until n) {
            // Find pivot
            var maxRow = col
            var maxVal = Math.abs(m[col][col])
            for (row in col + 1 until n) {
                if (Math.abs(m[row][col]) > maxVal) {
                    maxVal = Math.abs(m[row][col])
                    maxRow = row
                }
            }
            if (maxVal < 1e-10f) return null

            // Swap rows
            val tmp = m[col]; m[col] = m[maxRow]; m[maxRow] = tmp

            // Eliminate
            for (row in col + 1 until n) {
                val factor = m[row][col] / m[col][col]
                for (j in col until n + 1) {
                    m[row][j] -= factor * m[col][j]
                }
            }
        }

        // Back substitution
        val x = FloatArray(n)
        for (i in n - 1 downTo 0) {
            x[i] = m[i][n]
            for (j in i + 1 until n) {
                x[i] -= m[i][j] * x[j]
            }
            x[i] /= m[i][i]
        }
        return x
    }

    /**
     * Check if a point is inside a convex quad defined by 4 points.
     */
    private fun isPointInQuad(x: Float, y: Float, quad: Array<PointF>): Boolean {
        var positive = 0
        var negative = 0
        for (i in 0 until 4) {
            val x1 = quad[i].x; val y1 = quad[i].y
            val x2 = quad[(i + 1) % 4].x; val y2 = quad[(i + 1) % 4].y
            val cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
            if (cross > 0) positive++ else if (cross < 0) negative++
            if (positive > 0 && negative > 0) return false
        }
        return true
    }

    /**
     * Compute minimum distance from a point to any edge of the quad.
     */
    private fun distToQuadEdge(x: Float, y: Float, quad: Array<PointF>): Float {
        var minDist = Float.MAX_VALUE
        for (i in 0 until 4) {
            val d = pointToSegmentDist(
                x, y,
                quad[i].x, quad[i].y,
                quad[(i + 1) % 4].x, quad[(i + 1) % 4].y
            )
            if (d < minDist) minDist = d
        }
        return minDist
    }

    /**
     * Distance from point to line segment.
     */
    private fun pointToSegmentDist(
        px: Float, py: Float,
        ax: Float, ay: Float,
        bx: Float, by: Float
    ): Float {
        val dx = bx - ax; val dy = by - ay
        val lenSq = dx * dx + dy * dy
        if (lenSq < 1e-10f) {
            val ex = px - ax; val ey = py - ay
            return Math.sqrt((ex * ex + ey * ey).toDouble()).toFloat()
        }
        val t = ((px - ax) * dx + (py - ay) * dy) / lenSq
        val tc = t.coerceIn(0f, 1f)
        val ex = px - (ax + tc * dx)
        val ey = py - (ay + tc * dy)
        return Math.sqrt((ex * ex + ey * ey).toDouble()).toFloat()
    }

    /**
     * Bilinear sampling from pixel array.
     */
    private fun bilinearSample(
        pixels: IntArray, width: Int, height: Int,
        x: Float, y: Float
    ): Int {
        val x0 = x.toInt().coerceIn(0, width - 1)
        val y0 = y.toInt().coerceIn(0, height - 1)
        val x1 = (x0 + 1).coerceIn(0, width - 1)
        val y1 = (y0 + 1).coerceIn(0, height - 1)

        val fx = x - x0; val fy = y - y0

        val p00 = pixels[y0 * width + x0]
        val p10 = pixels[y0 * width + x1]
        val p01 = pixels[y1 * width + x0]
        val p11 = pixels[y1 * width + x1]

        fun lerp(a: Int, b: Int, c: Int, d: Int, channel: Int): Int {
            val va = (a shr channel and 0xFF).toFloat()
            val vb = (b shr channel and 0xFF).toFloat()
            val vc = (c shr channel and 0xFF).toFloat()
            val vd = (d shr channel and 0xFF).toFloat()
            return (va * (1 - fx) * (1 - fy) + vb * fx * (1 - fy) +
                    vc * (1 - fx) * fy + vd * fx * fy).roundToInt().coerceIn(0, 255)
        }

        val a = lerp(p00, p10, p01, p11, 24)
        val r = lerp(p00, p10, p01, p11, 16)
        val g = lerp(p00, p10, p01, p11, 8)
        val b = lerp(p00, p10, p01, p11, 0)

        return (a shl 24) or (r shl 16) or (g shl 8) or b
    }
}
