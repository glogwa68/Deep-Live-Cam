package com.deeplivecam.processing

import android.graphics.Bitmap
import android.graphics.PointF

/**
 * Pipeline for arbitrary part/region swapping.
 *
 * Unlike FaceSwapPipeline which uses ML models for face-specific swapping,
 * this pipeline uses perspective warping to paste any source image onto
 * any target region defined by 4 control points.
 *
 * Use cases:
 * - Swap body parts (arms, hands, etc.)
 * - Apply tattoos or designs to skin
 * - Replace clothing/accessories
 * - Overlay logos or textures on surfaces
 * - AR-style sticker placement
 */
class PartSwapPipeline {

    private var sourceImage: Bitmap? = null
    private var controlPoints: Array<PointF>? = null
    private var opacity: Float = 1.0f
    private var feather: Float = 0.08f

    /**
     * Set the source part image.
     */
    fun setSourceImage(bitmap: Bitmap) {
        sourceImage = bitmap
    }

    /**
     * Set the 4 destination control points.
     */
    fun setControlPoints(points: Array<PointF>) {
        controlPoints = points
    }

    /**
     * Set blend opacity (0.0 - 1.0).
     */
    fun setOpacity(value: Float) {
        opacity = value.coerceIn(0f, 1f)
    }

    /**
     * Set edge feathering amount (0.0 - 0.5).
     */
    fun setFeather(value: Float) {
        feather = value.coerceIn(0f, 0.5f)
    }

    fun hasSource(): Boolean = sourceImage != null
    fun hasControlPoints(): Boolean = controlPoints != null

    /**
     * Apply the part swap to a static image.
     */
    fun applyToImage(targetBitmap: Bitmap): RegionWarper.WarpResult {
        val src = sourceImage ?: return RegionWarper.WarpResult(targetBitmap, false)
        val pts = controlPoints ?: return RegionWarper.WarpResult(targetBitmap, false)

        return RegionWarper.warpAndBlend(
            source = src,
            target = targetBitmap,
            dstPoints = pts,
            opacity = opacity,
            feather = feather
        )
    }

    /**
     * Apply the part swap to a live camera frame.
     * For real-time use, the control points should be pre-set.
     */
    fun processFrame(frame: Bitmap): Bitmap {
        val src = sourceImage ?: return frame
        val pts = controlPoints ?: return frame

        val result = RegionWarper.warpAndBlend(
            source = src,
            target = frame,
            dstPoints = pts,
            opacity = opacity,
            feather = feather
        )
        return if (result.success) result.resultBitmap else frame
    }

    /**
     * Process frame with dynamically provided points (for tracked regions).
     */
    fun processFrame(frame: Bitmap, points: Array<PointF>): Bitmap {
        val src = sourceImage ?: return frame

        val result = RegionWarper.warpAndBlend(
            source = src,
            target = frame,
            dstPoints = points,
            opacity = opacity,
            feather = feather
        )
        return if (result.success) result.resultBitmap else frame
    }

    fun clear() {
        sourceImage = null
        controlPoints = null
    }
}
