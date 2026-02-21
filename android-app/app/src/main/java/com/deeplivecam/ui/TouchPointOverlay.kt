package com.deeplivecam.ui

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View

/**
 * Custom overlay view for placing and dragging 4 control points.
 *
 * Users drag the 4 corner points to define where the source image
 * should be warped onto the target. Points are connected with lines
 * to show the destination quad.
 */
class TouchPointOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyle: Int = 0
) : View(context, attrs, defStyle) {

    // 4 control points (in view coordinates)
    private val points = arrayOf(
        PointF(0f, 0f),  // top-left
        PointF(0f, 0f),  // top-right
        PointF(0f, 0f),  // bottom-right
        PointF(0f, 0f)   // bottom-left
    )

    private var initialized = false
    private var activePointIndex = -1
    private val touchRadius = 60f // Touch detection radius in pixels
    private val pointRadius = 18f // Visual point radius
    private val labelSize = 28f

    // Callbacks
    var onPointsChanged: ((Array<PointF>) -> Unit)? = null

    // Paint objects
    private val pointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#E43F5A")
        style = Paint.Style.FILL
    }

    private val pointStrokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }

    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#80E43F5A")
        style = Paint.Style.STROKE
        strokeWidth = 3f
        pathEffect = DashPathEffect(floatArrayOf(15f, 10f), 0f)
    }

    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#20E43F5A")
        style = Paint.Style.FILL
    }

    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = labelSize
        textAlign = Paint.Align.CENTER
        typeface = Typeface.DEFAULT_BOLD
    }

    private val activePointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FFFF44")
        style = Paint.Style.FILL
    }

    private val labels = arrayOf("1", "2", "3", "4")

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        if (!initialized) {
            resetToDefault()
            initialized = true
        }
    }

    /**
     * Reset points to a centered rectangle (40% of view).
     */
    fun resetToDefault() {
        val cx = width / 2f
        val cy = height / 2f
        val hw = width * 0.2f
        val hh = height * 0.2f

        points[0].set(cx - hw, cy - hh)  // top-left
        points[1].set(cx + hw, cy - hh)  // top-right
        points[2].set(cx + hw, cy + hh)  // bottom-right
        points[3].set(cx - hw, cy + hh)  // bottom-left

        invalidate()
        notifyPointsChanged()
    }

    /**
     * Set points programmatically.
     */
    fun setPoints(newPoints: Array<PointF>) {
        for (i in 0 until minOf(4, newPoints.size)) {
            points[i].set(newPoints[i])
        }
        invalidate()
    }

    /**
     * Get current points in view coordinates.
     */
    fun getPoints(): Array<PointF> = points.map { PointF(it.x, it.y) }.toTypedArray()

    /**
     * Get points normalized to image coordinates.
     * @param imageWidth actual image width
     * @param imageHeight actual image height
     */
    fun getPointsForImage(imageWidth: Int, imageHeight: Int): Array<PointF> {
        val scaleX = imageWidth.toFloat() / width
        val scaleY = imageHeight.toFloat() / height
        return points.map { PointF(it.x * scaleX, it.y * scaleY) }.toTypedArray()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // Draw filled quad
        val path = Path().apply {
            moveTo(points[0].x, points[0].y)
            lineTo(points[1].x, points[1].y)
            lineTo(points[2].x, points[2].y)
            lineTo(points[3].x, points[3].y)
            close()
        }
        canvas.drawPath(path, fillPaint)

        // Draw quad edges
        for (i in 0 until 4) {
            canvas.drawLine(
                points[i].x, points[i].y,
                points[(i + 1) % 4].x, points[(i + 1) % 4].y,
                linePaint
            )
        }

        // Draw points
        for (i in 0 until 4) {
            val paint = if (i == activePointIndex) activePointPaint else pointPaint
            canvas.drawCircle(points[i].x, points[i].y, pointRadius, paint)
            canvas.drawCircle(points[i].x, points[i].y, pointRadius, pointStrokePaint)

            // Draw label
            canvas.drawText(
                labels[i],
                points[i].x,
                points[i].y + labelSize / 3,
                labelPaint
            )
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                activePointIndex = findNearestPoint(event.x, event.y)
                if (activePointIndex >= 0) {
                    invalidate()
                    return true
                }
            }
            MotionEvent.ACTION_MOVE -> {
                if (activePointIndex >= 0) {
                    // Clamp to view bounds
                    points[activePointIndex].x = event.x.coerceIn(0f, width.toFloat())
                    points[activePointIndex].y = event.y.coerceIn(0f, height.toFloat())
                    invalidate()
                    notifyPointsChanged()
                    return true
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                activePointIndex = -1
                invalidate()
            }
        }
        return super.onTouchEvent(event)
    }

    private fun findNearestPoint(x: Float, y: Float): Int {
        var minDist = touchRadius
        var minIdx = -1
        for (i in 0 until 4) {
            val dx = x - points[i].x
            val dy = y - points[i].y
            val dist = Math.sqrt((dx * dx + dy * dy).toDouble()).toFloat()
            if (dist < minDist) {
                minDist = dist
                minIdx = i
            }
        }
        return minIdx
    }

    private fun notifyPointsChanged() {
        onPointsChanged?.invoke(getPoints())
    }
}
