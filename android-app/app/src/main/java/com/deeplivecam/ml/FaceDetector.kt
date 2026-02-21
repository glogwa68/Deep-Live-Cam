package com.deeplivecam.ml

import android.graphics.Bitmap
import android.graphics.PointF
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine

/**
 * Face detection using Google ML Kit (offline, bundled model).
 * Detects faces and extracts 5-point landmarks for alignment.
 */
class FaceDetector {

    private val detector by lazy {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
            .setMinFaceSize(0.1f)
            .build()
        FaceDetection.getClient(options)
    }

    data class DetectedFace(
        val boundingBox: android.graphics.Rect,
        val landmarks: FiveLandmarks?,
        val mlKitFace: Face
    )

    /**
     * 5-point landmarks used for ArcFace alignment:
     * left eye, right eye, nose tip, left mouth corner, right mouth corner
     */
    data class FiveLandmarks(
        val leftEye: PointF,
        val rightEye: PointF,
        val noseTip: PointF,
        val leftMouth: PointF,
        val rightMouth: PointF
    ) {
        fun toArray(): FloatArray = floatArrayOf(
            leftEye.x, leftEye.y,
            rightEye.x, rightEye.y,
            noseTip.x, noseTip.y,
            leftMouth.x, leftMouth.y,
            rightMouth.x, rightMouth.y
        )
    }

    /**
     * Detect all faces in a bitmap.
     */
    suspend fun detectFaces(bitmap: Bitmap): List<DetectedFace> = suspendCoroutine { cont ->
        val image = InputImage.fromBitmap(bitmap, 0)
        detector.process(image)
            .addOnSuccessListener { faces ->
                val detectedFaces = faces.map { face ->
                    DetectedFace(
                        boundingBox = face.boundingBox,
                        landmarks = extractFiveLandmarks(face),
                        mlKitFace = face
                    )
                }
                cont.resume(detectedFaces)
            }
            .addOnFailureListener { e ->
                cont.resumeWithException(e)
            }
    }

    /**
     * Detect the largest face in a bitmap (used for source face).
     */
    suspend fun detectLargestFace(bitmap: Bitmap): DetectedFace? {
        val faces = detectFaces(bitmap)
        return faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }
    }

    /**
     * Extract 5-point landmarks from ML Kit face detection result.
     */
    private fun extractFiveLandmarks(face: Face): FiveLandmarks? {
        val leftEye = face.getLandmark(FaceLandmark.LEFT_EYE)?.position ?: return null
        val rightEye = face.getLandmark(FaceLandmark.RIGHT_EYE)?.position ?: return null
        val noseTip = face.getLandmark(FaceLandmark.NOSE_BASE)?.position ?: return null
        val leftMouth = face.getLandmark(FaceLandmark.MOUTH_LEFT)?.position ?: return null
        val rightMouth = face.getLandmark(FaceLandmark.MOUTH_RIGHT)?.position ?: return null

        return FiveLandmarks(
            leftEye = leftEye,
            rightEye = rightEye,
            noseTip = noseTip,
            leftMouth = leftMouth,
            rightMouth = rightMouth
        )
    }

    fun close() {
        detector.close()
    }
}
