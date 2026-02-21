# ONNX Runtime
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**

# ML Kit
-keep class com.google.mlkit.** { *; }
-dontwarn com.google.mlkit.**

# Keep face swap engine classes
-keep class com.deeplivecam.ml.** { *; }
-keep class com.deeplivecam.processing.** { *; }
