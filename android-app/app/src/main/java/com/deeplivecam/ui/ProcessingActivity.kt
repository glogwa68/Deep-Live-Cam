package com.deeplivecam.ui

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.deeplivecam.databinding.ActivityMainBinding

/**
 * Placeholder activity for batch video processing.
 * Video processing is computationally expensive on mobile.
 * This serves as a future extension point.
 */
class ProcessingActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // For now, video processing redirects to main activity
        finish()
    }
}
