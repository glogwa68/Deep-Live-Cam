package com.deeplivecam

import android.app.Application

class DeepLiveCamApp : Application() {

    override fun onCreate() {
        super.onCreate()
        instance = this
    }

    companion object {
        lateinit var instance: DeepLiveCamApp
            private set
    }
}
