plugins {
    id("com.android.application") version "8.5.2"
    kotlin("android") version "1.9.24"
}

android {
    namespace = "com.sujal.depth"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.sujal.depth"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    kotlinOptions { jvmTarget = "17" }

    buildFeatures { viewBinding = true }

    packaging {
        jniLibs { useLegacyPackaging = true }
    }
}

kotlin { jvmToolchain(17) }

dependencies {
    val camerax = "1.3.4"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:1.3.4")

    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-ktx:1.9.2")
    implementation("com.google.android.material:material:1.12.0")

    // Switch to full PyTorch (1.13.1) for wider op support
    implementation("org.pytorch:pytorch_android:1.13.1")
}
