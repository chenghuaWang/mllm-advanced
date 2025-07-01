import os
import pymllm as mllm


def compile():
    COMMAND = [
        "cmake",
        "-G",
        "Ninja",
        "-B",
        "build",
        "-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_PATH/build/cmake/android.toolchain.cmake",
        "-DANDROID_PLATFORM=android-28",
        "-DANDROID_ABI=arm64-v8a",
        "-Dmllm_DIR=/root/mllm-install-android-arm64-v8a-qnn/lib/cmake/",
    ]
    print(" ".join(COMMAND))
    os.system(" ".join(COMMAND))
    COMMAND = [
        "cmake",
        "--build",
        "build",
    ]
    print(" ".join(COMMAND))
    os.system(" ".join(COMMAND))


if __name__ == "__main__":
    compile()
    adb = mllm.utils.ADBToolkit()
    print(adb.get_devices())
    adb.push_file(
        "./build/mllm-schedule-time-bt-cpu-npu",
        "/data/local/tmp/mllm-advanced/bin/playground/",
    )
    with adb.get_shell_context() as shell:
        shell.execute("cd /data/local/tmp/mllm-advanced/bin/playground/")
        shell.execute(
            "export LD_LIBRARY_PATH=/data/local/tmp/mllm-advanced/bin:$LD_LIBRARY_PATH"
        )
        shell.execute("export ADSP_LIBRARY_PATH=/data/local/tmp/mllm-advanced/lib64")
        res = shell.execute("./mllm-schedule-time-bt-cpu-npu")
        print(res)
