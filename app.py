from flask import Flask, request, render_template, send_file
import os
import subprocess

app = Flask(__name__)

# Đường dẫn đến thư mục gốc của LatentSync và ứng dụng
LATENTSYNC_DIR = "/home/cp/la/LatentSync"
APP_DIR = "/home/cp/la"  # Thư mục gốc của ứng dụng Flask
CHECKPOINT_DIR = os.path.join(LATENTSYNC_DIR, "checkpoints")
INFERENCE_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "latentsync_unet.pt")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    video_file = request.files["video"]
    audio_file = request.files["audio"]

    # Sử dụng đường dẫn tuyệt đối
    video_path = os.path.join(APP_DIR, "static", "temp_video.mp4")
    audio_path = os.path.join(APP_DIR, "static", "temp_audio.wav")
    output_path = os.path.join(APP_DIR, "static", "output.mp4")

    video_file.save(video_path)
    audio_file.save(audio_path)

    # Chạy LatentSync inference như một module
    cmd = [
        "python",
        "-m",
        "scripts.inference",
        "--unet_config_path",
        os.path.join(LATENTSYNC_DIR, "configs", "unet", "stage2.yaml"),
        "--inference_ckpt_path",
        INFERENCE_CKPT_PATH,
        "--video_path",
        video_path,
        "--audio_path",
        audio_path,
        "--video_out_path",
        output_path,
        "--inference_steps",
        "20",
        "--guidance_scale",
        "1.0",
        "--seed",
        "1247",
    ]
    result = subprocess.run(
        cmd, cwd=LATENTSYNC_DIR, capture_output=True, text=True
    )

    # Kiểm tra lỗi từ LatentSync
    if result.returncode != 0:
        error_msg = f"LatentSync failed with error:\n{result.stderr}"
        print(error_msg)
        return f"An error occurred: {error_msg}", 500

    # Xóa file tạm sau khi xử lý
    if os.path.exists(video_path):
        os.remove(video_path)
    if os.path.exists(audio_path):
        os.remove(audio_path)

    # Kiểm tra xem file output có tồn tại không
    if not os.path.exists(output_path):
        return "Error: Output file was not generated.", 500

    return send_file(output_path, mimetype="video/mp4")


if __name__ == "__main__":
    app.run(debug=True)
