import subprocess
import os

# 원본 파일 경로
input_file = "/home/khj6051/landr_4/part_14/packs-attic-techno/SS_AT_Top_Loop_Bare_125_Job.mp3"
# 저장할 파일 경로
output_file = "./fixed_SS_AT_Top_Loop_Bare_125_Job.mp3"

def copy_audio_file(input_path, output_path):
    try:
        # ffmpeg를 사용하여 파일을 그대로 복사 (코덱 변경 없이)
        subprocess.run(
            ["ffmpeg", "-i", input_path, "-acodec", "copy", output_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        print(f"✅ 파일 저장 완료: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 오류 발생: {e.stderr}")

# 실행
copy_audio_file(input_file, output_file)
