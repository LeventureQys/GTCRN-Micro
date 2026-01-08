import os
from pydub import AudioSegment
from pydub.utils import make_chunks

def agc_process_audio(input_path, output_path, target_dBFS=-10):
    """
    对单个WAV文件进行AGC处理（音量归一化）
    :param input_path: 输入WAV文件路径
    :param output_path: 输出WAV文件路径
    :param target_dBFS: 目标音量（dBFS），默认-10dB（适合大多数场景）
    """
    try:
        # 加载WAV文件
        audio = AudioSegment.from_wav(input_path)
        
        # 计算当前音频的音量差值
        change_in_dBFS = target_dBFS - audio.dBFS
        
        # 应用增益调整（AGC核心逻辑）
        normalized_audio = audio.apply_gain(change_in_dBFS)
        
        # 导出处理后的音频
        normalized_audio.export(output_path, format="wav")
        print(f"✅ 处理完成: {input_path} -> {output_path}")
        
    except Exception as e:
        print(f"❌ 处理失败 {input_path}: {str(e)}")

def batch_agc_process(input_folder, output_folder, target_dBFS=-10):
    """
    批量处理文件夹下的所有WAV文件
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    :param target_dBFS: 目标音量（dBFS）
    """
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历文件夹下所有文件
    for filename in os.listdir(input_folder):
        # 只处理WAV文件
        if filename.lower().endswith(".wav"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # 跳过目录，只处理文件
            if os.path.isfile(input_path):
                agc_process_audio(input_path, output_path, target_dBFS)

if __name__ == "__main__":
    # 配置参数
    INPUT_FOLDER = "./output"  # 待处理的WAV文件所在文件夹
    OUTPUT_FOLDER = "./agc_processed"  # 处理后的文件输出文件夹
    TARGET_DBFS = -20  # 目标音量（可调整：-15更柔和，-5更大声）
    
    # 执行批量AGC处理
    batch_agc_process(INPUT_FOLDER, OUTPUT_FOLDER, TARGET_DBFS)
    print("\n🎉 所有WAV文件AGC处理完成！")