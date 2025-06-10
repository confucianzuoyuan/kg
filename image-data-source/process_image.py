# 安装依赖
# pip install easyocr

from pathlib import Path
import time


def ocr_easyocr(img):
    import easyocr
    reader = easyocr.Reader(['ch_sim'], gpu=True)
    result = reader.readtext(img)
    result_string = ""
    for t in result:
        try:
            result_string += t[1]
        except:
            pass
    return result_string


start_time = time.time()


def find_jpg_files(root_dir):
    root_path = Path(root_dir)
    jpg_files = list(root_path.rglob('*.jpg'))  # 递归查找所有jpg文件
    return jpg_files

print(ocr_easyocr("images/product-1/1.jpg"))

# 示例用法
text = ""
root_directory = 'images'
jpg_list = find_jpg_files(root_directory)
for jpg in jpg_list:
    s = ocr_easyocr(str(jpg))
    text += (s + '\n')
end_time = time.time()
print('\n ==== OCR cost time: {} ===='.format(end_time-start_time))

with open("image_ocr_text.txt", "a", encoding="utf-8") as f:
    f.write(text)
