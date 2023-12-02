



import codecs

def detect_encoding(file_path):
    with codecs.open(file_path, 'rb') as f:
        content = f.read()
        encoding = f.encoding
        print(f"Detected encoding: {encoding}")

# 用实际文件路径替换 'your_file.txt'
detect_encoding('/home/zyyy/TF1-sentiment/Large_Scale_Sentiment_Classification_Data/Labeled_data_11754/11754.csv')
