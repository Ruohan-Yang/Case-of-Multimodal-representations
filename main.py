import torch
from torch.nn import MultiheadAttention
import numpy as np
from transformers import BertTokenizer, BertModel, ViTModel
import cv2
# cv2 需要 conda install opencv
# pycharm中安装opencv无法使用的问题解决办法 https://blog.csdn.net/m0_47278454/article/details/118469584

def extract_image_features(path, model):
    images = []
    for i in range(1, 11):  # 案例 batch 10
        img_path = path + str(i) + ".jpg"
        img = cv2.imread(img_path)  # (h,w,c)格式 (高 467, 宽 700, 通道 3)
        img = cv2.resize(img, (224, 224))   # (224, 224, 3)
        images.append(img)
    images = np.array([images]).squeeze()  # (batch, 224, 224, 3)
    images = torch.tensor(images).permute(0, 3, 1, 2)  # CHW格式 (batch, 3, 224, 224)
    output = model(images)
    # print(output)  # last_hidden_state, pooler_output
    return output.pooler_output

def extract_text_features(path, tokenizer, model):
    text_features = torch.Tensor()
    for i in range(1, 11):  # 案例 batch 10
        text_path = path + str(i) + ".txt"
        text = open(text_path, 'r').read()
        # print(text)
        # 预分词, 把句子切分成更小的“词”单元。
        # 分词是tokenizer.tokenize, 分词并转化为id是tokenier.encode
        # print(tokenizer.tokenize(text))
        input = tokenizer.encode(text)
        input = torch.tensor([input])
        # print(input)
        # print(input.shape)  # torch.Size([1, 24])
        output = model(input)
        # print(output)  # last_hidden_state, pooler_output
        # print(output[0].shape)  # last_hidden_state  torch.Size([1, 24, 768])
        # print(output[1].shape)  # pooler_output  torch.Size([1, 768])
        # last_hidden_state vs pooler_output的区别  https://blog.csdn.net/ningyanggege/article/details/132206331
        text_features = torch.cat((text_features, output.pooler_output), dim=0)
    return text_features

def merge_features(image_features, text_features):
    multiheadattention = MultiheadAttention(768, 8, dropout=0.1)
    text_image_features, _ = multiheadattention(image_features, text_features, text_features)
    return text_image_features

def load_model():
    ViT_PATH = './model_pretrained/vit-base-patch16-224-in21k'
    ViT_model = ViTModel.from_pretrained(ViT_PATH)
    BERT_PATH = './model_pretrained/bert-base-uncased'
    Bert_tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    Bert_model = BertModel.from_pretrained(BERT_PATH)
    # 预训练好的模型已经加载到本地'./model_pretrained/'。
    # 额外补充：
    # 再读VIT，还有多少细节是你不知道的 https://zhuanlan.zhihu.com/p/657666107
    # 如何下载和在本地使用Bert预训练模型  https://blog.csdn.net/weixin_38481963/article/details/110535583
    # 一行命令实现HuggingFace 国内高速下载 https://mp.weixin.qq.com/s/Fx6nfFt_RPwDHZ3V73PD1Q
    # 也可以考虑https://hf-mirror.com/镜像
    return ViT_model, Bert_tokenizer, Bert_model


if __name__ == '__main__':
    ViT_model, Bert_tokenizer, Bert_model = load_model()
    path = './data/'
    image_features = extract_image_features(path, ViT_model)
    print("image_features ", image_features)
    print("image_features.shape ", image_features.shape)  # (batch, 768)
    text_features = extract_text_features(path, Bert_tokenizer, Bert_model)
    print("text_features ", text_features)
    print("text_features.shape ", text_features.shape)  # (batch, 768)
    multi_features = merge_features(image_features, text_features)
    print("multi_features ", multi_features)
    print("multi_features.shape ", multi_features.shape)  # (batch, 768)
