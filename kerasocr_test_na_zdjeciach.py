# import cv2
# import keras_ocr
# pipeline = keras_ocr.pipeline.Pipeline()
# import numpy as np
# import time
# import nltk
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from sklearn.metrics import jaccard_score
# from sklearn.preprocessing import MultiLabelBinarizer
# from torchmetrics.text import CharErrorRate
# import os

# def read_gt(filename):
#     try:
#         # Otwieranie pliku w trybie odczytu
#         with open(filename, 'r', encoding='utf-8') as file:
#             text = ''
#             # Iteracja przez każdą linię w pliku
#             for line in file:
#                 # Podział linii na słowa rozdzielone przecinkami
#                 words = line.strip().split(',')
#                 # Wypisanie ostatniego słowa
#                 if words[-2] == 'Latin' or words[-2] == 'Symbols':
#                     text = text + words[-1] +  ' '
#                 else:
#                     return "niedotyczy"
    
#     except FileNotFoundError:
#         print(f"Plik {filename} nie został znaleziony.")
#     return text


# def read_text_keras_ocr(image):
#     text = ''
#     # preprocessed_image = preprocess_image2(image)
#     # preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)
#     results = pipeline.recognize([image])
#     for result in results[0]:
#         text += result[0] + ' '
#     # print(text.strip())
#     return text.strip()


# def jaccard_similarity(groundtrouth, recognition):
#     groundtrouth = set(groundtrouth.lower().split())
#     recognition = set(recognition.lower().split())

#     # Combine both sets for fitting the MultiLabelBinarizer
#     combined_set = groundtrouth | recognition
#     mlb = MultiLabelBinarizer().fit([combined_set])
#     y_true = mlb.transform([groundtrouth])
#     y_pred = mlb.transform([recognition])
#     return jaccard_score(y_true[0], y_pred[0])


# #https://www.geeksforgeeks.org/how-to-calculate-jaccard-similarity-in-python/


# def calculate_cer(predicted, ground_truth):
#     cer = CharErrorRate()
#     return cer(predicted, ground_truth) 
#     # https://lightning.ai/docs/torchmetrics/stable/text/char_error_rate.html

# def wer(ground_trouth, predicted):

#     # Split the reference and hypothesis into words
#     ref_words = ground_trouth.split()
#     hyp_words = predicted.split()
    
#     substitutions = sum(1 for ref, hyp in zip(ref_words, hyp_words) if ref != hyp)
#     deletions = len(ref_words) - len(hyp_words)
#     insertions = len(hyp_words) - len(ref_words)
# 	# Total number of words in the reference text
#     total_words = len(ref_words)
# 	# Calculating the Word Error Rate (WER)
#     wer = (substitutions + deletions + insertions) / total_words
#     return wer
# # https://thepythoncode.com/article/calculate-word-error-rate-in-python

# def calculate_bleu(groundtrouth, recognition):

#     # Tokenize the reference and hypothesis texts
#     recognition = recognition.lower().split()
#     groundtrouth = groundtrouth.lower().split()
    
#     # Calculate BLEU score
#     smoothing_function = SmoothingFunction().method1
#     bleu_score = sentence_bleu([groundtrouth], recognition, smoothing_function=smoothing_function)
    
#     return bleu_score

# count = 1000
# start = time.time()
# levenshtein_keras_ocr = 0
# word_error_rate = 0
# bleu_score = 0
# jaccard = 0
# # for image_path_ in os.listdir('/content/data'):
# #   image_path = os.path.join('/content/data', image_path_)

# #   gt = image_path[:-4].replace('_', ' ').lower()

# for i in range (1000, 2000):
#     # gt = read_gt("train/tr_img_0{}.txt".format(i))
#     image_path = 'ImagesPart1/tr_img_0{}.jpg'.format(i)
#     image = cv2.imread(image_path)

#     if image is None:
#         print(f"Nie można załadować obrazu jpg z podanej ścieżki: {image_path}")
#         image_path = 'ImagesPart1/tr_img_0{}.png'.format(i)
#         image = cv2.imread(image_path)
#     if image is None:
#         print(f"Nie można załadować obrazu z podanej ścieżki: {image_path}")
    

#     if image is not None:
#         tex = read_text_keras_ocr(image)
#         # print(tex)
#         # print(gt.lower())
#         gt = gt.lower().replace('\n', '').replace('!', '').replace('?', '').replace('.', '')
#         jaccard += jaccard_similarity(gt, tex)
#         levenshtein_keras_ocr += calculate_cer(gt, tex)
#         word_error_rate += wer(gt, tex)
#         bleu_score += calculate_bleu(gt, tex)
#     else:
#         count -= 1


# print('jaccard_keras:', jaccard / count)
# print('levenshtein_keras_ocr:', levenshtein_keras_ocr / count)
# print('word_error_rate:', word_error_rate / count)
# print('bleu_score:', bleu_score / count)
# print('time:', time.time() - start)