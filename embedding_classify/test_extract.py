from embedding_classify.extract_features import get_middle_layer_features

vec = get_middle_layer_features("Recent advances in machine learning have shown that large language models can "
                                "outperform traditional algorithms on complex reasoning tasks.")
print("Размерность вектора:", vec.shape)
print("Первые 5 значений:", vec[:5])
