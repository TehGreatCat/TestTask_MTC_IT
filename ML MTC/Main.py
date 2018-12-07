from sklearn import model_selection, metrics, preprocessing, ensemble
from sklearn.feature_extraction.text import TfidfVectorizer
import stop_words
from http.client import HTTPMessage
import pandas as pd, xgboost as xg, numpy, textblob, string, matplotlib.pyplot as plt

import os, pprint, itertools


def open_data_set(dataset_path):
    try:
        dataset = [open(os.path.join(dataset_path, filename), 'r').read()
                   for filename in os.listdir(dataset_path)
                   if filename.endswith(".txt")]
        return dataset
    except Exception as e:
        print(e)
    raise SystemExit


def remove_noise(input_text):
    # очистка текста от чисел и знаков препинания
    table = str.maketrans("", "", string.punctuation + "1234567890")
    punct_free_text = input_text.translate(table)
    # очистка текста от стоп-слов
    noise_list = stop_words.get_stop_words("en") + ["et", "al", "eg", "le"]
    words = punct_free_text.split()
    noise_free_words = [textblob.Word(word).lemmatize() for word in words if word not in noise_list]
    noise_free_text = " ".join(noise_free_words)
    return noise_free_text.strip()


def clear_tags(dataset, tag):       # remove ""abstract" and "introduction"
    if dataset:
        output = []
        for text_ in dataset:
            while tag in text_:
                ind1 = text_.find(tag)
                # print("ind1 = " + str(ind1))
                ind2 = text_[ind1 + len(tag):].find(tag) + len(tag) * 2 + ind1
                # print("ind2 = " + str(ind2))
                text_ = text_[:ind1] + text_[ind2 + 1:]
            output.append(text_.replace("\t", ' ').
                          replace("--", ' ').
                          replace("  ", ' '))
        return output       # list of texts ['text1', 'text2'...]
    else:
        print("No dataset")


def generate_dataset(dataset):
    if dataset:
        digested_dataset = []
        for text in dataset:
            for line in text.split('\n'):
                pair = [line[:line.find(' ')], remove_noise(line[line.find(' ') + 1:].lower())] # [label: digested_text]
                digested_dataset.append(pair)
        return digested_dataset


def train_classifier(classifier, train_feature_vect, train_lbl, test_feature_vect):
    classifier.fit(train_feature_vect, train_lbl)
    prediction = classifier.predict(test_feature_vect)
    return prediction


def cutLabel(data, label, n):
    output = []
    for x in data:
        if x[0] == label and n != 0:
            output.append(x)
            n -= 1
        elif x[0] != label:
            output.append(x)
    return output


def plot_ROC(classes, test_lbl, prediciton):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = metrics.roc_curve(numpy.array(pd.get_dummies(test_lbl))[:, i],
                                              numpy.array(pd.get_dummies(prediciton))[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    plt.plot([0, 1], [0, 1], 'k--', color='red', lw=3)
    colors = ['darkblue', 'orange', 'darkgreen', 'red', 'cyan']
    for i, color in zip(range(len(classes)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=3, label='{0}; Area = {1:0.2f})'.format(classes[i], roc_auc[i]))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False positive")
    plt.ylabel("True Positive")
    plt.legend(loc="lower center")
    plt.show()


def plotConfMatrix(test_lbl, result, classes):
    conf_mtrx = metrics.confusion_matrix(test_lbl, result)
    plt.figure()
    plt.imshow(conf_mtrx, interpolation='nearest', cmap="Reds")
    for i, j in itertools.product(range(conf_mtrx.shape[0]), range(conf_mtrx.shape[1])):
        plt.text(j, i, format(conf_mtrx[i, j], 'd'),
                 horizontalalignment="center")
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label',)
    plt.tight_layout()
    plt.show()

# def RNN(unique_words, len_of_sentence):
#     inputs = Input(name='inputs', shape=[len_of_sentence])  # input_shape = len(seq_mtrx[i])
#     layer = Embedding(input_dim=unique_words, output_dim=7, input_length=len_of_sentence)(inputs)
#     layer = LSTM(128)(layer)
#     layer = Dense(50, activation="relu")(layer)
#     layer = Dropout(0.25)(layer)
#     layer = Dense(1, name='out_layer')(layer)
#     layer = Activation('sigmoid')(layer)
#     model = models.Model(inputs=inputs, outputs=layer)
#     return model
#
#
# def test_NN(unique_words):
#     global train_txt, train_lbl, test_txt, test_lbl, TrainDF
#
#     encoder = preprocessing.LabelEncoder()
#     encoder.fit(numpy.unique(TrainDF['label']))
#     test_lbls = encoder.transform(test_lbl)
#     train_lbls = encoder.transform(train_lbl)
#
#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts(train_txt)
#     seq = tokenizer.texts_to_sequences(train_txt)
#     seq_mtrx = sequence.pad_sequences(seq)
#
#     print("seq_mtrx: ", len(seq_mtrx[0]))
#     model = RNN(unique_words, len(seq_mtrx[0]))
#     model.summary()
#     model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=0.0001), metrics=['accuracy'])
#     model.fit(seq_mtrx, train_lbls, batch_size=128, epochs=10, validation_split=0.25)
#     test_seq = tokenizer.texts_to_sequences(test_txt)
#     test_seq_mtrx = sequence.pad_sequences(test_seq)
#     accuracy = model.evaluate(test_seq_mtrx, test_lbls)
#     print(accuracy)


if __name__ == '__main__':
    # загрузка и предобработка датасета
    data = generate_dataset(clear_tags(open_data_set("SentenceCorpus\labeled_articles"), "###"))
    # обрезка кол-ва предложений с тэгом "MISC"
    data = cutLabel(data, "MISC", 900)
    # создание датафрейма
    TrainDF = pd.DataFrame()
    TrainDF['label'] = [element[0] for element in data]
    TrainDF['text'] = [element[1] for element in data]
    # удаление пустых строк
    TrainDF = TrainDF[TrainDF.label != '']
    # кол-во элементов в выборке
    print("elems in DF: ", len(TrainDF['text']))
    # labels и их частота
    print("dataframe labels: ", numpy.unique(TrainDF['label'], return_counts=True))
    # обрезка самых частых и самых редких слов
    freq_list = pd.Series(" ".join(TrainDF['text']).split()).value_counts()
    most_freq = list(freq_list[:2].index)  # 2
    least_freq = []
    # least_freq = list(freq_list[-sum(numpy.unique(freq_list, return_counts=True)[1][:3]):].index)  # 3 самых редких
    TrainDF['text'] = TrainDF['text'].apply(lambda x: " ".join(x for x in x.split() if x not in most_freq + least_freq))

    for i in range(1):
        # разделение на обучающие и тестовые выборки
        train_txt, test_txt, train_lbl, test_lbl = model_selection.train_test_split(TrainDF["text"], TrainDF["label"],
                                                                                    test_size=0.25)

        # кол-во элементов каждого класса в обучающей выборке
        print("train_labels: ", numpy.unique(train_lbl, return_counts=True))
        # print("test_labels: ", numpy.unique(test_lbl, return_counts=True))

        # tf-idf векторизация
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=512)
        tfidf_vect.fit(TrainDF['text'])
        xtrain_tfidf = tfidf_vect.transform(train_txt)
        xtest_tfidf = tfidf_vect.transform(test_txt)
        print("num of features: ", len(tfidf_vect.get_feature_names()))

        # обучение и классификация тестовой выборки
        result = train_classifier(xg.XGBClassifier(max_depth=15, learning_rate=0.1, n_estimators=128),
                                  xtrain_tfidf.tocsc(), train_lbl, xtest_tfidf.tocsc())
        # метрика
        print("XG accuracy :", round(metrics.accuracy_score(test_lbl, result), 3))
        print("XG Cohen's Kappa :", round(metrics.cohen_kappa_score(test_lbl, result), 3))
        print("------------------------------------")

        result2 = train_classifier(ensemble.RandomForestClassifier(n_estimators=128),
                                  xtrain_tfidf, train_lbl, xtest_tfidf)
        print("RFC accuracy :", round(metrics.accuracy_score(test_lbl, result2), 3))
        print("RFC Cohen's Kappa :", round(metrics.cohen_kappa_score(test_lbl, result2), 3))
        print("------------------------------------")

        plot_ROC(numpy.unique(TrainDF['label']), test_lbl, result)
        plotConfMatrix(test_lbl, result, numpy.unique(TrainDF['label']))
        plot_ROC(numpy.unique(TrainDF['label']), test_lbl, result2)
        plotConfMatrix(test_lbl, result2, numpy.unique(TrainDF['label']))
