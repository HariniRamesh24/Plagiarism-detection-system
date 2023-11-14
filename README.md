# Plagiarism-detection-system
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import re

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file_1 = request.files['firstfile']
        file_2 = request.files['secondfile']
        categories = ['file1', 'file2']
        # Read training data
        sample_contents=[]
        for file in [file_1, file_2]:
            text = file.read().decode('utf-8')
            # Clean the text by removing non-Tamil characters and symbols
            cleaned_text = re.sub(r'[^\u0B80-\u0BFF\n]', ' ', text)
            sample_contents.append(cleaned_text)

        # Tokenize and tag training data
        def tokenize(text, stopwords, max_len=20):
            return [token for token in gensim.utils.simple_preprocess(text, max_len=max_len) if token not in stopwords]

        cat_dict_tagged_train = {} # Contains clean tagged training data organized by category. To be used for the training corpus.
        offset = 0 # Used for managing IDs of tagged documents
        for k, v in zip(categories, sample_contents):
            cat_dict_tagged_train[k] = [gensim.models.doc2vec.TaggedDocument(tokenize(text, [], max_len=200), [i+offset]) for i, text in enumerate(v.split('\n'))]
            offset += len(v.split('\n'))

        # Eventually contains final versions of the training data to actually train the model
        train_corpus = [taggeddoc for taggeddoc_list in list(cat_dict_tagged_train.values()) for taggeddoc in taggeddoc_list]

        # Train the model
        model = gensim.models.doc2vec.Doc2Vec(vector_size=30, min_count=2, epochs=40, window=2)
        model.build_vocab(train_corpus)
        print(model)
        model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

        # Read test data
        metadata = {}
        inferred_vectors_test = {} # Contains, category-wise, inferred doc vecs for each document in the test set
        for cat, text in zip(categories, sample_contents):
            cleaned_text = re.sub(r'[^\u0B80-\u0BFF\n]', ' ', text)
            inferred_vectors_test[cat] = [model.infer_vector(tokenize(doc, [], max_len=200)) for doc in cleaned_text.split('\n')]
            metadata[cat] = len(inferred_vectors_test[cat])
        vectors = inferred_vectors_test[categories[0]] + inferred_vectors_test[categories[1]]

        def similarity(doc1, doc2):
            return cosine_similarity([doc1, doc2])

        s_vectors = list(zip(categories*metadata[categories[0]], vectors))

        # Check for plagiarism
        plagiarism_results = set()
        def check_plagiarism():
            for sample_a, text_a in s_vectors:
                new_vec = s_vectors.copy()
                curr_i = new_vec.index((sample_a, text_a))
                del new_vec[curr_i]
                for sample_b, text_b in new_vec:
                    sim_score = similarity(text_a, text_b)[0][1]
                    samp_pair = sorted((sample_a, sample_b))
                    score = (samp_pair[0], samp_pair[1], sim_score)
                    plagiarism_results.add(score)
            return plagiarism_results

        for data in check_plagiarism():
            print(data)

        return render_template('results.html', results=plagiarism_results)
    return render_template('home.html')



if __name__ == '__main__':
    app.run(debug=True)
