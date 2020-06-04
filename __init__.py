# python standard lib
import operator
import os
import pickle

# main lib
import librosa
# supporting libs
import numpy as np
import python_speech_features
from flask import Flask, request, jsonify
from hmmlearn.hmm import GaussianHMM
from pydub import AudioSegment

app = Flask(__name__)


def count_mfcc(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    feature = python_speech_features.mfcc(audio, sr)
    return feature


training_files = '/Users/Agung Kurniawan/Desktop/rame_rev_1'
pickle_1 = '/Users/Agung Kurniawan/Desktop/models/models_32.p'
pickle_2 = '/Users/Agung Kurniawan/Desktop/models/models_41.p'


@app.route('/train')
def train():
    models = []

    for label in [x for x in os.listdir(training_files) if
                  os.path.isdir(os.path.join(training_files, x))]:
        features = None

        print('calculating', label)

        for filename in os.listdir(os.path.join(training_files, label)):
            filepath = os.path.join(training_files, label, filename)

            feature = count_mfcc(filepath)

            if features is None:
                features = feature
            else:
                features = np.append(features, feature, axis=0)

        print(len(os.listdir(os.path.join(training_files, label))))

        model = GaussianHMM(n_components=3, n_iter=1000)
        model.fit(features)
        models.append((model, label))
        model = None

    # save training model
    pickle.dump(models, open(pickle_1, 'wb'))

    print('done training')
    return 'done training'


@app.route('/', methods=['GET', 'POST'])
def test_file():
    file = request.files['audio']
    expected_label = request.args.get('label', '')
    email = request.args.get('email', '')

    wav_file = convert_save_wav(expected_label, email, file)
    feature = count_mfcc(wav_file)
    prediction = predict(feature, expected_label)

    return prediction


test_files = "/Users/Agung Kurniawan/Desktop/test"


def convert_save_wav(expected_label, email, file_aac):
    print('converting to wav')
    input_file = AudioSegment.from_file(file_aac, format='aac')

    # if user never tests, create a folder for them
    if not os.path.exists(os.path.join(test_files, email)):
        os.mkdir(os.path.join(test_files, email))

    # then save it
    output_filename = os.path.join(test_files, email, expected_label + '.wav')
    input_file.export(output_filename, format='wav')

    print('converted to wav')
    return output_filename


def predict(feature, expected_label, current_pickle=pickle_1):
    print('predicting')
    print('running model in', current_pickle)

    models = pickle.load(open(current_pickle, "rb"))

    scores = {}

    for model, model_label in models:
        score = model.score(feature)
        scores[model_label] = score

    predicted_label = max(scores.items(), key=operator.itemgetter(1))[0]
    max_score = max(scores.items(), key=operator.itemgetter(1))[1]

    print('for', expected_label, 'predicted:', predicted_label)

    # if the prediction is wrong, try to use another pickle train file and predict again
    if (predicted_label != expected_label) and (current_pickle == pickle_1):
        return predict(feature, expected_label, pickle_2)
    else:
        print('returning the result...')

        return jsonify(
            result=(predicted_label == expected_label),
            label=predicted_label,
            score=max_score
        )


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='0.0.0.0', port=8080, debug=True)
    # [END gae_python37_app]
