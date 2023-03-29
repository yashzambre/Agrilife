import numpy as np
from PIL import Image
from feature_extraction import feature_extraction
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import os

app = Flask(__name__)

# Read image features
fe = feature_extraction()
base_path = '/home/grads/y/yashzambre/Desktop/YASH/plantpattern/'
img_path = base_path +'dataset/'
images = os.listdir(img_path)

feat_dir = base_path +'trained_features/'
train_feat = os.listdir(feat_dir)
features = []

for feat in train_feat:
    features.append(np.load(feat_dir + feat))

features = np.array(features)
(n,d) = features.shape


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['search_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path =  file.filename
        img.save(uploaded_img_path)

        # # Run search
        # query = fe.extract(img)
        # dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        # ids = np.argsort(dists)[:30]  # Top 30 results
        # scores = [(dists[id], img_paths[id]) for id in ids]
        
        test_feat = fe.extract(img)
        dists = np.zeros(n)

        dists = np.linalg.norm(features - test_feat, axis=1)

        idx = np.argsort(dists)[:30]
        scores = []
        msg1 = ''
        msg2 = ''
        if dists[idx[-1]] > 1:
            msg1 = "Ooops!! Looks like we do not have the animal in our dataset. But here are the animals that might resemble to the input based on extracted features !!"
        else:
            msg1 = "Here is most matched image and the top30 images"
            msg2 = 'http://127.0.0.1:8887/' + images[idx[-1]]
        for i in idx:
            img_dir = 'http://127.0.0.1:8887/' + images[i]
            scores.append((dists[i], img_dir))
        
        if msg2 == '':
            return render_template('index.html',
                                search_path='http://127.0.0.1:8887/'+uploaded_img_path,
                                scores=scores,
                                msg1 = msg1)
        else:
            return render_template('index.html',
                                search_path='http://127.0.0.1:8887/'+uploaded_img_path,
                                scores=scores,
                                msg1 = msg1,
                                msg2 = msg2)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("0.0.0.0", port=8080)