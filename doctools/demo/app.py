from flask import Flask, render_template, jsonify
import doctools.parser.cluster as pc
from doctools.postproc.correction.params import cluster_params as params

app = Flask(__name__)

group_path = '/OCRData2/praveen-intermediate/group/'


@app.route('/<book>/')
def correct(book):
    return render_template('main.html')

@app.route('/<book>/graph')
def graph(book):
    feat = 'images'
    data, status = pc.load(book, feat=feat, **params[feat])
    return jsonify(data["components"])


if __name__ == '__main__':
    app.run('0.0.0.0', port=8080)

