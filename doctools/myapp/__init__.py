from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///save.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(app)

class Books(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	name  = db.Column(db.String(20))
	language  = db.Column(db.String(1000))


@app.route('/books')

def books():
	queries = zip([q.name for q in Books.query.all()], [q.language for q in Books.query.all()])
	
	return render_template("books.html", queries=queries)

@app.route('/books/<book_id>')
def correct(book_id):
	return render_template("correction.html", book_id=book_id)
@app.route('/', methods=['GET'])
def index():
    return render_template('main.html', var = "Jinja testing")

if __name__ == "__main__":
	app.run(host='127.0.0.1', port=8080)