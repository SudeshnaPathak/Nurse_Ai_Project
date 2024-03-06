from vectorisation.vectorisation import vectorise , get_code1 , get_code2
from distutils.log import debug 
from fileinput import filename 
from flask import *
app = Flask(__name__) 

@app.route('/index') 
def main(): 
	return render_template("index.html") 

@app.route('/success', methods = ['POST']) 
def success(): 
	if request.method == 'POST': 
		f = request.files['file'] 
		f.save(f.filename)
		
	

@app.route('/') 
def check(): 
	return render_template("test.html" , name = vectorise("data/upload_files/", "medical_test")) 

@app.route('/submit', methods=['POST'])
def hello():
    ip = request.form.get("info")
    # Do something with the selected value
    return get_code1(ip)

@app.route('/info')
def index():
    return render_template("info.html")

@app.route('/time')
def time():
    return render_template("time.html")

@app.route('/select', methods=['POST'])
def home():
    op = request.form.get("time")
    # Do something with the selected value
    return get_code2(op)


if __name__ == '__main__': 
	app.run(debug=True)