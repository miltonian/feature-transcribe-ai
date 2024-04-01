from flask import Flask, render_template, request, jsonify
import feature_transcribe.prepare_code as pc
import feature_transcribe.feature_to_code as frc

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prepare_code', methods=['POST'])
def prepare_code():
    # Extract form data and call your Python function
    directory = request.form['directory']
    api_key = request.form['api_key']
    # Assuming you adapt prepare_code.py to be used as a module
    output_file = "embeddings_output.json"
    pc.main(directory, api_key, output_file)
    
    return jsonify({
        'response': 'Codebase Prepared Successfully'
    })

@app.route('/feature_to_code', methods=['POST'])
def feature_to_code():
    # Similar to prepare_code, extract data and call the script
    feature = request.form['feature']
    api_key = request.form['api_key']
    model = request.form['model']

    response = frc.main(feature, api_key, model)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
