from flask import Flask, request, jsonify
from flask_cors import CORS
from ragwhithsyllabus import run_mistral

app = Flask(__name__)
CORS(app)  # Active CORS pour toutes les routes

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try: 
        response = run_mistral(question)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)