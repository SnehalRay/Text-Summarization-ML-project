from flask import Flask, request, render_template, jsonify
from transformers import pipeline

app = Flask(__name__)

pipe = pipeline('summarization', model='./bart_samsum_ml_model')
argument = {'length_penalty': 0.8, 'num_beams': 8, "max_length": 512}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        # Get text from the form
        text_to_summarize = request.form['text']
        # Generate summary
        summary = pipe(text_to_summarize, **argument)[0]['summary_text']
        # Render the result page with the summary
        return render_template('result.html', summary=summary)
    except Exception as e:
        # If an error occurs, print it to the console and return a JSON with the error
        print(e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Run the Flask app
    app.run()
