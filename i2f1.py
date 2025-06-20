from flask import Flask, request, render_template_string, redirect, url_for, jsonify
import google.generativeai as genai
import os
import base64
from PIL import Image
import io
import json
import uuid
from datetime import datetime

app = Flask(__name__)

# Configure Gemini AI
genai.configure(api_key="AIzaSyC3kaKAFozoCl7oELP9CDiuzKQwsFbQDpY")
model = genai.GenerativeModel('gemini-1.5-flash')

# Store forms in memory (in production, use a database)
forms_storage = {}
submissions_storage = {}

# HTML Templates
UPLOAD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Image to Form Converter</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .upload-area {
            border: 2px dashed #ddd;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
            background-color: #fafafa;
        }
        .upload-area:hover {
            border-color: #007bff;
            background-color: #f0f8ff;
        }
        input[type="file"] {
            margin: 20px 0;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #0056b3;
        }
        .preview {
            margin: 20px 0;
            text-align: center;
        }
        .preview img {
            max-width: 100%;
            max-height: 300px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .forms-list {
            margin-top: 40px;
        }
        .form-item {
            background: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }
        .form-item a {
            color: #007bff;
            text-decoration: none;
            font-weight: bold;
        }
        .form-item a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📄 Image to Form Converter</h1>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area">
                <h3>Upload an Image</h3>
                <p>Select an image containing a form (bank details, application form, etc.)</p>
                <input type="file" id="imageFile" name="image" accept="image/*" required>
                <div class="preview" id="preview"></div>
            </div>
            <button type="submit">Convert to Form</button>
        </form>
        
        <div class="loading" id="loading">
            <p>🤖 AI is analyzing your image and creating the form...</p>
        </div>
        
        <div class="forms-list">
            <h3>Created Forms</h3>
            {% for form_id, form_data in forms.items() %}
            <div class="form-item">
                <a href="/form/{{ form_id }}">{{ form_data.title }}</a>
                <small style="color: #666; margin-left: 10px;">Created: {{ form_data.created_at }}</small>
                <a href="/submissions/{{ form_id }}" style="float: right; font-size: 12px;">View Submissions</a>
            </div>
            {% endfor %}
        </div>
    </div>

    <script>
        document.getElementById('imageFile').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('preview').innerHTML = 
                        '<img src="' + e.target.result + '" alt="Preview">';
                };
                reader.readAsDataURL(file);
            }
        });

        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData();
            const fileInput = document.getElementById('imageFile');
            formData.append('image', fileInput.files[0]);
            
            document.getElementById('loading').style.display = 'block';
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                if (result.success) {
                    window.location.href = '/form/' + result.form_id;
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                alert('Error uploading image: ' + error.message);
            }
            
            document.getElementById('loading').style.display = 'none';
        });
    </script>
</body>
</html>
"""

FORM_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ form_data.title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
        }
        .form-description {
            color: #666;
            margin-bottom: 30px;
            font-style: italic;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #333;
        }
        input[type="text"], input[type="email"], input[type="tel"], input[type="date"], 
        input[type="number"], textarea, select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
            box-sizing: border-box;
        }
        textarea {
            height: 80px;
            resize: vertical;
        }
        input[type="checkbox"], input[type="radio"] {
            margin-right: 8px;
        }
        .checkbox-group, .radio-group {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }
        .checkbox-item, .radio-item {
            display: flex;
            align-items: center;
        }
        button {
            background-color: #28a745;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin-right: 10px;
        }
        button:hover {
            background-color: #218838;
        }
        .back-btn {
            background-color: #6c757d;
        }
        .back-btn:hover {
            background-color: #545b62;
        }
        .share-section {
            background: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .share-url {
            background: white;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 3px;
            word-break: break-all;
            margin: 10px 0;
        }
        .required {
            color: red;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ form_data.title }}</h1>
        <p class="form-description">{{ form_data.description }}</p>
        
        <form id="dynamicForm" method="POST" action="/submit/{{ form_id }}">
            {% for field in form_data.fields %}
            <div class="form-group">
                <label for="{{ field.name }}">
                    {{ field.label }}
                    {% if field.required %}<span class="required">*</span>{% endif %}
                </label>
                
                {% if field.type == 'text' or field.type == 'email' or field.type == 'tel' or field.type == 'date' or field.type == 'number' %}
                    <input type="{{ field.type }}" 
                           id="{{ field.name }}" 
                           name="{{ field.name }}" 
                           {% if field.placeholder %}placeholder="{{ field.placeholder }}"{% endif %}
                           {% if field.required %}required{% endif %}>
                
                {% elif field.type == 'textarea' %}
                    <textarea id="{{ field.name }}" 
                              name="{{ field.name }}" 
                              {% if field.placeholder %}placeholder="{{ field.placeholder }}"{% endif %}
                              {% if field.required %}required{% endif %}></textarea>
                
                {% elif field.type == 'select' %}
                    <select id="{{ field.name }}" name="{{ field.name }}" {% if field.required %}required{% endif %}>
                        <option value="">-- Select an option --</option>
                        {% for option in field.options %}
                        <option value="{{ option }}">{{ option }}</option>
                        {% endfor %}
                    </select>
                
                {% elif field.type == 'checkbox' %}
                    <div class="checkbox-group">
                        {% for option in field.options %}
                        <div class="checkbox-item">
                            <input type="checkbox" id="{{ field.name }}_{{ loop.index }}" 
                                   name="{{ field.name }}" value="{{ option }}">
                            <label for="{{ field.name }}_{{ loop.index }}">{{ option }}</label>
                        </div>
                        {% endfor %}
                    </div>
                
                {% elif field.type == 'radio' %}
                    <div class="radio-group">
                        {% for option in field.options %}
                        <div class="radio-item">
                            <input type="radio" id="{{ field.name }}_{{ loop.index }}" 
                                   name="{{ field.name }}" value="{{ option }}" 
                                   {% if field.required %}required{% endif %}>
                            <label for="{{ field.name }}_{{ loop.index }}">{{ option }}</label>
                        </div>
                        {% endfor %}
                    </div>
                {% endif %}
            </div>
            {% endfor %}
            
            <button type="submit">Submit Form</button>
            <button type="button" class="back-btn" onclick="window.location.href='/'">Back to Home</button>
        </form>
        
        <div class="share-section">
            <h3>📤 Share this form</h3>
            <p>Copy this URL to share the form with others:</p>
            <div class="share-url" id="shareUrl">{{ request.url }}</div>
            <button type="button" onclick="copyUrl()">📋 Copy URL</button>
        </div>
    </div>

    <script>
        function copyUrl() {
            const urlElement = document.getElementById('shareUrl');
            const range = document.createRange();
            range.selectNode(urlElement);
            window.getSelection().removeAllRanges();
            window.getSelection().addRange(range);
            document.execCommand('copy');
            window.getSelection().removeAllRanges();
            alert('URL copied to clipboard!');
        }
        
        document.getElementById('dynamicForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            
            fetch(this.action, {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(result => {
                if (result.success) {
                    alert('Form submitted successfully!');
                    this.reset();
                } else {
                    alert('Error: ' + result.error);
                }
            })
            .catch(error => {
                alert('Error submitting form: ' + error.message);
            });
        });
    </script>
</body>
</html>
"""

SUBMISSIONS_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Form Submissions - {{ form_data.title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            margin-bottom: 20px;
        }
        .submission {
            background: #f8f9fa;
            padding: 20px;
            margin: 15px 0;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }
        .submission-header {
            font-weight: bold;
            color: #007bff;
            margin-bottom: 10px;
        }
        .field-value {
            margin: 8px 0;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }
        .field-label {
            font-weight: bold;
            color: #333;
        }
        .no-submissions {
            text-align: center;
            color: #666;
            font-style: italic;
            padding: 40px;
        }
        button {
            background-color: #6c757d;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin-bottom: 20px;
        }
        button:hover {
            background-color: #545b62;
        }
    </style>
</head>
<body>
    <div class="container">
        <button onclick="window.location.href='/'">← Back to Home</button>
        <h1>📊 Submissions for "{{ form_data.title }}"</h1>
        
        {% if submissions %}
            {% for submission_id, submission in submissions.items() %}
            <div class="submission">
                <div class="submission-header">
                    Submission #{{ loop.index }} - {{ submission.submitted_at }}
                </div>
                {% for field_name, value in submission.data.items() %}
                <div class="field-value">
                    <span class="field-label">{{ field_name }}:</span>
                    {% if value is iterable and value is not string %}
                        {{ value|join(', ') }}
                    {% else %}
                        {{ value }}
                    {% endif %}
                </div>
                {% endfor %}
            </div>
            {% endfor %}
        {% else %}
            <div class="no-submissions">
                No submissions yet. Share the form to start collecting responses!
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

def analyze_image_with_gemini(image_data):
    """Use Gemini AI to analyze the image and extract form fields"""
    try:
        # Convert image data to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        prompt = """
        Analyze this image which appears to be a form or document with fillable fields.
        Extract all the form fields you can identify and return them as a JSON object with the following structure:
        
        {
            "title": "A suitable title for this form",
            "description": "Brief description of what this form is for",
            "fields": [
                {
                    "name": "field_name_no_spaces",
                    "label": "Human readable field label",
                    "type": "text|email|tel|date|number|textarea|select|checkbox|radio",
                    "required": true/false,
                    "placeholder": "placeholder text if applicable",
                    "options": ["option1", "option2"] // only for select, checkbox, radio fields
                }
            ]
        }
        
        Guidelines:
        - Look for text labels, input boxes, checkboxes, signature areas, date fields, etc.
        - Infer appropriate field types based on the labels (e.g., "Email" should be type "email", "Phone" should be "tel")
        - For signature areas, use type "text" with placeholder "Your signature"
        - For address fields, you can break them into separate fields or use textarea
        - Make field names lowercase with underscores instead of spaces
        - Set required=true for fields that appear mandatory or important
        - For dropdown-like fields or multiple choice options, use "select", "checkbox", or "radio" as appropriate
        
        Return ONLY the JSON object, no other text.
        """
        
        response = model.generate_content([prompt, image])
        
        # Clean the response to extract JSON
        response_text = response.text.strip()
        
        # Remove any markdown code blocks
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        # Parse JSON
        form_data = json.loads(response_text.strip())
        return form_data
        
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        # Return a default form structure if AI analysis fails
        return {
            "title": "Extracted Form",
            "description": "Form fields extracted from uploaded image",
            "fields": [
                {
                    "name": "name",
                    "label": "Full Name",
                    "type": "text",
                    "required": True,
                    "placeholder": "Enter your full name"
                },
                {
                    "name": "email",
                    "label": "Email Address", 
                    "type": "email",
                    "required": True,
                    "placeholder": "Enter your email"
                }
            ]
        }

@app.route('/')
def index():
    return render_template_string(UPLOAD_TEMPLATE, forms=forms_storage)

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'})
        
        # Read image data
        image_data = file.read()
        
        # Analyze image with Gemini AI
        form_data = analyze_image_with_gemini(image_data)
        
        # Generate unique form ID
        form_id = str(uuid.uuid4())[:8]
        
        # Store form data
        forms_storage[form_id] = {
            'title': form_data['title'],
            'description': form_data['description'],
            'fields': form_data['fields'],
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Initialize submissions storage for this form
        submissions_storage[form_id] = {}
        
        return jsonify({'success': True, 'form_id': form_id})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/form/<form_id>')
def show_form(form_id):
    if form_id not in forms_storage:
        return "Form not found", 404
    
    form_data = forms_storage[form_id]
    return render_template_string(FORM_TEMPLATE, form_id=form_id, form_data=form_data, request=request)

@app.route('/submit/<form_id>', methods=['POST'])
def submit_form(form_id):
    try:
        if form_id not in forms_storage:
            return jsonify({'success': False, 'error': 'Form not found'})
        
        # Collect form data
        submission_data = {}
        for key, value in request.form.items():
            if key in request.form:
                # Handle multiple values (checkboxes)
                values = request.form.getlist(key)
                if len(values) > 1:
                    submission_data[key] = values
                else:
                    submission_data[key] = value
        
        # Generate submission ID
        submission_id = str(uuid.uuid4())[:8]
        
        # Store submission
        submissions_storage[form_id][submission_id] = {
            'data': submission_data,
            'submitted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify({'success': True, 'message': 'Form submitted successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/submissions/<form_id>')
def view_submissions(form_id):
    if form_id not in forms_storage:
        return "Form not found", 404
    
    form_data = forms_storage[form_id]
    submissions = submissions_storage.get(form_id, {})
    
    return render_template_string(SUBMISSIONS_TEMPLATE, 
                                form_data=form_data, 
                                submissions=submissions)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
