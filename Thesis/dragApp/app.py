from flask import Flask, render_template, send_from_directory, jsonify, request
import os
import base64
from werkzeug.serving import make_server

home = os.getcwd()

app = Flask(__name__)

# Set the directory where your images are stored
image_directory = home+'/Thesis/PipelineImages/5'

# Set the directory where you want to save the canvas images
save_directory = home+'/Thesis/PipelineImages/6'

@app.route('/')
def index():
    return render_template('index3.html')

@app.route('/get_image_urls')
def get_image_urls():
    # Get a list of image files in the specified directory
    image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    # Create a list of image URLs based on the directory path
    image_urls = [os.path.join('/image', img) for img in image_files]

    return jsonify(image_urls)

@app.route('/image/<path:filename>')
def image(filename):
    return send_from_directory(image_directory, filename)


@app.route('/get_image_positions')
def get_image_positions():
    # Define the path to your text file containing image positions
    text_file_path = home+'/Thesis/dragApp/image_positions.txt'  # Update with the actual path

    # Read the content of the text file and send it as a response
    with open(text_file_path, 'r') as file:
        data = file.read()

    return data

# Create a function to stop the server gracefully
def stop_server():
    try:
        os.kill(os.getpid(), 9)
    except Exception as e:
        print(f"Error stopping the server: {e}")

@app.route('/save_canvas', methods=['POST'])
def save_canvas():
    data = request.get_json()
    canvas_image_data = data['image']

    # Decode the base64 data and save the canvas image to the specified save directory
    with open(os.path.join(save_directory, 'canvas_image.jpg'), 'wb') as file:
        file.write(base64.b64decode(canvas_image_data.split(',')[1]))

    # Stop the server gracefully
    stop_server()

    return 'Canvas image saved successfully.'

if __name__ == '__main__':
    http_server = make_server('127.0.0.1', 5000, app)
    http_server.serve_forever()