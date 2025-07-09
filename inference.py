import os
import flask

from model import Classifier
from cd_model import Resnet50Model
import torch
from torchvision import transforms as v2
import PIL
from torchvision import tv_tensors

upload_path = './uploaded'
if not os.path.exists(upload_path):
    os.makedirs(upload_path)
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = flask.Flask(__name__)
device = torch.device('cpu')

app.config['UPLOAD_FOLDER'] = upload_path

"""
Render the initial page using index.html. Defaults to GET
Two decorators are used;  so both '/' and
'/home' will refer to the function home()
"""
@app.route('/')
@app.route('/home')
def home():
    return flask.render_template('home.html')


def preprocess(im_path):
    image = tv_tensors.Image(PIL.Image.open(im_path).convert('RGB'))
    transform_base = v2.Compose([v2.Resize((256, 256)),
                                v2.CenterCrop((256, 256)),
                                v2.ConvertImageDtype(torch.float32),
                                v2.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = torch.tensor(image, device=device)
    image = image[None, :]
    image = transform_base(image)
    return image

"""
Run the prediction by doing a forward pass
"""
def inference(image):
    model = Resnet50Model().to(device=device)
    model.load_state_dict(torch.load('./best_model.pt', weights_only=True, map_location=device))
    model.eval()
    
    logits, probas = model(image)
    _, predicted_label = torch.max(probas, 1)
    return predicted_label

"""
Check if the file extension is valid, splitting only
on one '.' character nearest to the right.
"""
def check_extensions(f_name):
    if '.' in f_name:
        extension = f_name.rsplit('.', 1)[1].lower()
        return extension in ALLOWED_EXTENSIONS
    return False

"""
Upon hitting the submit button on index.html, we should route to 
the results function to display the results of the prediction.
If there is some invalid submission, flash a message and return to 
the requesting page
"""
@app.route('/result', methods=['POST'])
def result():
    if flask.request.method == 'POST':
        to_predict_list = flask.request.form.to_dict()
        if 'file' not in flask.request.files:
            flask.flash('No file found')
            return flask.redirect(flask.request.url)
        file = flask.request.files['file']
        if not file.filename:
            flask.flash('No file uploaded')
            return flask.redirect(flask.request.url)
        if file and check_extensions(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            image = preprocess(file_path)
            pred = 'CAT' if inference(image=image) else 'DOG'
            return flask.render_template('result.html', prediction=pred)
            
            
            
            
    

