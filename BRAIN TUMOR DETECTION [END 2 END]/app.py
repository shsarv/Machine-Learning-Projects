import flask
import pickle
from io import BytesIO
from torch import argmax, load
from torch import device as DEVICE
from torch.cuda import is_available
from torch.nn import Sequential, Linear, SELU, Dropout, LogSigmoid
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.models import resnet50
import os
UPLOAD_FOLDER = os.path.join('static', 'photos')
app = flask.Flask(__name__, template_folder='templates')
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

LABELS = ['None', 'Meningioma', 'Glioma', 'Pitutary']

device = "cuda" if is_available() else "cpu"

resnet_model = resnet50(pretrained=True)

for param in resnet_model.parameters():
    param.requires_grad = True

n_inputs = resnet_model.fc.in_features
resnet_model.fc = Sequential(Linear(n_inputs, 2048),
                            SELU(),
                            Dropout(p=0.4),
                            Linear(2048, 2048),
                            SELU(),
                            Dropout(p=0.4),
                            Linear(2048, 4),
                            LogSigmoid())

for name, child in resnet_model.named_children():
    for name2, params in child.named_parameters():
        params.requires_grad = True

resnet_model.to(device)
resnet_model.load_state_dict(load('./models/bt_resnet50_model.pt', map_location=DEVICE(device)))
resnet_model.eval()

def preprocess_image(image_bytes):
  transform = Compose([Resize((512, 512)), ToTensor()])
  img = Image.open(BytesIO(image_bytes))
  return transform(img).unsqueeze(0)

def get_prediction(image_bytes):
  tensor = preprocess_image(image_bytes=image_bytes)
  y_hat = resnet_model(tensor.to(device))
  class_id = argmax(y_hat.data, dim=1)
  return str(int(class_id)), LABELS[int(class_id)]

@app.route('/', methods=['GET'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('DiseaseDet.html'))     

@app.route("/uimg",methods=['GET','POST'])
def uimg():
    if flask.request.method == 'GET':
        return(flask.render_template('uimg.html'))     
    if flask.request.method == 'POST':
        file = flask.request.files['file']
        img_bytes = file.read()   
        class_id, class_name = get_prediction(img_bytes)
        return(flask.render_template('pred.html',result = class_name,file = file))   
      
@app.errorhandler(500)
def server_error(error):
    return render_template('error.html'), 500

if __name__ == '__main__':
   	app.run(debug=True)