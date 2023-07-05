from flask import Flask, request, redirect, url_for, render_template
import os
from torchvision import models
import wikipedia
from torchvision import transforms
from PIL import Image
import torch

app = Flask(__name__)

if not os.path.exists('uploads'):
    os.makedirs('uploads')

UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('results', filename = filename, x = imageProcess(filename)))
    return render_template('index.html')

def imageProcess(filename):

    resnet = models.resnet101(pretrained=True)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
    img = Image.open('C:\\Users\\Lenovo\\Desktop\\Py Projects\\Idp1\\static\\uploads\\'+filename)
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    resnet.eval()
    out = resnet(batch_t)
    with open('C:\\Users\\Lenovo\\Downloads\\imagenet_classes.txt') as f:
        labels = [line.strip() for line in f.readlines()]
        _, index = torch.max(out, 1)

    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    labels[index[0]], percentage[index[0]].item()
    st = [x for x in labels[index[0]].split(',')]
    print(st)

    try:
        result = wikipedia.summary(st[1], sentences = 5)
    except:
        result = wikipedia.summary(st[1]+' mammal', sentences = 5)
    
    return st[1], result, percentage[index[0]].item()

@app.route('/results/<filename>')
def results(filename):
    return render_template('results.html', filename=filename, x = imageProcess(filename))

if __name__ == '__main__':
    app.run(debug=True)
