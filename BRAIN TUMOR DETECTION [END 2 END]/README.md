# how to run this app.
- clone the repo
- move to folder BRAIN TUMOR DETECTION,
- create a virtual env using `virtualenv 'envirnoment name'.`
- Now type `. \Scripts\activate` to activate your virtualenv venv
- install packages required by the app by running the command `pip install -r requirments.txt` if you found error of torch or torch-vision libraries so your can download it from below commands.
<b><i>for windows cpu torch library command with conda</i></b>: conda install pytorch torchvision cpuonly -c pytorch 
<b><i>for windows cpu torch library command with pip</i></b>: pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

For Linux, Mac,CUDA version for windows and much more you can visit <a href = "https://pytorch.org/">Link</a>
- now run the app using `flask run`.
- if all things works fine so you can view running web application in your browser at port 5000.


# Download the model file from [https://drive.google.com/file/d/1LJG_ITCWWtriLC5NPrWxIDwekWbhU_Rj/view?usp=sharing](https://drive.google.com/file/d/1LJG_ITCWWtriLC5NPrWxIDwekWbhU_Rj/view?usp=sharing) and add to model directory in order to run the project.
