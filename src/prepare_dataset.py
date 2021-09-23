import cv2
import os
import numpy as np
import collections
import matplotlib.pyplot as plt

classes = ["angry","disgust","fear","happy","neutral","sad","surprise"]
data_dir = "../images/train/"


def load_data_from_folder(data_dir):
    X = []
    y = []
    for c in classes:
        path = os.path.join(data_dir, c)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            X.append(img_array)
            y.append(c)
            
    X = np.array(X)
    y = np.array(y)
            
    return X, y
    
  
X, y = load_data_from_folder(data_dir)

plot_dict = collections.Counter(y)   

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(classes, [plot_dict[key] for key in plot_dict])
plt.xlabel("Classes")
plt.ylabel("Frequency")
plt.show()

from imblearn.combine import SMOTETomek
smk = SMOTETomek()
X = X.reshape(-1, 48*48*3)
X_res, y_res = smk.fit_resample(X,y)

print("y_res:", collections.Counter(y_res))

output_dir = "../images/train_alpha/"
inner_names = classes

os.makedirs(output_dir, exist_ok=True)
for inner_name in inner_names:
    os.makedirs(os.path.join(output_dir,inner_name), exist_ok=True)
    

X_res = X_res.reshape(-1,48,48, 3)

for i in range(len(inner_names)):
    for j in range(len(y_res)):
        if inner_names[i] == y_res[j]:
            file_name = output_dir + inner_names[i] + "/" + str(j) + ".jpg"
            result = cv2.imwrite(file_name, X_res[j])
            if not result:
                print("Error saving the file", file_name)
                
