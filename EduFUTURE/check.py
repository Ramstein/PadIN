from google.colab import files

files.upload()

# Let's make sure the kaggle.json file is present.
!ls - lha kaggle.json

# Next, install the Kaggle API client.
!pip install - q kaggle

# The Kaggle API client expects this file to be in ~/.kaggle,
# so move it there.
!mkdir - p ~ /.kaggle !cp kaggle.json ~ /.kaggle /

# This permissions change avoids a warning on Kaggle tool startup.
!chmod 600 ~ /.kaggle / kaggle.json


# List available datasets.
!kaggle datasets list

# Copy the stackoverflow data set locally.

!kaggle competitions download - c severstal - steel - defect - detection
from zipfile import ZipFile

'''preparing data required imports'''
import numpy as np
import os, copy, cv2, random, pickle, datetime



path = '/content'
fileNames = ['train_images.zip', 'test_images.zip', 'train.csv.zip']

for fileName in fileNames:
    if os.path.isfile(os.path.join(path, fileName)):
        if not os.path.isdir(os.path.join(path, fileName.split('.')[0])):
            os.mkdir(os.path.join(path, fileName.split('.')[0]))

            print('Extracting the file: ', fileName)

            with ZipFile(os.path.join(path, fileName), 'r') as zipObj:  # extracts in the sample_data/PetImages/Cat & Dog
                # Extract all the contents of zip file in different directory
                zipObj.extractall(os.path.join(path, fileName.split('.')[0]))
    else:
        print('File is not available.')


train_img = '/content/train_images'
test_img = '/content/test_images'
train_csv = '/content/train/train.csv'
sample_submission = '/content/sample_submission.csv'
!apt install plotly
!pip install mlxtend


# some basic imports
import pandas as pd
import numpy as np
import os
import cv2
# visualization
import matplotlib.pyplot as plt
# plotly offline imports
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly import subplots
import plotly.express as px
import plotly.figure_factory as ff
from plotly.graph_objs import *
from plotly.graph_objs.layout import Margin, YAxis, XAxis
init_notebook_mode()

#frequent pattern mining
from mlxtend.frequent_patterns import fpgrowth
# path where all the training images are


pd.read_csv(train_csv).head()


# load full data and label no mask as -1
train_df = pd.read_csv(train_csv).fillna(-1)
pd.read_csv(train_csv).head()

# image id and class id are two seperate entities and it makes it easier to split them up in two columns
train_df['ImageId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
train_df['ClassId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])
# lets create a dict with class id and encoded pixels and group all the defaults per image
train_df['ClassId_EncodedPixels'] = train_df.apply(lambda row: (row['ClassId'], row['EncodedPixels']), axis = 1)
grouped_EncodedPixels = train_df.groupby('ImageId')['ClassId_EncodedPixels'].apply(list)


# picking up 10 examples with at two faults for visualization
examples = []
for r in grouped_EncodedPixels.iteritems():
    if (len([x[1] for x in r[1] if x[1] != -1]) == 2) and (len(examples) < 10):
        examples.append(r[0])

# the masks are obviously encoded. we use the following function to
#decode the masks. I picked this function up from a Kaggle kernel,
# so thanks to the authors of https://www.kaggle.com/robertkag/rle-to-mask-converter
def rleToMask(rleString,height,width):
  rows,cols = height,width
  if rleString == -1:
      return np.zeros((height, width))
  else:
      rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
      rlePairs = np.array(rleNumbers).reshape(-1,2)
      img = np.zeros(rows*cols,dtype=np.uint8)
      for index,length in rlePairs:
        index -= 1
        img[index:index+length] = 255
      img = img.reshape(cols,rows)
      img = img.T
      return img


# visualize steel image with four classes of faults in seperate columns
def viz_two_class_from_path(img_path, img_id, masks, convert_to_float=False):
    img = cv2.imread(os.path.join(img_path, img_id))
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20,10))
    cmaps = ["Reds", "Blues", "Greens", "Purples"]
    axid = 0
    for idx, mask in enumerate(masks):
        class_id = idx + 1
        if mask == -1:
            pass
        else:
            mask_decoded = rleToMask(mask, 256, 1600)
            ax[axid].get_xaxis().set_ticks([])
            ax[axid].get_yaxis().set_ticks([])
            ax[axid].text(0.25, 0.25, 'Image Id: %s - Class Id: %s' % (img_id, class_id), fontsize=12)
            ax[axid].imshow(img)
            ax[axid].imshow(mask_decoded, alpha=0.15, cmap=cmaps[idx])
            axid += 1


# visualize the image we picked up earlier with mask
for example in examples:
    img_id = examples
    mask_1, mask_2, mask_3, mask_4 = grouped_EncodedPixels[example]
    masks = [mask_1[1], mask_2[1], mask_3[1], mask_4[1]]
    viz_two_class_from_path(train_img, example, masks)


# visualize steel image with four classes of faults in seperate columns
def viz_one_class_from_path(img_path, img_id, mask, class_id, convert_to_float=False):
    img = cv2.imread(os.path.join(img_path, img_id))
    mask_decoded = rleToMask(mask, 256, 1600)
    fig, ax = plt.subplots(figsize=(20, 10))
    cmaps = ["Reds", "Blues", "Greens", "Purples"]
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.imshow(img)
    ax.imshow(mask_decoded, alpha=0.15, cmap=cmaps[int(class_id) - 1])


def viz_per_class(train_df, class_id, sample_size=5):
    class_samples = train_df[(train_df['ClassId'] == class_id) & (train_df['EncodedPixels'] != -1)].sample(sample_size)
    class_img_ids = class_samples['ImageId'].values
    class_encoded_masks = class_samples['EncodedPixels'].values

    for img_id, mask in zip(class_img_ids, class_encoded_masks):
        viz_one_class_from_path(train_img, img_id, mask, class_id)


# ****#### Class 1 Defects
viz_per_class(train_df, '1', 2)

# #### Class 2 Defects

viz_per_class(train_df, '2', 2)

#### Class 3
viz_per_class(train_df, '3', 2)

#### Class 4
viz_per_class(train_df, '4', 2)


# Missing Labels & Defect Per Image
# calculate sum of the pixels for the mask per class id
train_df['mask_pixel_sum'] = train_df.apply(lambda x: rleToMask(x['EncodedPixels'], width=1600, height=256).sum(), axis=1)


# calculate the number of pictures without any label what so ever
annotation_count = grouped_EncodedPixels.apply(lambda x: 1 if len([1 for y in x if y[1] != -1]) > 0 else 0).value_counts()
annotation_count_labels = ['No Label' if x == 0 else 'Label' for x in annotation_count.index]
# calculate number of defects per image
defect_count_per_image = grouped_EncodedPixels.apply(lambda x: len([1 for y in x if y[1] != -1])).value_counts()
defect_count_labels = defect_count_per_image.index


trace0 = Bar(x=annotation_count_labels, y=annotation_count, name = 'Labeled vs Not Labeled')
trace1 = Bar(x=defect_count_labels, y=defect_count_per_image, name = 'Defects Per Image')
fig = subplots.make_subplots(rows=1, cols=2)
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=400, width=900, title='Defect Labels and Defect Frequency Per Image')
iplot(fig)


# ## Mask Size Per Defect Class
class_ids = ['1','2','3','4']
mask_count_per_class = [train_df[(train_df['ClassId']==class_id)&(train_df['mask_pixel_sum']!=0)]['mask_pixel_sum'].count() for class_id in class_ids]
pixel_sum_per_class = [train_df[(train_df['ClassId']==class_id)&(train_df['mask_pixel_sum']!=0)]['mask_pixel_sum'].sum() for class_id in class_ids]


# Create subplots: use 'domain' type for Pie subplot
fig = subplots.make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

fig.add_trace(Pie(labels=defect_count_labels, values=defect_count_per_image, name="Defects Count"), 1, 1)
fig.add_trace(Pie(labels=class_ids, values=mask_count_per_class, name="Mask Count"), 1, 2)
# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent+name")

fig.update_layout(
    title_text="Steel Defect Mask & Pixel Count",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Mask', x=0.18, y=0.5, font_size=20, showarrow=False),
                 dict(text='Pixel', x=0.80, y=0.5, font_size=20, showarrow=False)])
fig.show()


# plot a histogram and boxplot combined of the mask pixel sum per class Id
fig = px.histogram(train_df[train_df['mask_pixel_sum']!=0][['ClassId','mask_pixel_sum']],
                   x="mask_pixel_sum", y="ClassId", color="ClassId", marginal="box")

fig['layout'].update(title='Histogram and Boxplot of Sum of Mask Pixels Per Class')

fig.show()

## Frequent Pattern Mining

# create a series with fault classes
class_per_image = grouped_EncodedPixels.apply(lambda encoded_list: [x[0] for x in encoded_list if x[1] != -1])


# create a list of dict with count of each fault class
class_per_image_list = []
for r in class_per_image.iteritems():
    class_count = {'1':0,'2':0,'3':0,'4':0}
    # go over each class and
    for image_class in r[1]:
        class_count[image_class] = 1
    class_per_image_list.append(class_count)


# do FP calculation with all image
class_per_image_df = pd.DataFrame(class_per_image_list)
class_fp_df = fpgrowth(class_per_image_df, use_colnames=True, min_support=0.001)
class_fp_df = class_fp_df.sort_values(by=['support'])


# subset to images with at least one mask
class_per_fault_image_df = class_per_image_df[(class_per_image_df.T != 0).any()]
class_fp_faulty_df = fpgrowth(class_per_fault_image_df, use_colnames=True, min_support=0.001)
class_fp_faulty_df = class_fp_faulty_df.sort_values(by=['support'])


# a simple function to do horizontal barplot
def bar_plot_h(x, y, title, x_label, y_label):
    y_pos = np.arange(len(y))
    plt.barh(y_pos, x)
    plt.yticks(y_pos, y)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)

plt.figure(figsize=(15,5))
# plot for FP for all images
plt.subplot(1,2,1)
combinations = [', '.join(x) for x in class_fp_df['itemsets'].values]
support = class_fp_df['support'].values
bar_plot_h(support, combinations, 'Fault Classes Appearing Frequently - All Samples', 'Fault Classes', 'Support')
# plot for FP for images with at least one fault
plt.subplot(1,2,2)
combinations = [', '.join(x) for x in class_fp_faulty_df['itemsets'].values]
support = class_fp_faulty_df['support'].values
bar_plot_h(support, combinations, 'Fault Classes Appearing Frequently - Faulty Samples Only', 'Fault Classes', 'Support')

