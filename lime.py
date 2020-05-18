#%% Import libraries
import numpy as np
import keras
from keras.applications.imagenet_utils import decode_predictions

import skimage.segmentation
import skimage.io
import sklearn.metrics
from sklearn.linear_model import LinearRegression

import copy

#%%
#Xi = skimage.io.imread("https://arteagac.github.io/blog/lime_image/img/cat-and-dog.jpg")
Xi = skimage.io.imread("https://image.shutterstock.com/image-photo/little-girl-maltese-puppy-outdoor-260nw-1431006791.jpg")
Xi = skimage.transform.resize(Xi, (299,299)) 
Xi = (Xi - 0.5)*2 #Inception pre-processing
skimage.io.imshow(Xi / 2 + 0.5) # Show image before inception preprocessing

#%% Predict class for image using InceptionV3
np.random.seed(222)
inceptionV3_model = keras.applications.inception_v3.InceptionV3() #Load pretrained model
preds = inceptionV3_model.predict(Xi[np.newaxis,:,:,:])
top_pred_classes = preds[0].argsort()[-5:][::-1] # Save ids of top 5 classes
decode_predictions(preds)[0] #Print top 5 classes

"""
Step 1: Generate random perturbations for input image
"""
#%% Generate segmentation for image
superpixels = skimage.segmentation.quickshift(Xi, kernel_size=4, max_dist=200, ratio=0.2)
num_superpixels = np.unique(superpixels).shape[0]
skimage.io.imshow(skimage.segmentation.mark_boundaries(Xi / 2 + 0.5, superpixels))

#%% Generate perturbations
num_perturb = 150
perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))

#%% Create function to apply perturbations to images
def perturb_image(img, perturbation, segments): 
    active_pixels = np.where(perturbation == 1)[0]
    mask = np.zeros(segments.shape)
    for active in active_pixels:
        mask[segments == active] = 1 
    
    perturbed_image = copy.deepcopy(img)
    perturbed_image = perturbed_image * mask[:,:,np.newaxis]
    return perturbed_image

#%% Show example of perturbations
print(perturbations[0])
skimage.io.imshow(perturb_image(Xi / 2 + 0.5, perturbations[0], superpixels))

"""
Step 2 Predict class for perturbations
"""
predictions = []
idx = 0
for pert in perturbations:
    perturbed_img = perturb_image(Xi, pert, superpixels)
    pred = inceptionV3_model.predict(perturbed_img[np.newaxis,:,:,:])
    predictions.append(pred)
    print("%d / %d processed." % (idx + 1, perturbations.shape[0]))
    idx += 1

predictions = np.array(predictions)
print(predictions.shape)

"""
Step 3: Compute weights (importance) for the perturbations
"""
#%% Compute distances to original image
original_image = np.ones(num_superpixels)[np.newaxis,:] #Perturbation with all superpixels enabled
distances = sklearn.metrics.pairwise_distances(perturbations, original_image, metric='cosine').ravel()
print(distances.shape)
distances.min()
distances.max()

#%% Transform distances to a value between 0 an 1 (weights) using a kernel function
kernel_width = 0.25
weights = np.sqrt(np.exp(-(distances**2) / kernel_width**2)) #Kernel function
print(weights.shape)

"""
Step 4: Fit a explainable linear model using the perturbations, predictions and weights
"""
#%% Estimate linear model
class_to_explain = top_pred_classes[0] #Labrador class
simpler_model = LinearRegression()
simpler_model.fit(X=perturbations, y=predictions[:,:,class_to_explain], sample_weight=weights)
coeff = simpler_model.coef_[0]

#%% Use coefficients from linear model to extract top features
num_top_features = 4
top_features = np.argsort(coeff)[-num_top_features:] 

#%% Show only the superpixels corresponding to the top features
mask = np.zeros(num_superpixels) 
mask[top_features] = True #Activate top superpixels
skimage.io.imshow(perturb_image(Xi / 2 + 0.5, mask, superpixels))