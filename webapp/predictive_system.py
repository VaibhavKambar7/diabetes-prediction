import numpy as np
import pickle

# loading the saved model
#Replace this directory with the directory of the downloaded trained model in your system from your notebook.
loaded_model = pickle.load(open('C:/Users/vaibhav/Desktop/diabetes_mini_proj/trained_model.sav','rb'))


#input_data = (0,112,40,35,148,43.1,2.288,33)
input_data = (2, 120, 70, 30, 80, 25.5, 0.45, 35)


input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print("The Person IS NOT DIABETIC")
else:
  print("The Person IS DIABETIC")
