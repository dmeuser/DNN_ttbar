# Small script to convert h5 to onnx
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
import argparse
import onnx
import keras2onnx

if __name__ == "__main__":
   parser = argparse.ArgumentParser()

   parser.add_argument('inputModel', type=str, help='<Required> h5 input model, which will be converted')

   args = parser.parse_args()
   
   if args.inputModel.endswith(".h5")==False:
      print("%s is no valid h5 file",args.inputModel)
      exit

   model = load_model(args.inputModel)
   onnx_model_name = args.inputModel.replace(".h5",".onnx")
   onnx_model = keras2onnx.convert_keras(model, model.name)
   onnx.save_model(onnx_model, onnx_model_name)
