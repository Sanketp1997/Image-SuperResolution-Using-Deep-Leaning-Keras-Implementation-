import models
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description="Up-Scales an image using Image Super Resolution Model")
parser.add_argument("imgpath", type=str, nargs="+", help="Path to input image")

args = parser.parse_args()
#print ("hello")
with tf.device('/CPU:0'):
    path = args.imgpath
    for p in path:
	    model = models.DistilledResNetSR(2)
        
model.upscale(p, save_intermediate="true", mode="patch", patch_size=8, suffix="out")