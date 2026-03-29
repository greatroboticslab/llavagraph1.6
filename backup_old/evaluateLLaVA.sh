# define our variables
# LoRA adapter folder
MODELPATH="/data/ilminur/12.14llava/llavagraph1.5/checkpointsV6"

# ADD THIS: This is the original base model you used for training
MODELBASE="/data/ilminur/12.14llava/llavagraph1.5/models_setup/llava-v1.5-7b"

IMAGEFOLDER="/data/ilminur/12.14llava/llavagraph1.5/eval1.26/test"


mkdir -p results/llava_V6


# random noise
python eval1.26/evaluateLLaVA.py --model-path $MODELPATH --model-base $MODELBASE --image-folder $IMAGEFOLDER/RandomNoise --output-file results/llava_V6/randomNoise.json 

# sine waves
python eval1.26/evaluateLLaVA.py --model-path $MODELPATH --model-base $MODELBASE --image-folder $IMAGEFOLDER/SineWave --output-file results/llava_V6/sineWave.json 

#square waves
python eval1.26/evaluateLLaVA.py --model-path $MODELPATH --model-base $MODELBASE --image-folder $IMAGEFOLDER/SquareWave --output-file results/llava_V6/squareWave.json 

