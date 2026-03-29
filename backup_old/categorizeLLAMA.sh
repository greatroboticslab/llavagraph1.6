# define our variables
MODELPATH="/data/ilminur/12.14llava/llavagraph1.5/models_setup/Llama-3.2-3B-Instruct"

mkdir -p results/llama_V6


# random noise
python eval1.26/categorizeLLAMA_V6.py --model-path $MODELPATH --conversation-file results/llava_V6/randomNoise.json --output-file results/llama_V6/randomNoise.json 


# sine waves
python eval1.26/categorizeLLAMA_V6.py --model-path $MODELPATH --conversation-file results/llava_V6/sineWave.json --output-file results/llama_V6/sineWave.json 

#square waves
python eval1.26/categorizeLLAMA_V6.py --model-path $MODELPATH --conversation-file results/llava_V6/squareWave.json --output-file results/llama_V6/squareWave.json 
