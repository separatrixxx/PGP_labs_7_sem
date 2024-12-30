nvcc --std=c++11 -Werror cross-execution-space-call -lm kp.cu -o kp
./kp < best_config.txt
mkdir -p images

if [ "$(ls -A images 2>/dev/null)" ]; then
    rm images/*
fi

for file in data/*
do 
    f=${file#data/}
    echo "Converting $file to images/${f%.data}.jpg"
    python3 converter/conv.py "$file" "images/${f%.data}.jpg"
done
