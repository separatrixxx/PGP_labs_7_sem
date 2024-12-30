NVCC = /usr/local/cuda/bin/nvcc
MPICPP = /usr/local/bin/mpic++

NVCCFLAGS = --std=c++11 -Werror cross-execution-space-call -lm
CFLAGS = -std=c++11

TARGET = kp

SRC = kp.cu

INCLUDE_DIR = libs

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -I$(INCLUDE_DIR) $(SRC) -o $(TARGET)

clean:
	rm -rf $(OUTPUT_DIR)

run: all
	./$(TARGET)
