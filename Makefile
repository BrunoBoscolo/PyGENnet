# Compiler and flags
CC = gcc
CFLAGS = -Iinclude -Wall -O3
LDFLAGS = -lm

# Source files and object files
SRCS = src/main.c src/neural_network.c src/evolution.c
OBJS = $(SRCS:.c=.o)

# Target executable
TARGET = main

# Default rule
all: $(TARGET)

# Rule to link the object files into the target executable
$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

# Rule to compile source files into object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
